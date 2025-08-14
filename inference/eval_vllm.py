import math as pymath
import json, os, re, sys, math, argparse
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset, get_dataset_config_names
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    GenerationConfig,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
)
from pathlib import Path
from vllm import LLM, SamplingParams
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

##############################################################################
# 1. Prompt builder & parser & extractor & matcher                           #
##############################################################################
from answer_extractor import AnswerExtractor
from step_parser import StepParser, build_chat_messages, build_chat_messages2
from answer_matcher import MathAnswerScorer
from config import PRMConfig
from prm_model import ProcessRewardModel

##############################################################################
# 2. Dataset loaders & field mapping                                         #
##############################################################################
FIELD_MAP: Dict[str, Tuple[str, str]] = { # dataset name  : (question_field, answer_field)
    "gsm8k": ("question", "answer"),
    "math": ("problem", "solution"),
    "omni": ("problem", "answer"),
    "olympiad": ("question", "final_answer"),
    "aime": ("problem", "answer"),
}

def load_olympiadbench_english(split: str = "train"):
    all_cfgs = get_dataset_config_names("Hothan/OlympiadBench")
    en_cfgs = [cfg for cfg in all_cfgs if "_en_" in cfg or cfg.endswith("_en")]
    ds_list = []
    for cfg in en_cfgs:
        try:
            ds = load_dataset("Hothan/OlympiadBench", cfg, split=split)
            ds_list.append(ds)
        except Exception as e:
            print(f"⚠️  {cfg} load failed: {e}")
    if len(ds_list) == 0:
        raise ValueError("Fail to load English configs")
    full_ds = concatenate_datasets(ds_list)
    return full_ds

def get_loader(ds_name: str, split: str, batch_size: int):
    """Return torch DataLoader yielding (idx, question, gold_answer)."""
    if ds_name == "math":
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
    elif ds_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
    elif ds_name == "omni":
        ds = load_dataset("KbsdJames/Omni-MATH", split=split)
    elif ds_name == "olympiad":
        ds = load_olympiadbench_english(split)
    elif ds_name =="aime":
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    else:
        raise ValueError(f"Unsupported dataset {ds_name}")

    q_key, a_key = FIELD_MAP[ds_name]
    def collate(indices):
        items = [ds[i] for i in indices]
        idxs = indices
        qs = [item[q_key] for item in items]
        golds = [item[a_key] for item in items]
        return idxs, qs, golds

    return DataLoader(range(len(ds)), batch_size=batch_size, shuffle=False, collate_fn=collate), len(ds)

##############################################################################
# 3. Inference utilities (Naive, Best-Of-N, Majority Voting)                 #
##############################################################################
def batched_generate_vllm(prompts: List[str], llm: "LLM", max_tokens: int, N: int=1) -> List[str]:
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.2, top_p=0.9, n=N) # naive one trajectory decoding
    outs = llm.generate(prompts, sp)
    results = []
    for o in outs:
        texts = [cand.text for cand in o.outputs]
        results.append(texts)
    return results

# 3‑A. Best‑of‑N **path‑level** search
def best_of_n_paths_batch(qs: List[str], tokenizer, llm: "LLM", ds_name: str, parser: StepParser, prm, prm_tok, device: torch.device, N: int = 8, max_tokens: int = 2048) -> List[str]:
    prompts = [build_chat_messages(q, tokenizer, ds_name) for q in qs]
    decoded_lists = batched_generate_vllm(prompts, llm, max_tokens, N=N)
    diagnostics: List[Dict[str, Any]] = []
    chosen_texts: List[str] = []
    for decodes in decoded_lists:
        cand_info = []
        best_body, best_score = "", -1e9
        for d in decodes:
            body = d.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            raw_steps = parser.parse(body)
            steps = parser.parse_clean(raw_steps)
            steps = [s for s in steps if s and s.strip()]
            if len(steps) == 0:
                continue
            s = score_path(steps, prm, prm_tok, device)
            cand_info.append({"body": body, "score": s, "n_steps": len(steps)})
            if s > best_score:
                best_score, best_body = s, body
        # fallback if all decodes invalid
        chosen_texts.append(best_body if best_body else decodes[0])
        diagnostics.append({"chosen": best_body, "chosen_score": best_score, "candidates": cand_info})
    return chosen_texts, diagnostics

# 3‑B. Best‑of‑N **step‑level** search (Monte‑Carlo greedy)
ANSWER_KEYWORDS = ("answer", "final answer", "therefore", "result", "the final answer is",)
def generate_stepwise_with_prm(question: str, tokenizer, llm: "LLM", ds_name: str, parser: StepParser, prm, prm_tok, device: torch.device, N: int = 8, max_tokens_per_step: int = 128, max_total_steps: int = 30) -> str:
    steps: List[str] = []
    for step_idx in range(max_total_steps):
        messages = []
        # system prompt & few‑shot & problem statement
        prompt_base = build_chat_messages2(question, tokenizer, ds_name)
        # Remove the problematic line that tries to apply chat template to empty list
        base_text = prompt_base + "\n" + "\n".join(steps)
        prompt = base_text + "\n"  # generation prompt after last newline
        # Generate N continuations of at most max_tokens_per_step tokens
        cand_list = batched_generate_vllm([prompt], llm, max_tokens_per_step, N=N)[0]
        best_cand, best_score = None, -1e9
        for cand in cand_list:
            line = cand.split("\n")[0].strip()
            if not line:
                continue
            s = score_step(line, prm, prm_tok, device)
            if s > best_score:
                best_score, best_cand = s, line
        if best_cand is None:
            break  # unable to proceed
        steps.append(best_cand)
        lower_line = best_cand.lower()
        if any(lower_line.startswith(k) for k in ANSWER_KEYWORDS):
            break
    return "\n".join(steps)

def best_of_n_steps_batch(qs: List[str], tokenizer, llm: "LLM", ds_name: str, parser: StepParser,
    prm, prm_tok, device: torch.device, N: int = 8, max_tokens_per_step: int = 384) -> List[str]:
    return [generate_stepwise_with_prm(q, tokenizer, llm, ds_name, parser, prm, prm_tok, device, N, max_tokens_per_step) for q in qs]

# 3-C. Majority voting (majority voting with weighted PRM)
def _softmax_weights(scores: List[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    max_s = max(scores)
    exps = [math.exp((s - max_s) / max(1e-6, temperature)) for s in scores]
    z = sum(exps)
    return [e / z for e in exps] if z > 0 else [1.0 / len(scores)] * len(scores)

@torch.no_grad()
def generate_candidates_batch(qs: List[str], tokenizer, llm: "LLM", ds_name: str, parser: StepParser, extractor: AnswerExtractor,
    prm, prm_tok, device: torch.device, N: int = 4, max_tokens: int = 3096) -> List[List[Dict[str, Any]]]:
    prompts = [build_chat_messages(q, tokenizer, ds_name) for q in qs]
    decoded_lists = batched_generate_vllm(prompts, llm, max_tokens, N)
    all_cands = []
    for decodes in decoded_lists:
        cands = []
        for d in decodes:
            body = d.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            raw_steps = parser.parse(body)
            steps = parser.parse_clean(raw_steps)
            prm_score = score_path(steps, prm, prm_tok, device) if steps else float('-inf')
            ans = extractor.extract_pred_answer(body.split("Answer:")[-1])
            cands.append({"body": body, "answer": ans, "steps": steps, "prm_score": prm_score})
        all_cands.append(cands)
    return all_cands

def choose_by_majority(cands: List[Dict[str, Any]], use_prm_weights: bool = False, temperature: float = 1.0, tie_break: str = "prm") -> Tuple[str, Dict[str, Any]]:
    if not cands:
        return "", {"reason": "no_candidates"}
    weights = _softmax_weights([c['prm_score'] for c in cands], temperature) if use_prm_weights else [1.0] * len(cands)
    vote_by_answer = defaultdict(float)
    members = defaultdict(list)
    for i, (cand, w) in enumerate(zip(cands, weights)):
        vote_by_answer[cand['answer']] += w
        members[cand['answer']].append(i)
    max_votes = max(vote_by_answer.values(), default=0)
    winners = [a for a, v in vote_by_answer.items() if abs(v - max_votes) < 1e-12]
    if not winners:
        idx = max(range(len(cands)), key=lambda i: cands[i]['prm_score']) if use_prm_weights else 0
        return cands[idx]['body'], {"reason": "fallback"}
    win_ans = winners[0] if len(winners) == 1 else (
        max(winners, key=lambda a: max(cands[i]['prm_score'] for i in members[a])) if tie_break == 'prm' else winners[0]
    )
    idxs = members[win_ans]
    chosen_idx = max(idxs, key=lambda i: cands[i]['prm_score']) if use_prm_weights else idxs[0]
    diag = {"votes": dict(vote_by_answer), "use_prm_weights": use_prm_weights, "temperature": temperature, "winning_answer": win_ans, "chosen_score": cands[chosen_idx]['prm_score']}
    return cands[chosen_idx]['body'], diag

def majority_vote_batch(qs: List[str], tokenizer, llm: "LLM", ds_name: str, parser: StepParser, extractor: AnswerExtractor, prm, prm_tok, device: torch.device,
    N: int, max_tokens: int, use_prm_weights: bool, temperature: float) -> Tuple[List[str], List[Dict[str, Any]]]:
    batch_cands = generate_candidates_batch(qs, tokenizer, llm, ds_name, parser, extractor, prm, prm_tok, device, N, max_tokens)
    texts, diags = [], []
    for cands in batch_cands:
        txt, dg = choose_by_majority(cands, use_prm_weights, temperature)
        texts.append(txt); diags.append(dg)
    return texts, diags

##############################################################################
# 4. PRM utilities                                                           #
##############################################################################
class MinimalPRM(nn.Module):
    def __init__(self, backbone: nn.Module, *, mlp_ratio: int = 4, value_head_prefix: str = "value_head"):
        super().__init__()
        self.backbone = backbone
        hidden = self.backbone.config.hidden_size
        mlp_hidden = max(1, hidden // mlp_ratio)
        head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden, bias=False, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1, bias=False, dtype=torch.bfloat16),
        )
        setattr(self, value_head_prefix, head)
        self.value_head_prefix = value_head_prefix

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if attention_mask is None:
            pad_id = getattr(self.backbone.config, "pad_token_id", 0)
            attention_mask = (input_ids != pad_id).long()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1]
        eos_idx = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(-1, keepdim=True)
        last_hidden = last_hidden.float()
        value_head = getattr(self, self.value_head_prefix)
        if last_hidden.dtype != value_head[0].weight.dtype:
            last_hidden = last_hidden.to(value_head[0].weight.dtype)
        values = value_head(last_hidden).squeeze(-1)
        reward = values.gather(1, eos_idx).squeeze(1)
        return reward

@torch.no_grad()
def load_prm_case(prm_root: str, base_model_name: str, device: torch.device, *, mlp_ratio: int = 4, value_head_prefix: str = "value_head", adapter_subdir: str = "adapter"):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb,
    )
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    adapter_dir = os.path.join(prm_root, adapter_subdir)
    adapter_cfg = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        raise FileNotFoundError(f"Case B expects LoRA adapters at {adapter_dir} (adapter_config.json not found)")

    backbone = PeftModel.from_pretrained(backbone, adapter_dir)
    print(f"[PRM/LM] LoRA adapters attached from: {adapter_dir}")

    model = MinimalPRM(backbone, mlp_ratio=mlp_ratio, value_head_prefix=value_head_prefix).to(device).eval()

    candidates = [
        os.path.join(prm_root, "final_model", "model.safetensors"),
        os.path.join(prm_root, "value_head.safetensors"),
        os.path.join(prm_root, "value_head.bin"),
        os.path.join(prm_root, "model.safetensors"),
        os.path.join(prm_root, "pytorch_model.bin"),
    ]
    loaded = False
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            if p.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load
                sd = safe_load(p)
            else:
                sd = torch.load(p, map_location="cpu")
            head_keys = {}
            for k, v in sd.items():
                if k.startswith(f"value_head."):
                    new_key = k.replace("value_head.", "")
                    head_keys[new_key] = v
            if head_keys:
                missing, unexpected = getattr(model, "value_head").load_state_dict(head_keys, strict=False)
                print(f"[PRM] value head loaded from {os.path.basename(p)}; missing={missing}, unexpected={unexpected}")
                loaded = True
                break
        except Exception as e:
            print(f"[PRM] reading {p} failed: {e}")
    if not loaded:
        print("[PRM] ⚠️  value head weights not found; using randomly initialized head.")
    return model, tok

@torch.no_grad()
def load_prm_lm(checkpoint_path: str, base_model_name: str, device: torch.device, *, head_prefix: str = "value_head", mlp_ratio: int = 4):
    adapter_dir = os.path.join(checkpoint_path, "adapter")
    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        if base_model_name is None:
            raise ValueError("Case B requires --prm_base_model to rebuild the backbone.")
        try:
            return load_prm_case(checkpoint_path, base_model_name, device, mlp_ratio=mlp_ratio, value_head_prefix=head_prefix)
        except Exception as e:
            print("[PRM] Case B load failed:", e)
    raise RuntimeError(
        "No compatible PRM checkpoint layout found. Expected either a unified state_dict (Case A) "
        "or an adapter directory with adapter_config.json (Case B)."
    )

def load_prm_scr(checkpoint_path: str, base_model_name_or_cfg: Union[str, PretrainedConfig], device: torch.device,):
    # ---- 1) Load checkpoint ----
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # ---- 2) Resolve PRM config (back/forward compatible: "cfg" or "config") ----
    prm_cfg = PRMConfig(**ckpt.get("cfg", {}))
    # ---- 3) Resolve base hidden size & base name_or_path ----
    if hasattr(base_model_name_or_cfg, "hidden_size"):
        base_cfg = base_model_name_or_cfg
    else:
        base_cfg = AutoConfig.from_pretrained(
            base_model_name_or_cfg, trust_remote_code=True
        )
    hidden_size = getattr(base_cfg, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            f"Base config has no hidden_size: {getattr(base_cfg, 'name_or_path', base_cfg)}"
        )
    base_name_or_path = getattr(base_cfg, "name_or_path", None)
    if base_name_or_path is None and isinstance(base_model_name_or_cfg, str):
        base_name_or_path = base_model_name_or_cfg
    # ---- 4) Build PRM and load weights (robust to different keys) ----
    prm = ProcessRewardModel(hidden_size, cfg=prm_cfg)
    state = ckpt.get("prm_state") or ckpt.get("state_dict") or ckpt
    missing, unexpected = prm.load_state_dict(state, strict=False)
    if missing:
        print(f"[PRM/LR] Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[PRM/LR] Unexpected keys in state_dict: {unexpected}")
    prm = prm.float().to(device).eval()
    # ---- 5) Tokenizer (prefer checkpoint's tokenizer_config, else fallback to base) ----
    tok_cfg = ckpt.get("tokenizer_config") or {}
    tok_name_or_path = tok_cfg.get("name_or_path") or base_name_or_path
    if tok_name_or_path is None:
        raise ValueError(
            "Could not resolve tokenizer name_or_path from checkpoint or base model."
        )
    prm_tokenizer = AutoTokenizer.from_pretrained(
        tok_name_or_path, use_fast=False, trust_remote_code=True
    )
    return prm, prm_tokenizer

def load_prm_hf(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except:  # token‑classification or generic backbone with custom head
        model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer

_BASELINE_CACHE = {}
def _get_baseline_for_tokenizer(tok, device: torch.device):
    name = getattr(tok, "name_or_path", None) \
        or tok.__dict__.get("init_kwargs", {}).get("name_or_path")
    if name in _BASELINE_CACHE and _BASELINE_CACHE[name] is not None:
        return _BASELINE_CACHE[name]

    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16).eval()
    _BASELINE_CACHE[name] = model
    return model

def score_step(step: str, prm, prm_tok, device: torch.device) -> float:
    if step is None or len(step.strip()) == 0:
        return -1e9
    cls_name = prm.__class__.__name__

    # ---- 1) SCR: PRM expects hidden‑state embedding ----------------------------------
    if cls_name == "ProcessRewardModel":
        toks = prm_tok(step.strip(), return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        baseline = _get_baseline_for_tokenizer(prm_tok, device)
        out = baseline(**toks, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :].float()  # (B, H)
        score_t = prm(last_hidden).squeeze(-1)
        return float(score_t.item())
    # ---- 2) LM: MinimalPRM with value_head -------------------------------------------
    if hasattr(prm, "value_head_prefix"):
        inputs = prm_tok(step, truncation=True, return_tensors="pt")
        inputs = {k: v.long().to(device) for k, v in inputs.items()}
        logits = prm(**inputs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim > 1:
            logits = logits.squeeze()
        return float(logits.item())
    # ---- 3) HF: either sequence‑ or token‑classification head -------------------------
    with torch.no_grad():
        toks = prm_tok(step, truncation=True, return_tensors="pt").to(device)
        outputs = prm(**toks)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits  # type: ignore[arg-type]
        # (bs, num_labels) or (bs, seq_len, num_labels)
        if logits.ndim == 3:
            logits_last = logits[:, -1, :]  # positive label at last token
        else:
            logits_last = logits  # sequence‑level classification
        probs = F.softmax(logits_last, dim=-1)
        # convention: label id 1 == "good / correct step"
        return float(probs[0, 1].item())

def score_path(steps: List[str], prm, prm_tok, device: torch.device) -> float:
    if len(steps) == 0:
        return -1e9
    scores = [score_step(s, prm, prm_tok, device) for s in steps]
    return sum(scores) / len(scores)

##############################################################################
# 5. Main – end‑to‑end evaluation                                            #
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "gsm8k", "omni", "olympiad"]) 
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--prm_type", choices=["lm", "scr", "hf"], default="scr")
    parser.add_argument("--max_samples", type=int, default=1000, help="0 = all")
    parser.add_argument("--search_mode", choices=["single","path_bon","step_bon","maj","wmaj"], default="path_bon")
    parser.add_argument("--reward_type", choices=["ori","contri","cmi", "pav", "orimi"], default="orimi")
    parser.add_argument("--N", type=int, default=8, help="Number of candidates per search")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Load Policy Model Using vllm ##
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.45, quantization="bitsandbytes", max_model_len=4096,)

    ## Load Math Dataset ##
    loader, total = get_loader(args.dataset, "test" if args.dataset != "olympiad" else "train", args.batch_size)
    max_n = total if args.max_samples == 0 else min(args.max_samples, total)

    ## Load PRM Model ##
    if args.search_mode == "single":
        print("No need PRM model for inference!")
        pass
    else: 
        if args.prm_type == "lm":
            prm_path = f"/home/leena/ccc_eval/mcts_prm/checkpoints/pt_lm/{args.reward_type}/final_model"
            prm, prm_tok = load_prm_lm(prm_path, args.model, device)
        elif args.prm_type == "hf":
            prm_hf = "Qwen/Qwen2.5-Math-PRM-7B"
            prm, prm_tok = load_prm_hf(prm_hf, device)
            print("I'm baseline model!")
        elif args.prm_type == "scr":
            prm_path = f"/home/leena/ccc_eval/mcts_prm/checkpoints/scratch/{args.reward_type}/best_prm.pt"
            base_model = "Qwen/Qwen2.5-Math-7B"
            prm, prm_tok = load_prm_scr(prm_path, base_model, device)
        else:
            raise ValueError(f"Unknown --prm_type {args.prm_type}")
        print(f"Finish Loading {args.reward_type} Model and {args.dataset} Dataset!")

    ## Helpers ##
    extractor = AnswerExtractor()
    scorer = MathAnswerScorer()
    parser_obj = StepParser()
    
    ## Evaluation loop ##
    correct = 0
    seen = 0
    mode = args.search_mode
    diagnostics_all = []  # Initialize diagnostics list
    pbar = tqdm(loader, total = math.ceil(max_n / args.batch_size))
    for idxs, qs, golds in pbar:
        if seen >= max_n:
            break
        take = min(len(qs), max_n - seen) # trim overflow inside batch
        qs, golds = qs[:take], golds[:take]

        if mode == "single":
            raw_batches = [build_chat_messages(q, tokenizer, args.dataset) for q in qs]
            decoded = batched_generate_vllm(raw_batches, llm, 3000, N=1)
            raw_texts = [d[0] for d in decoded]
        elif args.search_mode == "path_bon":
            raw_texts, diags = best_of_n_paths_batch(qs, tokenizer, llm, args.dataset, parser_obj, prm, prm_tok, device, args.N, 3000)
            diagnostics_all.extend(diags)
        elif args.search_mode == "step_bon":
            raw_texts = best_of_n_steps_batch(qs, tokenizer, llm, args.dataset, parser_obj, prm, prm_tok, device, args.N)
        elif args.search_mode in ("maj","wmaj"):
            use_w = (args.search_mode=="wmaj")
            raw_texts, diags = majority_vote_batch(qs, tokenizer, llm, args.dataset, parser_obj, extractor, prm, prm_tok, device,
                N=args.N, max_tokens=3000, use_prm_weights=use_w, temperature=1.0)
            diagnostics_all.extend(diags)
        else:
            raise ValueError(f"Unknown search_mode {mode}")

        # Extract answers & score
        pred_answers = [extractor.extract_pred_answer(rt.split("Answer:")[-1]) for rt in raw_texts]
        gold_answers = [extractor.extract_gold_answer(g, args.dataset) for g in golds]
        batch_corr = [scorer.answers_match(p, g) for p, g in zip(pred_answers, gold_answers)]
        correct += sum(batch_corr)
        seen += take
        pbar.set_postfix(acc=f"{correct/seen:.3%}")

    if args.prm_type == "hf":
        case = prm_hf
    else:
        case = args.reward_type

    print(f"\n====================== Summary ({case}) ======================")
    print(f"Dataset      : {args.dataset}")
    print(f"Search mode  : {args.search_mode} (N={args.N})")
    print(f"Correct      : {correct}")
    print(f"Samples seen : {seen}")
    print(f"Accuracy     : {correct/seen:.3%}")

    # save results/diagnostics for analysis
    summary = {
        "dataset": args.dataset,
        "search_mode": args.search_mode,
        "N": args.N,
        "correct": correct,
        "total": seen,
        "accuracy": correct/seen,
        "accuracy_str": f"{correct/seen:.3%}"
    }
    
    filename = f"./test/eval_summ_{args.prm_type}_{args.search_mode}_{args.reward_type}_{args.dataset}_1000.json"
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)

    if diagnostics_all:
        with open(f"./test/diag_{args.prm_type}_{args.search_mode}_{args.reward_type}_{args.dataset}_1000.json", "w", encoding="utf8") as f:
            json.dump(diagnostics_all, f, ensure_ascii=False, indent=2)
        print("Diagnostics saved to file")


if __name__ == "__main__":
    main()
