import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ---------- Project-local helpers (must exist in your repo) ----------
from answer_extractor import AnswerExtractor
from step_parser import StepParser

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# 1) Dataset field mapping & loaders
FIELD_MAP: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {
    "gsm8k":    ("question", "answer",        "answer"),         # GSM8K는 answer 본문에 풀이 + #### 정답
    "math":     ("problem",  "solution",      None),             # 최종답은 solution 내부에 있음
    "omni":     ("problem",  "solution",      "answer"),         # 풀이 + 별도 정답
    "olympiad": ("question", "solution",      "final_answer"),   # 풀이 + 별도 정답(대개 list/string)
    "aime":     ("problem",  "solution",      "answer"),         # 풀이 + 별도 정답
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
    if ds_name == "math":
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
    elif ds_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
    elif ds_name == "omni":
        ds = load_dataset("KbsdJames/Omni-MATH", split=split)
    elif ds_name == "olympiad":
        ds = load_olympiadbench_english(split)
    elif ds_name == "aime":
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    else:
        raise ValueError(f"Unsupported dataset {ds_name}")

    q_key, sol_key, ans_key = FIELD_MAP[ds_name]

    def _get(item: dict, key: Optional[str]) -> str:
        if key is None: return ""
        v = item.get(key, "")
        return "" if v is None else str(v)

    def collate(indices):
        items = [ds[i] for i in indices]
        idxs = indices
        qs   = [_get(items[i], q_key)   for i in range(len(items))]
        sols = [_get(items[i], sol_key) for i in range(len(items))]
        fins = [_get(items[i], ans_key) for i in range(len(items))]
        return idxs, qs, sols, fins
    
    return DataLoader(range(len(ds)), batch_size=batch_size, shuffle=False, collate_fn=collate), len(ds)

# 2) PRM loading & scoring
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
    print(f"[PRM/CaseB] LoRA adapters attached from: {adapter_dir}")

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

_BASELINE_CACHE = {}
def _get_baseline_for_tokenizer(tok, device: torch.device):
    name = getattr(tok, "name_or_path", None) \
        or tok.__dict__.get("init_kwargs", {}).get("name_or_path")
    if name in _BASELINE_CACHE and _BASELINE_CACHE[name] is not None:
        return _BASELINE_CACHE[name]

    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16).eval()
    _BASELINE_CACHE[name] = model
    return model

@torch.no_grad()
def score_step_text(step: str, prm, prm_tok, device: torch.device) -> float:
    if step is None or len(step.strip()) == 0:
        return -1e9
    # ---- SCR형(PRM이 텐서 임베딩을 기대) 분기: 클래스명으로 식별 ----
    if prm.__class__.__name__ == "ProcessRewardModel":
        # 1) 토큰화
        toks = prm_tok(step.strip(), return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        # 2) baseline LM (지연 로드 & 캐시)
        baseline = _get_baseline_for_tokenizer(prm_tok, device)
        # 3) hidden state 추출 (마지막 토큰)
        out = baseline(**toks, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :].float()  # (B, H)
        # 4) PRM 통과
        score_t = prm(last_hidden).squeeze(-1)                 # (B,)
        return float(score_t.item())
    # ---- LM형(PRM이 텍스트 입력을 기대) 기존 경로 ----
    inputs = prm_tok(step, truncation=True, return_tensors="pt")
    inputs = {k: v.long().to(device) for k, v in inputs.items()}
    logits = prm(**inputs)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.ndim > 1:
        logits = logits.squeeze()
    return float(logits.item())

# @torch.no_grad()
# def score_step_text(text: str, prm, prm_tok, device: torch.device) -> float:
#     if text is None or len(text.strip()) == 0:
#         return -1e9
#     enc = prm_tok(text, truncation=True, return_tensors="pt")
#     enc = {k: v.long().to(device) for k, v in enc.items()}
#     out = prm(**enc)
#     if isinstance(out, (tuple, list)):
#         out = out[0]
#     if out.ndim > 1:
#         out = out.squeeze()
#     return float(out.item())

@torch.no_grad()
def compute_step_rewards(prm, prm_tok, device: torch.device, prompt_prefix: str, steps: List[str]) -> List[float]:
    rewards: List[float] = []
    for s in steps:
        text = f"{prompt_prefix}\n{s}"
        rewards.append(score_step_text(text, prm, prm_tok, device))
    return rewards

# 3) Step extraction from gold solution (robust across datasets)
def extract_gold_steps_and_answer(ds_name: str, question: str, solution_text: str, final_text: str, extractor: AnswerExtractor, parser: StepParser,) -> Tuple[List[str], str]:
    ds  = (ds_name or "").lower()
    sol = (solution_text or "").strip()
    fin = (final_text or "").strip()

    # 1) Final answer: final → solution → empty
    gold_answer = None
    if fin:
        gold_answer = extractor.extract_gold_answer(fin, dataset=ds)
    if not gold_answer and sol:
        gold_answer = extractor.extract_gold_answer(sol, dataset=ds)
    if not gold_answer:
        gold_answer = fin or ""

    # 2) Steps from solution (now sentence-level by default)
    steps_clean = parser.parse_clean(sol, ds_name=ds, gold_answer=gold_answer, sentence_level=True) if sol else []

    if not steps_clean:
        if gold_answer:
            steps_clean = [f"The final answer is {gold_answer}."]
        elif sol:
            steps_clean = [sol]
        else:
            steps_clean = []

    steps = [f"Step {i+1}: {s}" for i, s in enumerate(steps_clean)]

    if ds == "gsm8k" and gold_answer:
        steps.append(f"Step {len(steps)+1}: The final answer is {gold_answer}.")

    # print("Parsed Steps:", steps)
    return steps, gold_answer


# 4) Perturbation utilities
SELF_REFLECTION_BANK = [
    "Let me think about this problem carefully.",
    "I need to check my calculations.",
    "This step seems important for the solution.",
    "Let me verify the previous step.",
    "I should double-check my work.",
    "This is a crucial part of the solution.",
    "Let me organize my thoughts.",
    "I need to be careful with the math.",
]

WRONG_STEP_BANK = [
    "5 + 3 = 9",
    "10 * 2 = 15",
    "20 / 4 = 6",
    "x/2 = 6, so x = 10",
    "2^2 = 5",
    "\\sqrt(36) = 5",
    "Perimeter of square side 5 = 15",
]

IRRELEVANT_BANK = [
    "The weather is nice today.",
    "I like mathematics very much.",
    "This reminds me of my school days.",
    "The sky is blue and beautiful.",
    "I should drink more water.",
    "Patterns exist in everything.",
    "I should make a grocery list.",
]

def renumber_steps(steps: List[str], start_idx: int = 0) -> List[str]:
    new_steps: List[str] = []
    for i in range(len(steps)):
        s = steps[i]
        content = s.split(":", 1)[1] if ":" in s else s
        new_steps.append(f"Step {start_idx + i + 1}:{content}")
    return new_steps

def create_perturbed_steps(gold_steps: List[str], perturbation_type: str = "irrelevant", insert_position: int = 2) -> List[str]:
    if not gold_steps:
        return gold_steps
    perturbed_steps = gold_steps.copy()
    # Safe clamp of insertion point
    insert_position = max(0, min(insert_position, len(gold_steps)-1))

    if perturbation_type == "self_reflection":
        insert = f"Step {insert_position + 1}: {random.choice(SELF_REFLECTION_BANK)}"
        perturbed_steps.insert(insert_position, insert)
        perturbed_steps = renumber_steps(perturbed_steps)
    elif perturbation_type == "wrong_step":
        insert = f"Step {insert_position + 1}: {random.choice(WRONG_STEP_BANK)}"
        perturbed_steps.insert(insert_position, insert)
        perturbed_steps = renumber_steps(perturbed_steps)
    elif perturbation_type == "irrelevant":
        insert = f"Step {insert_position + 1}: {random.choice(IRRELEVANT_BANK)}"
        perturbed_steps.insert(insert_position, insert)
        perturbed_steps = renumber_steps(perturbed_steps)
    elif perturbation_type == "repetition":
        if len(gold_steps) > 1 and insert_position - 1 >= 0:
            repeat_step = gold_steps[insert_position - 1]
            content = repeat_step.split(":", 1)[1] if ":" in repeat_step else repeat_step
            insert = f"Step {insert_position + 1}:{content}"
            perturbed_steps.insert(insert_position, insert)
            perturbed_steps = renumber_steps(perturbed_steps)
    # print("Perturbed Steps:", perturbed_steps)
    return perturbed_steps

@dataclass
class PerturbResult:
    steps: List[str]
    rewards: List[float]

    @property
    def total(self) -> float:
        return float(sum(self.rewards))

    @property
    def avg(self) -> float:
        return float(sum(self.rewards) / max(1, len(self.rewards)))

    @property
    def count(self) -> int:
        return len(self.steps)

@torch.no_grad()
def analyze_step_rewards_with_perturbations(prm_model: nn.Module, prm_tokenizer, device: torch.device,
    ds_name: str, question: str, gold_steps: List[str], gold_ans: str, perturb_types: List[str],) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    prompt_prefix = f"You are a math expert. Let's solve the following problem step by step.\nProblem: {question}\n"

    gold_rewards = compute_step_rewards(prm_model, prm_tokenizer, device, prompt_prefix, gold_steps)
    results["gold"] = PerturbResult(gold_steps, gold_rewards)

    for ptype in perturb_types:
        p_steps = create_perturbed_steps(gold_steps, ptype, insert_position=2)
        p_rewards = compute_step_rewards(prm_model, prm_tokenizer, device, prompt_prefix, p_steps)
        results[ptype] = PerturbResult(p_steps, p_rewards)

    # Convert to JSON-friendly dict
    out: Dict[str, Any] = {
        "gold": {
            "steps": results["gold"].steps,
            "rewards": results["gold"].rewards,
            "total_reward": results["gold"].total,
            "avg_reward": results["gold"].avg,
            "step_count": results["gold"].count,
        }
    }
    for ptype in perturb_types:
        pr: PerturbResult = results[ptype]
        out[ptype] = {
            "steps": pr.steps,
            "rewards": pr.rewards,
            "total_reward": pr.total,
            "avg_reward": pr.avg,
            "step_count": pr.count,
        }
    return out


# 5) CLI & main evaluation loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "gsm8k", "omni", "olympiad", "aime"]) 
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=1000, help="0 = all")
    parser.add_argument("--reward_type", choices=["ori","contri","cmi", "pav"], default="pav")
    parser.add_argument("--prm_base_model", type=str, default="Qwen/Qwen2.5-Math-7B", help="Backbone for PRM (Case B)")
    parser.add_argument("--prm_type", choices=["lm", "scr"], default="lm")
    parser.add_argument("--perturb_types", type=str, nargs="*", default=["self_reflection","wrong_step","irrelevant","repetition"]) 
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    split = "test" if args.dataset != "olympiad" else "train"
    loader, total = get_loader(args.dataset, split, args.batch_size)
    max_n = total if args.max_samples == 0 else min(args.max_samples, total)

    # Load PRM (Case B LoRA)
    if args.prm_type == "lm":
        prm_root = f"/home/leena/ccc_eval/mcts_prm/checkpoints/pt_lm/{args.reward_type}/final_model"
        prm, prm_tok = load_prm_lm(prm_root, args.prm_base_model, device)
    else:
        prm_path = f"/home/leena/ccc_eval/mcts_prm/checkpoints/scratch/{args.reward_type}/best_prm.pt"
        prm, prm_tok = load_prm_scr(prm_path, args.prm_base_model, device)
    print(f"Finish Loading {args.reward_type} Model and {args.dataset} Dataset!")

    extractor = AnswerExtractor()
    parser_obj = StepParser()

    results_per_sample: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    seen = 0
    for idxs, qs, sols, fins in loader:
        if seen >= max_n: break
        lim = min(len(qs), max_n - seen)
        qs, sols, fins, idxs = qs[:lim], sols[:lim], fins[:lim], idxs[:lim]

        for i in range(lim):
            q_txt  = qs[i]
            sol_txt= sols[i]
            fin_txt= fins[i]
            try:
                gold_steps, gold_ans = extract_gold_steps_and_answer(args.dataset, q_txt, sol_txt, fin_txt, extractor, parser_obj)
            except Exception as e:
                print(f"[WARN] extract failed at index {idxs[i]}: {e}")
                # Fallback: minimal single step from raw answer text
                fallback_ans = fin_txt or (sol_txt.strip().splitlines()[-1] if sol_txt.strip() else "")
                gold_steps = [f"Step 1: The final answer is {fallback_ans}."] if fallback_ans else ["Step 1: "]
                gold_ans   = fallback_ans

            perturb_result = analyze_step_rewards_with_perturbations(
                prm_model=prm,
                prm_tokenizer=prm_tok,
                device=device,
                ds_name=args.dataset,
                question=q_txt,
                gold_steps=gold_steps,
                gold_ans=gold_ans,
                perturb_types=args.perturb_types,
            )

            gold_avg = perturb_result["gold"]["avg_reward"]
            row = {
                "id": int(idxs[i]),
                "dataset": args.dataset,
                "question": q_txt,
                "gold_answer": gold_ans,
                "gold_steps": gold_steps,
                "gold_total": float(perturb_result["gold"]["total_reward"]),
                "gold_avg": float(gold_avg),
                "gold_step_count": int(perturb_result["gold"]["step_count"]),
            }
            for ptype in args.perturb_types:
                r = perturb_result[ptype]
                row[f"{ptype}_total"] = float(r["total_reward"]) 
                row[f"{ptype}_avg"] = float(r["avg_reward"]) 
                row[f"{ptype}_count"] = int(r["step_count"]) 
                row[f"{ptype}_drop"] = float(gold_avg - r["avg_reward"]) 
            results_per_sample.append({
                "id": int(idxs[i]),
                "dataset": args.dataset,
                "question": q_txt,
                "gold_answer": gold_ans,
                "analysis": perturb_result,
            })
            summary_rows.append(row)

        seen += lim
        print(f"Processed {seen}/{max_n} samples...")

    # Aggregate summary
    def avg_of(key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in summary_rows]
        return float(sum(vals) / max(1, len(vals)))

    overall_gold_avg = avg_of("gold_avg")
    perturbation_summary: Dict[str, Dict[str, float]] = {}
    for ptype in args.perturb_types:
        mean_avg = avg_of(f"{ptype}_avg")
        mean_count = avg_of(f"{ptype}_count")
        drop = overall_gold_avg - mean_avg
        drop_pct = (drop / overall_gold_avg * 100.0) if overall_gold_avg > 0 else 0.0
        perturbation_summary[ptype] = {
            "avg_reward": mean_avg,
            "avg_step_count": mean_count,
            "reward_drop": drop,
            "drop_percentage": drop_pct,
        }

    detailed_results = {
        "summary": {
            "dataset": args.dataset,
            "total_samples": len(summary_rows),
            "overall_gold_avg_reward": overall_gold_avg,
            "perturbation_summary": perturbation_summary,
        },
        "samples": results_per_sample,
    }

    print(f"Overall Gold Avg Reward ({args.reward_type}): {overall_gold_avg:.4f}")
    for ptype, info in perturbation_summary.items():
        print(
            f"{ptype}: avg={info['avg_reward']:.4f}, steps={info['avg_step_count']:.2f}, "
            f"drop={info['reward_drop']:.4f} ({info['drop_percentage']:.1f}%)"
        )

    save_path = f"/home/leena/ccc_eval/mcts_prm/inference/analysis/perb_{args.prm_type}_{args.dataset}_{args.reward_type}_1000.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved perturbation analysis to: {save_path}")

if __name__ == "__main__":
    main()
