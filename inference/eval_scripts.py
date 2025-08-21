import re, math, random
from fractions import Fraction
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple, Type
from decimal import Decimal, InvalidOperation
import os, json, torch
import argparse
import importlib
from collections import defaultdict, Counter
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from datasets import concatenate_datasets, load_dataset, get_dataset_config_names
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

from gsm_scorer import GSM8KScorer
from math_scorer import MATHScorer
from mmlu_scorer import MMLUScorer
# from aime_scorer import AIMEScorer

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# Generate Solution Helpers (Policy Sampling Trajectory) #
##############################################################################
SYSTEM_PROMPT = (
    "You are Qwen-Math, a meticulous math tutor. "
    "Solve problems with clear, numbered steps and give only ONE final answer line."
)
FEWSHOT_EXAMPLES: List[Dict[str, Any]] = []

def format_fewshot_block(example: Dict[str, Any]) -> str:
    steps = example.get("steps") or []
    body = "Problem: " + example["question"].strip() + "\n"
    for i, s in enumerate(steps, 1):
        s = s if s.strip().lower().startswith("step") else f"Step {i}: {s}"
        body += s.strip() + "\n"
    body += f"Answer: {example['answer']}\n"
    return body.strip()

def build_user_prompt(question: str) -> str:
    header = (
        "Please reason step by step. Follow this EXACT format:\n"
        "Step 1: <short reasoning>\n"
        "Step 2: <short reasoning>\n"
        "...\n"
        "Answer: <final numeric answer>\n"
        "Constraints:\n"
        "- Keep each step concise, one idea per line.\n"
        "- Do not include extra explanations after the final Answer line.\n"
    )
    parts = [header]
    if FEWSHOT_EXAMPLES:
        parts.append("Here are solved examples:\n")
        for ex in FEWSHOT_EXAMPLES:
            parts.append(format_fewshot_block(ex))
        parts.append("\nNow solve this new problem in the same format.\n")

    parts.append("Problem: " + question.strip())
    return "\n".join(parts).strip()

def to_prm_chat_prompt(tokenizer, question: str, steps: List[str], rw_token: str = "<RW>", eval_style: Optional[str] = "default") -> str:
    assistant_text = rw_token.join(steps) + rw_token
    if eval_style == "qwen_eval":
        user = f"{question.strip()}\n\nPlease reason step by step, and put your final answer within `Answer: \\boxed{{}}`."
    else:
        user = build_user_prompt(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
        {"role": "assistant", "content": assistant_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def to_chat_prompt(tokenizer, question: str, eval_style: Optional[str] = "default") -> str:
    if eval_style == "qwen_eval":
        user = f"{question.strip()}\n\nPlease reason step by step, and put your final answer within `Answer: \\boxed{{}}`."
    else:
        user = build_user_prompt(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def batched_generate_vllm(questions: List[str], llm: LLM, tokenizer, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 512, seed: Optional[int] = 123, eval_style: Optional[str] = "default") -> List[List[str]]:
    assert n >= 1
    do_sample = (n > 1) or (temperature and temperature > 1e-8)

    prompts = [to_chat_prompt(tokenizer, q, eval_style=eval_style) for q in questions]
    sp = SamplingParams(
        temperature = (temperature if do_sample else 0.0),
        top_p       = (top_p if do_sample else 1.0),
        max_tokens  = max_tokens,
        n           = n,
        seed        = seed,
        repetition_penalty = 1.05,
        skip_special_tokens = True,
        # stop=[]  # 필요시 스톱 토큰 지정
    )
    outs = llm.generate(prompts, sp, use_tqdm=False)
    result: List[List[str]] = []
    for out in outs:
        gens_i = [o.text.strip() for o in out.outputs]
        # print("Check Generation:", gens_i)
        result.append(gens_i)
    return result

def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield i, lst[i:i+size]

def evaluate_vllm(dataset, llm: LLM, tokenizer, limit: Optional[int] = None, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, seed: int = 123, max_tokens: int = 512, 
                  batch_size: int = 32, save_incorrect_path: Optional[str] = None, scorer: Any = None) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []

    N = len(dataset)
    if limit is not None:
        N = min(N, limit)

    for start, batch in _chunk(list(range(N)), batch_size):
        # 1) collect batch inputs
        qs: List[str] = [dataset[i]["question"] for i in batch]
        gold_raws: List[str] = [dataset[i]["answer"]   for i in batch]
        golds: List[str] = [scorer.extract_gold(x) for x in gold_raws]

        # 2) generate (BATCHED): gens_batch shape: [bsz][n]
        gens_batch: List[List[List[str]]] = batched_generate_vllm(qs, llm, tokenizer,n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)

        # 3) score each item in batch
        for j, i_ex in enumerate(batch):
            q = qs[j]
            gold = golds[j]
            gens = gens_batch[j]
            preds = [scorer.extract_pred(t) for t in gens]

            is_correct = scorer.grade(preds[0], gold)
            total += 1
            correct += int(is_correct)

            logs.append({
                "idx": i_ex,
                "question": q,
                "gold": gold,
                "gens": gens,
                "preds": preds,
                "correct_first": bool(is_correct),
            })

            if not is_correct:
                incorrect_samples.append({
                    "idx": i_ex,
                    "question": q,
                    "gold": gold,
                    "pred_chosen": preds[0] if preds else "",
                    "preds_all": preds,
                    "gens_all": gens,
                })

        if (total % 20) == 0:
            acc = 100.0 * correct / total
            print(f"[{total}/{N}] running acc = {acc:.2f}%")

    acc = 100.0 * correct / max(total, 1)
    print(f"GSM8K Accuracy = {acc:.2f}%  on {total} examples.")

    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples

##############################################################################
# 2. Dataset/Scorer loaders & field mapping                                         #
##############################################################################
FIELD_MAP: Dict[str, Tuple[str, str]] = { # dataset name  : (question_field, answer_field)
    "gsm8k": ("question", "answer", None),
    "math": ("problem", "solution", None),
    "omni": ("problem", "answer", None),
    "olympiad": ("question", "final_answer", None),
    "aime": ("problem", "answer", None),
    "mmlu": ("question", "answer", "choices"),
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
    elif ds_name == "mmlu":
        ds = load_dataset("TIGER-Lab/MMLU-STEM", split=split)
    else:
        raise ValueError(f"Unsupported dataset {ds_name}")
    
    q_key, a_key, els = FIELD_MAP[ds_name]
    def _std(ex):
        out = {"question": ex[q_key], "answer": ex[a_key]}
        if els is not None and els in ex:
            out["choices"] = ex[els] # for mmlu
        return out
    ds = ds.map(_std, remove_columns=[])
    return ds, len(ds)

def resolve_scorer(ds_name: str):
    if not ds_name or not isinstance(ds_name, str):
        raise ValueError("ds_name must be a non-empty string")

    raw = ds_name.strip()
    ds_upper = re.sub(r'[^0-9A-Za-z]+', '_', raw).upper()   # e.g., 'gsm8k' -> 'GSM8K'
    instance_name = ds_upper + "Scorer"

    sym = globals().get(instance_name)
    required = ("extract_gold", "extract_pred", "grade")
    if all(callable(getattr(sym, m, None)) for m in required):
        return sym

    raise ImportError(f"Could not resolve scorer for dataset '{instance_name}'. ")

##############################################################################
# Step parsing utilities #
##############################################################################
STEP_RE = re.compile(r"^(?:step\s*\d+\s*:\s*)(.*)$", re.IGNORECASE)
ANSWER_LINE_RE = re.compile(r"^\s*answer\s*:\s*(.*)$", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

def split_steps(text: str) -> List[str]:
    """Parse a model completion into a list of steps (strings).
    Priority:
      1) Existing <extra_0> separators
      2) Blank-line separation ("\n\n")
      3) Lines starting with "Step k:" markers
      4) Fallback: entire reasoning (before the Answer line) as one step
    """
    if "<extra_0>" in text:
        parts = [p.strip() for p in text.split("<extra_0>") if p.strip()]
    else:
        # Cut off after the explicit Answer line if present
        lines = text.splitlines()
        body_lines: List[str] = []
        for ln in lines:
            if ANSWER_LINE_RE.match(ln):
                break
            body_lines.append(ln)
        body = "\n".join(body_lines)
        # 2) try blank line separation
        if "\n\n" in body:
            parts = [p.strip() for p in body.split("\n\n") if p.strip()]
        else:
            # 3) try explicit step markers
            parts_tmp: List[str] = []
            cur: List[str] = []
            for ln in body.splitlines():
                if STEP_RE.match(ln):
                    if cur:
                        parts_tmp.append(" ".join(cur).strip())
                        cur = []
                    cur.append(STEP_RE.sub(r"\\1", ln).strip())
                else:
                    cur.append(ln.strip())
            if cur:
                parts_tmp.append(" ".join(cur).strip())
            parts = [p for p in parts_tmp if p]
    if not parts:
        parts = [text.strip()]
    # print("Split steps:", parts, flush=True)
    return parts

##############################################################################
# PRM scoring wrappers #
##############################################################################
def make_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
    """Compute per-step positive probabilities at <extra_0> positions."""
    probs = F.softmax(logits, dim=-1)
    all_scores: List[List[float]] = []
    bsz = probs.size(0)
    for i in range(bsz):
        # select rows where token_masks[i] is True → [num_steps, num_labels]
        masked = probs[i][token_masks[i].squeeze(0)]  # (S, C) or empty
        if masked.numel() == 0:
            all_scores.append([])
        else:
            # take P(label=1)
            pos = masked[:, 1].detach().cpu().tolist()
            all_scores.append(pos)
    return all_scores

def aggregate(scores: List[float], how: str = "mean") -> float:
    if not scores:
        return 0.0
    if how == "mean":
        return float(sum(scores) / len(scores))
    if how == "last":
        return float(scores[-1])
    if how == "sum":
        return float(sum(scores))
    if how == "median":
        s = sorted(scores)
        m = len(s) // 2
        return float((s[m] if len(s) % 2 else 0.5*(s[m-1] + s[m])))
    return float(sum(scores) / len(scores))

def score_candidates_with_prm(prm_model, prm_tokenizer, system_prompt: str, question: str, candidates: List[str], rw_token: str = "<RW>", eval_style: Optional[str] = "default") -> Tuple[List[List[float]], List[float]]:
    is_reward_model = hasattr(prm_model, "predict_rewards_at_rw")
    all_step_scores: List[List[float]] = []
    all_agg_scores: List[float] = []

    if is_reward_model:
        if rw_token not in prm_tokenizer.get_vocab():
            raise ValueError(f"Reward token '{rw_token}' is not in the tokenizer vocabulary; "
                             "please add it before training your reward model.")
        for txt in candidates:
            steps = split_steps(txt) 
            conv_str = to_prm_chat_prompt(prm_tokenizer, question, steps, rw_token, eval_style=eval_style)
            inputs = prm_tokenizer(conv_str, return_tensors="pt")
            dev = prm_model.get_input_embeddings().weight.device
            input_ids = inputs["input_ids"].to(dev)
            attention_mask = inputs["attention_mask"].to(dev)
            rewards = prm_model.predict_rewards_at_rw(input_ids=input_ids, attention_mask=attention_mask)
            if len(rewards) != 1:
                    raise RuntimeError(f"Unexpected reward output shape: {rewards}")
            step_rewards = rewards[0].detach().cpu().tolist()
            all_step_scores.append(step_rewards)
            all_agg_scores.append(aggregate(step_rewards, how="mean"))
            # reward sanity check
            num_steps = len(steps)
            num_rw = int((input_ids[0] == prm_model.rw_token_id).sum().item())
            if num_steps != num_rw:
                print(f"[WARN] steps({num_steps}) != RW tokens({num_rw}) for this sample")
    else:
        prm_inputs: List[torch.Tensor] = []
        for txt in candidates:
            steps = split_steps(txt)
            rw_ori_token = "<extra_0>"
            conv_str = to_prm_chat_prompt(prm_tokenizer, question, steps, rw_ori_token, eval_style=eval_style)
            input_ids = prm_tokenizer.encode(conv_str, return_tensors="pt").squeeze(0)
            prm_inputs.append(input_ids)
        max_len = max(x.size(0) for x in prm_inputs)
        input_ids = torch.full((len(prm_inputs), max_len), prm_tokenizer.pad_token_id or 0, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        step_sep_id = prm_tokenizer.encode(rw_ori_token)[0]
        for i, seq in enumerate(prm_inputs):
            input_ids[i, :seq.size(0)] = seq
            attention_mask[i, :seq.size(0)] = 1
        token_masks = (input_ids == step_sep_id)
        input_ids = input_ids.to(prm_model.device if hasattr(prm_model, "device") else prm_model.parameters().__next__().device)
        attention_mask = attention_mask.to(input_ids.device)
        outputs = prm_model(input_ids=input_ids, attention_mask=attention_mask)
        # element is the logits tensor of shape (B, T, C)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        step_scores_batch = make_step_rewards(logits, token_masks)       # List[List[float]] for all candidates
        all_step_scores = step_scores_batch
        all_agg_scores  = [aggregate(s, how="mean") for s in step_scores_batch]
    return all_step_scores, all_agg_scores

def pick_best_by_prm(prm_model,prm_tokenizer,question: str, candidates: List[str],agg: str = "mean", rw_token: str = "<RW>", eval_style: Optional[str] = "default") -> Tuple[int, List[List[float]], List[float]]:
    step_scores, agg_scores = score_candidates_with_prm(
            prm_model=prm_model,
            prm_tokenizer=prm_tokenizer,
            system_prompt=SYSTEM_PROMPT,
            question=question,
            candidates=candidates,
            rw_token=rw_token,
            eval_style=eval_style,
    )
    agg_scores_custom = [aggregate(s, how=agg) for s in step_scores]
    best_idx = int(max(range(len(agg_scores_custom)), key=lambda i: agg_scores_custom[i]))
    return best_idx, step_scores, agg_scores_custom

def evaluate_dataset_bon_vllm(dataset, llm: LLM, policy_tokenizer, prm_model, prm_tokenizer, limit: Optional[int] = None, n: int = 8, temperature: float = 0.6, top_p: float = 0.95, seed: int = 123, 
    max_tokens: int = 512, batch_size: int = 32, agg: str = "mean", save_incorrect_path: Optional[str] = None, eval_style: Optional[str] = "default", scorer: Any = None) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    get_q = lambda ex: ex["question"]
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []

    N = len(dataset)
    if limit is not None:
        N = min(N, limit)
    all_indices = list(range(N))

    for start, batch in _chunk(all_indices[:N], batch_size):
        # 1) collect batch inputs
        qs: List[str]   = [get_q(dataset[i]) for i in batch]
        gold_raws: List[str] = [dataset[i]["answer"] for i in batch]
        golds: List[str]     = [scorer.extract_gold(x) for x in gold_raws]
        # 2) generate candidates (BATCHED): gens_batch shape: [bsz][n]
        gens_batch: List[List[List[str]]] = batched_generate_vllm(qs, llm, policy_tokenizer, n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed, eval_style=eval_style)
        # 3) for each example: PRM-score candidates, pick best, grade
        for j, i_ex in enumerate(batch):
            q = qs[j]
            gold = golds[j]
            gens: List[str] = gens_batch[j]
            preds: List[str] = [scorer.extract_pred(t) for t in gens]
            # Pick best candidate by PRM
            best_idx, step_scores, agg_scores = pick_best_by_prm(prm_model, prm_tokenizer, q, gens, agg=agg, rw_token="<RW>", eval_style=eval_style)
            chosen_pred = preds[best_idx] if preds else ""
            is_correct = scorer.grade(chosen_pred, gold)
            total += 1
            correct += int(is_correct)
            # print log
            logs.append({
                "idx": i_ex,
                "question": q,
                "gold": gold,
                "gens": gens,
                "preds": preds,
                "prm_step_scores": step_scores,
                "prm_agg_scores": agg_scores,
                "chosen_idx": best_idx,
                "pred_chosen": chosen_pred,
                "correct": bool(is_correct),
            })
            if not is_correct:
                incorrect_samples.append({
                    "idx": i_ex,
                    "question": q,
                    "gold": gold,
                    "pred_chosen": chosen_pred,
                    "preds_all": preds,
                    "gens_all": gens,
                    "prm_step_scores": step_scores,
                    "prm_agg_scores": agg_scores,
                })

        if (total % 20) == 0:
            acc = 100.0 * correct / max(total, 1)
            print(f"[{total}/{N}] running BoN acc = {acc:.2f}% (agg={agg}, N={n})", flush = True)

    acc = 100.0 * correct / max(total, 1)
    print(f"Final BoN Accuracy = {acc:.2f}% on {total} examples. (agg={agg}, N={n})")

    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples

##############################################################################
# Majority Voting (with optional PRM weights)
##############################################################################
def _softmax_weights(scores: List[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    # Numerical stability via max-subtraction; protect very small temperature
    t = max(1e-6, float(temperature))
    max_s = max(scores)
    exps = [math.exp((s - max_s) / t) for s in scores]
    z = float(sum(exps))
    if z <= 0.0 or not math.isfinite(z):
        return [1.0 / len(scores)] * len(scores)
    return [e / z for e in exps]

def _build_candidates_with_scores(question: str, gens: List[str], prm_model, prm_tokenizer, agg: str = "mean", rw_token: str = "<RW>",) -> Tuple[List[Dict[str, Any]], List[float]]:
    # PRM scores for all candidates at once
    step_scores, agg_scores = score_candidates_with_prm(prm_model, prm_tokenizer, SYSTEM_PROMPT, question, gens, rw_token=rw_token)
    cands: List[Dict[str, Any]] = []
    for txt, agg_s, step_s in zip(gens, agg_scores, step_scores):
        ans = SCORER.extract_pred(txt)
        cands.append({
            "body": txt,
            "answer": ans,
            "answer_norm": SCORER.normalize_number(ans) if ans else "",
            "steps": split_steps(txt),
            "prm_score": float(agg_s),
            "prm_step_scores": step_s,
        })
    return cands, agg_scores

def choose_by_majority(cands: List[Dict[str, Any]], use_prm_weights: bool = False, temperature: float = 1.0, tie_break: str = "prm", ignore_empty: bool = True,) -> Tuple[int, Dict[str, Any]]:
    if not cands:
        return -1, {"reason": "no_candidates"}
    # Optionally drop empty answers from voting
    vote_pool = [c for c in cands if (c.get("answer_norm") or c.get("answer"))] if ignore_empty else list(cands)
    if not vote_pool:  # all empty → fallback to highest PRM
        best_idx = int(max(range(len(cands)), key=lambda i: cands[i]['prm_score']))
        return best_idx, {"reason": "fallback_all_empty"}

    weights = _softmax_weights([c['prm_score'] for c in vote_pool], temperature) if use_prm_weights else [1.0] * len(vote_pool)

    vote_by_answer: Dict[str, float] = defaultdict(float)
    members: Dict[str, List[int]] = defaultdict(list)
    for local_i, (cand, w) in enumerate(zip(vote_pool, weights)):
        key = cand.get("answer_norm") or cand.get("answer") or "__EMPTY__"
        vote_by_answer[key] += float(w)
        # store ORIGINAL index into cands, not local vote_pool index
        global_i = cands.index(cand)
        members[key].append(global_i)

    max_votes = max(vote_by_answer.values(), default=0.0)
    winners = [a for a, v in vote_by_answer.items() if abs(v - max_votes) < 1e-12]

    if not winners:
        # Should not happen because vote_pool non-empty; still guard
        fallback = int(max(range(len(cands)), key=lambda i: cands[i]['prm_score'])) if use_prm_weights else 0
        return fallback, {"reason": "fallback_no_winner"}

    if len(winners) == 1:
        win_ans = winners[0]
    else:
        if tie_break == 'prm':
            # Choose the answer bucket whose best member has highest PRM
            def best_prm_for_answer(ans_key: str) -> float:
                return max(cands[i]['prm_score'] for i in members[ans_key])
            win_ans = max(winners, key=best_prm_for_answer)
        else:
            win_ans = winners[0]

    # Choose the candidate within the winning bucket
    idxs = members[win_ans]
    chosen_idx = max(idxs, key=lambda i: cands[i]['prm_score']) if use_prm_weights else idxs[0]

    diag = {
        "votes": {k: float(v) for k, v in vote_by_answer.items()},
        "use_prm_weights": bool(use_prm_weights),
        "temperature": float(temperature),
        "tie_break": tie_break,
        "winning_answer": win_ans,
        "chosen_idx": int(chosen_idx),
        "chosen_score": float(cands[chosen_idx]['prm_score']),
    }
    return int(chosen_idx), diag

def evaluate_dataset_majority_vllm(dataset, llm: LLM, policy_tokenizer, prm_model, prm_tokenizer, *, limit: Optional[int] = None, n: int = 8, temperature: float = 0.6, top_p: float = 0.95, seed: int = 123, 
    max_tokens: int = 512, batch_size: int = 32, agg: str = "mean", use_prm_weights: bool = False, maj_temp: float = 1.0,     # softmax temperature for PRM vote weights
    tie_break: str = "prm", save_incorrect_path: Optional[str] = None, scorer: Any = None) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    
    get_q = lambda ex: ex["question"]
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []

    N = len(dataset)
    if limit is not None:
        N = min(N, limit)
    all_indices = list(range(N))

    for start, batch in _chunk(all_indices[:N], batch_size):
        qs: List[str] = [get_q(dataset[i]) for i in batch]
        gold_raws: List[str] = [dataset[i]["answer"] for i in batch]
        golds: List[str] = [scorer.extract_gold(x) for x in gold_raws]

        gens_batch: List[List[List[str]]] = batched_generate_vllm(
            qs, llm, policy_tokenizer, n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
        )

        for j, i_ex in enumerate(batch):
            q = qs[j]
            gold = golds[j]
            gens: List[str] = gens_batch[j]

            # Build candidates with PRM scores
            cands, _ = _build_candidates_with_scores(q, gens, prm_model, prm_tokenizer, agg=agg, rw_token="<RW>")
            # Choose by majority (optionally PRM-weighted)
            chosen_idx, diag = choose_by_majority(cands, use_prm_weights=use_prm_weights, temperature=maj_temp, tie_break=tie_break)
            if chosen_idx < 0:
                # Defensive fallback: choose the best PRM candidate if majority failed
                chosen_idx = int(max(range(len(cands)), key=lambda i: cands[i]['prm_score'])) if cands else -1

            chosen_pred = cands[chosen_idx]["answer"] if (0 <= chosen_idx < len(cands)) else ""
            is_correct = scorer.grade(chosen_pred, gold)
            total += 1
            correct += int(is_correct)

            logs.append({
                "idx": i_ex,
                "question": q,
                "gold": gold,
                "gens": gens,
                "cands": cands,  # includes answers & prm scores
                "chosen_idx": chosen_idx,
                "pred_chosen": chosen_pred,
                "majority_diag": diag,
                "correct": bool(is_correct),
            })
            if not is_correct:
                incorrect_samples.append({
                    "idx": i_ex,
                    "question": q,
                    "gold": gold,
                    "pred_chosen": chosen_pred,
                    "gens_all": gens,
                    "cands": cands,
                    "majority_diag": diag,
                })

        if (total % 20) == 0:
            acc = 100.0 * correct / max(total, 1)
            print(
                f"[{total}/{N}] running MAJ acc = {acc:.2f}% (N={n}, use_prm_weights={use_prm_weights}, maj_temp={maj_temp})",
                flush=True,
            )

    acc = 100.0 * correct / max(total, 1)
    print(
        f"Final GSM8K Majority Accuracy = {acc:.2f}% on {total} examples. (N={n}, use_prm_weights={use_prm_weights}, maj_temp={maj_temp})"
    )

    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples

##############################################################################
# Main APIs #
##############################################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='math', choices=['gsm8k', 'math', 'mmlu', 'aime', "omni"])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--prm', type=str, default='Qwen/Qwen2.5-Math-PRM-7B')
    ap.add_argument('--prm_type', type=str, default='cust', choices=['hug','cust'])
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.8)
    ap.add_argument('--max_new_tokens', type=int, default=2048)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--infer_type', type=str, default='bon', choices=['bon', 'single', 'maj'])
    ap.add_argument('--bon_agg', type=str, default='mean', choices=['mean', 'last', 'sum', 'median'])
    ap.add_argument('--maj_tie_break', type=str, default='first', choices=['prm','first'])
    ap.add_argument('--maj_use_prm_weights', action='store_true')
    args = ap.parse_args()

    # Load models
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"   # "mistralai/Mathstral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    policy = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16", 
        tensor_parallel_size=4,
        gpu_memory_utilization=0.65, 
        max_model_len=3096,
        # quantization="bitsandbytes", 
        enforce_eager=False,
        enable_prefix_caching=True,  
        distributed_executor_backend="mp",  
    )

    # Load Datasets
    datas, total = get_loader(args.dataset, "test" if args.dataset != "olympiad" else "train", args.batch_size)

    # Load Scorer
    SCORER = resolve_scorer(args.dataset)

    # Prepare PRM (needed for BoN and Majority)
    prm_model = None
    prm_tokenizer = None
    if args.infer_type in ("bon", "maj"):
        if args.prm_type == 'cust':
            path = "/home/leena/prm_shaping/checkpoints/1.5b/mi_shapley/final_model"
            prm_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
            # Expecting a PRMRewardWrapper class available in your codebase
            from wrapper import PRMRewardWrapper  # type: ignore
            prm_model = PRMRewardWrapper.from_pretrained(path, tokenizer=prm_tokenizer)
        else:
            prm_tokenizer = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
            prm_model = AutoModel.from_pretrained(
                args.prm, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
            ).eval()
    
    # Evaluation
    incorr_path = f"/home/leena/prm_shaping/analysis/{args.dataset}_1.5b_shap_sign_{args.infer_type}.json"
    if args.infer_type == 'single':
        print(f"Evaluating {args.dataset} with single generation...", flush=True)
        acc, logs, incorrect = evaluate_vllm(datas, policy, tokenizer,
            n=1, temperature=args.temperature, top_p=args.top_p,
            seed=args.seed, max_tokens=args.max_new_tokens, limit=args.limit, 
            batch_size=args.batch_size, save_incorrect_path=incorr_path, scorer=SCORER,
        )
    elif args.infer_type == 'bon':
        print(f"Evaluating {args.dataset} with PRM best-of-N...", flush=True)
        acc, logs, incorrect = evaluate_dataset_bon_vllm(datas, policy, tokenizer, prm_model, prm_tokenizer,
            n=args.n, temperature=args.temperature, top_p=args.top_p, limit=args.limit, 
            seed=args.seed, max_tokens=args.max_new_tokens, batch_size=args.batch_size,
            agg=args.bon_agg, save_incorrect_path=incorr_path, scorer=SCORER,
        )
    elif args.infer_type == 'maj':
        print(f"Evaluating {args.dataset} with Majority Voting...", flush=True)
        acc, logs, incorrect = evaluate_dataset_majority_vllm(datas, policy, tokenizer, prm_model, prm_tokenizer,
            n=args.n, temperature=args.temperature, top_p=args.top_p, limit=args.limit, 
            seed=args.seed, max_tokens=args.max_new_tokens, batch_size=args.batch_size,
            agg=args.bon_agg, use_prm_weights=False,
            maj_temp=1.0, tie_break=args.maj_tie_break,
            save_incorrect_path=incorr_path, scorer=SCORER,
        )
    else:
        raise ValueError(f"Unknown inference type: {args.infer_type}")

    # Optionally dump logs next to incorrect file if provided
    path_logs = os.path.splitext(incorr_path)[0] + "_logs.json"
    with open(path_logs, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"Saved full logs to: {path_logs}")
