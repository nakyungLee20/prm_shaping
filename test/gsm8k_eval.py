import re, math, random
from fractions import Fraction
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal, InvalidOperation
import os, json, torch
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
from datasets import load_dataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# Generate Solution Helpers #
##############################################################################
SYSTEM_PROMPT = (
    "You are Qwen-Math, a meticulous math tutor. "
    "Solve problems with clear, numbered steps and give only ONE final answer line."
)
FEWSHOT_EXAMPLES: List[Dict[str, Any]] = []

def build_user_prompt(question: str) -> str:
    header = (
        "Solve the following problem. Follow this EXACT format:\n"
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

def to_chat_prompt(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(question)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def generate_solutions(question: str, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, max_new_tokens: int = 512, seed: int = 123) -> List[str]:
    assert n >= 1
    do_sample = (n > 1) or (temperature and temperature > 1e-8)

    prompt = to_chat_prompt(question)
    inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    gen_cfg = GenerationConfig(
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        num_return_sequences=n,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_cfg.to_dict())

    # decode only the generated part
    gen_only = out[:, inputs["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    return [t.strip() for t in texts]

##############################################################################
# Answer Extraction & Grader Helpers #
##############################################################################
# extract pred answer
ANS_LINE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
NUM_FINDER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_MIXED_FRAC = re.compile(r"[-+]?\d+\s+\d+/\d+")
_PURE_FRAC  = re.compile(r"[-+]?\d+/\d+")
_DEC_SCI    = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?")
# find gold answer
_GSM_GOLD = re.compile(r"####\s*([^\n]+)")
# truncate tail part of the generation
_TRAIL_PUNCT = re.compile(r"[)\].,;:\s]+$")
_LEAD_PUNCT  = re.compile(r"^[([{\s]+")
_UNIT_TAIL = re.compile(
    r"\s*(dollars?|cents?|percent|perc\.?|pts?|points?|years?|year|hrs?|hours?|mins?|minutes?|secs?|seconds?)\s*$",
    re.IGNORECASE,
)

def _strip_wrappers(s: str) -> str:
    s = _LEAD_PUNCT.sub("", s)
    s = _TRAIL_PUNCT.sub("", s)
    return s.strip()

def _simple_moneylike_last(text: str) -> str:
    m = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not m:
        return ""
    last = m[-1]
    tok = [x for x in last if x][0]
    tok = tok.strip()
    for rgx in (",", r"\$", r"(?s).*#### ", r"\.$"):
        tok = re.sub(rgx, "", tok)
    return tok.strip()

def _find_last_numberish(text: str) -> str:
    for rx in (_MIXED_FRAC, _PURE_FRAC, _DEC_SCI):
        hits = list(rx.finditer(text))
        if hits:
            tok = hits[-1].group(0)
            return _post_clean_numberish(tok)
    fallback = _simple_moneylike_last(text)
    return fallback or ""

def _post_clean_numberish(s: str) -> str:
    if not s:
        return ""
    t = s.strip()
    for rx in (_MIXED_FRAC, _PURE_FRAC, _DEC_SCI):
        hits = rx.findall(t)
        if hits:
            cand = hits[-1] if isinstance(hits, list) else hits
            t = cand if isinstance(cand, str) else cand[-1]
            break

    t = t.replace("$", "").replace(",", "")
    t = _UNIT_TAIL.sub("", t)
    t = _strip_wrappers(t)
    t = re.sub(r"%\s*$", "", t)
    return t.strip()

def _parse_numeric(s: str):
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    # 혼합분수
    m = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", t)
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if whole < 0 else 1
        whole = abs(whole)
        try:
            return Fraction(whole * den + num, den) * sign
        except ZeroDivisionError:
            return None
    # 순수 분수
    m = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", t)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    # 일반 수
    try:
        v = float(Decimal(t))
        return v
    except (InvalidOperation, ValueError):
        return None

def normalize_number(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = _strip_wrappers(s)
    s = s.replace(",", "").replace("$", "").strip()
    # 백분율 처리: 뒤에 %/percent 있으면 제거(수치 비교는 grade 단계에서 가능)
    s = re.sub(r"\s*%\s*$", "", s)
    s = re.sub(r"\s*percent\s*$", "", s, flags=re.IGNORECASE)
    # 혼합분수: "a b/c" -> (a*den + num)/den
    m = re.fullmatch(r"\s*([+-]?\d+)\s+(\d+)\s*/\s*(\d+)\s*\Z", s)
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if whole < 0 else 1
        whole = abs(whole)
        try:
            frac = Fraction(whole * den + num, den) * sign
            return f"{frac.numerator}/{frac.denominator}"
        except ZeroDivisionError:
            return ""
    # 순수 분수: "a/b" -> 기약
    m = re.fullmatch(r"\s*([+-]?\d+)\s*/\s*(\d+)\s*\Z", s)
    if m:
        try:
            frac = Fraction(int(m.group(1)), int(m.group(2)))
            return f"{frac.numerator}/{frac.denominator}"
        except ZeroDivisionError:
            return ""
    # 소수/정수/과학표기
    try:
        v = float(Decimal(s))
        if math.isfinite(v) and abs(v - round(v)) < 1e-12:
            return str(int(round(v)))
        return f"{v:.12g}"
    except (InvalidOperation, ValueError):
        return s

def gsm_extract_gold(gold_field: str) -> str:
    if not gold_field:
        return ""
    m = _GSM_GOLD.search(gold_field)
    ans = (m.group(1) if m else gold_field).strip()
    ans = _strip_wrappers(ans)
    return ans

def gsm_extract_pred(text: str):
    if not text:
        return ""
    # 1) find "Answer: ..." line
    m = ANS_LINE.search(text)
    if m:
        cand = _post_clean_numberish(m.group(1))
        if cand:
            return cand
    # 2) find "\boxed{...}" qwen style
    m = _BOXED.search(text)
    if m:
        cand = _post_clean_numberish(m.group(1))
        if cand:
            return cand
    # 3) fallback
    last = _find_last_numberish(text)
    return last or ""

def grade_gsm8k_answer(pred: str, gold: str, atol: float = 1e-6) -> bool:
    p_norm = normalize_number(pred)
    g_norm = normalize_number(gold)
    if p_norm == g_norm:
        return True
    
    p_val = _parse_numeric(p_norm)
    g_val = _parse_numeric(g_norm)
    if p_val is None or g_val is None:
        return False
    
    if isinstance(p_val, Fraction) and isinstance(g_val, Fraction):
        return p_val == g_val
    
    try:
        pv = float(p_val) if not isinstance(p_val, float) else p_val
        gv = float(g_val) if not isinstance(g_val, float) else g_val
        return math.isfinite(pv) and math.isfinite(gv) and abs(pv - gv) <= atol
    except Exception:
        return False

def evaluate_gsm8k(dataset, limit: int = None, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, seed: int = 123, max_new_tokens: int = 512, save_incorrect_path: str | None = None,) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []

    for i, ex in enumerate(dataset):
        if limit and i >= limit:
            break

        q = ex["question"]
        gold_raw = ex["answer"]
        gold = gsm_extract_gold(gold_raw)
        gens = generate_solutions(q, n=n, temperature=temperature, top_p=top_p,max_new_tokens=max_new_tokens, seed=seed + i)
        preds = [gsm_extract_pred(t) for t in gens]
        # print("Pred:", preds, "|", "Gold:", gold)

        is_correct = grade_gsm8k_answer(preds[0], gold)
        total += 1
        correct += int(is_correct)
        logs.append({
            "idx": i,
            "question": q,
            "gold": gold,
            "gens": gens,
            "preds": preds,
            "correct_first": bool(is_correct),
        })

        # Save incorrect answers for debugging
        if not is_correct:
            incorrect_samples.append({
                "idx": i,
                "question": q,
                "gold": gold,
                "pred_chosen": preds[0] if preds else "",
                "preds_all": preds,
                "gens_all": gens,
            })

        if (i + 1) % 20 == 0:
            acc = 100.0 * correct / total
            print(f"[{i+1}] running acc = {acc:.2f}%")

    acc = 100.0 * correct / max(total, 1)
    print(f"Done. Accuracy (first sample) = {acc:.2f}%  on {total} examples.")

    # Save to JSON
    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples

##############################################################################
# Main APIs #
##############################################################################
# Load models
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"  # "mistralai/Mathstral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load Datasets
gsm8k = load_dataset("openai/gsm8k", "main", split="test")

# Evaluation
acc, logs, incorrect = evaluate_gsm8k(gsm8k, n=1, max_new_tokens=512,save_incorrect_path="/home/leena/ccc_eval/prm_rs/analysis/incorrect_gsm8k.json")

