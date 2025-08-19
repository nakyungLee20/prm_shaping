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
from vllm import LLM, SamplingParams

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# Generate Solution Helpers #
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

def to_chat_prompt(tokenizer, question: str, eval_style: str = "default") -> str:
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

##############################################################################
# Answer Extraction & Grader Helpers #
##############################################################################
# extract pred answer
ANS_LINE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
HINT_ANS = re.compile(r"(?i)\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+equal\s+to|is|=|equals|:)\s*([^\n]+)")
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
    
    boxed = _BOXED.search(t)
    if boxed:
        t = boxed.group(1)
    
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
    # 1) find "\boxed{...}" qwen style
    m_all = list(_BOXED.finditer(text))
    if m_all:
        cand = _post_clean_numberish(m_all[-1].group(1))
        if cand:
            return cand
    # 2) find "Answer: ..." line
    m = ANS_LINE.search(text)
    if m:
        cand = _post_clean_numberish(m.group(1))
        if cand:
            return cand
    # 3) other types of answers
    m2 = HINT_ANS.search(text)
    if m2:
        cand = _post_clean_numberish(m2.group(1))
        if cand:
            return cand
    # 4) fallback
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

def evaluate_gsm8k_vllm(dataset, llm: LLM, tokenizer, limit: Optional[int] = None, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, seed: int = 123, max_tokens: int = 512, batch_size: int = 32, save_incorrect_path: Optional[str] = None,) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        golds: List[str] = [gsm_extract_gold(x) for x in gold_raws]

        # 2) generate (BATCHED): gens_batch shape: [bsz][n]
        gens_batch: List[List[List[str]]] = batched_generate_vllm(qs, llm, tokenizer,n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed + start)

        # 3) score each item in batch
        for j, i_ex in enumerate(batch):
            q = qs[j]
            gold = golds[j]
            gens = gens_batch[j]
            preds = [gsm_extract_pred(t) for t in gens]

            is_correct = grade_gsm8k_answer(preds[0], gold)
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
# Main APIs #
##############################################################################
if __name__ == "__main__":
    # Load models
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"   # "mistralai/Mathstral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16", 
        tensor_parallel_size=1,        # 멀티GPU면 2+ 로
        gpu_memory_utilization=0.80, 
        max_model_len=4096,
        quantization="bitsandbytes", 
        # enforce_eager=True, 
    )

    # Load Datasets
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")

    # Evaluation
    incorr_path = "/home/leena/prm_shaping/analysis/incorr_gsm8k_vllm_0814.json"
    acc, logs, incorrect = evaluate_gsm8k_vllm(gsm8k, llm=llm, tokenizer=tokenizer, n=1, max_tokens=1024, batch_size=32, save_incorrect_path=incorr_path)

