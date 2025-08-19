import os, re, json, math, random
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple
import sympy as sp
from sympy import Eq, Pow, simplify, sympify
from sympy.parsing.latex import parse_latex
import numpy as np
from dataclasses import dataclass

from datasets import concatenate_datasets, load_dataset, get_dataset_config_names
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

##############################################################################
# Prompt helpers (reuse your GSM8K style; tweak Answer boxing for MATH)
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

def _fewshot_section() -> str:
    if not FEWSHOT_EXAMPLES:
        return ""
    parts = ["Here are solved examples:\n"]
    for ex in FEWSHOT_EXAMPLES:
        parts.append(format_fewshot_block(ex))
    parts.append("\nNow solve this new problem in the same format.\n")
    return "\n".join(parts)

def build_user_prompt(question: str) -> str:
    header = (
        "Please reason step by step. Follow this EXACT format:\n"
        "Step 1: <short reasoning>\n"
        "Step 2: <short reasoning>\n"
        "...\n"
        "Answer: <final simplified answer>\n"
        "Constraints:\n"
        "- Keep each step concise, one idea per line.\n"
        "- Put the final answer in LaTeX box: `Answer: \\boxed{...}` if possible.\n"
        "- Do not include extra explanations after the final Answer line.\n"
    )
    return (header + "\n" + _fewshot_section() + "Problem: " + question.strip()).strip()

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

def batched_generate_vllm(questions: List[str], llm: LLM, tokenizer, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 512, seed: Optional[int] = 123, eval_style: Optional[str] = "default",) -> List[List[str]]:
    assert n >= 1
    do_sample = (n > 1) or (temperature and temperature > 1e-8)

    prompts = [to_chat_prompt(tokenizer, q, eval_style=eval_style) for q in questions]
    sp = SamplingParams(
        temperature=(temperature if do_sample else 0.0),
        top_p=(top_p if do_sample else 1.0),
        max_tokens=max_tokens,
        n=n,
        seed=seed,
        repetition_penalty=1.05,
        skip_special_tokens=True,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    outs = llm.generate(prompts, sp, use_tqdm=False)

    result: List[List[str]] = []
    for out in outs:
        gens_i = [o.text.strip() for o in out.outputs]
        result.append(gens_i)
    return result

##############################################################################
# Extraction utilities (pred & gold)
##############################################################################
BOXED_OPEN = re.compile(r"\\boxed\\*?\s*\{")
DOLLAR_INLINE = re.compile(r"\$(.*?)\$")
_SPECIAL_STRIP = [
    (re.compile(r"\\left\s*"), ""),
    (re.compile(r"\\right\s*"), ""),
    (re.compile(r"\\,|\\;|\\:"), " "),
    (re.compile(r"\\!"), ""),
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\operatorname\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
]
_TOP_COMMA_SPLIT = re.compile(r",(?=(?:[^(){}\[\]]|[(){}\[\]])*$)")  # commas not inside brackets
ANS_LINE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.I | re.M)

def _extract_all_boxed(text: str) -> List[str]:
    # Return ALL boxed contents in order
    out: List[str] = []
    if not text:
        return out
    for m in BOXED_OPEN.finditer(text):
        i = m.end()
        depth, j = 1, i
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "\\":
                j += 2; continue
            if ch == "{": depth += 1
            elif ch == "}": depth -= 1
            j += 1
        if depth == 0:
            out.append(text[i:j-1])
    return out

def _strip_and_replace(s: str) -> str:
    if not s:
        return ""
    # 1) 줄바꿈 먼저 정규화 (예: "$-\n 2$")
    t = s.replace("\n", " ")
    # 2) 달러 언이스케이프
    t = t.replace("\\$", "$")
    # 3) 공통 LaTeX wrapper 제거
    for rx, rep in _SPECIAL_STRIP:
        t = rx.sub(rep, t)
    # 4) 기호/문장부호 정리
    t = t.replace("∶", ":").replace("，", ",").replace("$", "")
    t = t.replace("\\approx", "=").replace("\\simeq", "=").replace("\\sim", "=")
    t = t.replace("^{\\prime}", "'").replace("^\\prime", "'")
    # 5) degree 마커 제거: ^\circ 와 ^{\circ} 모두
    t = t.replace("^{\\circ}", "")
    t = t.replace("^\\circ", "")
    t = re.sub(r"\^\{?\\?circ\}?", "", t)
    # 6) 퍼센트 기호 제거 (수치 비교는 numeric_equal에서 처리)
    t = t.replace("%", "")
    # 7) 백슬래시 정규화 (regex 말고 안전한 문자열 치환)
    t = t.replace("\\\\", "\\")
    # 8) 공백 정리
    return re.sub(r"\s+", " ", t).strip()

def _remove_unit(s: str, unit: Optional[str]) -> str:
    if not s or not unit:
        return s or ""
    # remove plain and LaTeX-styled unit at the right end
    unit = unit.strip()
    patt = re.compile(rf"\s*(?:{re.escape(unit)}|\\mathrm\{{\s*{re.escape(unit)}\s*\}})\s*$", re.IGNORECASE)
    return patt.sub("", s)

def _split_top_level_commas(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in _TOP_COMMA_SPLIT.split(s) if x.strip()]

def _balance_braces(s: str) -> str:
    if not s:
        return ""
    open_n = s.count("{")
    close_n = s.count("}")
    while close_n > open_n and s.endswith("}"):
        s = s[:-1]
        close_n -= 1
    return s

def _latex_cleanup(expr: str) -> str:
    # minimally prepare LaTeX-ish strings for sympy
    t = _strip_and_replace(_balance_braces(expr))
    t = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1)/(\2)", t)
    t = t.replace("\\cdot", "*").replace("\\times", "*")
    t = t.replace("^", "**").replace("\\pm", "±")
    t = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", t)
    t = re.sub(r"(?<![A-Za-z\\])sqrt\{([^}]*)\}", r"sqrt(\1)", t)
    # implicit multiplication: 5sqrt(2) -> 5*sqrt(2)
    t = re.sub(r"(\d|\))\s*([A-Za-z]+)\s*\(", r"\1*\2(", t)
    return t.strip()

def _as_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def _variants_pm(s: str) -> List[str]:
    return [s.replace("±", "+"), s.replace("±", "-")] if "±" in s else [s]

def ob_extract_pred(text: str) -> str:
    if not text:
        return ""
    boxed = _extract_all_boxed(text)
    if boxed:
        return ",".join(boxed)
    # else, try last-line pattern or answer line
    m = ANS_LINE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        dollars = DOLLAR_INLINE.findall(lines[-1])
        if dollars:
            return ",".join(dollars)
        return lines[-1]
    return ""

def ob_extract_gold(rec: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    unit = rec.get("unit")
    ans_type = rec.get("answer_type")
    ans = rec.get("final_answer")
    if isinstance(ans, list) and ans:
        gold = ",".join([str(x) for x in ans])
        return gold, unit, ans_type
    if isinstance(ans, str) and ans.strip():
        return ans.strip(), unit, ans_type
    # fallback: try to parse from solution
    sol = rec.get("solution") or rec.get("solutions") or ""
    if isinstance(sol, list) and sol:
        sol = sol[-1]
    if isinstance(sol, str):
        m = ANS_LINE.search(sol)
        if m:
            return m.group(1).strip(), unit, ans_type
    return "", unit, ans_type

##############################################################################
# Final grader
##############################################################################
@dataclass
class OBJudge:
    default_precision: float = 1e-8
    
    def preprocess(self, gt: str, pred: str, unit: Optional[str]) -> Tuple[str, str]:
        # 1) prefer all boxed contents; else try $...$ on last line; else raw
        def grab(expr: str) -> str:
            if not expr:
                return ""
            boxed = _extract_all_boxed(expr)
            if boxed:
                return ",".join(boxed)
            m = ANS_LINE.search(expr)
            if m:
                return m.group(1).strip()
            last = expr.strip().splitlines()[-1]
            parts = DOLLAR_INLINE.findall(last)
            return ",".join(parts) if parts else expr

        gt2 = _strip_and_replace(grab(gt))
        pr2 = _strip_and_replace(grab(pred))
        # remove trailing unit if any
        gt2 = _remove_unit(gt2, unit)
        pr2 = _remove_unit(pr2, unit)
        return gt2, pr2

    # ---- equality routines ----
    def _interval_equal(self, a: str, b: str) -> bool:
        # Compare union of intervals separated by \cup, endpoints via expression_equal
        parts_a = [x.strip() for x in a.split("\\cup")]
        parts_b = [x.strip() for x in b.split("\\cup")]
        if len(parts_a) != len(parts_b):
            return False
        for ia, ib in zip(parts_a, parts_b):
            if ia[0] != ib[0] or ia[-1] != ib[-1]:
                return False
            inner_a = ia.strip('[]()')
            inner_b = ib.strip('[]()')
            ea, eb = [x.strip() for x in inner_a.split(',')], [x.strip() for x in inner_b.split(',')]
            if len(ea) != len(eb):
                return False
            for xa, xb in zip(ea, eb):
                if not self._expression_equal(xa, xb):
                    return False
        return True

    def _numeric_equal(self, a: str, b: str) -> bool:
        # allow % semantics: x, x/100, x*100
        fa, fb = _as_float(a), _as_float(b)
        if fa is None or fb is None:
            return False
        for ref in (fa/100.0, fa, fa*100.0):
            if abs(ref - fb) <= self.default_precision * 1.01:
                return True
        return False

    def _try_parse(self, s: str):
        if not s:
            return None
        if sp is not None and parse_latex is not None and re.search(r"\\[A-Za-z]+", s):
            try:
                return parse_latex(_strip_and_replace(s))
            except Exception:
                pass
        if sp is not None:
            try:
                return sympify(_latex_cleanup(s), rational=True)
            except Exception:
                return None
        return None

    def _expression_equal(self, a: str, b: str) -> bool:
        if a == b and a != "":
            return True
        if sp is None:
            return False
        A = self._try_parse(a)
        B = self._try_parse(b)
        if A is None or B is None:
            return False
        try:
            if simplify(A - B) == 0:
                return True
        except Exception:
            pass
        # numeric expressions without symbols
        try:
            if not (A.has(sp.Symbol) or B.has(sp.Symbol)):
                return abs(A.evalf() - B.evalf()) <= self.default_precision * 1.01
        except Exception:
            pass
        # sample a few points if variables exist
        try:
            vars_ = sorted(list((A.free_symbols | B.free_symbols)), key=lambda s: s.name)
            if not vars_:
                return False
            for val in (-2, -1, 0, 1, 2, 3):
                subs = {v: val for v in vars_}
                if abs(complex(A.evalf(subs=subs)) - complex(B.evalf(subs=subs))) > 1e-8:
                    return False
            return True
        except Exception:
            return False

    def _equation_equal(self, a: str, b: str) -> bool:
        if sp is None:
            return False
        if "=" not in a or "=" not in b:
            return False
        def simp(eq: str):
            lhs, rhs = eq.split("=", 1)
            return simplify(parse_latex(lhs) - parse_latex(rhs)) if parse_latex else simplify(sympify(_latex_cleanup(lhs)) - sympify(_latex_cleanup(rhs)))
        try:
            A = simp(a); B = simp(b)
            d1 = simplify(A / B)
            d2 = simplify(B / A)
            return (getattr(d1, 'is_Integer', False) and d1 != 0) or (getattr(d2, 'is_Integer', False) and d2 != 0)
        except Exception:
            return False

    def judge(self, gt: str, pred: str, *, unit: Optional[str] = None, precision: Optional[float] = None) -> bool:
        prec = precision if precision is not None else self.default_precision
        self.default_precision = prec
        gt0, pr0 = self.preprocess(gt, pred, unit)
        # split by commas at top level (order-insensitive)
        g_list = _split_top_level_commas(gt0)
        p_list = _split_top_level_commas(pr0)
        if len(g_list) != len(p_list):
            return False
        # try to match each ground-truth with some prediction
        used = [False] * len(p_list)
        for g in g_list:
            ok = False
            for j, p in enumerate(p_list):
                if used[j]:
                    continue
                # expand ± on either side
                cand_g = _variants_pm(g)
                cand_p = _variants_pm(p)
                matched = False
                for cg in cand_g:
                    for cp in cand_p:
                        if cg == cp and cg != "":
                            matched = True; break
                        if (cg.startswith(("(", "[")) and cg.endswith((")", "]")) and
                            cp.startswith(("(", "[")) and cp.endswith((")", "]")) and self._interval_equal(cg, cp)):
                            matched = True; break
                        if self._numeric_equal(cg, cp) or self._expression_equal(cg, cp) or self._equation_equal(cg, cp):
                            matched = True; break
                    if matched:
                        break
                if matched:
                    used[j] = True
                    ok = True
                    break
            if not ok:
                return False
        return all(used)

##############################################################################
# Batch evaluation over MATH dataset
##############################################################################
def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield i, lst[i:i+size]

def precision_for_type(answer_type: Optional[str]) -> float:
    # Be slightly looser for Numerical (accounts for rounding); otherwise keep tight.
    if not answer_type:
        return 1e-8
    t = answer_type.lower()
    if "numerical" in t or "number" in t:
        return 1e-6
    return 1e-8

def evaluate_obench_vllm(dataset, llm: LLM, tokenizer, limit: Optional[int] = None, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, seed: int = 123, max_tokens: int = 2048, batch_size: int = 16, save_incorrect_path: Optional[str] = None,) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []

    N = len(dataset)
    if limit is not None:
        N = min(N, limit)

    judge = OBJudge()

    indices = []
    for i in range(N):
        rec = dataset[i]
        if rec.get("final_answer") is not None:
            indices.append(i)
    if not indices:
        print("No evaluable records with final_answer found.")
        return 0.0, [], []

    for start, idxs in _chunk(indices, batch_size):
        qs: List[str] = []
        recs: List[Dict[str, Any]] = []
        for k in idxs:
            rec = dataset[k]
            recs.append(rec)
            q = rec.get("question") or rec.get("problem") or ""
            qs.append(q)

        gens_batch: List[List[List[str]]] = batched_generate_vllm(
            qs, llm, tokenizer, n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, seed=seed + start,
        )

        for j, k in enumerate(idxs):
            rec = recs[j]
            q = qs[j]
            gold, unit, ans_type = ob_extract_gold(rec)
            precision = precision_for_type(ans_type)
            gens = gens_batch[j]
            preds = [ob_extract_pred(t) for t in gens]

            ok = False
            if preds:
                ok = judge.judge(gold, preds[0], unit=unit, precision=precision)
            total += 1
            correct += int(bool(ok))

            logs.append({
                "idx": k,
                "question": q,
                "gold": gold,
                "unit": unit,
                "answer_type": ans_type,
                "gens": gens,
                "preds": preds,
                "correct_first": bool(ok),
            })
            if not ok:
                incorrect_samples.append({
                    "idx": k,
                    "question": q,
                    "gold": gold,
                    "unit": unit,
                    "answer_type": ans_type,
                    "pred": preds[0] if preds else "",
                    "gens": gens,
                })

        if (total % 20) == 0:
            acc = 100.0 * correct / total
            print(f"[{total}/{N}] running acc = {acc:.2f}%")

    acc = 100.0 * correct / max(total, 1)
    print(f"OlympiadBench Accuracy = {acc:.2f}% on {total} examples.")

    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples

if __name__ == "__main__":

    # scorer = OBJudge()
    # tests_ok_minimal = [
    #     (r"\boxed{1/2}",           "1/2", True),     # 박스 추출
    #     ("Answer: 0.5",            "1/2", True),     # 'Answer:' 라인 + 0.5 ↔ 1/2
    #     (r"\boxed{5\sqrt{2}}",     r"\sqrt{50}", True),  # 심볼릭 동치
    #     (r"30^{\circ}",            "30", True),      # degree 마커 제거
    #     (r"50%",                   "0.5", True),     # 퍼센트 해석 (50/100=0.5)
    # ]
    # tests_multi_answer = [
    #     (r"\boxed{2,3}",           "3,2", True),     # 콤마 구분, 순서 무관
    #     (r"\boxed{2}\n...\n\boxed{3}", "2,3", True), # 여러 box → 내부적으로 '2,3'으로 합쳐짐
    #     (r"(1,2]",                 "(1,2]", True),   # 동일 문자열(튜플/인터벌 헷갈림 방지용)
    #     (r"(1,2] \cup [3,4)",      r"(1,2]\cup[3,4)", True),  # union 간격/공백 무시
    # ]
    # tests_should_fail = [
    #     (r"(1,2]",                 "(1,2)", False),  # 닫힘/열림 다름
    #     (r"2,3",                   "2,3,4", False),  # 개수 불일치
    #     (r"\boxed{1/3}",           "0.5", False),    # 값 불일치
    # ]

    # def run_cases(cases, *, name="cases", unit=None, precision=None):
    #     print(f"\n== {name} ==")
    #     for i, (pred, gold, want) in enumerate(cases, 1):
    #         got = scorer.judge(gold, pred, unit=unit, precision=precision)  # gold, pred 순서 주의!
    #         ok  = "OK" if got == want else "MISMATCH"
    #         print(f"[{i:02d}] {repr(pred)} vs {repr(gold)} -> {got} (want {want})  [{ok}]")

    # run_cases(tests_ok_minimal, name="A. minimal")
    # run_cases(tests_multi_answer, name="A. multi-answer")
    # run_cases(tests_should_fail, name="A. should-fail")

    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16", 
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90, 
        max_model_len=4096,
        # quantization="bitsandbytes", 
        enforce_eager=False,
        enable_prefix_caching=True,  
        # distributed_executor_backend="mp",      
    )

    # Load Datasets
    dataset = load_olympiadbench_english("train")
    def _as_dict(x):
        if isinstance(x, dict):
            return x
        try:
            return dict(x)
        except Exception:
            return {k: x[k] for k in x.keys()}  # type: ignore

    dataset = dataset.map(lambda r: _as_dict(r))

    # Evaluation
    print("Starting evaluation on OlympiadBench dataset...", flush=True)
    incorr_path = "/home/leena/prm_shaping/analysis/incorr_olym_vllm_0819.json"
    acc, logs, incorrect = evaluate_obench_vllm(
        dataset, llm=llm, tokenizer=tokenizer, n=1,
        max_tokens=3096,
        batch_size=32,
        temperature=0, top_p=1.0,
        save_incorrect_path=incorr_path,
    )

    # Optionally dump logs next to incorrect file if provided
    path_logs = os.path.splitext(incorr_path)[0] + "_logs.json"
    with open(path_logs, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"Saved full logs to: {path_logs}")
