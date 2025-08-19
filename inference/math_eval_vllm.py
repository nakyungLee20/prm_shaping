import os, re, json, math, random
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple
import sympy as sp
from sympy.parsing.latex import parse_latex
import numpy as np

from datasets import load_dataset
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
_UNITS_RX = re.compile(r"\\text\{[^}]*\}\s*$")
_DEG_RX   = re.compile(r"\^\{?\\?circ\}?|°")  # ^{\circ}, ^\circ, or °
_MOD_TAIL = re.compile(r"\s*(?:\\?mod|\\bmod)\s*[0-9]+\s*$", re.IGNORECASE)
_RATIO_RX = re.compile(r"^\s*([+-]?\d+)\s*:\s*([+-]?\d+)\s*$")
_ABS_LATEX = re.compile(r"\\left\|([^|]+)\\right\|")
_LATEX_STRIP = [
    (re.compile(r"\\left\s*"), ""),
    (re.compile(r"\\right\s*"), ""),
    (re.compile(r"\\!"), ""),
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\operatorname\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    # thin/med spaces
    (re.compile(r"\\,|\\;|\\:|\\\s+"), " "),
]

BOXED = re.compile(r"\\boxed\{([^}]*)\}")
BOXED_STAR = re.compile(r"\\boxed\*\{([^}]*)\}")
ANS_LINE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
HINT_ANS = re.compile(r"(?i)\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+equal\s+to|is|=|equals|:)\s*([^\n]+)")

# Useful latex-stripping patterns
_LEAD_PUNCT  = re.compile(r"^[([\{\s]+")
_TRAIL_PUNCT = re.compile(r"[)\].,;:\s]+$")
_UNIT_TAIL = re.compile(
    r"\s*(dollars?|cents?|percent|perc\.?|pts?|points?|years?|year|hrs?|hours?|mins?|minutes?|secs?|seconds?|cm|m|km|ft|feet|in|inch|inches|units?)\s*$",
    re.IGNORECASE,
)

NUM_LIKE = re.compile(r"[-+]?\d+(?:\.\d+)?")
MIXED_FRAC = re.compile(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)")
PURE_FRAC  = re.compile(r"([+-]?\d+)\s*/\s*(\d+)")
DEC_SCI    = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?")

NO_SOLUTION_TOKENS = {
    "no solution", "nosolution", "none", "empty set", "emptyset", "∅", "varnothing", "\\varnothing", "{}",
}
DNE_TOKENS = {"dne", "does not exist", "undefined", "not defined"}
INF_TOKENS = {"infinity", "\\infty", "∞"}

def _fix_fracs(s: str) -> str:
    # "\\frac1b" or "\\frac12" -> "\\frac{1}{b}" and "\\frac{1}{2}"; tolerant to short tails
    parts = s.split("\\frac")
    out = parts[0]
    for tail in parts[1:]:
        out += "\\frac"
        if not tail:
            continue
        if tail[0] == "{":  # already braced
            out += tail
            continue
        if len(tail) == 1:
            out += "{" + tail + "}"
            continue
        a, b, rest = tail[0], tail[1], tail[2:]
        if b != "{":
            out += "{" + a + "}{" + b + "}" + rest
        else:
            out += "{" + a + "}" + b + rest
    return out

def _fix_sqrt(s: str) -> str:
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    out = parts[0]
    for tail in parts[1:]:
        if tail and tail[0] != "{":
            out += "\\sqrt{" + tail[0] + "}" + tail[1:]
        else:
            out += "\\sqrt" + tail
    return out

def _expand_pm(expr: str) -> List[str]:
    if "±" not in expr:
        return [expr]
    # replace first ± with + and - variants
    a = expr.replace("±", "+", 1)
    b = expr.replace("±", "-", 1)
    # if multiple ±, a recursive expansion could be added; for now, handle first two passes
    if "±" in a or "±" in b:
        return sum((_expand_pm(x) for x in (a, b)), [])
    return [a, b]

def _strip_wrappers(s: str) -> str:
    s = _LEAD_PUNCT.sub("", s)
    s = _TRAIL_PUNCT.sub("", s)
    return s.strip()

def _remove_right_units(s: str) -> str:
    s = _UNITS_RX.sub("", s)
    return s

def _basic_clean(s: str) -> str:
    if not s:
        return ""
    t = s.replace("\n", " ")
    t = t.replace("\\$", "$")
    t = t.replace("\\\\", "\\")
    for rx, rep in _LATEX_STRIP:
        t = rx.sub(rep, t)
    t = _remove_right_units(t)
    t = _DEG_RX.sub("", t)  # drop degree marker (treat 30^\circ as 30)
    t = t.replace("%", "")
    # fix sqrt/bracing and frac short-hands
    t = _fix_sqrt(_fix_fracs(t))
    # unify spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t

BOXED_OPEN = re.compile(r'\\boxed\*?\s*\{')  # \boxed{...} 와 \boxed*{...} 모두
def _find_boxed_all(text: str):
    out = []
    for m in BOXED_OPEN.finditer(text):
        i = m.end()  # 첫 '{' 다음 위치
        depth, j = 1, i
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == '\\':         # 이스케이프된 문자/괄호 건너뛰기
                j += 2
                continue
            if ch == '{': depth += 1
            elif ch == '}': depth -= 1
            j += 1
        if depth == 0:            # 정상적으로 닫힘
            out.append(text[i:j-1])
    return out

def _post_clean_numberish(s: str) -> str:
    if not s:
        return ""
    t = s
    # Prefer explicit boxed last
    boxed = _find_boxed_all(t)
    if boxed:
        t = boxed[-1]
    else:
        pass
    # pick last numeric-ish token
    hits = list(DEC_SCI.finditer(t))
    if hits:
        tok = hits[-1].group(0)
    else:
        # fractions / mixed fraction
        m3 = list(MIXED_FRAC.finditer(t))
        if m3:
            tok = m3[-1].group(0)
        else:
            pf = list(PURE_FRAC.finditer(t))
            tok = pf[-1].group(0) if pf else t
    tok = tok.replace("$", "").replace(",", "")
    tok = _UNIT_TAIL.sub("", tok)
    tok = _MOD_TAIL.sub("", tok)
    tok = _strip_wrappers(tok)
    return tok

def math_extract_pred(text: str) -> str:
    """Extract model's final answer string from the generated text."""
    if not text:
        return ""
    # 1) try boxed
    boxed = _find_boxed_all(text)
    if boxed:
        cand = _post_clean_numberish(boxed[-1])  # 마지막 \boxed 내용 선택
        if cand:
            return cand
    m_star = list(BOXED_STAR.finditer(text))
    if m_star:
        cand = _post_clean_numberish(m_star[-1].group(1))
        if cand:
            return cand
    # 2) "Answer: ..."
    m = ANS_LINE.search(text)
    if m:
        cand = _post_clean_numberish(m.group(1))
        if cand:
            return cand
    # 3) hints like "final answer is ..."
    m2 = HINT_ANS.search(text)
    if m2:
        cand = _post_clean_numberish(m2.group(1))
        if cand:
            return cand
    # 4) fallback: last numberish / or last math-like token
    return _post_clean_numberish(text)

def math_extract_gold(rec: Dict[str, Any]) -> str:
    # Some variants provide an `answer` field already
    if isinstance(rec, dict):
        for key in ["answer", "final_answer", "finalAnswer", "ans"]:
            ans = rec.get(key)
            if ans and str(ans).strip():
                return _strip_wrappers(str(ans))
        sol = rec.get("solution") or rec.get("solutions") or rec.get("proof") or ""
    else:
        sol = ""
    if sol:
        # prefer last boxed
        boxed = _find_boxed_all(sol)
        if boxed:
            cand = _post_clean_numberish(boxed[-1])  # 마지막 \boxed 내용 선택
            if cand:
                return cand
        m_all = list(BOXED.finditer(sol))
        if m_all:
            return _strip_wrappers(m_all[-1].group(1))
        m_star = list(BOXED_STAR.finditer(sol))
        if m_star:
            return _strip_wrappers(m_star[-1].group(1))
        m = ANS_LINE.search(sol)
        if m:
            return _strip_wrappers(m.group(1))
        # last inline math $...$
        inlines = list(re.finditer(r"\$([^$]+)\$", sol))
        if inlines:
            return _strip_wrappers(inlines[-1].group(1))
    # last resort: pick last numeric-ish token
    return _post_clean_numberish(sol)

##############################################################################
# Numeric + symbolic normalization and comparison
##############################################################################
def _normalize_special_tokens(s: str) -> str:
    t = _basic_clean(s).lower().strip()
    if t in NO_SOLUTION_TOKENS:
        return "<NO_SOLUTION>"
    if t in DNE_TOKENS:
        return "<DNE>"
    if t in INF_TOKENS:
        return "<INF>"
    return _basic_clean(s)

def _ratio_to_frac(s: str) -> Optional[str]:
    m = _RATIO_RX.fullmatch(s)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if b == 0:
        return None
    f = Fraction(a, b)
    return f"{f.numerator}/{f.denominator}"

def normalize_number(x: str) -> str:
    if x is None:
        return ""
    s = _normalize_special_tokens(x)
    # take the last \boxed{...}
    b = _find_boxed_all(s)
    if b:
        s = b[-1]
    # "Answer: ..." / "final answer is ..." remove prefix
    m = ANS_LINE.search(s)
    if m:
        s = m.group(1)
    else:
        m2 = HINT_ANS.search(s)
        if m2:
            s = m2.group(1)
    # 변수/함수 할당 앞부분 제거: x=..., f(x)=... -> ...
    s = re.sub(r'^\s*[A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?\s*=\s*', '', s)
    # \frac{a}{b} -> a/b  (수치 정규화에서도 처리)
    s = re.sub(r'\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}', r'\1/\2', s)
    s = s.replace(",", "").replace("$", "").strip()
    s = re.sub(r"\s*%\s*$", "", s)
    s = _MOD_TAIL.sub("", s)
    # ratio a:b
    r = _ratio_to_frac(s)
    if r is not None:
        return r
    # mixed fraction a b/c
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
    # pure fraction -> reduced
    m = re.fullmatch(r"\s*([+-]?\d+)\s*/\s*(\d+)\s*\Z", s)
    if m:
        try:
            frac = Fraction(int(m.group(1)), int(m.group(2)))
            return f"{frac.numerator}/{frac.denominator}"
        except ZeroDivisionError:
            return ""
    # decimal/int (keep limited precision)
    try:
        v = float(Decimal(s))
        if math.isfinite(v) and abs(v - round(v)) < 1e-12:
            return str(int(round(v)))
        return f"{v:.12g}"
    except (InvalidOperation, ValueError):
        return s

def _latex_cleanup(expr: str) -> str:
    if not expr:
        return ""
    t = _basic_clean(expr)
    # map latex to ascii-friendly forms
    # frac -> (a)/(b): sympy-friendly without latex parser
    t = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1)/(\2)", t)
    t = t.replace("\\cdot", "*").replace("\\times", "*")
    t = t.replace("^", "**")
    # ±/\pm -> special token retained for later branching
    t = t.replace("\\pm", "±")
    # absolute value
    t = _ABS_LATEX.sub(lambda m: f"Abs({m.group(1)})", t)
    # binomial/choose
    t = re.sub(r"\\binom\{([^}]*)\}\{([^}]*)\}", r"binomial(\1,\2)", t)
    t = re.sub(r"\{([^}]*)\}\\choose\{([^}]*)\}", r"binomial(\1,\2)", t)
    # sqrt{..}
    t = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", t)
    t = re.sub(r"(?<![A-Za-z\\])sqrt\{([^}]*)\}", r"sqrt(\1)", t)
    # strip leftover latex spacing
    t = re.sub(r"(\d|\))\s*([A-Za-z]+)\s*\(", r"\1*\2(", t)
    t = re.sub(r"\\\s+", " ", t)
    return t.strip()

def _sympy_parse(expr: str):
    if expr is None:
        return None
    raw = expr.strip()
    if not raw:
        return None
    # looks_latex = bool(re.search(r"\\[a-zA-Z]+|\{|\}|\^|\\frac|\\sqrt|\\boxed", raw))
    looks_latex = bool(re.search(r"\\[a-zA-Z]+", raw))
    # Try latex first
    if looks_latex and parse_latex is not None:
        try:
            return parse_latex(_basic_clean(raw))
        except Exception:
            pass
    # Fallback to ascii sympify
    try:
        return sp.sympify(_latex_cleanup(raw), rational=True)
    except Exception:
        return None

##############################################################################
# Container handling (tuples, sets, intervals) #
##############################################################################
def _as_container(s: str) -> Tuple[Optional[str], List[str]]:
    """Detect simple containers: tuple '(a,b,...)', set '{a,b,...}', interval '(..., ...)' with brackets.
    Returns (kind, items) where kind in {"tuple","set","interval",None}.
    """
    t = _basic_clean(s)
    if not t:
        return None, []
    if t.startswith("{") and t.endswith("}"):
        return "set", _split_top_level_commas(t[1:-1])
    if t.startswith("(") and t.endswith(")"):
        # Heuristic: if contains any of [ or ] then treat as interval like (a,b], else tuple
        inner = t[1:-1]
        if any(ch in inner for ch in "[]"):
            return "interval", [inner]
        return "tuple", _split_top_level_commas(inner)
    # Explicit interval notation like [a,b), [a,b], etc.
    if (t.startswith("[") or t.startswith("(")) and (t.endswith("]") or t.endswith(")")):
        return "interval", [t]
    # set keyword: e.g., "{1,2}" may already be handled; otherwise none
    return None, []

def _compare_sets(pred_items: List[str], gold_items: List[str], atol: float) -> bool:
    # Order-insensitive; allow one-to-one matching with removal
    used = [False]*len(gold_items)
    for p in pred_items:
        hit = False
        for j, g in enumerate(gold_items):
            if not used[j] and grade_math_answer(p, g, atol=atol):
                used[j] = True
                hit = True
                break
        if not hit:
            return False
    # All pred matched and counts identical
    return all(used) and len(pred_items) == len(gold_items)

def _compare_tuples(pred_items: List[str], gold_items: List[str], atol: float) -> bool:
    if len(pred_items) != len(gold_items):
        return False
    return all(grade_math_answer(p, g, atol=atol) for p, g in zip(pred_items, gold_items))

def _split_top_level_commas(s: str) -> List[str]:
    items, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            items.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    items.append("".join(buf))
    return [x.strip() for x in items if x.strip()]

##############################################################################
# Numeric parsing & symbolic equality
##############################################################################
def _maybe_unbox_for_compare(s: str) -> str:
    if not s:
        return s
    boxed = _find_boxed_all(s)
    return boxed[-1] if boxed else s

def _parse_numeric(s: str):
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    # Special tokens
    if t in {"<NO_SOLUTION>", "<DNE>", "<INF>"}:
        return t
    # ratio handled earlier by normalize_number
    # mixed fraction a b/c
    m = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", t)
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if whole < 0 else 1
        whole = abs(whole)
        try:
            return Fraction(whole * den + num, den) * sign
        except ZeroDivisionError:
            return None
    # pure fraction
    m = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", t)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    # decimal / int / sci
    try:
        v = float(Decimal(t))
        return v
    except (InvalidOperation, ValueError):
        return None

def _sympy_equal(a_str: str, b_str: str) -> bool:
    a = _sympy_parse(a_str)
    b = _sympy_parse(b_str)
    if a is None or b is None:
        return False
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        # Sets or intervals
        if isinstance(a, (sp.Set, sp.Interval)) or isinstance(b, (sp.Set, sp.Interval)):
            return sp.simplify(a) == sp.simplify(b)
    except Exception:
        pass
    # Numeric sampling fallback for expressions with symbols
    try:
        vars = sorted(list((a.free_symbols | b.free_symbols)), key=lambda s: s.name)  # type: ignore[attr-defined]
        if not vars:
            return False
        for val in [-2, -1, 0, 1, 2, 3]:
            subs = {v: val for v in vars}
            av = complex(a.evalf(subs=subs))  # type: ignore[call-arg]
            bv = complex(b.evalf(subs=subs))  # type: ignore[call-arg]
            if abs(av - bv) > 1e-8:
                return False
        return True
    except Exception:
        return False
    
def grade_math_answer(pred: str, gold: str, atol: float = 1e-6) -> bool:
    """Equivalence check for MATH/GSM-style answers.
    Strategy:
      0) Normalize special tokens (NO_SOLUTION/DNE/INF) and short-circuit.
      1) Handle container forms: sets {..}, tuples (..,..), intervals.
      2) Try exact string normalization on numbers/fractions/ratios.
      3) Try exact rational/float compare (with tolerance for floats).
      4) Try symbolic equivalence via SymPy (if available).
      5) Try un-normalized sympy as last resort.
    """
    if pred is None or gold is None:
        return False
    pred = _maybe_unbox_for_compare(pred)
    gold = _maybe_unbox_for_compare(gold)
    # Special canonical tokens
    p0 = _normalize_special_tokens(pred)
    g0 = _normalize_special_tokens(gold)
    if p0 in {"<NO_SOLUTION>", "<DNE>", "<INF>"} or g0 in {"<NO_SOLUTION>", "<DNE>", "<INF>"}:
        return p0 == g0
    # Container-aware comparison
    pk, pitems = _as_container(p0)
    gk, gitems = _as_container(g0)
    if pk and gk:
        if pk != gk:
            # Allow set vs tuple when both represent unordered multi-answers by heuristic: if both lengths>1 and elements match as a set
            if {pk, gk} == {"set", "tuple"}:
                return _compare_sets(pitems, gitems, atol)
            return False
        if pk == "set":
            return _compare_sets(pitems, gitems, atol)
        if pk == "tuple":
            return _compare_tuples(pitems, gitems, atol)
        if pk == "interval":
            # Delegate to sympy
            return _sympy_equal(p0, g0)
    # Non-container path with ± expansion
    p_cands = _expand_pm(p0)
    g_cands = _expand_pm(g0)
    
    for pc in p_cands:
        pn = normalize_number(pc)
        for gc in g_cands:
            gn = normalize_number(gc)
            # 1) Equal normalized strings
            if pn == gn and pn != "":
                return True
            # 2) Numeric compare
            p_num = _parse_numeric(pn)
            g_num = _parse_numeric(gn)
            if p_num is not None and g_num is not None:
                if isinstance(p_num, Fraction) and isinstance(g_num, Fraction):
                    if p_num == g_num:
                        return True
                else:
                    try:
                        pv = float(p_num) if not isinstance(p_num, float) else p_num
                        gv = float(g_num) if not isinstance(g_num, float) else g_num
                        if math.isfinite(pv) and math.isfinite(gv) and abs(pv - gv) <= atol:
                            return True
                    except Exception:
                        pass
            # 3) Symbolic compare on normalized
            if _sympy_equal(pn, gn):
                return True
            # 4) Symbolic compare on raw
            if _sympy_equal(pc, gc):
                return True
    return False

##############################################################################
# Batch evaluation over MATH dataset
##############################################################################
def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield i, lst[i:i+size]

def evaluate_math_vllm(dataset, llm: LLM, tokenizer, limit: Optional[int] = None, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, seed: int = 123, max_tokens: int = 2048, batch_size: int = 16, save_incorrect_path: Optional[str] = None,) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []

    N = len(dataset)
    if limit is not None:
        N = min(N, limit)

    for start, idxs in _chunk(list(range(N)), batch_size):
        qs: List[str] = []
        recs: List[Dict[str, Any]] = []
        for k in idxs:
            rec = dataset[k]
            recs.append(rec)
            # Common MATH fields: `problem`, `solution`, maybe `answer`
            q = rec.get("problem") or rec.get("question") or ""
            qs.append(q)

        gens_batch: List[List[List[str]]] = batched_generate_vllm(
            qs, llm, tokenizer, n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, seed=seed + start,
        )

        for j, k in enumerate(idxs):
            rec = recs[j]
            q = qs[j]
            gold = math_extract_gold(rec)
            gens = gens_batch[j]
            preds = [math_extract_pred(t) for t in gens]

            is_correct = grade_math_answer(preds[0], gold)
            total += 1
            correct += int(is_correct)

            logs.append({
                "idx": k,
                "question": q,
                "gold": gold,
                "gens": gens,
                "preds": preds,
                "correct_first": bool(is_correct),
            })

            if not is_correct:
                incorrect_samples.append({
                    "idx": k,
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
    print(f"MATH Accuracy = {acc:.2f}% on {total} examples.")

    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples

if __name__ == "__main__":

    # tests = [
    #     ("\\boxed{1/2}", "1/2", True),
    #     ("Answer: 0.5", "1/2", True),
    #     ("final answer is \\frac12", "1/2", True),
    #     ("(1,2)", "(1,2)", True),
    #     ("{2,1}", "{1,2}", True),
    #     ("(1,2)", "{1,2}", False),
    #     ("±\\sqrt{2}", "-sqrt(2)", True),
    #     ("3:4", "3/4", True),
    #     ("30^{\\circ}", "30", True),
    #     ("\\boxed{5\\sqrt{2}}", "sqrt{50}", True),
    #     ("no solution", "\\varnothing", True),
    #     ("infinity", "\\infty", True),
    #     ("x=3", "3", True),
    #     ("$-\n 2$", "-2", True),
    #     ("\\frac{x + 2}{7}}", "x/7 + 2/7", True),
    #     (".5", "(\\boxed{\\frac{1}{2}}\\)", True),
    # ]
    # for i, (p, g, want) in enumerate(tests, 1):
    #     got = grade_math_answer(p, g)
    #     print(f"[{i}] {p!r} vs {g!r} -> {got} (want {want})")

    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16", 
        tensor_parallel_size=4,
        gpu_memory_utilization=0.80, 
        max_model_len=4096,
        quantization="bitsandbytes", 
        enforce_eager=True, 
    )

    # Load Datasets
    dataset = load_dataset("HuggingFaceTB/MATH", "all", split="test")
    # Harmonize records to dicts (for gold extraction)
    def _as_dict(x):
        if isinstance(x, dict):
            return x
        # datasets lib may hand out row objects; coerce conservatively
        try:
            return dict(x)
        except Exception:
            return {k: x[k] for k in x.keys()}  # type: ignore

    dataset = dataset.map(lambda r: _as_dict(r))

    # Evaluation
    incorr_path = "/home/leena/prm_shaping/analysis/incorr_gsm8k_vllm_0818.json"
    acc, logs, incorrect = evaluate_math_vllm(dataset, llm=llm, tokenizer=tokenizer, n=1,
        max_tokens=3096,
        batch_size=16,
        save_incorrect_path=incorr_path,
    )

    # Optionally dump logs next to incorrect file if provided
    path_logs = os.path.splitext(incorr_path)[0] + "_logs.json"
    with open(path_logs, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"Saved full logs to: {path_logs}")
