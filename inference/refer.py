# -*- coding: utf-8 -*-
"""
Qwen-style scorers for math benchmarks
-------------------------------------

This module provides four dataset-specific scorers that mirror the parsing
and grading behavior used in Qwen2.5-Math's evaluation code
(https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation), adapted into a
simple, reusable class interface:

    - MATHScorer     : for the MATH dataset (AoPS/competition_math)
    - OBScorer       : for OlympiadBench
    - OMNIScorer     : for Omni-MATH
    - MMLUScorer     : for MMLU (STEM subset; multiple-choice A–D)

Each scorer exposes three public methods:
    - extract_gold(gold_field: str) -> str
    - extract_pred(text: str) -> str
    - grade(pred: str, gold: str) -> bool

Design notes
~~~~~~~~~~~~
* We follow Qwen's evaluation/parser.py patterns:
  - Prefer the **last \\boxed{...}** span when present.
  - Otherwise, accept explicit "Answer:" / "final answer is"-style hints.
  - Fall back to the **last number** in the text for numeric tasks.
  - For multiple choice (MMLU), extract the last capital letter among A–E.
* Numeric comparison uses tolerant matching (abs diff <= atol) and exact
  fraction equality when possible. We normalize text by removing $, commas,
  trailing percent and lightweight unit tokens.
* Optional symbolic check (via sympy) is included: if both sides parse to
  valid SymPy expressions, we attempt equality via `.equals()`.
* We intentionally avoid heavyweight dependencies — the symbolic path is
  best-effort and gracefully disabled if `sympy` / `antlr` is unavailable.

The goal is to cover the corner cases handled by Qwen's public evaluation
scripts while keeping the interface compact for integration into your own
evaluation harness.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional, Tuple

# ---- Optional symbolic backend (best-effort) ---------------------------------
try:
    import sympy as _sp  # type: ignore
    try:
        # Try LaTeX parser if available; otherwise fall back to sympify
        from sympy.parsing.latex import parse_latex as _parse_latex  # type: ignore
        _HAS_LATEX = True
    except Exception:  # antlr missing, etc.
        _parse_latex = None
        _HAS_LATEX = False
    _HAS_SYMPY = True
except Exception:
    _sp = None
    _parse_latex = None
    _HAS_LATEX = False
    _HAS_SYMPY = False


# ---- Common regexes & utilities ----------------------------------------------
_BOXED = re.compile(r"\\boxed\s*\{")  # detect a boxed-start and then parse
_ANS_LINE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_HINT_ANS = re.compile(
    r"(?i)\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+equal\s+to|is|=|equals|:)\s*([^\n]+)"
)
# numbers: mixed fraction, pure fraction, decimal/scientific
_MIXED_FRAC = re.compile(r"[-+]?\d+\s+\d+/\d+")
_PURE_FRAC = re.compile(r"[-+]?\d+/\d+")
_DEC_SCI = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?")

_LEAD_PUNCT = re.compile(r"^[([\{\s]+")
_TRAIL_PUNCT = re.compile(r"[)\].,;:\s]+$")
_UNIT_TAIL = re.compile(
    r"\s*(dollars?|cents?|percent|perc\.?|pts?|points?|years?|year|hrs?|hours?|mins?|minutes?|secs?|seconds?|cm|m|km|mm|in|ft|yd|kg|g|mg|lb|lbs)\s*$",
    re.IGNORECASE,
)

# Choice letter (for multiple-choice datasets)
_CHOICE_RE = re.compile(r"\b([A-E])\b")

# Chinese "答案是"
_ZH_ANSWER = re.compile(r"答案是\s*([^\n]+)")


def _strip_wrappers(s: str) -> str:
    s = _LEAD_PUNCT.sub("", s)
    s = _TRAIL_PUNCT.sub("", s)
    return s.strip()


def _strip_units_commas_currency_percent(x: str) -> str:
    s = x.replace(",", "").replace("$", "").strip()
    s = re.sub(r"\s*%\s*$", "", s)
    s = re.sub(r"\s*percent\s*$", "", s, flags=re.IGNORECASE)
    s = _UNIT_TAIL.sub("", s)
    return _strip_wrappers(s)


def _latex_frac_to_plain(s: str) -> str:
    """Convert simple LaTeX fractions like \frac{a}{b} (and nested spaces)
    into plain a/b. Handles a limited but common subset sufficient for grading.
    """
    # Remove $ … $ and \( … \)
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\\(|\\\)", "", s)
    s = re.sub(r"\\,|\\!|\\;|\\:|\\tfrac", " ", s)

    def repl(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        return f"{a}/{b}"

    # Normalize nested spaces and braces
    s = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_last_boxed(text: str) -> Optional[str]:
    """Find the **last** \boxed{...} span, allowing nested braces (simple stack).
    Mirrors Qwen's manual scan in `parser.extract_answer` for \boxed.
    """
    idx = 0
    last_payload: Optional[str] = None
    while True:
        m = _BOXED.search(text, idx)
        if not m:
            break
        i = m.end()  # position after '{'
        depth = 1
        buf = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == '{':
                depth += 1
                buf.append(c)
            elif c == '}':
                depth -= 1
                if depth == 0:
                    i += 1
                    break
                buf.append(c)
            else:
                buf.append(c)
            i += 1
        if depth == 0:
            last_payload = ''.join(buf)
            idx = i
        else:
            # Unbalanced; stop
            break
    if last_payload is None:
        return None
    return _strip_units_commas_currency_percent(last_payload)


def _find_last_numberish(text: str) -> str:
    # Prefer mixed fraction / fraction / decimal-scientific in that order
    for rx in (_MIXED_FRAC, _PURE_FRAC, _DEC_SCI):
        hits = list(rx.finditer(text))
        if hits:
            tok = hits[-1].group(0)
            return _strip_units_commas_currency_percent(tok)
    # Fallback: last money/number token
    m = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not m:
        return ""
    last = m[-1]
    tok = [x for x in last if x][0]
    tok = tok.strip()
    for rgx in (",", r"\$", r"(?s).*#### ", r"\.$"):
        tok = re.sub(rgx, "", tok)
    return tok.strip()


def _choice_answer_clean(s: str) -> str:
    """Extract the last standalone choice letter A–E from a string.
    Accepts patterns like "(A)", "Option C", "Answer: D.", etc.
    """
    cand = _CHOICE_RE.findall(s.upper())
    if cand:
        return cand[-1]
    # Also try common words
    m = re.search(r"(?:option|choice|answer)\s*[:=\-]?\s*([A-E])", s, re.I)
    if m:
        return m.group(1).upper()
    return s.strip().strip(".")


def _parse_numeric(s: Optional[str]) -> Optional[object]:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    # mixed frac: a b/c
    m = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", t)
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if whole < 0 else 1
        whole = abs(whole)
        try:
            return Fraction(whole * den + num, den) * sign
        except ZeroDivisionError:
            return None
    # pure frac: a/b
    m = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", t)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    # decimal/scientific
    try:
        v = float(Decimal(t))
        return v
    except (InvalidOperation, ValueError):
        return None


def _normalize_number(x: Optional[str]) -> str:
    if x is None:
        return ""
    s = _strip_units_commas_currency_percent(str(x))
    # mixed → irreducible fraction form "n/d"; otherwise decimal canonicalization
    m = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", s)
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if whole < 0 else 1
        whole = abs(whole)
        try:
            frac = Fraction(whole * den + num, den) * sign
            return f"{frac.numerator}/{frac.denominator}"
        except ZeroDivisionError:
            return ""
    m = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", s)
    if m:
        try:
            frac = Fraction(int(m.group(1)), int(m.group(2)))
            return f"{frac.numerator}/{frac.denominator}"
        except ZeroDivisionError:
            return ""
    try:
        v = float(Decimal(s))
        if math.isfinite(v) and abs(v - round(v)) < 1e-12:
            return str(int(round(v)))
        return f"{v:.12g}"
    except (InvalidOperation, ValueError):
        return s


def _symbolic_equal(lhs: str, rhs: str) -> bool:
    """Best-effort symbolic equivalence check using SymPy if available.
    It attempts LaTeX parse first (if available), then falls back to sympify.
    Non-fatal: returns False if parsing fails or sympy is not installed.
    """
    if not _HAS_SYMPY:
        return False
    a, b = _latex_frac_to_plain(lhs), _latex_frac_to_plain(rhs)
    try:
        if _HAS_LATEX and ("\\" in lhs or "\\" in rhs or "$" in lhs or "$" in rhs):
            x = _parse_latex(a) if _parse_latex else _sp.sympify(a)
            y = _parse_latex(b) if _parse_latex else _sp.sympify(b)
        else:
            x = _sp.sympify(a)
            y = _sp.sympify(b)
        try:
            return bool(x.equals(y))
        except Exception:
            # last resort: simplify difference
            return _sp.simplify(x - y) == 0
    except Exception:
        return False


def _grade_numeric_like(p: str, g: str, *, atol: float) -> bool:
    p_norm, g_norm = _normalize_number(p), _normalize_number(g)
    if p_norm == g_norm and p_norm != "":
        return True
    p_val, g_val = _parse_numeric(p_norm), _parse_numeric(g_norm)
    if p_val is None or g_val is None:
        # try symbolic if numeric failed
        return _symbolic_equal(p.strip(), g.strip())
    if isinstance(p_val, Fraction) and isinstance(g_val, Fraction):
        return p_val == g_val
    try:
        pv = float(p_val) if not isinstance(p_val, float) else p_val
        gv = float(g_val) if not isinstance(g_val, float) else g_val
        return math.isfinite(pv) and math.isfinite(gv) and abs(pv - gv) <= atol
    except Exception:
        return False


# ---- Base class ---------------------------------------------------------------
class _BaseScorer:
    def __init__(self, atol: float = 1e-6):
        self.atol = float(atol)

    # --- dataset-agnostic extractors ---
    def _extract_from_text(self, text: str, *, use_last_number: bool = True) -> str:
        if not text:
            return ""
        # 1) last \boxed{...}
        boxed = _parse_last_boxed(text)
        if boxed:
            return boxed
        # 2) explicit answer lines
        m = _ANS_LINE.search(text)
        if m:
            cand = _strip_units_commas_currency_percent(m.group(1))
            if cand:
                return cand
        m2 = _HINT_ANS.search(text)
        if m2:
            cand = _strip_units_commas_currency_percent(m2.group(1))
            if cand:
                return cand
        # Chinese: 答案是 ...
        mz = _ZH_ANSWER.search(text)
        if mz:
            cand = _strip_units_commas_currency_percent(mz.group(1))
            if cand:
                return cand
        # 3) last number fallback
        if use_last_number:
            return _find_last_numberish(text)
        return ""

    # --- public API ---
    def extract_gold(self, gold_field: str) -> str:
        raise NotImplementedError

    def extract_pred(self, text: str) -> str:
        raise NotImplementedError

    def grade(self, pred: str, gold: str) -> bool:
        raise NotImplementedError


# ---- MATH (competition_math) --------------------------------------------------
class MATHScorer(_BaseScorer):
    """Scorer for the MATH benchmark.

    Ground truth typically comes from the full solution text containing a
    final boxed answer (LaTeX). We adopt the "last boxed" rule, falling back
    to last-number extraction when necessary.
    """

    def extract_gold(self, gold_field: str) -> str:
        # gold_field is usually the LaTeX solution text
        # Prefer last boxed; otherwise fall back to last numberish
        boxed = _parse_last_boxed(gold_field)
        if boxed:
            return boxed
        return _strip_units_commas_currency_percent(_find_last_numberish(gold_field))

    def extract_pred(self, text: str) -> str:
        return self._extract_from_text(text, use_last_number=True)

    def grade(self, pred: str, gold: str) -> bool:
        return _grade_numeric_like(pred, gold, atol=self.atol)


# ---- OlympiadBench ------------------------------------------------------------
class OBScorer(_BaseScorer):
    """Scorer for OlympiadBench.

    The dataset stores final answers as strings (often wrapped by `$ ... $`).
    According to Qwen's parser, gold is `example["final_answer"][0].strip("$")`.
    We therefore strip math wrappers/units. Prediction uses last-boxed/answer-line
    logic identical to MATH.
    """

    def extract_gold(self, gold_field: str) -> str:
        s = str(gold_field).strip().strip("$")
        s = _latex_frac_to_plain(s)
        return _strip_units_commas_currency_percent(s)

    def extract_pred(self, text: str) -> str:
        return self._extract_from_text(text, use_last_number=True)

    def grade(self, pred: str, gold: str) -> bool:
        return _grade_numeric_like(pred, gold, atol=self.atol)


# ---- Omni-MATH ----------------------------------------------------------------
class OMNIScorer(_BaseScorer):
    """Scorer for Omni-MATH.

    Omni-MATH recommends the same rule as MATH: take the **last boxed** answer.
    If the gold field is a solution/answer string, we first look for boxed
    content; otherwise we fall back to the last numeric token.
    """

    def extract_gold(self, gold_field: str) -> str:
        boxed = _parse_last_boxed(gold_field)
        if boxed:
            return boxed
        return _strip_units_commas_currency_percent(_find_last_numberish(gold_field))

    def extract_pred(self, text: str) -> str:
        return self._extract_from_text(text, use_last_number=True)

    def grade(self, pred: str, gold: str) -> bool:
        return _grade_numeric_like(pred, gold, atol=self.atol)


# ---- MMLU (STEM subset) -------------------------------------------------------
class MMLUScorer(_BaseScorer):
    """Scorer for MMLU (STEM subjects) – multiple choice.

    Ground truth is a single letter in {A,B,C,D} (Qwen maps integer indices to
    A–D). We extract the **last** occurrence of a standalone choice letter (A–E)
    from predictions to be robust to chain-of-thought and extra text.
    """

    def extract_gold(self, gold_field: str) -> str:
        # Already a letter in most loaders
        s = str(gold_field).strip().upper().strip(".$)")
        if s and s[0] in "ABCDE":
            return s[0]
        # If wrapped like "$A$" or "(B)"
        return _choice_answer_clean(s)

    def extract_pred(self, text: str) -> str:
        return _choice_answer_clean(text)

    def grade(self, pred: str, gold: str) -> bool:
        p = self.extract_pred(pred).upper()
        g = self.extract_gold(gold).upper()
        return p[:1] == g[:1] and p[:1] in "ABCDE"


__all__ = [
    "MATHScorer",
    "OBScorer",
    "OMNIScorer",
    "MMLUScorer",
    # helpers exposed for testing/debugging
    "_normalize_number",
    "_parse_numeric",
    "_symbolic_equal",
]
