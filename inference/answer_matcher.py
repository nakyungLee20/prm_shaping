import re, math, unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import List, Optional

import sympy as sp
from sympy import Pow, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    parse_expr, implicit_multiplication_application,
    standard_transformations,
)

class MathAnswerScorer:
    """
    Judge equivalence between two MATH-style answers.
    -------
    >>> scorer = MathAnswerScorer()
    >>> scorer.answers_match("1/2,3/4", "\\frac34, \\frac12") # True
    """
    def __init__(self, *, default_atol: float = 1e-8):
        self._default_atol = default_atol
        self._pi = parse_latex("\\pi")

        # SymPy parsing transforms
        self._sympy_tf = standard_transformations + (implicit_multiplication_application,)
        self._special_signal_map = {
            "\\left": "",  "\\right": "",  "∶": ":",  "，": ",",   "$": "",
            "\\approx": "=", "\\simeq": "=", "\\sim": "=",  "^\\prime": "'",
            "^{\\prime}": "'", "^\\circ": "", "%": "",
        }
        self._NUM_SIMPLE = re.compile(r"^[\+\-]?\d*\.?\d+(e[\+\-]?\d+)?$", re.I)
        self._INTERVAL_RE = re.compile(r"^[\(\[]\s*[^,]+?\s*,\s*[^,]+?\s*[\)\]]$")
        self._ANGLE_RE    = re.compile(r"^[-+]?\d+(?:\.\d+)?\s*°$")
        self._SET_RE      = re.compile(r"^\{[^}]*\}$")
        self._TEXT_ALNUM  = re.compile(r"[^\w]")
        self._BOOLEAN_WORDS = { "yes","no","true","false","none","all","some","both","neither"}
        self._CHOICES = {"A","B","C","D","E"}
        self._MATRIX_RE = re.compile(r"\\begin\{[pb]matrix\}")
        self._CMPLX_RE = re.compile(
            r"^\s*([+\-]?\d*\.?\d+(?:e[+\-]?\d+)?)\s*"
            r"([+\-])\s*"
            r"(\d*\.?\d+(?:e[+\-]?\d+)?)\s*[ij]\s*$",
            re.I,
        )
        self._UNIT_MAP = {
            "mm":  1e-3,  "cm": 1e-2,  "m": 1.0, "km": 1e3, # length
            "mg":  1e-6,  "g":  1e-3,  "kg": 1.0, # mass
            "ms":  1e-3,  "s":  1.0,   "min": 60, "h": 3600, # time
            "pa":1, "kpa":1e3, "mpa":1e6, "bar":1e5, # pressure
        }
        self._UNIT_RE = re.compile(r"^\s*([+\-]?\d*\.?\d+(?:e[+\-]?\d+)?)\s*([a-zA-Z]+)\s*$")

    # ===================================================================== #
    #  Top‑level driver                                                     #
    # ===================================================================== #
    def answers_match( self, pred: str, gold: str, *, atol: float | None = None) -> bool:
        atol = self._default_atol if atol is None else atol
        if pred is None or gold is None:
            return False
        # ---------- Fast exact match after minimal canonicalisation ----- #
        if self._strip_string(pred) == self._strip_string(gold):
            return True
        # ---------- Fast exact match after minimal canonicalisation ----- #
        if self._is_choice_letter(gold):
            return self._choice_equal(pred, gold)
        # ---------- Multiple Answer match (comma, pm) ----- #
        plist = self._expand_plus_minus(self._split_by_comma(pred))
        glist = self._expand_plus_minus(self._split_by_comma(gold))
        # depth incoreect -> false
        if len(plist) != len(glist):
            return False
        # greedy permutation matching for tuple answers
        gold_remaining = glist.copy()
        for p in plist:
            for g in gold_remaining:
                if self._base_match(p, g, atol):
                    gold_remaining.remove(g)
                    break
            else:
                return False 
        return True 

    #  (_base_match)
    def _base_match(self, pred: str, gold: str, atol: float) -> bool:
        # ---------- 1) Normalised text equivalence ------------------------ #
        pn, gn = self._normalize_math_answer(pred), self._normalize_math_answer(gold)
        if pn and gn and pn == gn:
            return True
        # ---------- 2) Numeric comparison (tolerant) ---------------------- #
        if self._numerical_equal(pred, gold, atol):
            return True
        # ---------- 3) Algebraic equivalence via SymPy -------------------- #
        if self._sympy_equal(pred, gold):
            return True
        # ---------- 4) Expression / equation / interval / set / angle ----- #
        if self._complex_equal(pred, gold, atol):
            return True
        # ---------- 5) Fallback textual match (“yes/no/none/...”) --------- #
        return self._compare_text_answers(pred, gold)

    ############################################################################
    #  Helper suites                                                           #
    ############################################################################
    # <A> Choice #
    def _is_choice_letter(self, s: str) -> bool:
        return s.strip().upper() in self._CHOICES

    def _choice_equal(self, pred: str, gold: str) -> bool:
        cleaned = self._choice_answer_clean(pred)
        return cleaned.upper() == gold.strip().upper()

    def _choice_answer_clean(self, txt: str) -> str:
        txt = txt.strip().rstrip(".").rstrip("/").lstrip(":")
        found = re.findall(r"\b(A|B|C|D|E)\b", txt.upper())
        return (found or [txt.strip().strip(".")])[-1]
    
    # <B> Low‑level canonicalisation (mostly ported from MATH evaluation code) #
    def _strip_string(self, string: str) -> str:
        s = string.replace("\n", "")
        s = s.replace("\\!", "")
        s = s.replace("\\\\", "\\")
        s = s.replace("tfrac", "frac").replace("dfrac", "frac")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("^{\\circ}", "").replace("^\\circ", "")
        s = s.replace("\\$", "")
        s = self._remove_right_units(s)
        s = s.replace("\\%", "").replace("%", "")
        s = s.replace(" .", " 0.").replace("{.", "{0.")
        if s.startswith("."):
            s = "0" + s
        if s.count("=") == 1 and len(s.split("=")[0]) <= 2:
            s = s.split("=")[1]
        s = self._fix_sqrt(s)
        s = s.replace(" ", "")
        s = self._fix_fracs(s)
        if s == "0.5":
            s = "\\frac{1}{2}"
        s = self._fix_a_slash_b(s)
        return s

    def _fix_fracs(self, string: str) -> str:
        """Turn "\frac12" → "\frac{1}{2}"."""
        parts = string.split("\\frac")
        new_str = parts[0]
        for tail in parts[1:]:
            new_str += "\\frac"
            if not tail:
                continue
            if tail[0] == "{":
                new_str += tail
            else:
                if len(tail) < 2:
                    return string
                a, b, rest = tail[0], tail[1], tail[2:]
                if b != "{":
                    new_str += f"{{{a}}}{{{b}}}{rest}"
                else:
                    new_str += f"{{{a}}}{b}{rest}"
        return new_str

    def _fix_a_slash_b(self, string: str) -> str:
        parts = string.split("/")
        if len(parts) != 2:
            return string
        a, b = parts
        try:
            a_int, b_int = int(a), int(b)
            if string == f"{a_int}/{b_int}":
                return f"\\frac{{{a_int}}}{{{b_int}}}"
        except ValueError:
            pass
        return string

    def _remove_right_units(self, s: str) -> str:
        return s.split("\\text{ ")[0] if "\\text{ " in s else s

    def _fix_sqrt(self, s: str) -> str:
        if "\\sqrt" not in s:
            return s
        parts = s.split("\\sqrt")
        new = parts[0]
        for tail in parts[1:]:
            if tail and tail[0] != "{":
                new += f"\\sqrt{{{tail[0]}}}{tail[1:]}"
            else:
                new += "\\sqrt" + tail
        return new

    # <C> From OlympiadBench auto_scoring_judge code #
    def _split_by_comma(self, expr: str) -> List[str]:
        level, start, parts = 0, 0, []
        for i,ch in enumerate(expr):
            if ch in "([{": level += 1
            elif ch in ")]}": level -= 1
            elif ch == "," and level == 0:
                parts.append(expr[start:i].strip())
                start = i + 1
        parts.append(expr[start:].strip())
        return [p for p in parts if p]

    def _expand_plus_minus(self, lst: List[str]) -> List[str]:
        out = []
        for s in lst:
            if "\\pm" in s:
                out.append(s.replace("\\pm", "+", 1))
                out.append(s.replace("\\pm", "-", 1))
            else:
                out.append(s)
        return out

    # <D> Numeric equal and parsing (percent/ratio allow) #
    def _numerical_equal(self, a: str, b: str, atol: float) -> bool:
        # quantity compare
        if self._quantity_equal(a, b, atol):
            return True
        # complex number
        ca, cb = self._parse_complex(a), self._parse_complex(b)
        if ca is not None and cb is not None:
            try:
                diff = ca - cb
                # 유한성 검사: (abs는 hypot을 써서 바로 Overflow가 날 수 있음)
                if math.isfinite(diff.real) and math.isfinite(diff.imag):
                    if abs(diff) <= atol * 1.01:
                        return True
            except OverflowError:
                pass
        # real/percent
        ra = self._parse_number(a)
        rb = self._parse_number(b)
        if ra is None or rb is None:
            return False
        candidates = [ra, ra / 100.0, ra * 100.0]  # % 변형 후보
        return any(abs(c - rb) <= atol * 1.01 for c in candidates)

    def _parse_number(self, text: str) -> Optional[float]:
        if not text: return None
        s = re.sub(r"[^0-9eE./+\-]", "", text.strip())
        if "/" in s and s.count("/") == 1: # Fraction
            p,q = s.split("/")
            if self._NUM_SIMPLE.fullmatch(p) and self._NUM_SIMPLE.fullmatch(q):
                try:                     # 정수 분수인 경우
                    return float(Fraction(int(p), int(q)))
                except ValueError:       # 정수가 아니면 float 로 처리
                    try:
                        val = float(p) / float(q)
                        return val if math.isfinite(val) else None
                    except (ValueError, ZeroDivisionError):
                        return None
        # Simple / scientific
        if self._NUM_SIMPLE.fullmatch(s):
            try: 
                val = float(Decimal(s))
                return val if math.isfinite(val) else None
            except InvalidOperation: return None
        return None

    def _parse_complex(self, txt: str) -> Optional[complex]:
        m = self._CMPLX_RE.match(txt.replace("−", "-"))
        if m:
            try:
                re_part = float(m.group(1))
                im_part = float(m.group(3)) * (1 if m.group(2) == "+" else -1)
            except OverflowError:
                return None
            if not (math.isfinite(re_part) and math.isfinite(im_part)):
                return None
            return complex(re_part, im_part)
        try:
            val = self._to_sympy(txt)
            if val is not None and val.is_number:
                ve = sp.N(val)
                # 실수면
                if ve.is_real:
                    try:
                        rf = float(ve)   # 여기서 Overflow 가능
                    except (TypeError, ValueError, OverflowError):
                        return None
                    return complex(rf, 0.0) if math.isfinite(rf) else None
                # 복소수면
                re_sym, im_sym = ve.as_real_imag()
                try:
                    rf = float(re_sym)
                    inf = float(im_sym)
                except (TypeError, ValueError, OverflowError):
                    return None
                if math.isfinite(rf) and math.isfinite(inf):
                    return complex(rf, inf)
        except Exception:
            pass
        return None

    def _parse_quantity(self, txt:str):
        m = self._UNIT_RE.match(txt.lower())
        if not m: return None
        val, unit = float(m.group(1)), m.group(2)
        if unit not in self._UNIT_MAP: return None
        return val * self._UNIT_MAP[unit], unit  # (value in base, unit-name)

    def _quantity_equal(self, a:str, b:str, atol:float)->bool:
        qa = self._parse_quantity(a)
        qb = self._parse_quantity(b)
        if qa and qb:   # 같은 차원(동일 base-key)만 허용  and qa[1]==qb[1]
            return math.isclose(qa[0], qb[0], rel_tol=1e-4, abs_tol=atol)
        return False

    # <E> Normalisation helpers #
    def _normalize_math_answer(self, ans: str) -> Optional[str]:
        if not ans:
            return None
        ans = ans.strip()
        ans = self._normalize_latex(ans)
        ans = self._normalize_math_symbols(ans)
        ans = re.sub(r"\s+", " ", ans).strip()
        return ans or None

    def _normalize_latex(self, text: str) -> str:
        text = re.sub(r"\\dfrac\{([^}]*)\}\{([^}]*)\}", r"\\frac{\1}{\2}", text)
        text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
        text = re.sub(r"\\left\(", "(", text)
        text = re.sub(r"\\right\)", ")", text)
        text = re.sub(r"\\left\[", "[", text)
        text = re.sub(r"\\right\]", "]", text)
        text = re.sub(r"\\left\\{", "{", text)
        text = re.sub(r"\\right\\}", "}", text)
        replacements = {
            "\\infty": "∞",
            "\\cup": "∪",
            "\\cap": "∩",
            "\\subset": "⊂",
            "\\supset": "⊃",
            "\\in": "∈",
            "\\notin": "∉",
            "\\leq": "≤",
            "\\geq": "≥",
            "\\neq": "≠",
            "\\approx": "≈",
            "\\circ": "°",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
        return text

    def _normalize_math_symbols(self, text: str) -> str:
        text = re.sub(r"infinity|inf", "∞", text, flags=re.IGNORECASE)
        text = re.sub(r"degrees?|deg", "°", text, flags=re.IGNORECASE)
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
        return text
    
    # <F> SymPy equivalence # 
    def _to_sympy(self, expr: str) -> Optional[sp.Expr]:
        if not expr: return None
        try: return parse_latex(expr)
        except Exception: pass
        try: return parse_expr(expr, transformations=self._sympy_tf, evaluate=True)
        except Exception: return None

    def _sympy_equal(self, a: str, b: str) -> bool:
        ea, eb = self._to_sympy(a), self._to_sympy(b)
        if ea is None or eb is None: return False
        try:
            if ea.free_symbols != eb.free_symbols:
                return False
        except AttributeError:
            return False
        try: return simplify(ea - eb) == 0
        except Exception: return False

    # <G> Complex answer comparison (intervals, sets, equations, etc.) #
    def _complex_equal(self, a: str, b: str, atol: float) -> bool:
        # set
        if self._SET_RE.match(a) and self._SET_RE.match(b):
            return self._set_equal(a,b)
        # interval
        if self._INTERVAL_RE.match(a) and self._INTERVAL_RE.match(b):
            return self._interval_equal(a,b, atol)
        # angle
        if self._ANGLE_RE.match(a) and self._ANGLE_RE.match(b):
            return math.isclose(float(a.rstrip("°")), float(b.rstrip("°")), abs_tol=atol)
        # matrix
        if self._is_matrix(a) or self._is_matrix(b):
            return self._matrix_equal(a, b, atol)
        # Equations / expressions
        if "=" in a or "=" in b:
            try:
                return self._equation_equal(a, b)
            except Exception:
                return False
        else:
            try:
                return self._expression_equal(a, b, atol)
            except Exception:
                return False

    # ---- interval  ------------------------------------------------------ #
    def _interval_equal(self, i1: str, i2: str, atol: float) -> bool:
        if i1 == i2: return True
        p1, p2 = i1.split("\\cup"), i2.split("\\cup")
        if len(p1)!=len(p2): return False
        for s1,s2 in zip(p1,p2):
            if s1[0]!=s2[0] or s1[-1]!=s2[-1]: return False
            a1,b1 = map(str.strip, s1.strip("()[]").split(","))
            a2,b2 = map(str.strip, s2.strip("()[]").split(","))
            if not (self._base_match(a1,a2,atol) and self._base_match(b1,b2,atol)):
                return False
        return True

    # ---- set ------------------------------------------------------------ #
    def _set_equal(self, s1: str, s2: str) -> bool:
        elems = lambda s: {re.sub(r"\s+","",e) for e in s.strip("{} ").split(",") if e.strip()}
        return elems(s1) == elems(s2)

    # ------ matrix ------------------------------------------------------ #
    def _is_matrix(self, s: str) -> bool:
        return bool(self._MATRIX_RE.search(s))

    def _str_to_pmatrix(self, txt: str) -> str:
        txt = txt.strip().strip('{} ')
        rows_raw = re.split(r'\}\s*,\s*\{', txt)
        body = '\\\\'.join('&'.join(c.strip() for c in row.split(',')) for row in rows_raw)
        return f"\\begin{{pmatrix}}{body}\\end{{pmatrix}}"

    def _matrix_equal(self, p: str, g: str, atol: float) -> bool:
        # 둘 중 하나만 pmatrix 이면 brace→pmatrix 변환 시도
        if self._is_matrix(p) and not self._is_matrix(g):
            g = self._str_to_pmatrix(g)
        elif self._is_matrix(g) and not self._is_matrix(p):
            p = self._str_to_pmatrix(p)

        if not(self._is_matrix(p) and self._is_matrix(g)):
            return False

        def _rows(s):
            body = re.sub(r"\\begin\{[pb]matrix\}|\\end\{[pb]matrix\}", "", s)
            return [r.strip() for r in body.split("\\\\") if r.strip()]

        pr, gr = _rows(p), _rows(g)
        if len(pr)!=len(gr): return False
        for pr_row, gr_row in zip(pr,gr):
            pr_el = [x.strip() for x in pr_row.split("&")]
            gr_el = [x.strip() for x in gr_row.split("&")]
            if len(pr_el)!=len(gr_el): return False
            for a,b in zip(pr_el, gr_el):
                if not self._base_match(a,b,atol):
                    return False
        return True
    
    # ---- expression ----------------------------------------------------- #
    def _expression_equal(self, e1: str, e2: str, atol: float) -> bool:
        get_rhs = lambda e: e.split("=")[-1].strip() if "=" in e else e.strip()
        sym1, sym2 = self._to_sympy(get_rhs(e1)), self._to_sympy(get_rhs(e2))
        if sym1 is None or sym2 is None: return False
        if sym1 == sym2: return True
        sym1, sym2 = sym1.subs(self._pi, math.pi), sym2.subs(self._pi, math.pi)
        if not sym1.free_symbols and not sym2.free_symbols:
            if self._can_compute_power(sym1) and self._can_compute_power(sym2):
                try: return abs(float(sym1.evalf()) - float(sym2.evalf())) <= atol * 1.01
                except Exception: return False
            return False
        try: diff = simplify(sym1 - sym2).evalf(); return abs(float(diff)) < 1e-3
        except Exception: return False
    
    def _can_compute_power(self, expr: sp.Expr) -> bool:
        if isinstance(expr, Pow):
            base, exp = expr.as_base_exp()
            if base.is_number and exp.is_number:
                return abs(exp.evalf()) <= 1000
            return False
        return True

    # ---- equation ------------------------------------------------------- #
    def _equation_equal(self, eq1: str, eq2: str) -> bool:
        def simplify_eq(eq: str):
            lhs, rhs = eq.split("=")
            lhs_expr = self._to_sympy(lhs)
            rhs_expr = self._to_sympy(rhs)
            if lhs_expr is None or rhs_expr is None:
                raise ValueError("Invalid equation")
            return simplify(lhs_expr - rhs_expr)

        s1 = simplify_eq(eq1)
        s2 = simplify_eq(eq2)
        if s1 == 0 and s2 == 0:
            return True
        try:
            div1 = simplify(s1 / s2)
            div2 = simplify(s2 / s1)
            return (div1.is_Integer and div1 != 0) or (div2.is_Integer and div2 != 0)
        except Exception:
            return False

    # <H> textual answers #
    def _compare_text_answers(self, a: str, b: str) -> bool:
        a = self._normalize_latex(a)
        b = self._normalize_latex(b)
        ac = self._TEXT_ALNUM.sub("", a).lower()
        bc = self._TEXT_ALNUM.sub("", b).lower()
        return ac == bc and ac in self._BOOLEAN_WORDS

