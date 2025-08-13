from datasets import load_dataset
import re, ast
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from typing import Optional, Sequence, Tuple, List

class AnswerExtractor:
    """
    Robust gold/ pred-answer extractor for GSM8K · Math · Omni · OlympiadBench.
    """
    def __init__(self):
        self.answer_keywords = (
            "answer", "final answer", "therefore", "result",
            "the final answer is"
        )
        self.number_patterns = [
            r'(\d+\.\d+)', r'(\d+\/\d+)', r'(\d+)',
            r'(\d+\+\d+)', r'(\d+\-\d+)', r'(\d+\*\d+)'
        ]

    def extract_gold_answer(self, text: str,dataset: str | None = None) -> Optional[str]:
        if text is None:
            return None
        if isinstance(text, (list, tuple)):
            return self._flatten_and_clean(text)

        ds  = (dataset or "").lower()
        txt = str(text).strip()
        # 1) GSM8K  #### 정답
        if ds == "gsm8k" or re.search(r"\n####\s*[^\n]+", txt):
            m = re.search(r"\n####\s*([^\n]+)", txt)
            return self._strip(m.group(1)) if m else None
        # 2) OlympiadBench  [ '$…$', '$…$' ]
        if ds =="olympiad" or ( txt.startswith('[') and txt.endswith(']')):
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, (list, tuple)):
                    return self._flatten_and_clean(parsed)
                return self._strip(str(parsed))
            except (SyntaxError, ValueError):
                return self._strip(txt)
        # 3) Omni  (그대로)
        if ds == "omni" or ds == "aime":
            return self._strip(txt)
        # 4) Math 또는 그 밖 → 공통 pred-extract 로 처리
        return self.extract_pred_answer(txt)

    def extract_pred_answer(self, text: str) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return ""
        # 1) 마지막 balanced \boxed{…} / \fbox{…}
        boxed = self._extract_last_boxed(text)
        if boxed:
            return self._strip(boxed)
        # 2) Answer: … / Therefore: … (다음 줄까지 포함)
        ans_line = self._extract_answer_line(text)
        if ans_line:
            return self._strip(ans_line)
        # 3) 마지막 줄에서 inline LaTeX
        last_expr = self._extract_last_latex(text)
        if last_expr:
            return self._strip(last_expr)
        # 4) 마지막 숫자/수식
        num = self._extract_last_number(text)
        return self._strip(num) if num else ""

    # ─────────────────── Internals ───────────────────
    def _extract_last_boxed(self, text: str) -> Optional[str]:
        start_pat = re.compile(r'(\\boxed|\\fbox)\s*\{')
        starts = list(start_pat.finditer(text))
        if not starts:
            return None
        i = starts[-1].end()
        depth = 1
        while i < len(text) and depth:
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            i += 1
        return text[starts[-1].end(): i-1].strip() if depth == 0 else None

    def _extract_answer_line(self, text: str) -> Optional[str]:
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            low = ln.lower()
            if any(k in low for k in self.answer_keywords):
                # 콜론 뒤 같은 줄?
                m = re.search(r'[:\-]\s*(.+)$', ln)
                if m and m.group(1).strip():
                    return m.group(1).strip()
                # 아니면 다음 비어있지 않은 줄
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    return lines[j].strip()
        return None

    # def _extract_last_latex(self, text: str) -> Optional[str]:
    #     for ln in reversed(text.splitlines()):
    #         ln = ln.strip()
    #         if not ln:
    #             continue
    #         # $ … $  \[…\]  \(…\)
    #         m = re.findall(r'\$(.*?)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]', ln)
    #         if m:
    #             return [seg for seg in m[-1] if seg][0]
    #         # bare \frac  \sqrt …
    #         m2 = re.search(r'(\\[a-zA-Z]+(?:\{[^{}]+\})+)', ln)
    #         if m2:
    #             return m2.group(1)
    #         break
    #     return None

    def _extract_last_latex(self, text: str) -> Optional[str]:
        if not text:
            return None
        for ln in reversed(text.splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            # 1) inline LaTeX
            m_iter = list(re.finditer(r'\$(.*?)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]', ln))
            if m_iter:
                last = m_iter[-1]
                for g in last.groups():
                    if g and g.strip():
                        return g.strip()
            # 2) bare pattern
            m2 = re.search(r'(\\[a-zA-Z]+(?:\{[^{}]+\})+)', ln)
            if m2:
                s = (m2.group(1) or "").strip()
                if s:
                    return s
            break
        return None

    def _extract_last_number(self, text: str) -> Optional[str]:
        for ln in reversed(text.splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            for pat in self.number_patterns:
                m = re.findall(pat, ln)
                if m:
                    return m[-1]
        return None

    def _strip(self, s: str | None) -> str | None:
        if s is None:
            return None
        s = s.strip()
        s = re.sub(r'^\$+\s*', '', s)            # leading $
        s = re.sub(r'\s*\$+$', '', s)            # trailing $
        s = re.sub(r'^\\\(|\\\)$', '', s)        # \( … \)
        s = re.sub(r'^\\\[|\\\]$', '', s)        # \[ … \]
        s = s.replace('\\\\', '\\')              # \\ → \
        return s.strip(" ,;:")

    def _flatten_and_clean(self, seq: List[str]) -> str:
        return ", ".join(self._strip(x) for x in seq)
