import re
from typing import List, Optional, Tuple, Dict, Union, Iterable
import torch

def build_chat_messages(question: str,tokenizer,dataset: str, shots: Optional[List[tuple[str, str, str]]] = None,) -> str:
    system_prompt = (
        "You are an **expert mathematical‑reasoning assistant**.\n\n"
        "## Format rules\n"
        "1. Begin *every* reasoning line with the exact prefix `Step k:` where `k = 1, 2, …`. No other prefix is allowed.\n"
        "2. Show *all* intermediate calculations using standard symbols (×, ÷, ±, √).\n"
        "3. Conclude with **one** line of the form `Answer: <final numeric result>` and **stop immediately** - no explanations, no closing remarks.\n"
        "4. Each step must be concise *yet mathematically rigorous*.\n"
        "5. Avoid markdown bullet lists or narrative words such as ‘First’,  ‘Next’, ‘Finally’.\n\n"
        "Follow these rules exactly - evaluations are case- and format‑sensitive.\n"
        "Respond *only* in the specified format."
    )
    default_shots: List[tuple[str, str, str]] = [
        (
            "gsm8k, math, olympiad, omni",
            "Problem: What is the next number in the sequence 2, 4, 8, 16?",
            "Step 1: Identify the pattern – each term is multiplied by 2.\n"
            "Step 2: 16 × 2 = 32\n"
            "Answer: 32",
        ),
        (
            "gsm8k, math",
            "Problem: Solve for x: 3x + 7 = 22",
            "Step 1: Subtract 7 from both sides: 3x = 15\n"
            "Step 2: Divide by 3: x = 5\n"
            "Answer: 5",
        ),
        (
            "olympiad, omni",
            "Problem: Determine whether v₁ = [1,2] and v₂ = [3,6] are linearly independent.",
            "Step 1: Observe v₂ = 3 · v₁, so v₂ is a scalar multiple of v₁.\n"
            "Step 2: Therefore the vectors are linearly dependent.\n"
            "Answer: Dependent",
        ),
    ]

    if shots is None:
        shots = default_shots

    messages = [{"role": "system", "content": system_prompt}]
    for tag, q, a in shots:
        if dataset.lower() in tag.lower():
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": f"Problem: {question}"})
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

def build_chat_messages2(question: str, tokenizer, dataset: str, shots: Optional[List[Tuple[str, str, str]]] = None,) -> str:
    system_prompt = (
        "Solve the math problem step by step. "
        "Only generate **one new step** based on the current prefix. "
        "If you reach the final solution, output exactly: 'Answer: [final answer]' and stop. "
        "Do not generate extra text or additional steps after the answer."
    )

    default_shots: List[Tuple[str, str, str]] = [
        (
            "gsm8k, math, olympiad, omni",
            "Problem: What is the next number in the sequence 2, 4, 8, 16?\nStep 1: Identify the pattern – each term is multiplied by 2.\n",
            "Step 2: 16 × 2 = 32\n" "Answer: 32",
        ),
        (
            "gsm8k, math, olympiad, omni",
            "Problem: Solve for x: 3x + 7 = 22\nStep 1: Subtract 7 from both sides: 3x = 15\nStep 2: Divide by 3: x = 5\n",
            "Answer: 5",
        ),
        # (
        #     "olympiad, omni",
        #     "Problem: Determine whether v₁ = [1,2] and v₂ = [3,6] are linearly independent.",
        #     "Step 1: Observe v₂ = 3 · v₁, so v₂ is a scalar multiple of v₁.\n"
        #     "Step 2: Therefore the vectors are linearly dependent.\n" "Answer: Dependent",
        # ),
    ]

    if shots is None:
        shots = default_shots

    messages = [{"role": "system", "content": system_prompt}]
    for tag, q, a in shots:
        if dataset.lower() in tag.lower():
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": f"Problem: {question}"})
    return tokenizer.apply_chat_template(messages, tokenize=False)


class StepParser:
    # 1) "Step n:" / "Step One:" / "Step IV-" …
    _STEP_RE = re.compile(
        r"^\s*Step\s*(?:\d+|[IVXLCDM]+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\b\s*[:\-.]",
        re.I,
    )
    # 2) 1. / 1) / 1-   OR  I. / II) / III-
    _ENUM_RE = re.compile(r"^\s*(?:\d+|[IVXLCDM]+)[.)-]\s+", re.I)
    # 3) Narrative adverbs
    _NARR_RE = re.compile(
        r"^\s*(First|Firstly|Second|Secondly|Third|Thirdly|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Next|Then|After that|Therefore|However|Since|Thus|Finally|Lastly|In conclusion)\b[,:]?",
        re.I,
    )
    # 4) Bullets etc.
    _BULLET_RE = re.compile(r"^\s*[-*•]\s+")
    _JUNK_RE = re.compile(r"^(?:<\|endoftext\|>\s*)+")
    _LIST_LIKE_RE = re.compile(r"^\s*\[.*\]\s*$", re.S)

    # Abbreviation-aware sentence splitting
    _DEF_ABBREV = r"Mr|Mrs|Dr|Prof|vs|etc|e\.g|i\.e|Fig|Eq|No|cf|al"
    # Split on period/question/exclamation followed by whitespace; also allow explicit split on ".  " sequences.
    # NOTE: We do not consume the next token; we only consume the delimiter and following spaces.
    _SENT_SPLIT_RE = re.compile(
        r"(?:(?<!\d)\.(?!\d)|[!?])\s+|\.[ ]{2,}"
    )

    # Answer-ish patterns
    _GSM8K_HASH_RE = re.compile(r"^\s*####\s*(.+)\s*$")

    # Math span delimiters for masking during sentence split
    _MATH_DELIMS: List[Tuple[str, str]] = [(r"\\(", r"\\)"), (r"\\[", r"\\]"), (r"$$", r"$$")]

    _DELIMS = (_STEP_RE, _ENUM_RE, _NARR_RE, _BULLET_RE)

    # ------------------------------------------------------------------
    @classmethod
    def _is_delim(cls, line: str) -> bool:
        return any(r.match(line) for r in cls._DELIMS)

    @classmethod
    def _clean_line(cls, line: str) -> str:
        return cls._JUNK_RE.sub("", line).strip()

    @classmethod
    def _normalize_text(cls, obj: Union[str, List[str], tuple]) -> str:
        # 1) List/Tuple → join with blank line
        if isinstance(obj, (list, tuple)):
            parts = [str(x) for x in obj]
            text = "\n\n".join(parts)
        else:
            text = str(obj)
            # String that looks like a list → literal_eval
            if cls._LIST_LIKE_RE.match(text):
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, (list, tuple)):
                        text = "\n\n".join(str(x) for x in parsed)
                except Exception:
                    pass
        # 2) Escaped newlines → real newlines (if no real newlines yet)
        if "\\n" in text and "\n" not in text:
            text = text.replace("\\n", "\n")
        return text

    # -------------------- Math masking for sentence split --------------------
    @classmethod
    def _mask_math(cls, s: str) -> Tuple[str, Dict[str, str]]:
        """Mask LaTeX math regions so sentence splitter won't cut inside them."""
        out = s
        mapping: Dict[str, str] = {}
        counter = 0

        def _mask_once(text: str, start_pat: str, end_pat: str) -> str:
            nonlocal counter
            start_re = re.compile(re.escape(start_pat))
            end_re = re.compile(re.escape(end_pat))
            i = 0
            res = []
            while True:
                m = start_re.search(text, i)
                if not m:
                    res.append(text[i:])
                    break
                res.append(text[i:m.start()])
                j = end_re.search(text, m.end())
                if not j:
                    # no closing; emit rest and stop
                    res.append(text[m.start():])
                    break
                token = f"\x00MATH{counter}\x01"
                mapping[token] = text[m.start(): j.end()]
                counter += 1
                res.append(token)
                i = j.end()
            return "".join(res)

        # Apply masks for all delims
        for st, ed in cls._MATH_DELIMS:
            out = _mask_once(out, st, ed)
        return out, mapping

    @staticmethod
    def _unmask_math(s: str, mapping: Dict[str, str]) -> str:
        if not mapping:
            return s
        out = s
        for k, v in mapping.items():
            out = out.replace(k, v)
        return out
    
    @classmethod
    def _strip_inline_answerish(cls, s: str, *, ds_name: Optional[str] = None, gold_answer: Optional[str] = None) -> str:
        t = s
        # 1) gsm8k: 문장/라인 중간에 들어간 "#### <ans>"부터 끝까지 잘라내기
        if (ds_name or "").lower() == "gsm8k":
            # #### 이후는 전부 잘라냄
            t = re.split(r"\s*####\s*", t, maxsplit=1)[0].rstrip()
        # 2) "Answer: ..." / "Final answer: ..."가 스텝 끝에 달라붙은 경우 제거
        t = re.sub(
            r"(?:^|[\s:.])(?:final\s*answer|answer)\s*[:\-]*\s*[^.\n]*\s*$",
            "", t, flags=re.I
        ).rstrip()
        # 3) gold_answer가 스텝 꼬리에 그대로 붙은 경우(중복 표기) 제거 (선택)
        if gold_answer:
            ga = re.escape(gold_answer.strip())
            t = re.sub(rf"\s*(?:=|\b)\s*{ga}\s*\.?\s*$", "", t, flags=re.I)

        return t.strip()

    # ---------------------------- Splitting ----------------------------
    @classmethod
    def _split_blocks(cls, text: str) -> List[str]:
        """Split by explicit step-like delimiters, preserving the leading marker in the block."""
        blocks: List[str] = []
        buf: List[str] = []

        def flush():
            if buf:
                # join with single spaces to avoid accidental sentence merges at newlines
                blocks.append(" ".join([b for b in buf if b]).strip())
                buf.clear()

        for raw_ln in text.splitlines():
            ln = cls._clean_line(raw_ln)
            if not ln:
                # allow blank lines to be part of the current block
                continue
            if cls._is_delim(ln):
                flush()
            buf.append(ln)
        flush()
        return [b for b in blocks if b]

    @classmethod
    def _sentence_split_safe(cls, text: str) -> List[str]:
        if not text:
            return []
        # Protect abbreviations: replace "." with placeholder
        tmp = re.sub(rf"\b({cls._DEF_ABBREV})\.\s+", lambda m: m.group(1) + "<PERIOD> ", text)
        # Mask math spans
        masked, mp = cls._mask_math(tmp)
        parts: List[str] = []
        for para in re.split(r"\n\s*\n", masked):
            para = para.strip()
            if not para:
                continue
            # Use finditer to get split points; then slice manually to avoid losing context
            spans = []
            last = 0
            for m in cls._SENT_SPLIT_RE.finditer(para):
                end = m.end()  # consume delimiter + spaces
                seg = para[last:end].strip()
                if seg:
                    spans.append(seg)
                last = end
            tail = para[last:].strip()
            if tail:
                spans.append(tail)
            for s in spans:
                s = s.replace("<PERIOD>", ".")
                s = cls._unmask_math(s, mp).strip()
                if s:
                    parts.append(s)
        # Secondary split on "; " or ": " for very long segments
        refined: List[str] = []
        for seg in parts:
            if len(seg) > 240 and ("; " in seg or ": " in seg):
                for t in re.split(r";\s+|:\s+", seg):
                    t = t.strip()
                    if t:
                        refined.append(t)
            else:
                refined.append(seg)
        return refined

    # ---------------------------- Merging short steps ----------------------------
    @staticmethod
    def _concat(a: str, b: str) -> str:
        if not a:
            return b.strip()
        if not b:
            return a.strip()
        a = a.rstrip()
        b = b.lstrip()
        # collapse multiple spaces across the boundary
        joined = (a + " " + b)
        return re.sub(r"\s+", " ", joined).strip()

    @classmethod
    def _merge_short_steps(cls, steps: List[str], min_chars: int = 30, merge_short_to: str = "prev",) -> List[str]:
        if min_chars is None or min_chars <= 0 or not steps:
            return steps
        to_prev = (merge_short_to.lower() != "next")

        def is_short(s: str) -> bool:
            return len(s.strip()) < min_chars

        if to_prev:
            out: List[str] = []
            i = 0
            n = len(steps)
            while i < n:
                s = steps[i]
                if is_short(s):
                    if out:
                        out[-1] = cls._concat(out[-1], s)
                        i += 1
                        continue
                    else:
                        # first item is short → attach to next if possible
                        if i + 1 < n:
                            steps[i + 1] = cls._concat(s, steps[i + 1])
                            i += 1
                            # skip appending s separately
                            continue
                        else:
                            out.append(s)
                            i += 1
                            continue
                else:
                    out.append(s)
                    i += 1
            return out
        else:
            # merge to next
            out: List[str] = []
            i = 0
            n = len(steps)
            while i < n:
                s = steps[i]
                if is_short(s) and i + 1 < n:
                    steps[i + 1] = cls._concat(s, steps[i + 1])
                    # do not append s now
                    i += 1
                    continue
                else:
                    out.append(s)
                    i += 1
            return out

    # ---------------------------- Public APIs ----------------------------
    @classmethod
    def parse(cls, obj: Union[str, List[str], tuple]) -> List[str]:
        """Return raw step-like blocks (leading markers kept)."""
        text = cls._normalize_text(obj)
        # 1) delimiter-based blocks
        blocks = cls._split_blocks(text)
        if blocks:
            return blocks
        # 2) blank-line split
        sans = "\n".join(cls._clean_line(l) for l in text.splitlines())
        paras = [p.strip() for p in re.split(r"\n\s*\n", sans) if p.strip()]
        if paras:
            return paras
        # 3) sentence split (abbr/math aware)
        sents = cls._sentence_split_safe(sans)
        if sents:
            return sents
        # 4) line-by-line
        return [ln for ln in sans.splitlines() if ln.strip()]

    @classmethod
    def _strip_leading_marker(cls, text: str) -> Tuple[str, str]:
        """Return (marker, body). Marker empty if none found."""
        txt = cls._clean_line(text)
        for p in cls._DELIMS:
            m = p.match(txt)
            if m:
                return txt[: m.end()].rstrip(), txt[m.end():].lstrip()
        return "", txt

    @classmethod
    def _norm_text(cls, s: Optional[str]) -> str:
        return re.sub(r"[\s$]", "", (s or "").strip().lower())

    @classmethod
    def _drop_trailing_answerish(cls, steps: List[str], *, ds_name: Optional[str] = None, gold_answer: Optional[str] = None,) -> List[str]:
        ds = (ds_name or "").lower()
        ga = cls._norm_text(gold_answer)

        def is_answerish(t: str) -> bool:
            t0 = (t or "").strip()
            if cls._GSM8K_HASH_RE.match(t0):
                return True
            if ga:
                nt = cls._norm_text(t0)
                if nt.endswith(ga) and len(nt) - len(ga) <= 10:
                    return True
            return False

        j = len(steps) - 1
        while j >= 0 and is_answerish(steps[j]):
            j -= 1
        return steps[: j + 1]

    @classmethod
    def parse_clean(cls, obj: Union[str, Iterable[str]],  *,
        ds_name: Optional[str] = None,
        gold_answer: Optional[str] = None, sentence_level: bool = True,
        min_step_chars: int = 40,  merge_short_to: str = "prev",) -> List[str]:
        text = obj if isinstance(obj, str) else "\n\n".join(str(x) for x in obj)
        raw_blocks = cls.parse(text)

        cleaned: List[str] = []
        for b in raw_blocks:
            marker, core = cls._strip_leading_marker(b)
            if not core:
                continue
            if sentence_level:
                sents = cls._sentence_split_safe(core)
                if len(sents) >= 2:
                    cleaned.extend(sents)
                else:
                    cleaned.append(core)
            else:
                cleaned.append(core)

        # Drop trailing answer-ish lines first
        cleaned = cls._drop_trailing_answerish(cleaned, ds_name=ds_name, gold_answer=gold_answer)
        
        if ds_name == "gsm8k":
            cleaned = [cls._strip_inline_answerish(step, ds_name=ds_name, gold_answer=gold_answer) for step in cleaned]
            cleaned = [s for s in cleaned if s]
        
        if min_step_chars and min_step_chars > 0:
            cleaned = cls._merge_short_steps(cleaned, min_chars=min_step_chars, merge_short_to=merge_short_to)
        return cleaned

