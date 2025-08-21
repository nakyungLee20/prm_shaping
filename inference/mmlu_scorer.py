import re, math
from fractions import Fraction
from decimal import Decimal, InvalidOperation
import string
from typing import List, Optional, Tuple, Dict, Union


class MmluScorer:
    # Captures letters A-J (up to 10 options) and numbers 0..10 (0 rarely used)
    _LETTER = r"([A-J])"
    _NUMBER = r"(10|[0-9])"
    # Strong patterns: prefer explicit late matches
    _PATTERNS = [
    re.compile(r"(?i)\\boxed\s*\{\s*" + _LETTER + r"\s*\}"),
    re.compile(r"(?i)\\boxed\s*\{\s*" + _NUMBER + r"\s*\}"),
    re.compile(r"(?i)\b(?:final\s+)?answer\s*(?:is|=|:)\s*\(?" + _LETTER + r"\)?\b"),
    re.compile(r"(?i)\b(?:final\s+)?answer\s*(?:is|=|:)\s*\(?" + _NUMBER + r"\)?\b"),
    re.compile(r"(?i)\bthe\s+correct\s+answer\s*(?:is|=|:)\s*\(?" + _LETTER + r"\)?\b"),
    re.compile(r"(?i)\bthe\s+correct\s+answer\s*(?:is|=|:)\s*\(?" + _NUMBER + r"\)?\b"),
    re.compile(r"(?i)\b(?:option|choice)\s*" + _LETTER + r"\b(?:\s*is\s*correct)?"),
    re.compile(r"(?i)\(([A-J])\)\s*$", re.MULTILINE), # line ending with (C)
    ]
    # Weak patterns as backup
    _WEAK_PATTERNS = [
    re.compile(r"(?i)\b(?:option|choice|ans(?:wer)?)\s*" + _LETTER + r"\b"),
    re.compile(r"(?i)\b(?:pick|select)\s*" + _LETTER + r"\b"),
    ]
    # A line like: "Answer: ..."
    _ANS_LINE = re.compile(r"(?i)^\s*answer\s*[:\-]\s*(.+?)\s*$", re.MULTILINE)

    def __init__(self, labels: str = string.ascii_uppercase):
        self.labels = labels

    # ----------------------- helpers -----------------------
    @staticmethod
    def _norm_text(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _label_from_index(self, idx: int) -> Optional[str]:
        if 0 <= idx < len(self.labels):
            return self.labels[idx]
        return None

    def _index_from_label(self, label: str) -> Optional[int]:
        if not label:
            return None
        u = label.upper()
        pos = self.labels.find(u)
        return pos if pos != -1 else None

    def _label_from_number(self, number: int, n_choices: int) -> Optional[str]:
        # Interpret numbers as 1-based indices by default; clamp to available choices
        if 1 <= number <= n_choices:
            return self._label_from_index(number - 1)
        # If someone outputs 0 (rare), treat as first option when plausible
        if number == 0 and n_choices > 0:
            return self._label_from_index(0)
        return None
    
    def _number_from_label(self, label: str) -> Optional[int]:
        # label → 1-based
        idx = self._index_from_label(label)
        return (idx + 1) if idx is not None else None
    
    def _best_text_match(self, text: str, choices: List[str]) -> Optional[str]:
        if not choices:
            return None
        T = self._norm_text(text)
        last_hit: Tuple[int, int] | None = None  # (char_index, choice_idx)
        norm_choices = [self._norm_text(c) for c in choices]
        for i, c in enumerate(norm_choices):
            if not c:
                continue
            pos = T.rfind(c)
            if pos != -1:
                if (last_hit is None) or (pos > last_hit[0]):
                    last_hit = (pos, i)
        if last_hit is None:
            return None
        return self._label_from_index(last_hit[1])
    
    # ----------------------- gold parsing -----------------------
    def extract_gold(self, gold_field: Union[str, int], choices: List[str]) -> str:
        """Return canonical gold label 'A'/'B'/... given the dataset gold field."""
        n = len(choices) if choices else 0
        if isinstance(gold_field, int):
            lab = self._label_from_number(gold_field, n)
            return lab or ""
        if gold_field is None:
            return ""
        t = str(gold_field).strip()
        if re.fullmatch(r"[A-Ja-j]", t):
            idx = self._index_from_label(t.upper())
            if idx is not None and (n == 0 or idx < n):
                return t.upper()
            return t.upper()
        m = re.fullmatch(r"\s*(10|[0-9])\s*", t)
        if m:
            num = int(m.group(1))
            lab = self._label_from_number(num, n)
            return lab or ""
        if n > 0:
            norm_choices = [self._norm_text(c) for c in choices]
            nt = self._norm_text(t)
            for i, c in enumerate(norm_choices):
                if c == nt:
                    return self._label_from_index(i) or ""
        return ""

    def extract_gold_index(self, gold_field: Union[str, int], choices: List[str]) -> Optional[int]:
        """Return 1-based index for gold when possible."""
        n = len(choices) if choices else 0
        if isinstance(gold_field, int):
            return gold_field if gold_field >= 0 else None
        if gold_field is None:
            return None
        t = str(gold_field).strip()
        if re.fullmatch(r"[A-Ja-j]", t):
            k = self._number_from_label(t.upper())
            return k
        m = re.fullmatch(r"\s*(10|[0-9])\s*", t)
        if m:
            return int(m.group(1))
        # match by choice text
        lab = self.extract_gold(gold_field, choices)
        if lab:
            return self._number_from_label(lab)
        return None
    
    # ----------------------- pred parsing -----------------------
    def extract_pred(self, text: str, choices: List[str]) -> str:
        if text is None:
            return ""
        # If already a clean label or number, handle directly
        t0 = str(text).strip()
        if re.fullmatch(r"[A-Ja-j]", t0):
            lab = t0.upper()
            idx = self._index_from_label(lab)
            n = len(choices) if choices else 0
            if idx is None or (n and idx >= n):
                return ""
            return lab
        if re.fullmatch(r"\s*(10|[0-9])\s*", t0):
            n = len(choices) if choices else 0
            lab = self._label_from_number(int(t0), n)
            return lab or ""
        text = str(text)
        n = len(choices) if choices else 0
        # 1) Strong patterns (letter/number/boxed)
        for rx in self._PATTERNS:
            matches = list(rx.finditer(text))
            if not matches:
                continue
            g = matches[-1].group(1)
            if g.isalpha():
                lab = g.upper()
                idx = self._index_from_label(lab)
                if idx is not None and (n == 0 or idx < n):
                    return lab
            else:
                num = int(g)
                lab = self._label_from_number(num, n)
                if lab:
                    return lab
        # 2) "Answer: ..." line
        m = self._ANS_LINE.search(text)
        if m:
            token = m.group(1).strip()
            mL = re.fullmatch(r"\(?\s*([A-Ja-j])\s*\)?", token)
            if mL:
                lab = mL.group(1).upper()
                idx = self._index_from_label(lab)
                if idx is not None and (n == 0 or idx < n):
                    return lab
            mN = re.fullmatch(r"\(?\s*(10|[0-9])\s*\)?", token)
            if mN:
                num = int(mN.group(1))
                lab = self._label_from_number(num, n)
                if lab:
                    return lab
            lab = self._best_text_match(token, choices)
            if lab:
                return lab
        # 3) Weak patterns
        for rx in self._WEAK_PATTERNS:
            matches = list(rx.finditer(text))
            if matches:
                lab = matches[-1].group(1).upper()
                idx = self._index_from_label(lab)
                if idx is not None and (n == 0 or idx < n):
                    return lab
        # 4) Choice-text last occurrence
        lab = self._best_text_match(text, choices)
        if lab:
            return lab
        return ""

    def extract_pred_index(self, text: str, choices: List[str]) -> Optional[int]:
        """Return 1-based index parsed from text when possible (supports boxed and explicit patterns)."""
        if text is None:
            return None
        t0 = str(text).strip()
        if re.fullmatch(r"[A-Ja-j]", t0):
            return self._number_from_label(t0.upper())
        if re.fullmatch(r"\s*(10|[0-9])\s*", t0):
            return int(t0)
        # Search strong patterns (letters or numbers including boxed)
        for rx in self._PATTERNS:
            matches = list(rx.finditer(str(text)))
            if matches:
                g = matches[-1].group(1)
                if g.isalpha():
                    return self._number_from_label(g.upper())
                return int(g)
        # Try the Answer: line
        m = self._ANS_LINE.search(str(text))
        if m:
            tok = m.group(1).strip()
            mL = re.fullmatch(r"\(?\s*([A-Ja-j])\s*\)?", tok)
            if mL:
                return self._number_from_label(mL.group(1).upper())
            mN = re.fullmatch(r"\(?\s*(10|[0-9])\s*\)?", tok)
            if mN:
                return int(mN.group(1))
            # map via choice text if exact match
            lab = self._best_text_match(tok, choices)
            if lab:
                return self._number_from_label(lab)
        # Fallback: map by choice text found in the body
        lab = self._best_text_match(str(text), choices)
        if lab:
            return self._number_from_label(lab)
        return None

    # ----------------------- grading -----------------------
    def grade(self, pred: Union[str, int], gold: Union[str, int], choices: List[str], *, prefer_numeric: bool = True) -> bool:
        if prefer_numeric:
            p_idx = self.extract_pred_index(pred if isinstance(pred, str) else str(pred), choices)
            g_idx = self.extract_gold_index(gold, choices)
            if p_idx is not None and g_idx is not None:
                return p_idx == g_idx
        # Fallback to labels
        p_lab = self.extract_pred(pred if isinstance(pred, str) else str(pred), choices)
        g_lab = self.extract_gold(gold, choices)
        return (p_lab != "" and g_lab != "") and (p_lab == g_lab)


MMLUScorer = MmluScorer()

# Test # 
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    from datasets import load_dataset
    import random

    # model load
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
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    def format_mc_question(question: str, choices: List[str]) -> str:
        # 숫자 인덱스(1-based)로 옵션을 제시 → 숫자 우선 파싱을 유도
        lines = [question.strip(), "", "Choices:"]
        lines += [f"{i+1}. {c}" for i, c in enumerate(choices)]
        return "\n".join(lines)
    
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
        return header + "\n\nProblem: " + question.strip()

    def to_chat_prompt(tokenizer, question: str, eval_style: Optional[str] = "default") -> str:
        SYSTEM_PROMPT = (
            "You are Qwen-Math, a meticulous math tutor. "
            "Solve problems with clear, numbered steps and give only ONE final answer line."
        )
        if eval_style == "qwen_eval":
            user = f"{question.strip()}\n\nPlease reason step by step, and put your final answer within `Answer: \\boxed{{}}`."
        else:
            user = build_user_prompt(question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # load dataset
    ds = load_dataset("TIGER-Lab/MMLU-STEM", split="test")
    ds = ds.select(range(0, 20))

    # build prompts
    prompts = []
    batch_choices = []
    batch_gold = []
    batch_subjects = []
    for item in ds:
        q = item["question"]
        choices = list(item["choices"])
        gold = int(item["answer"])  # dataset is usually 1-based
        qt = format_mc_question(q, choices)
        prompt = to_chat_prompt(tokenizer, qt, eval_style="qwen_eval")
        prompts.append(prompt)
        batch_choices.append(choices)
        batch_gold.append(gold)
        batch_subjects.append(item.get("subject", ""))

    # ---------------- Generation ----------------
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.7,
            top_p=0.8,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_texts = tokenizer.batch_decode(gen_out, skip_special_tokens=True)

    # Helper to strip the prompt prefix to keep only the generated continuation
    def strip_prefix(prefix: str, text: str) -> str:
        return text[len(prefix):] if text.startswith(prefix) else text

    gen_texts = [strip_prefix(p, t) for p, t in zip(prompts, full_texts)]

    # ---------------- Scoring ----------------
    scorer = MmluScorer()
    results = []
    n_correct = 0
    use_index_helpers = True
    for i, (txt, choices, gold, subj) in enumerate(zip(gen_texts, batch_choices, batch_gold, batch_subjects)):
        ok = scorer.grade(txt, gold, choices)
        pred_label = scorer.extract_pred(txt, choices)
        gold_label = scorer.extract_gold(gold, choices)
        n_correct += int(ok)
        results.append({
            "subject": subj,
            "gold": gold,                 # 원본 gold (dataset의 1-based int)
            "gold_label": gold_label,     # 라벨로 변환된 정답 (예: 'C')
            "pred_label": pred_label,     # 라벨로 변환된 모델 출력
            "pred_text": txt.strip(),     # 원문 생성
            "correct": ok,
        })
        
    # ---------------- Report ----------------
    acc = n_correct / len(results)
    print(f"\nEvaluated {len(results)} examples | Accuracy = {acc:.3f}\n")
    
    for idx, r in enumerate(results):
        print(f"[{idx}: gold={r['gold']} gold_label={r['gold_label']} pred_label={r['pred_label'] or '-':<2}  correct={r['correct']}")
        # 필요하면 생성 전문 확인
        print('GEN:', r['pred_text'])

