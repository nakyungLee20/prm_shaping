import re, math, random
import numpy as np
import sympy as sp
from typing import Optional, List, Tuple, Dict, Iterable
import torch
from datasets import load_dataset
from tqdm import tqdm
from utils import system_prompt, _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer

class MISampleReward:
    ANSWER_PATTERN = re.compile(
        r"""^[\s>#*\-]*          # optional markdown/bullet symbols
            Answer               # word 'Answer'
            \s*[:.\-]\s*         # separator
            (.+?)\s*$            # capture everything after
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )
    _ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")
    
    def __init__(self, config: "PRMConfig", model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._H_CACHE: Dict[Tuple[str, str, str], float] = {}

    # Function to parse a solution text into steps and final answer.
    def _extract_answer(self, text: str) -> Optional[str]:
        match = self.ANSWER_PATTERN.search(text)
        if match:
            return _sanitize_enhanced(match.group(1))
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            candidate = lines[-1]
            if re.search(r"\d", candidate):  # contains digit
                return _sanitize_enhanced(candidate)
        for line in reversed(text.splitlines()):
            if line.strip().lower().startswith("answer"):
                return _sanitize_enhanced(line.split("Answer", 1)[-1])
        return None
    
    @staticmethod
    def _robust_z(x: np.ndarray, clip_z: float = 3.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-8   # 0-div 방지
        z = (x - med) / (1.4826 * mad)
        if clip_z is not None:
            z = np.clip(z, -clip_z, clip_z)
        return z

    def normalize_mi(self, scores: List[float],*,
        mode: str = "signed",   # "signed"([-1,1]), "unit"([0,1]), "relu", "minmax", "raw"
        tau: float = 1.5, clip_z: float = 3.0, deadzone: float = 0.2,   # signed/unit에서 |s|<=deadzone을 0 근처로 수축
        q_low: float = 5.0,      # minmax용 로버스트 하한 퍼센타일
        q_high: float = 95.0,    # minmax용 로버스트 상한 퍼센타일
        round_to: Optional[int] = 4,) -> List[float]:
        """
        - signed : robust z → tanh(z/tau) ∈ [-1,1] (음수 허용)
        - unit   : signed 결과를 (s+1)/2 → [0,1]
        - relu   : max(x,0)
        - minmax : 로버스트 퍼센타일 기반 [0,1] 스케일링
        - raw    : 그대로
        """
        x = np.asarray(scores, dtype=float)

        if mode == "raw":
            y = x
        elif mode == "relu":
            y = np.maximum(x, 0.0)
        elif mode in ("signed", "unit"):
            z = self._robust_z(x, clip_z=clip_z)
            s = np.tanh(z / max(tau, 1e-8))  # [-1,1]
            if deadzone and deadzone > 0.0:
                mag = np.maximum(0.0, np.abs(s) - deadzone) / (1.0 - deadzone)
                s = np.sign(s) * mag
            y = s if mode == "signed" else 0.5 * (s + 1.0)  # [0,1]
        elif mode == "minmax":
            lo = np.percentile(x, q_low)
            hi = np.percentile(x, q_high)
            if hi <= lo:
                y = np.zeros_like(x)
            else:
                y = (x - lo) / (hi - lo)
                y = np.clip(y, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        if round_to is not None:
            y = np.round(y.astype(float), round_to)
        return y.tolist()
    
    @torch.no_grad()
    def _sample_answers(self, prompt: str, n: int = 64, max_new_tokens: int = 200, temperature: float = 0.8, top_p: float = 0.95,) -> List[str]:
        """모델에서 N개 샘플을 생성하고 'Answer: ...' 한 줄을 추출."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            do_sample=True,
            num_return_sequences=n,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        answers = [self._extract_answer(t) for t in texts]
        return answers

    def _empirical_entropy_bits(self, items: List[str]) -> float:
        """경험적 분포 Ĥ(X) in bits."""
        items = [x for x in items if x != ""]
        if not items:
            return 0.0
        N = len(items)
        cnt = Counter(items)
        H = 0.0
        for c in cnt.values():
            p = c / N
            H -= p * math.log(p, 2)
        return H

    def compute_step_mi_sampling_marginal(self, question: str, steps: List[str], n_samples: int = 64,
        max_new_tokens: int = 200, temperature: float = 0.8, top_p: float = 0.95,):
        """
        샘플링 기반 Marginal MI: MI_i ≈ Ĥ(base) - Ĥ(base + S_i)
        """
        sys_prompt = (
            'Solve the given problem with step by step reasoning in the format of '
            '"Step k: <k-th rationale>" and write final answer in the format of '
            '"Answer: <final answer>".\nProblem: '
        )
        question = re.sub(r' +', ' ', question)
        base = sys_prompt + question + "\n\n"

        H_base = self._empirical_entropy_bits(
            self._sample_answers(base, n=n_samples, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        )

        mis = []
        for s in steps:
            with_i = base + s.rstrip() + "\n"
            H_with = self._empirical_entropy_bits(
                self._sample_answers(with_i, n=n_samples, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            )
            mis.append(H_base - H_with)
        return mis

    def compute_step_mi_sampling_loo(self, question: str, steps: List[str], n_samples: int = 64,
        max_new_tokens: int = 200, temperature: float = 0.8, top_p: float = 0.95,):
        """
        샘플링 기반 LOO: Δ_i ≈ Ĥ(all \ i) - Ĥ(all)
        """
        sys_prompt = (
            'Solve the given problem with step by step reasoning in the format of '
            '"Step k: <k-th rationale>" and write final answer in the format of '
            '"Answer: <final answer>".\nProblem: '
        )
        question = re.sub(r' +', ' ', question)
        base = sys_prompt + question + "\n\n"

        with_all = base + "".join(s.rstrip() + "\n" for s in steps)
        H_all = self._empirical_entropy_bits(
            self._sample_answers(with_all, n=n_samples, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p))

        contribs = []
        for i in range(len(steps)):
            without_i = base + "".join(
                steps[j].rstrip() + "\n" for j in range(len(steps)) if j != i
            )
            H_wo = self._empirical_entropy_bits(
                self._sample_answers(without_i, n=n_samples, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            )
            contribs.append(H_wo - H_all)
        return contribs

    # Streaming versions for memory-efficient processing
    def gsm8k_reward_dataset_streaming(self, *, split: str = "train", start: int = 0, take: Optional[int] = 0,
        norm: str = "unit",            # "signed" | "unit" | "relu" | "minmax" | "raw"
        norm_kwargs: Optional[Dict] = None,  # {"tau":..., "clip_z":..., "deadzone":..., ...}
        round_to: Optional[int] = 4,) -> Iterable[Dict]:

        ds = load_dataset("openai/gsm8k", "main", split=split)
        # ds = ds.select(range(start, start + take)) if take else ds
        ds = ds.select(range(start, len(ds)))
        print("Dataset", len(ds), "Loading!")
        
        for sample in tqdm(ds, desc="Building GSM8K MI reward-dataset"):
            q_txt   = sample["question"]
            g_sol   = sample["answer"]
            lines, gold_ans = [], None
            for ln in g_sol.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                m = self._ANSWER_RE.match(ln)
                if m:
                    gold_ans = _sanitize_enhanced(m.group(1))
                    break
                lines.append(ln)
            if gold_ans is None:
                raise ValueError("gold answer not found for sample")
            steps = [f"Step {i+1}: {t}" for i, t in enumerate(lines)]

            mi_sample_margin = self.compute_step_mi_sampling_marginal(qtxt, steps, n_samples=64)
            mi_sample_loo = self.compute_step_mi_sampling_loo(qtxt, steps, n_samples=64)
            mi_norm = self.normalize_mi(mi_sample_loo, mode=norm, **(norm_kwargs or {}), round_to=round_to,)

            entry = {
                    "question":      q_txt,
                    "completion":    steps,
                    "mi_sample_loo":   mi_sample_loo,
                    "mi_sample_margin":   mi_sample_margin,
                    "mi_sam_norm": mi_norm,
                    "gold_answer":   gold_ans,
                }
            yield entry

    def math_reward_dataset_streaming(self, *, split: str = "train", start: int = 0, take: Optional[int] = 0,
        norm: str = "unit",            # "signed" | "unit" | "relu" | "minmax" | "raw"
        norm_kwargs: Optional[Dict] = None,  # {"tau":..., "clip_z":..., "deadzone":..., ...}
        round_to: Optional[int] = 4,) -> Iterable[Dict]:

        sent_split = re.compile(r'\.(?!\d)(?=\s|$)')
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
        # ds = ds.select(range(start, start + take)) if take else ds
        ds = ds.select(range(start, len(ds)))
        print("Dataset", len(ds), "Loading!")

        for sample in tqdm(ds, desc="Building MATH MI reward-dataset"):
            full_sol = sample["solution"]
            q_txt = sample["problem"]
            boxed_content = _extract_boxed_answer(full_sol)
            gold_ans = _sanitize_enhanced(boxed_content) if boxed_content else None
            if gold_ans is None:
                lines = [line.strip() for line in full_sol.splitlines() if line.strip()]
                for line in reversed(lines):
                    if re.search(r'[\d\-+*/()=]', line):
                        gold_ans = _sanitize_enhanced(line)
                        break
            
            sol_wo_box = re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', full_sol)
            raw_steps = [s.strip() for s in sent_split.split(sol_wo_box) if s.strip()]
            steps = [f"Step {i+1}: {s}" for i, s in enumerate(raw_steps)]

            mi_sample_margin = self.compute_step_mi_sampling_marginal(qtxt, steps, n_samples=64)
            mi_sample_loo = self.compute_step_mi_sampling_loo(qtxt, steps, n_samples=64)
            mi_norm = self.normalize_mi(mi_sample_loo, mode=norm, **(norm_kwargs or {}), round_to=round_to,)

            entry = {
                    "question":      q_txt,
                    "completion":    steps,
                    "mi_sample_loo":   mi_sample_loo,
                    "mi_sample_margin":   mi_sample_margin,
                    "mi_sam_norm": mi_norm,
                    "gold_answer":   gold_ans,
                }
            yield entry

