import re, math, random
import numpy as np
import sympy as sp
from typing import Optional, List, Tuple, Dict, Iterable
import torch
from datasets import load_dataset
from tqdm import tqdm
from utils import system_prompt, _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer

class MIReward2:
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
    def _entropy_bits_total(self, prompt: str, target: str) -> Tuple[float, float, int]:
        """
        Return (H_total_bits, H_bits_per_token, target_len) for H(A | prompt).
        - H_total_bits: sum_t H(p_t) over target tokens, in bits
        - H_bits_per_token: H_total_bits / target_len (0 if target_len==0)
        - target_len: # of target tokens included in conditional entropy
        """
        # Build the exact concatenated string used by the model
        full = prompt + target
        # Tokenize full (with specials) and target (NO specials so it matches suffix)
        full_enc = self.tokenizer(full, return_tensors="pt", add_special_tokens=True).to(self.device)
        tgt_ids  = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        L_full   = full_enc["input_ids"].shape[1]
        L_tgt    = len(tgt_ids)
        if L_tgt == 0:
            return 0.0, 0.0, 0
        # Target token region is the suffix of FULL
        Lp = L_full - L_tgt  # start idx of target tokens inside FULL
        # Forward pass: logits [1, L, V]
        logits = self.model(**full_enc).logits.float()
        # H(p) = -sum_j p_j log p_j ; use log_softmax for stability
        LOG2E = 1.0 / math.log(2.0)
        H_bits_sum = 0.0
        # Iterate only target time steps
        for t in range(Lp, L_full):
            lp = torch.log_softmax(logits[0, t], dim=-1)   # [V], natural log
            # p*log p in nats; convert to bits with LOG2E
            H_bits_sum += (-(lp.exp() * lp).sum().item()) * LOG2E
        return H_bits_sum, (H_bits_sum / L_tgt), L_tgt
    
    def _H_cached(self, prompt: str, target: str) -> float:
        """H_total bits with simple string-key cache."""
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}
        key = ("H|", prompt, "\u241E", target)  # 분리자 U+241E
        if key in self._H_CACHE:
            return self._H_CACHE[key]
        H, _, _ = self._entropy_bits_total(prompt, target)
        self._H_CACHE[key] = H
        return H

    def compute_step_mi_loo(self, question: str, steps: List[str], gold_answer: str):
        """
        LOO 공헌도: Δ_i = H(all \ i) - H(all) = (스텝 i를 빼면 엔트로피가 얼마나 커지는가) 반환: List[float] (각 스텝의 Δ_i)
        """
        sys_prompt = (
            'Solve the given problem with step by step reasoning in the format of '
            '"Step k: <k-th rationale>" and write final answer in the format of '
            '"Answer: <final answer>".\nProblem: '
        )
        question = re.sub(r' +', ' ', question)
        answer_text = gold_answer.strip()
        if not answer_text.lower().startswith("answer:"):
            answer_text = "Answer: " + answer_text

        base = sys_prompt + question + "\n\n"
        with_all = base + "".join(s.rstrip() + "\n" for s in steps)
        H_all = self._H_cached(with_all, answer_text)

        contribs = []
        for i in range(len(steps)):
            without_i = base + "".join(steps[j].rstrip() + "\n" for j in range(len(steps)) if j != i)
            H_wo = self._H_cached(without_i, answer_text)
            contribs.append(H_wo - H_all)  # 제거 시 ↑만큼이 i의 공헌
        return contribs

    def compute_step_mi_marginal(self, question: str, steps: List[str], gold_answer: str):
        """
        단독 효과: MI_i = H(base) - H(base + S_i) 순서 의존성 ↓. 반환: List[float]
        """
        sys_prompt = (
            'Solve the given problem with step by step reasoning in the format of '
            '"Step k: <k-th rationale>" and write final answer in the format of '
            '"Answer: <final answer>".\nProblem: '
        )
        question = re.sub(r' +', ' ', question)
        answer_text = gold_answer.strip()
        if not answer_text.lower().startswith("answer:"):
            answer_text = "Answer: " + answer_text

        base = sys_prompt + question + "\n\n"
        H_base = self._H_cached(base, answer_text)

        mis = []
        for s in steps:
            with_i = base + s.rstrip() + "\n"
            H_with = self._H_cached(with_i, answer_text)
            mis.append(H_base - H_with)
        return mis

    def compute_step_mi_shapley(self, question: str, steps: List[str], gold_answer: str, n_perm: int = 16, seed: int = 42,):
        """
        Shapley 근사: 여러 랜덤 순열 π 평균의 증분 MI 
        φ_i ≈ E_π[ H(base + prefix_before_i) - H(base + prefix_before_i + S_i) ]
        """
        sys_prompt = (
            'Solve the given problem with step by step reasoning in the format of '
            '"Step k: <k-th rationale>" and write final answer in the format of '
            '"Answer: <final answer>".\nProblem: '
        )
        question = re.sub(r' +', ' ', question)
        answer_text = gold_answer.strip()
        if not answer_text.lower().startswith("answer:"):
            answer_text = "Answer: " + answer_text
        base = sys_prompt + question + "\n\n"

        N = len(steps)
        rng = random.Random(seed)
        shap = [0.0] * N
        for _ in range(n_perm):
            idxs = list(range(N))
            rng.shuffle(idxs)
            # 순열을 따라 prefix를 늘려가며 계산
            prompt = base
            H_prev = self._H_cached(prompt, answer_text)
            for k, idx in enumerate(idxs):
                # 현재 스텝을 추가했을 때
                prompt_with = prompt + steps[idx].rstrip() + "\n"
                H_with = self._H_cached(prompt_with, answer_text)
                # 해당 순열에서의 마진(=증분 MI)
                marg = H_prev - H_with
                shap[idx] += marg
                # prefix 업데이트 (다음 스텝의 "이전"이 됨)
                prompt = prompt_with
                H_prev = H_with
        # 평균
        shap = [v / n_perm for v in shap]
        return shap

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

            mi_loo = self.compute_step_mi_loo(q_txt, steps, gold_ans)
            mi_shapley = self.compute_step_mi_shapley(q_txt, steps, gold_ans)
            mi_margin = self.compute_step_mi_marginal(q_txt, steps, gold_ans)
            mi_norm = self.normalize_mi(mi_loo, mode=norm, **(norm_kwargs or {}), round_to=round_to,)

            entry = {
                    "question":      q_txt,
                    "completion":    steps,
                    "mi_loo":   mi_loo,
                    "mi_shapley":   mi_shapley,
                    "mi_margin": mi_margin,
                    "mi_norm": mi_norm,
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
            full_sol   = sample["solution"]
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

            mi_loo = self.compute_step_mi_loo(q_txt, steps, gold_ans)
            mi_shapley = self.compute_step_mi_shapley(q_txt, steps, gold_ans)
            mi_margin = self.compute_step_mi_marginal(q_txt, steps, gold_ans)
            mi_norm = self.normalize_mi(mi_loo, mode=norm, **(norm_kwargs or {}), round_to=round_to,)

            entry = {
                    "question":      q_txt,
                    "completion":    steps,
                    "mi_loo":   mi_loo,
                    "mi_shapley":   mi_shapley,
                    "mi_margin": mi_margin,
                    "mi_norm": mi_norm,
                    "gold_answer":   gold_ans,
                }
            yield entry

