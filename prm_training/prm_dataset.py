import random, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class StepwisePRMDataset(Dataset):
    """mcr rewards가 반환한 entries(list[dict])를 (input_ids, scalar_reward) 샘플들로 변환한다.
    한 entry = {question, completion[steps], rewards[float], …} →  (Problem + Step1, r1), (Problem + Step1 \nStep2, r2) …"""
    def __init__(
        self,
        entries: List[dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        reward_type: str = "mi_loo",
        *,
        cache_encodings: bool = True,
        preprocess: bool = True,
        apply_norm: bool = True,         # 정규화 적용 여부
        incorrect: bool = True,        # incorrect step injected 여부
        norm_mode: str = "unit",       # "signed" | "unit" | "relu" | "minmax" | "raw"
        norm_kwargs: Optional[Dict] = None,  # {"tau":1.5,"clip_z":3.0,"deadzone":0.2,"q_low":5,"q_high":95,"round_to":4}
    ):
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.reward_type = reward_type
        self.cache       = {} if cache_encodings else None
        self.samples: List[Tuple[str, float]] = []

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")

        print(f"reward_type: {self.reward_type} (normalize={self.apply_norm}, mode={self.norm_mode})")

        for e in entries:
            q_txt   = e["question"]
            steps   = e["completion"]
            ans = e["gold_answer"]

            if self.reward_type in ("mi_loo", "mi_shapley", "mi_margin"):
                raw_rewards = e[self.reward_type]
            elif self.reward_type == "ori":
                raw_rewards = e["ori_rewards"]
            elif self.reward_type == "contri":
                raw_rewards = e["contributions"]
            elif self.reward_type == "cmi":
                raw_rewards = [c + m for c, m in zip(e["contributions"], e["mi_filtered"])]
            elif self.reward_type == "orimi":
                raw_rewards = [o + m for o, m in zip(o_rewards, e["mi_filtered"])]
            elif self.reward_type == "pav":
                contributions = e["contributions"]
                raw_rewards = [contributions[0]] + [contributions[i] - contributions[i-1] for i in range(1, len(contributions))]
            else:
                raise ValueError(f"Unsupported reward_type for this entry schema: {self.reward_type}")
            
            if self.apply_norm:
                if self.incorrect:
                    rewards = self.recompute_mi_norm_with_ignore(raw_rewards, e["incorrect_mask"], mode=self.norm_mode, **self.norm_kwargs)
                else:
                    rewards = self.normalize_mi(raw_rewards, mode=self.norm_mode, **self.norm_kwargs)
            else:
                rewards = raw_rewards

            n = min(len(steps), len(scores))
            if n < len(steps) or n < len(scores):
                # 필요시 경고를 찍고 자름
                print(f"[warn] steps({len(steps)}) != scores({len(scores)}), truncating to {n}")
                steps  = steps[:n]
                scores = scores[:n]
            
            prefix_lines = [f"You are a math expert. Solve the problem step by step.\n\nProblem: {q_txt}"]
            for step_txt, r in zip(steps, rewards):
                prefix_lines.append(step_txt)
                full_txt = "\n".join(prefix_lines)
                if preprocess:
                    full_txt = self._clean(full_txt)
                self.samples.append((full_txt, float(r)))   # (text, reward)

    # --------------------------------------------------------------------- utils
    @staticmethod
    def _clean(txt: str) -> str:
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    # --------------------------------------------------------------------- normalized
    @staticmethod
    def _robust_z(x: np.ndarray, clip_z: float = 3.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-8
        z = (x - med) / (1.4826 * mad)
        if clip_z is not None:
            z = np.clip(z, -clip_z, clip_z)
        return z

    @staticmethod
    def _normalize_signed(x: np.ndarray, tau=1.5, clip_z=3.0, deadzone=0.2):
        z = _robust_z(x, clip_z=clip_z)
        s = np.tanh(z / max(tau, 1e-8))   # [-1,1]
        if deadzone and deadzone > 0.0:
            mag = np.maximum(0.0, np.abs(s) - deadzone) / (1.0 - deadzone)
            s = np.sign(s) * mag
        return s

    def normalize_mi(self, scores: List[float],*,
        mode: str = "signed",   # "signed"([-1,1]), "unit"([0,1]), "relu", "minmax", "raw"
        tau: float = 1.5, clip_z: float = 3.0,
        deadzone: float = 0.2,   # signed/unit에서 |s|<=deadzone을 0 근처로 수축
        q_low: float = 5.0,      # minmax용 로버스트 하한 퍼센타일
        q_high: float = 95.0,    # minmax용 로버스트 상한 퍼센타일
        round_to: Optional[int] = 4,) -> List[float]:
        """
        문항별 점수 벡터를 정규화.
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

    def recompute_mi_norm_with_ignore(self, raw: List[float], incorrect_mask: List[int],*, mode: str = "unit",         # "signed" | "unit" | "minmax" | "relu"
        tau: float = 1.5, clip_z: float = 3.0, deadzone: float = 0.2, q_low: float = 5.0, q_high: float = 95.0, round_to: int = 4) -> List[float]:
        x = np.asarray(raw, dtype=float)
        mask = np.asarray(incorrect_mask, dtype=int)
        keep = (mask == 0)

        x_keep = x[keep] if np.any(keep) else x
        if mode == "relu":
            y = np.maximum(x_keep, 0.0)
            out = np.zeros_like(x, dtype=float)
            out[keep] = y
            out[~keep] = 0.0
        elif mode == "signed":
            y = _normalize_signed(x_keep, tau=tau, clip_z=clip_z, deadzone=deadzone)
            out = np.zeros_like(x, dtype=float)
            out[keep] = y
            out[~keep] = -1.0
        elif mode == "unit":
            y = _normalize_signed(x_keep, tau=tau, clip_z=clip_z, deadzone=deadzone)
            y = 0.5 * (y + 1.0)
            out = np.zeros_like(x, dtype=float)
            out[keep] = y
            out[~keep] = 0.0
        elif mode == "minmax":
            lo = np.percentile(x_keep, q_low) if len(x_keep) else np.min(x)
            hi = np.percentile(x_keep, q_high) if len(x_keep) else np.max(x)
            if hi <= lo:
                scale = np.zeros_like(x_keep)
            else:
                scale = np.clip((x_keep - lo) / (hi - lo), 0.0, 1.0)
            out = np.zeros_like(x, dtype=float)
            out[keep] = scale
            out[~keep] = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if round_to is not None:
            out = np.round(out.astype(float), round_to)
        return out.tolist()

    # --------------------------------------------------------------------- dunder
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, reward = self.samples[idx]
        if self.cache is not None and text in self.cache:
            ids = self.cache[text]
        else:
            ids = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            if self.cache is not None:
                self.cache[text] = ids
        return ids, torch.tensor(reward, dtype=torch.float32)
