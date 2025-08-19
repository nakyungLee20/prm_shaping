import re, copy, random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# -------------------- 유틸: 정규화 (incorrect 제외) --------------------
def _robust_z(x: np.ndarray, clip_z: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    z = (x - med) / (1.4826 * mad)
    if clip_z is not None:
        z = np.clip(z, -clip_z, clip_z)
    return z

def _normalize_signed_core(x: np.ndarray, tau=1.5, clip_z=3.0, deadzone=0.2):
    z = _robust_z(x, clip_z=clip_z)
    s = np.tanh(z / max(tau, 1e-8))   # [-1,1]
    if deadzone and deadzone > 0.0:
        mag = np.maximum(0.0, np.abs(s) - deadzone) / (1.0 - deadzone)
        s = np.sign(s) * mag
    return s

def normalize_vector(scores: List[float], incorrect_mask: Optional[List[int]] = None, *,
                     mode: str = "unit", tau=1.5, clip_z=3.0, deadzone=0.2, q_low=5.0, q_high=95.0, round_to=4) -> List[float]:
    x = np.asarray(scores, dtype=float)
    if incorrect_mask is None:
        keep = np.ones_like(x, dtype=bool)
    else:
        keep = (np.asarray(incorrect_mask) == 0)

    x_keep = x[keep] if np.any(keep) else x
    if mode == "relu":
        y_keep = np.maximum(x_keep, 0.0)
        out = np.zeros_like(x, dtype=float); out[keep] = y_keep; out[~keep] = 0.0
    elif mode == "signed":
        y_keep = _normalize_signed_core(x_keep, tau=tau, clip_z=clip_z, deadzone=deadzone)
        out = np.zeros_like(x, dtype=float); out[keep] = y_keep; out[~keep] = -1.0
    elif mode == "unit":
        y_keep = _normalize_signed_core(x_keep, tau=tau, clip_z=clip_z, deadzone=deadzone)
        y_keep = 0.5 * (y_keep + 1.0)
        out = np.zeros_like(x, dtype=float); out[keep] = y_keep; out[~keep] = 0.0
    elif mode == "minmax":
        lo = np.percentile(x_keep, q_low) if len(x_keep) else np.min(x)
        hi = np.percentile(x_keep, q_high) if len(x_keep) else np.max(x)
        if hi <= lo:
            scale = np.zeros_like(x_keep)
        else:
            scale = np.clip((x_keep - lo) / (hi - lo), 0.0, 1.0)
        out = np.zeros_like(x, dtype=float); out[keep] = scale; out[~keep] = 0.0
    elif mode == "raw":
        out = x
    else:
        raise ValueError(f"Unknown norm mode: {mode}")

    if round_to is not None:
        out = np.round(out.astype(float), round_to)
    return out.tolist()

PROMPT_TEMPLATES = [
    "You are a math expert. Solve the problem step by step.\n\nProblem: {q}",
    "Solve carefully, one step at a time.\n\nProblem: {q}",
    "Think step by step and show your reasoning.\n\nProblem: {q}",
]

class PRMDataset(Dataset):
    """
    PRM 학습용 전처리:
    - mode="unroll": (prefix upto step_i + <RW>) → target=r_i  (스텝별 샘플)
    - mode="pack": (전체 스텝 + 각 스텝 뒤 <RW>) → targets=[r_1,..,r_n], rw_positions=[...]
    """
    def __init__(
        self,
        entries: List[dict],
        tokenizer,
        *,
        reward_key: Optional[str] = None,
        reward_source_priority: List[str] = ("mi_norm", "mi_loo", "mi_shapley", "mi_margin", "ori_rewards", "contributions"),
        apply_norm: bool = True,
        norm_mode: str = "unit",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_incorrect_mask_for_norm: bool = True,
        add_rw_token: bool = True,
        rw_token: str = "<RW>",
        mode: str = "pack",  # "unroll" | "pack"
        max_length: int = 1024,
        truncate_strategy: str = "tail",  # "tail" (뒤 유지) | "head"
        step_dropout: float = 0.0,  # unroll 모드에서 특정 스텝 샘플을 확률적으로 드롭(과적합 완화)
        incorrect_weight: float = 1.0,  # collator에서 loss weight에 활용 가능
        rand_prompt_variant: bool = True,  # prefix 템플릿 랜덤
    ):
        assert mode in ("unroll", "pack")
        self.tok = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.truncate_strategy = truncate_strategy
        self.add_rw_token = add_rw_token
        self.rw_token = rw_token
        self.step_dropout = step_dropout
        self.incorrect_weight = incorrect_weight
        self.rand_prompt_variant = rand_prompt_variant

        # special token 등록
        if self.add_rw_token and rw_token not in self.tok.get_vocab():
            self.tok.add_special_tokens({"additional_special_tokens": [rw_token]})

        self.rw_id = self.tok.convert_tokens_to_ids(rw_token) if self.add_rw_token else None
        self.samples = []  # mode="unroll": list of dict; mode="pack": same but targets/rw_positions are lists

        for e in entries:
            q = e["question"]
            steps: List[str] = e["completion"]
            incorrect_mask: Optional[List[int]] = e.get("incorrect_mask")
            # 1) Reward Source
            reward_vec = None
            if reward_key is not None:
                if reward_key in e and isinstance(e[reward_key], list) and len(e[reward_key]) == len(steps):
                    reward_vec = [float(x) for x in e[reward_key]]
                else:
                    continue
            else:
                for key in reward_source_priority:
                    if key in e and isinstance(e[key], list) and len(e[key]) == len(steps):
                        reward_vec = [float(x) for x in e[key]]
                        break
                if reward_vec is None:
                    continue
            # 2) Normalize
            if apply_norm:
                reward_vec = normalize_vector(
                    reward_vec,
                    incorrect_mask if use_incorrect_mask_for_norm else None,
                    mode=norm_mode,
                    **(norm_kwargs or {})
                )
            # 3) Diversify prefix
            if self.rand_prompt_variant:
                tpl = random.choice(PROMPT_TEMPLATES)
            else:
                tpl = PROMPT_TEMPLATES[0]
            prefix0 = tpl.format(q=q)

            if self.mode == "unroll":   # stepwise parsed dataset
                prefix_lines = [prefix0]
                for i, (st, r) in enumerate(zip(steps, reward_vec)):
                    if self.step_dropout > 0 and random.random() < self.step_dropout:
                        continue
                    prefix_lines.append(st)
                    text = "\n".join(prefix_lines)
                    if self.add_rw_token:
                        text = text.rstrip() + f"\n{self.rw_token}"
                    enc = self._encode(text, self.max_length, self.truncate_strategy)
                    rw_pos = self._find_rw_position(enc) if self.add_rw_token else (len(enc["input_ids"]) - 1)
                    self.samples.append({
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "rw_positions": [int(rw_pos)],            # list
                        "targets": [float(r)],                    # list
                        "is_incorrect": [int(incorrect_mask[i]) if incorrect_mask else 0],  # list
                        "meta": {"q": q, "step_idx": i},
                    })
            else:                       # total + token inserted dataset
                lines = [prefix0]
                rw_positions = []
                for i, st in enumerate(steps):
                    lines.append(st)
                    if self.add_rw_token:
                        lines.append(self.rw_token)
                text = "\n".join(lines)
                enc = self._encode(text, self.max_length, self.truncate_strategy)
                rw_positions = self._find_all_rw_positions(enc) if self.add_rw_token else []
                n_valid = min(len(rw_positions), len(reward_vec))
                rw_positions = rw_positions[:n_valid]
                targets = [float(x) for x in reward_vec[:n_valid]]
                is_incorrect = [int(x) for x in (incorrect_mask[:n_valid] if incorrect_mask else [0]*n_valid)]
                if n_valid == 0:
                    continue
                self.samples.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "rw_positions": rw_positions,
                    "targets": targets,
                    "is_incorrect": is_incorrect,
                    "meta": {"q": q, "n_steps": n_valid},
                })

    # -------------------- helpers --------------------
    def _encode(self, text: str, max_length: int, strategy: str):
        enc = self.tok(text, max_length=max_length, truncation=True, padding=False, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

    def _find_rw_position(self, enc):
        ids = enc["input_ids"].tolist()
        if self.rw_id is not None and self.rw_id in ids:
            return ids.index(self.rw_id)
        # fallback: 마지막 토큰
        return len(ids) - 1

    def _find_all_rw_positions(self, enc):
        ids = enc["input_ids"].tolist()
        if self.rw_id is None:
            return []
        return [i for i, t in enumerate(ids) if t == self.rw_id]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

# -------------------- Collators --------------------
class PRMPackCollator:
    def __init__(self, pad_token_id: int, rw_token_id: int = None, strict: bool = False):
        self.pad = pad_token_id
        self.rw_id = rw_token_id
        self.strict = strict

    def _ensure_pack_schema(self, b: dict) -> dict:
        # 1) rw_positions 복구: (a) 이미 있으면 OK (b) 'rw_pos'면 리스트로 승격 (c) input_ids에서 <RW> 스캔
        if "rw_positions" not in b:
            if "rw_pos" in b:
                b["rw_positions"] = [int(b["rw_pos"])]
            elif self.rw_id is not None:
                ids = b["input_ids"]
                if torch.is_tensor(ids):
                    pos = (ids == self.rw_id).nonzero(as_tuple=False).flatten().tolist()
                else:
                    # ids가 텐서가 아니라면 텐서로 변환 후 탐색
                    ids_t = torch.tensor(ids, dtype=torch.long)
                    pos = (ids_t == self.rw_id).nonzero(as_tuple=False).flatten().tolist()
                b["rw_positions"] = pos
            else:
                if self.strict:
                    raise KeyError("Sample has no 'rw_positions' and no 'rw_pos', and rw_token_id not provided.")
                b["rw_positions"] = []
        # 2) targets 복구
        if "targets" not in b:
            if "target" in b:
                b["targets"] = [float(b["target"])]
            else:
                if self.strict:
                    raise KeyError("Sample has no 'targets' or 'target'.")
                b["targets"] = [0.0] * len(b["rw_positions"])
        # 3) is_incorrect 복구
        if "is_incorrect" not in b:
            b["is_incorrect"] = [0] * min(len(b["rw_positions"]), len(b["targets"]))
        # 4) 길이 정합
        n = min(len(b["rw_positions"]), len(b["targets"]), len(b["is_incorrect"]))
        b["rw_positions"] = b["rw_positions"][:n]
        b["targets"]      = b["targets"][:n]
        b["is_incorrect"] = b["is_incorrect"][:n]
        return b

    def __call__(self, batch):
        batch = [self._ensure_pack_schema(b) for b in batch]

        def pad_stack(key, pad_val):
            seqs = [b[key] for b in batch]
            seqs = [s if torch.is_tensor(s) else torch.tensor(s, dtype=torch.long) for s in seqs]
            maxlen = max(x.size(0) for x in seqs)
            out = torch.full((len(seqs), maxlen), pad_val, dtype=seqs[0].dtype)
            for i, x in enumerate(seqs):
                out[i, :x.size(0)] = x
            return out
        
        input_ids      = pad_stack("input_ids", self.pad)
        attention_mask = pad_stack("attention_mask", 0)
        # attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)  # 이미 토크나이저가 1/0 부여했다면 동일 pad 크기로 맞추기
        rw_positions   = [torch.tensor(b["rw_positions"], dtype=torch.long) for b in batch]
        targets_list   = [torch.tensor(b["targets"], dtype=torch.float32) for b in batch]
        incor_list     = [torch.tensor(b["is_incorrect"], dtype=torch.long) for b in batch]
        meta           = [b.get("meta", {}) for b in batch]

        keep = [i for i in range(len(batch)) if rw_positions[i].numel() > 0]
        if len(keep) < len(batch):
            print("Keep", keep)

        return {"input_ids":input_ids, "attention_mask":attention_mask,
                "rw_positions":rw_positions, "targets_list":targets_list,
                "is_incorrect_list":incor_list, "meta":meta}


class PRMUnrollCollator:
    def __init__(self, dtype=torch.long):
        self.dtype = dtype
    def __call__(self, batch):
        input_ids      = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        rw_pos         = torch.tensor([b["rw_pos"] for b in batch], dtype=torch.long)
        targets        = torch.tensor([b["target"] for b in batch], dtype=torch.float32)
        is_incorrect   = torch.tensor([b["is_incorrect"] for b in batch], dtype=torch.long)
        meta           = [b["meta"] for b in batch]
        return {"input_ids":input_ids, "attention_mask":attention_mask,
                "rw_pos":rw_pos, "targets":targets, "is_incorrect":is_incorrect, "meta":meta}

