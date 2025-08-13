import json, copy, random, re
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter

# ======================= Expanded Banks =======================
SELF_REFLECTION_BANK = [
    "Let me think about this problem carefully.",
    "I need to check my calculations.",
    "This step seems important for the solution.",
    "Let me verify the previous step.",
    "I should double-check my work.",
    "This is a crucial part of the solution.",
    "Let me organize my thoughts.",
    "I need to be careful with the math.",
    "I should confirm the units are consistent.",
    "Let me restate the given conditions precisely.",
    "I might have made an algebraic slip; let me re-derive.",
    "I should check edge cases and constraints.",
    "Let me simplify the expression before substituting values.",
    "I should verify if I used the correct formula.",
    "Let me check whether I applied the operation in the right order.",
    "I should compute a quick sanity check with approximations.",
    "Let me recompute using an alternative method to confirm.",
    "I should cross-check with the final requirement of the question.",
    "Let me verify that each transformation is logically valid.",
    "I should test with a small example to validate the pattern.",
]

WRONG_STEP_BANK = [
    # 산술/대수
    "5 + 3 = 9",
    "10 * 2 = 15",
    "20 / 4 = 6",
    "x/2 = 6, so x = 10",
    "2^2 = 5",
    "\\sqrt(36) = 5",
    "Perimeter of square side 5 = 15",
    # 추가: 분수/부호/분배/지수/로그/근사
    "1/3 + 1/6 = 1/9",
    "(-2)^2 = -4",
    "2(x + 3) = 2x + 3",
    "x^a * x^b = x^{a-b}",
    "\\log(ab) = \\log a - \\log b",
    "\\sqrt{a+b} = \\sqrt a + \\sqrt b",
    "1/0 = 0",
    "0^0 = 0",
    "7/10 ≈ 0.9",
    # 기하
    "Area of triangle = base + height",
    "Circumference of a circle with r=3 is 3r",
    "Area of a circle with r=3 is 2\\pi r^2",
    "Pythagorean theorem: a + b = c",
    # 확률/통계
    "P(A \\cap B) = P(A) + P(B)",
    "Variance of cX is Var(X) + c",
    "Mean of [2,4,9] is 6",
    # 미적분
    "d/dx (x^2) = 2",
    "∫ x dx = x^2 + C (missing 1/2)",
    "Derivative of sin x is -cos x",
    "Product rule: (fg)' = f' + g'",
    # 방정식 처리
    "From 2x = 6, x = 2 (dividing by 2 and subtracting 2)",
    "If x/y = 2/3, then x = y (cross-multiplication error)",
]

IRRELEVANT_BANK = [
    "The weather is nice today.",
    "I like mathematics very much.",
    "This reminds me of my school days.",
    "The sky is blue and beautiful.",
    "I should drink more water.",
    "Patterns exist in everything.",
    "I should make a grocery list.",
    "I wonder what to cook for dinner.",
    "This pencil needs sharpening.",
    "I should clean my desk later.",
    "The soundtrack from that movie is stuck in my head.",
    "My cat was very energetic this morning.",
    "I might take a walk after finishing this.",
    "I should reply to that email soon.",
    "I forgot to water the plants yesterday.",
    "I wonder if it's going to rain tomorrow.",
    "I should back up my files.",
    "This coffee tastes a bit strong.",
    "I need to charge my phone.",
]

# ======================= Utils =======================
def renumber_steps(steps: List[str], start_idx: int = 0) -> List[str]:
    new_steps: List[str] = []
    for i in range(len(steps)):
        s = steps[i]
        content = s.split(":", 1)[1] if ":" in s else s
        new_steps.append(f"Step {start_idx + i + 1}:{content}")
    return new_steps

def _is_numeric_list(v: Any, expected_len: int) -> bool:
    if not isinstance(v, list) or len(v) != expected_len:
        return False
    try:
        _ = [float(x) for x in v]
        return True
    except Exception:
        return False

def _sample_perturbation_type(perturbation_probs: Optional[Dict[str, float]] = None, rng: Optional[random.Random] = None,) -> str:
    R = rng or random
    types = ["wrong_step", "irrelevant", "self_reflection"]
    if perturbation_probs:
        # normalize
        items = [(t, max(0.0, float(perturbation_probs.get(t, 0.0)))) for t in types]
        total = sum(w for _, w in items)
        if total <= 0:
            weights = [1/3, 1/3, 1/3]
        else:
            weights = [w/total for _, w in items]
    else:
        weights = [1/3, 1/3, 1/3]
    return random.choices(types, weights=weights, k=1)[0]

def create_perturbed_steps(steps: List[str], typ: str, insert_pos: int, rng: Optional[random.Random] = None) -> (List[str], int):
    R = rng or random
    assert 0 <= insert_pos <= len(steps)
    new_steps = steps.copy()
    if typ == "wrong_step":
        ins = f"Step {insert_pos + 1}: {random.choice(WRONG_STEP_BANK)}"
    elif typ == "irrelevant":
        ins = f"Step {insert_pos + 1}: {random.choice(IRRELEVANT_BANK)}"
    else:
        ins = f"Step {insert_pos + 1}: {random.choice(SELF_REFLECTION_BANK)}"
    new_steps.insert(insert_pos, ins)
    new_steps = renumber_steps(new_steps)
    return new_steps, insert_pos

def _robust_z(x: np.ndarray, clip_z: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    z = (x - med) / (1.4826 * mad)
    if clip_z is not None:
        z = np.clip(z, -clip_z, clip_z)
    return z

def _normalize_signed(x: np.ndarray, tau=1.5, clip_z=3.0, deadzone=0.2):
    z = _robust_z(x, clip_z=clip_z)
    s = np.tanh(z / max(tau, 1e-8))   # [-1,1]
    if deadzone and deadzone > 0.0:
        mag = np.maximum(0.0, np.abs(s) - deadzone) / (1.0 - deadzone)
        s = np.sign(s) * mag
    return s

def recompute_mi_norm_with_ignore(raw: List[float], incorrect_mask: List[int],*, mode: str = "unit",         # "signed" | "unit" | "minmax" | "relu"
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

# ======================= Main API functions =======================
def inject_incorrect_step( entry: Dict[str, Any],*, min_raw_value: float = -1e6, norm_mode_for_mi_norm: str = "signed", norm_kwargs: Optional[Dict[str, Any]] = None,
    perturbation_probs: Optional[Dict[str, float]] = None, rng: Optional[random.Random] = None,) -> Dict[str, Any]:
    
    R = rng or random
    e = copy.deepcopy(entry)
    steps = e.get("completion", [])
    if not isinstance(steps, list) or len(steps) == 0:
        return e

    L = len(steps)
    insert_pos = random.randint(0, L)
    ptype = _sample_perturbation_type(perturbation_probs)

    # 1) 텍스트 삽입
    new_steps, ins = create_perturbed_steps(steps, ptype, insert_pos)
    e["completion"] = new_steps
    e["perturbation"] = ptype
    e["perturbation_pos"] = ins
    e["incorrect_mask"] = [0] * len(new_steps)
    e["incorrect_mask"][ins] = 1
    if perturbation_probs:
        e["perturbation_probs_used"] = perturbation_probs

    # 2) 모든 수치형 벡터에 min_raw_value 삽입 (mi_norm 제외)
    numeric_keys = []
    for k, v in list(e.items()):
        if k == "mi_norm":
            continue
        if _is_numeric_list(v, L):
            numeric_keys.append(k)

    for k in numeric_keys:
        old = [float(x) for x in e[k]]
        e[k] = old[:ins] + [float(min_raw_value)] + old[ins:]

    # 3) mi_norm 재계산 (주입 스텝 무시하고 통계 산출 → 주입 스텝은 바닥값 강제)
    base_priority = ["mi_loo", "mi_shapley", "mi_margin"]  # "ori_rewards", "contributions", "cmi"
    base_key = next((k for k in base_priority if k in e and _is_numeric_list(e[k], len(new_steps))), None)

    if base_key is not None:
        nk = norm_kwargs or {}
        e["mi_norm"] = recompute_mi_norm_with_ignore(
            e[base_key],
            e["incorrect_mask"],
            mode=norm_mode_for_mi_norm,
            tau=nk.get("tau", 1.5),
            clip_z=nk.get("clip_z", 3.0),
            deadzone=nk.get("deadzone", 0.2),
            q_low=nk.get("q_low", 5.0),
            q_high=nk.get("q_high", 95.0),
            round_to=nk.get("round_to", 4),
        )
        e["norm_mode"] = norm_mode_for_mi_norm
        e["norm_kwargs"] = nk
    return e

def process_dataset_copy(data: List[Dict[str, Any]],*,min_raw_value: float = -1e6, norm_mode_for_mi_norm: str = "signed",
    norm_kwargs: Optional[Dict[str, Any]] = None, perturbation_probs: Optional[Dict[str, float]] = None, seed: int = 42,) -> (List[Dict[str, Any]], Dict[str, Any]):
    
    random.seed(seed)
    out = []
    stats = {"total": 0, "types": Counter(), "positions": Counter()}
    for e in data:
        ne = inject_incorrect_step( e,min_raw_value=min_raw_value, norm_mode_for_mi_norm=norm_mode_for_mi_norm, norm_kwargs=norm_kwargs, perturbation_probs=perturbation_probs,)
        out.append(ne)
        stats["total"] += 1
        stats["types"][ne["perturbation"]] += 1
        stats["positions"][ne["perturbation_pos"]] += 1
    # Counter -> dict
    stats["types"] = dict(stats["types"])
    stats["positions"] = dict(stats["positions"])
    return out, stats

# ======================= File I/O helpers =======================
def print_stats(stats: Dict[str, Any]):
    print(f"Processed entries: {stats.get('total', 0)}")
    print("Type distribution:")
    for k, v in sorted(stats.get("types", {}).items()):
        print(f"  - {k}: {v}")
    print("Insert position distribution (0-based after renumbering):")
    for k, v in sorted(stats.get("positions", {}).items(), key=lambda x: int(x[0])):
        print(f"  - pos {k}: {v}")

def process_file_copy(input_path: str, output_path: str, **kwargs):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Top-level JSON must be a list"
    new_data, stats = process_dataset_copy(data, **kwargs)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print_stats(stats)


def main():
    input_file = "/home/leena/ccc_eval/mcts_prm/cmi_samples/test_json.json"
    output_file = "/home/leena/ccc_eval/mcts_prm/cmi_samples/test_incorr.json"
    process_file_copy(
        input_path=input_file,
        output_path=output_file,
        min_raw_value=-1e5,
        norm_mode_for_mi_norm="unit",                # "signed" | "unit" | "minmax" | "relu"
        norm_kwargs={"tau":1.5, "clip_z":3.0, "deadzone":0.2, "q_low":5, "q_high":95, "round_to":4},
        perturbation_probs={"wrong_step":0.4, "irrelevant":0.4, "self_reflection":0.2},
    )

if __name__ == "__main__":
    main()