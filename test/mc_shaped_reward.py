import math
import sympy as sp
import re
from typing import Optional, List
import torch
from datasets import load_dataset
from tqdm import tqdm
from utils import system_prompt, _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer

class MCRewardShaped:
    ANSWER_PATTERN = re.compile(
        r"""^[\s>#*\-]*          # optional markdown/bullet symbols
            Answer               # word 'Answer'
            \s*[:.\-]\s*         # separator
            (.+?)\s*$            # capture everything after
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )
    _ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")
    _MASK_PATTERN = re.compile(
        r"""
        (?:
        # {ops_pattern}|                # operator patterns
            \b\d+(?:\.\d+)?\b         # integers / decimals
          | \b\d+/\d+\b                 # simple fractions
        #   | \b[a-zA-Z]\b                 # single‑letter variables
        )
        """,
        re.VERBOSE,
    )
    
    def __init__(self, config: "PRMConfig", model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # 모델이 지원하는 generate 파라미터 확인
        self.supported_params = self._get_supported_generate_params()

    def _get_supported_generate_params(self):
        """모델이 지원하는 generate 파라미터 목록 반환"""
        # 기본적으로 지원되는 파라미터들
        base_params = {
            'max_new_tokens', 'do_sample', 'num_return_sequences', 
            'temperature', 'top_p', 'pad_token_id', 'eos_token_id',
            'repetition_penalty', 'length_penalty', 'no_repeat_ngram_size'
        }
        
        # 모델 타입에 따라 추가 지원 파라미터 확인
        model_type = type(self.model).__name__.lower()
        model_config = getattr(self.model, 'config', None)
        
        # Qwen 계열 모델들 확인
        qwen_models = ['qwen', 'qwq', 'qwen2', 'qwen2.5']
        is_qwen = any(qm in model_type for qm in qwen_models)
        
        if model_config:
            model_name = getattr(model_config, 'model_type', '').lower()
            is_qwen = is_qwen or any(qm in model_name for qm in qwen_models)
            
            # 모델 이름에서도 확인
            if hasattr(model_config, 'name_or_path'):
                model_path = model_config.name_or_path.lower()
                is_qwen = is_qwen or any(qm in model_path for qm in qwen_models)
        
        if is_qwen:
            # Qwen 모델들은 presence_penalty를 지원하지 않을 수 있음
            print(f"Detected Qwen model: {model_type}, excluding presence_penalty")
            return base_params
        else:
            # 다른 모델들은 presence_penalty를 지원할 수 있음
            base_params.add('presence_penalty')
            return base_params

    def _filter_generate_params(self, **kwargs):
        """지원되지 않는 파라미터를 필터링"""
        return {k: v for k, v in kwargs.items() if k in self.supported_params}

    # Function to parse a solution text into steps and final answer.
    def _extract_answer(self, text: str) -> Optional[str]:
        """Try multiple heuristics / regexes to pull out an answer string."""
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
    
    def compute_step_rewards(self, question, sys_prompt, steps, gold_answer):
        rewards = []
        total_steps = len(steps)

        # Pre‑encode static prefix (sys_prompt + question) once for efficiency
        base_prompt = f"{sys_prompt}\n\nProblem: {question}\n"
        base_ids = self.tokenizer.encode(base_prompt, return_tensors="pt").to(self.device)

        for i in range(total_steps):
            prefix_tokens = self.tokenizer.encode("\n".join(steps[: i + 1]) + "\n", return_tensors="pt").to(self.device) # steps up to current step i (0-indexed)
            # Decide how to prompt the next part:
            if i < total_steps - 1:
                next_label = f"Step {i + 2}:"
            else:
                next_label = "Answer: "
            cont_ids = self.tokenizer.encode(next_label, return_tensors="pt").to(self.device)
            # Build full prefix ids (avoid Python concat inefficiency by cat)
            prefix_ids = torch.cat([base_ids, prefix_tokens, cont_ids], dim=-1)
            
            # 지원되는 파라미터만 사용
            generate_params = self._filter_generate_params(
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                num_return_sequences=self.config.num_rollouts,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            rollout_outputs = self.model.generate(prefix_ids, **generate_params)
            new_token_start = prefix_ids.shape[-1] 
            # Check each rollout's final answer against the gold answer
            correct_count = 0
            for idx, seq in enumerate(rollout_outputs):
                completion = self.tokenizer.decode(seq[new_token_start:], skip_special_tokens=True)
                pred_answer = self._extract_answer(completion)
                print(f"[{i+1}-th Step, {idx}-th Original Rollout]", completion, "Pred Answer", pred_answer, "Gold Answer", gold_answer)
                if pred_answer is not None and _numeric_equiv_enhanced(pred_answer, gold_answer):
                    correct_count += 1
            reward = correct_count / float(self.config.num_rollouts)
            rewards.append(reward)
        return rewards
    
    # Using perurbed rollouts to compute step rewards
    def model_masking(self, text: str, *, max_new_tokens: int = 64) -> str:
        prompt = "In the sentence below, mask any word or expression that seems crucial for solving the math step. This may include key numbers, variables, or action words (like operations), but you should decide what matters. Replace each important item with '[MASKED]'. Keep everything else unchanged. Return ONE line.\n\nSentence: \"{sent}\"\nRewritten:".format(sent=text)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 지원되는 파라미터만 사용
        generate_params = self._filter_generate_params(
            max_new_tokens=max_new_tokens,
            temperature=0.5, 
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        out_ids = self.model.generate(input_ids, **generate_params)
        return self.tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def perturb_step_rewards(self, question: str, sys_prompt: str, steps: List[str], gold_answer: str, use_llm: bool = True) -> List[float]:
        ptb_rewards: List[float] = []
        total_steps = len(steps)
        base_prompt = f"{sys_prompt}\n\nProblem: {question}\n"
        base_ids = self.tokenizer.encode(base_prompt, return_tensors="pt").to(self.device)

        for i in range(total_steps):
            # 1. Perturb *only* step i
            orig_step = steps[i] 
            step_match = re.match(r"^[\s>#*\-]*Step\s*\d+\s*[:.\-]\s*", orig_step, flags=re.I)
            prefix = step_match.group(0) if step_match else ""
            # ② 나머지 부분(body)만 마스킹
            body   = steps[i][len(prefix):]                       # 접두사 뒷부분
            if use_llm:
                masked_body = self.model_masking(body)
            else:
                masked_body = self._MASK_PATTERN.sub("[MASKED]", body)
            # ③ 접두사 + 마스킹된 body
            masked_step = prefix + masked_body    
            ptb_prefix_steps = steps[:i] + [masked_step]
            print("perturbed step:", ptb_prefix_steps)

            prefix_tokens = self.tokenizer.encode("\n".join(ptb_prefix_steps) + "\n", return_tensors="pt").to(self.device)
            next_label = f"Step {i + 2}:" if i < total_steps - 1 else "Answer:"
            cont_ids = self.tokenizer.encode(next_label, return_tensors="pt").to(self.device)
            prefix_ids = torch.cat([base_ids, prefix_tokens, cont_ids], dim=-1)

            # 지원되는 파라미터만 사용
            generate_params = self._filter_generate_params(
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                num_return_sequences=self.config.num_rollouts,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            rollout_outputs = self.model.generate(prefix_ids, **generate_params)
            new_token_start = prefix_ids.shape[-1]
            correct_count = 0
            for idx, seq in enumerate(rollout_outputs):
                completion = self.tokenizer.decode(seq[new_token_start:], skip_special_tokens=True)
                pred_answer = self._extract_answer(completion)
                # print(f"Masked [{i+1}-th Step, {idx}-th Rollout]", completion, "Pred Answer", pred_answer)
                if pred_answer is not None and _numeric_equiv_enhanced(pred_answer, gold_answer):
                    correct_count += 1
            ptb_rewards.append(correct_count / float(self.config.num_rollouts))
        return ptb_rewards
    
    # Using Mutual Information to compute step rewards
    def entropy_bits_exact(self, prompt: str, target: str) -> float:
        """True H(A|prompt) in bits/token, by ∑_t H(p_t). Memory-intensive: stores full probs tensor."""
        LOG2E = 1 / math.log(2)
        full   = prompt + target
        inputs = self.tokenizer(full, return_tensors="pt").to(self.device)
        Lp     = len(self.tokenizer(prompt)["input_ids"])

        with torch.no_grad():
            logits = self.model(**inputs).logits.float()      # [1,L,V]

        probs = logits.softmax(-1)                      # [...,V]
        token_H = -(probs * probs.log()).sum(-1) * LOG2E  # bits/token

        mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        mask[:, Lp:] = True                             # answer tokens
        return token_H[mask].sum().item() / mask.sum().item()
    
    def compute_step_mi(self, question: str, steps: List[str], gold_answer: str):
        sys_prompt = """Solve the given problem with step by step reasoning in the format of "Step k: <k-th rationale>" and write final answer in the format of "Answer: <answer>".\nProblem: """
        question = re.sub(r' +', ' ', question) 
        gold_answer = "Answer: " + gold_answer
        context = sys_prompt + question + "\n\n"

        mi_incremental = []
        cumulative_prompt = context
        for i, step in enumerate(steps):
            h_before = self.entropy_bits_exact(cumulative_prompt, gold_answer)
            cumulative_prompt += step+"\n"
            h_after = self.entropy_bits_exact(cumulative_prompt, gold_answer)
            # I(S_i ; A | context, S_1,...,S_{i-1}) = H(A|prev) - H(A|prev,S_i)
            incremental_mi = h_before - h_after
            mi_incremental.append(incremental_mi)
        return mi_incremental
    
    # Build datasets based on input datas
    def gsm8k_reward_dataset(self, *, split: str = "train", start: int = 0, take: int | None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        if take is not None:
            ds = ds.select(range(start, start+take))
        else:
            ds = ds.select(range(start, len(ds)))

        dataset    = []
        for sample in tqdm(ds, desc="Building GSM-8K reward-dataset"):
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

            ori = self.compute_step_rewards(q_txt, system_prompt("rollout"), steps, gold_ans)
            ptb = self.perturb_step_rewards(q_txt, system_prompt("rollout"), steps, gold_ans, self.config.use_llm)
            mi = self.compute_step_mi(q_txt, steps, gold_ans)

            naive = [round(o + max(m, 0), 4) for o, m in zip(ori, mi)]
            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]

            entry = {
                    "question":      q_txt,
                    "completion":    steps,
                    "ori_rewards":   ori,
                    "ptb_rewards":   ptb,
                    "contributions": contrib,
                    "mi_rewards":   mi,
                    "naive_rewards": naive,
                    "gold_answer":   gold_ans,
                }
            dataset.append(entry)
        return dataset

    def math_reward_dataset(self, *, split: str = "train", start: int = 0, take: int | None):
        sent_split = re.compile(r'\.(?!\d)(?=\s|$)')   # 소수점·수식 내부 마침표 무시
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
        if take is not None:
            ds = ds.select(range(start, start+take))
        else:
            ds = ds.select(range(start, len(ds)))

        dataset    = []
        for sample in tqdm(ds, desc="Building MATH reward-dataset"):
            full_sol   = sample["solution"]

            boxed_content = _extract_boxed_answer(full_sol)
            gold_ans = _sanitize_enhanced(boxed_content) if boxed_content else None
            if gold_ans is None:
                # Fallback: look for last mathematical expression
                lines = [line.strip() for line in full_sol.splitlines() if line.strip()]
                for line in reversed(lines):
                    if re.search(r'[\d\-+*/()=]', line):
                        gold_ans = _sanitize_enhanced(line)
                        break
            
            # Remove all \\boxed{...} for step extraction  
            sol_wo_box = re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', full_sol)
            raw_steps = [s.strip() for s in sent_split.split(sol_wo_box) if s.strip()]
            steps = [f"Step {i+1}: {s}" for i, s in enumerate(raw_steps)]

            # Calculate rewards
            ori = self.compute_step_rewards(sample["problem"], system_prompt("rollout"), steps, gold_ans)
            ptb = self.perturb_step_rewards(sample["problem"], system_prompt("rollout"), steps, gold_ans, self.config.use_llm)
            mi = self.compute_step_mi(sample["problem"], steps, gold_ans)

            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]
            naive = [round(o + max(m, 0), 4) for o, m in zip(ori, mi)]

            entry = {
                "question":      sample["problem"],
                "completion":    steps,
                "ori_rewards":   ori,
                "ptb_rewards":   ptb,
                "contributions": contrib,
                "mi_rewards":   mi,
                "naive_rewards": naive,
                "gold_answer":   gold_ans,
            }
            dataset.append(entry)
        return dataset
    
    def gsm8k_mi_reward_dataset(self, *, split: str = "train", start: int = 0, take: int | None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        if take is not None:
            ds = ds.shuffle(seed=self.config.seed).select(range(start, start+take))
        else:
            ds = ds.shuffle(seed=self.config.seed).select(range(start, len(ds)))

        dataset    = []
        for sample in tqdm(ds, desc="Building GSM-8K reward-dataset"):
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

            ori = self.compute_step_rewards(q_txt, system_prompt("rollout"), steps, gold_ans)
            mi = self.compute_step_mi(q_txt, steps, gold_ans)
            naive = [round(o + max(m, 0), 4) for o, m in zip(ori, mi)]

            entry = {
                    "question":      q_txt,
                    "completion":    steps,
                    "ori_rewards":   ori,
                    "mi_rewards":   mi,
                    "naive_rewards": naive,
                    "gold_answer":   gold_ans,
                }
            dataset.append(entry)
        return dataset

    def math_mi_reward_dataset(self, *, split: str = "train", start: int = 0, take: int | None):
        sent_split = re.compile(r'\.(?!\d)(?=\s|$)')   # 소수점·수식 내부 마침표 무시
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
        if take is not None:
            ds = ds.select(range(start, start+take))
        else:
            ds = ds.select(range(start, len(ds)))

        dataset    = []
        for sample in tqdm(ds, desc="Building MATH reward-dataset"):
            full_sol   = sample["solution"]

            boxed_content = _extract_boxed_answer(full_sol)
            gold_ans = _sanitize_enhanced(boxed_content) if boxed_content else None
            if gold_ans is None:
                # Fallback: look for last mathematical expression
                lines = [line.strip() for line in full_sol.splitlines() if line.strip()]
                for line in reversed(lines):
                    if re.search(r'[\d\-+*/()=]', line):
                        gold_ans = _sanitize_enhanced(line)
                        break
            
            # Remove all \\boxed{...} for step extraction  
            sol_wo_box = re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', full_sol)
            raw_steps = [s.strip() for s in sent_split.split(sol_wo_box) if s.strip()]
            steps = [f"Step {i+1}: {s}" for i, s in enumerate(raw_steps)]

            # Calculate rewards
            ori = self.compute_step_rewards(sample["problem"], system_prompt("rollout"), steps, gold_ans)
            mi = self.compute_step_mi(sample["problem"], steps, gold_ans)
            naive = [round(o + max(m, 0), 4) for o, m in zip(ori, mi)]

            entry = {
                "question":      sample["problem"],
                "completion":    steps,
                "ori_rewards":   ori,
                "mi_rewards":   mi,
                "naive_rewards": naive,
                "gold_answer":   gold_ans,
            }
            dataset.append(entry)
        return dataset

    # Streaming versions for memory-efficient processing
    def math_reward_dataset_streaming(self, *, split: str = "train", start: int = 0, take: int | None):
        """Streaming version that yields entries one by one"""
        sent_split = re.compile(r'\.(?!\d)(?=\s|$)')   # 소수점·수식 내부 마침표 무시
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
        if take is not None:
            ds = ds.select(range(start, start+take))
        else:
            ds = ds.select(range(start, len(ds)))

        for sample in tqdm(ds, desc="Building MATH reward-dataset"):
            full_sol   = sample["solution"]

            boxed_content = _extract_boxed_answer(full_sol)
            gold_ans = _sanitize_enhanced(boxed_content) if boxed_content else None
            if gold_ans is None:
                # Fallback: look for last mathematical expression
                lines = [line.strip() for line in full_sol.splitlines() if line.strip()]
                for line in reversed(lines):
                    if re.search(r'[\d\-+*/()=]', line):
                        gold_ans = _sanitize_enhanced(line)
                        break
            
            # Remove all \\boxed{...} for step extraction  
            sol_wo_box = re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', full_sol)
            raw_steps = [s.strip() for s in sent_split.split(sol_wo_box) if s.strip()]
            steps = [f"Step {i+1}: {s}" for i, s in enumerate(raw_steps)]

            # Calculate rewards
            ori = self.compute_step_rewards(sample["problem"], system_prompt("rollout"), steps, gold_ans)
            ptb = self.perturb_step_rewards(sample["problem"], system_prompt("rollout"), steps, gold_ans, self.config.use_llm)
            mi = self.compute_step_mi(sample["problem"], steps, gold_ans)

            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]
            naive = [round(o + max(m, 0), 4) for o, m in zip(ori, mi)]

            entry = {
                "question":      sample["problem"],
                "completion":    steps,
                "ori_rewards":   ori,
                "ptb_rewards":   ptb,
                "contributions": contrib,
                "mi_rewards":   mi,
                "naive_rewards": naive,
                "gold_answer":   gold_ans,
            }
            yield entry

    def gsm8k_reward_dataset_streaming(self, *, split: str = "train", start: int = 0, take: int | None):
        """Streaming version that yields entries one by one"""
        ds = load_dataset("openai/gsm8k", "main", split=split)
        if take is not None:
            ds = ds.select(range(start, start+take))
        else:
            ds = ds.select(range(start, len(ds)))

        for sample in tqdm(ds, desc="Building GSM-8K reward-dataset"):
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

            ori = self.compute_step_rewards(q_txt, system_prompt("rollout"), steps, gold_ans)
            ptb = self.perturb_step_rewards(q_txt, system_prompt("rollout"), steps, gold_ans, self.config.use_llm)
            mi = self.compute_step_mi(q_txt, steps, gold_ans)

            naive = [round(o + max(m, 0), 4) for o, m in zip(ori, mi)]
            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]

            entry = {
                    "question":      q_txt,
                    "completion":    steps,
                    "ori_rewards":   ori,
                    "ptb_rewards":   ptb,
                    "contributions": contrib,
                    "mi_rewards":   mi,
                    "naive_rewards": naive,
                    "gold_answer":   gold_ans,
                }
            yield entry
