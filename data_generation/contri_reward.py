import math
import re
from typing import List, Optional
import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Project-level helpers
from config import PRMConfig
from utils import _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer, system_prompt

class ContriRewardvLLM:
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
            \b\d+(?:\.\d+)?\b         # integers / decimals
          | \b\d+/\d+\b                 # simple fractions
        )
        """,
        re.VERBOSE,
    )
    
    def __init__(self, config: "PRMConfig", model_name: str = "mistralai/Mathstral-7B-v0.1"):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            quantization="bitsandbytes",
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.rollout_params = SamplingParams(
            temperature=0.5,
            top_p=0.9,
            max_tokens=self.config.max_new_tokens,
            n=self.config.num_rollouts,
            repetition_penalty=1.1,
        )
        self.masking_params = SamplingParams(
            temperature=0.5,
            top_p=0.9,
            max_tokens=self.config.max_new_tokens,
            n=self.config.num_rollouts,
            repetition_penalty=1.1,
        )
        print(f"vLLM model loaded: {model_name}")

    def _batched_generate(self, prompts: List[str], params: SamplingParams):
        return self.llm.generate(prompts, params)

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

    def _score_batch(self, outputs, gold_answer: str) -> List[float]:
        """Convert vLLM batched outputs → reward list (fraction of correct roll‑outs)."""
        rewards = []
        for result in outputs:
            correct = sum(
                1 for comp in result.outputs
                if (ans := self._extract_answer(comp.text)) and _numeric_equiv_enhanced(ans, gold_answer)
            )
            rewards.append(correct / float(self.config.num_rollouts))
        return rewards

    def compute_step_rewards_batch(self, question: str, sys_prompt: str, steps: List[str], gold_answer: str) -> List[float]:
        base_prompt = f"{sys_prompt}\n\nProblem: {question}\n"
        prompts = [
            base_prompt + "\n".join(steps[:i + 1]) + "\n" + (f"Step {i + 2}:" if i < len(steps) - 1 else "Answer:")
            for i in range(len(steps))
        ]
        outputs = self._batched_generate(prompts, self.rollout_params)
        # print("generated rollout outputs: ", outputs)
        return self._score_batch(outputs, gold_answer)
        
    def model_masking_batch(self, texts: List[str]) -> List[str]:
        mask_prompts = [
            (
                "In the sentence below, mask any word or expression that seems crucial (such as a variable or a number or a operator etc.) "
                "for solving the math problem by replacing it with '[MASKED]'.\n"
                f"Sentence: \"{t}\"\nRewritten:"
            )
            for t in texts
        ]
        outputs = self._batched_generate(mask_prompts, self.masking_params)
        return [out.outputs[0].text.strip() for out in outputs]

    def perturb_step_rewards_batch(self, question: str, sys_prompt: str, steps: List[str], gold_answer: str, use_llm: bool = True) -> List[float]:
        base_prompt = f"{sys_prompt}\n\nProblem: {question}\n"
        bodies = []
        prefixes = []
        for step in steps:
            m = re.match(r"^[\s>#*\-]*Step\s*\d+\s*[:.\-]\s*", step, flags=re.I)
            prefixes.append(m.group(0) if m else "")
            bodies.append(step[len(prefixes[-1]):])

        if use_llm:
            masked_bodies = self.model_masking_batch(bodies)
        else:
            masked_bodies = [self._MASK_PATTERN.sub("[MASKED]", b) for b in bodies]
            
        prompts = []
        for i in range(len(steps)):
            masked_step = prefixes[i] + masked_bodies[i]
            staged_steps = steps[:i] + [masked_step]
            label = f"Step {i + 2}:" if i < len(steps) - 1 else "Answer:"
            prompts.append(base_prompt + "\n".join(staged_steps) + "\n" + label)

        outputs = self._batched_generate(prompts, self.rollout_params)
        return self._score_batch(outputs, gold_answer)

    def gsm8k_reward_dataset_vllm(self, *, split: str = "train", start: int = 0, take: int | None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        # ds = ds.select(range(start, start + take)) if take else ds
        ds = ds.select(range(start, len(ds)))
        print("Generated dataset size: ", len(ds))

        for sample in tqdm(ds, desc="Building GSM8K contri reward-dataset"):
            q_txt, g_sol = sample["question"], sample["answer"]
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

            ori = self.compute_step_rewards_batch(q_txt, system_prompt("rollout"), steps, gold_ans)
            ptb = self.perturb_step_rewards_batch(q_txt, system_prompt("rollout"), steps, gold_ans, self.config.use_llm)
            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]

            entry = {
                "question": q_txt,
                "completion": steps,
                "ori_rewards": ori,
                "ptb_rewards": ptb,
                "contributions": contrib,
                "gold_answer": gold_ans,
            }
            yield entry

    def math_reward_dataset_vllm(self, *, split: str = "train", start: int = 0, take: int | None):
        sent_split = re.compile(r'\.(?!\d)(?=\s|$)')
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
        ds = ds.select(range(start, take)) if take else ds
        # ds = ds.select(range(start, len(ds)))
        print("Generated dataset size: ", len(ds))
        
        for sample in tqdm(ds, desc="Building MATH contri reward-dataset"):
            full_sol = sample["solution"]
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

            ori = self.compute_step_rewards_batch(sample["problem"], system_prompt("rollout"), steps, gold_ans)
            ptb = self.perturb_step_rewards_batch(sample["problem"], system_prompt("rollout"), steps, gold_ans, self.config.use_llm)
            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]

            entry = {
                "question": sample["problem"],
                "completion": steps,
                "ori_rewards": ori,
                "ptb_rewards": ptb,
                "contributions": contrib,
                "gold_answer": gold_ans,
            }
            yield entry
