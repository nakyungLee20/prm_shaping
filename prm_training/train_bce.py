# pip install transformers peft accelerate datasets
import torch, torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments)
from transformers.modeling_outputs import ModelOutput
from peft import LoraConfig, get_peft_model

STEP_SEP = "<extra_0>"

class PRMHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 2):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):  # (bs, seq, h)
        return self.classifier(hidden_states)  # (bs, seq, num_labels)

class PRMModel(nn.Module):
    """
    Backbone: Causal LM (Qwen, Llama, Mistral...)
    Head: token-level classifier -> logits_prm (bs, seq, C)
    Forward inputs:
      input_ids, attention_mask, step_mask(bool), step_labels(int or float)
    Returns: dict(loss, logits_prm)
    """
    def __init__(self, base_model_name: str, use_lora: bool = True, num_labels: int = 2):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        if use_lora:
            lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                              target_modules=["q_proj","k_proj","v_proj","o_proj"])
            self.backbone = get_peft_model(self.backbone, lora)

        hidden_size = self.backbone.config.hidden_size
        self.prm_head = PRMHead(hidden_size, num_labels=num_labels)

        # choose loss: 2-class CE (hard labels) or BCE (soft labels)
        self.use_bce = (num_labels == 1)

    def forward(self, input_ids=None, attention_mask=None,
                step_mask=None, step_labels=None):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True)
        last_h = out.hidden_states[-1]            # (bs, seq, h)
        logits = self.prm_head(last_h)            # (bs, seq, C) or (bs, seq, 1)

        loss = None
        if step_labels is not None and step_mask is not None:
            # select only <extra_0> positions
            active = step_mask.bool().view(-1)    # (bs*seq,)
            if logits.shape[-1] == 1:
                # BCE with logits (soft labels allowed)
                pos_logits = logits.view(-1, 1)[active].squeeze(-1)    # (N,)
                labels = step_labels.view(-1)[active].float()          # (N,)
                loss = nn.functional.binary_cross_entropy_with_logits(pos_logits, labels)
            else:
                # CrossEntropy (hard labels: 0/1)
                pos_logits = logits.view(-1, logits.size(-1))[active]  # (N, C)
                labels = step_labels.view(-1)[active].long()           # (N,)
                loss = nn.functional.cross_entropy(pos_logits, labels)

        return {"loss": loss, "logits_prm": logits}

@dataclass
class CollatedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    step_mask: torch.Tensor
    step_labels: torch.Tensor

class PRMDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_len=2048, num_labels=2):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len
        self.num_labels = num_labels

        # ensure STEP_SEP token exists
        if STEP_SEP not in self.tok.get_vocab():
            self.tok.add_special_tokens({"additional_special_tokens":[STEP_SEP]})

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["query"]},
            {"role": "assistant", "content": STEP_SEP.join(ex["steps"]) + STEP_SEP},
        ]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len)

        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        step_id = self.tok.encode(STEP_SEP, add_special_tokens=False)[0]
        mask = (input_ids == step_id)  # True at each step boundary

        # align labels to the number of <extra_0> occurrences
        idxs = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        assert len(idxs) == len(ex["labels"]), f"mismatch steps={len(idxs)} labels={len(ex['labels'])}"

        # pack labels to full seq (we'll index by mask later)
        if self.num_labels == 1:  # soft label BCE
            labels_full = torch.zeros_like(input_ids, dtype=torch.float32)
            labels_full[mask] = torch.tensor(ex["labels"], dtype=torch.float32)
        else:                      # CE 0/1
            labels_full = torch.full_like(input_ids, fill_value=-100)  # unused elsewhere
            labels_full[mask] = torch.tensor(ex["labels"], dtype=torch.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "step_mask": mask.to(torch.int8),
            "step_labels": labels_full,
        }

def collate_fn(batch):
    keys = ["input_ids","attention_mask","step_mask","step_labels"]
    max_len = max(x["input_ids"].size(0) for x in batch)
    out = {}
    for k in keys:
        tensors = [x[k] for x in batch]
        pad_val = 0 if k != "step_labels" else (0 if tensors[0].dtype==torch.float32 else -100)
        out[k] = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_val)
    return CollatedBatch(**out)

# ---- usage (toy) ----
base = "Qwen/Qwen2.5-Math-7B"  # 보통 base 수치모델에서 시작, 또는 Instruct도 가능
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
train_data = [
  {
    "system": "Please reason step by step, and put your final answer within \\boxed{}.",
    "query": "…문제…",
    "steps": ["step1 text", "step2 text", "step3 text"],
    "labels": [1,0,1],   # hard labels (CE). soft label 쓰려면 num_labels=1로 바꾸세요
  },
]
ds = PRMDataset(train_data, tok, max_len=4096, num_labels=2)
model = PRMModel(base, use_lora=True, num_labels=2)

args = TrainingArguments(
    output_dir="prm_out",
    learning_rate=5e-5,  # 7B면 1e-5 ~ 5e-5 사이부터 탐색
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    fp16=False, bf16=True,
    logging_steps=10, save_steps=500, warmup_ratio=0.03,
)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate_fn)
trainer.train()
