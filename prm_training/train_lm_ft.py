import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import random
import string
import re
import wandb
from torch.utils.data import Subset
from prm_dataset import StepwisePRMDataset   

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_PROJECT"]="mc_prm"
os.environ["WANDB_WATCH"]="false"

# PRM Custom Class
class FTLM(nn.Module):
    def __init__(self, base_model_name: str, lora_rank: int = 16, lora_alpha: int = 32, mlp_ratio: int = 4, value_head_prefix: str = "value_head", 
                 normalize_reward: bool = False):
        super().__init__()
        
        # Use AutoModelForCausalLM for pure CausalLM fine-tuning
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
        )

        # Add Lora Adapter for efficient fine-tuning
        self.backbone = prepare_model_for_kbit_training(self.backbone)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        # Value head for reward prediction
        hidden = self.backbone.config.hidden_size
        mlp_hidden = hidden // mlp_ratio
        head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1, bias=False),
        )
        self.value_head_prefix = value_head_prefix
        setattr(self, value_head_prefix, head)

        # head 가중치 학습 가능하도록 보장
        for p in head.parameters():
            p.requires_grad_(True)
        
        self.normalize_reward = normalize_reward
        self.register_buffer("mean", torch.zeros(1), persistent=False)
        self.register_buffer("std",  torch.ones(1),  persistent=False)

        # 캐시 비활성 (gradient checkpointing 호환)
        self.backbone.config.use_cache = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None, return_hidden: bool = False):
        if attention_mask is None:
            attention_mask = (input_ids != self.backbone.config.pad_token_id).long()

        # position_ids = cumulative mask
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )

        last_hidden = outputs.hidden_states[-1]        # (B, T, H)
        # Index of last non-pad token  → (B, 1)
        eos_idx = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(-1, keepdim=True)

        values = getattr(self, self.value_head_prefix)(last_hidden).squeeze(-1)   # (B, T)
        reward = values.gather(1, eos_idx).squeeze(1)                             # (B,)

        if labels is not None:
            loss = F.mse_loss(reward, labels.float())
            return loss, reward
        else:
            # if (not self.training) and self.normalize_reward:
            #     reward = (reward - self.mean) / (self.std + 1e-8)
            return (reward, last_hidden) if return_hidden else reward

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_parameter_stats(self):
        trainable_params = 0
        all_param = 0
        module_stats = {}
        
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
                module_name = name.split('.')[0]
                if module_name not in module_stats:
                    module_stats[module_name] = {'trainable': 0, 'total': 0}
                module_stats[module_name]['trainable'] += param.numel()
                module_stats[module_name]['total'] += param.numel()
            else:
                module_name = name.split('.')[0]
                if module_name not in module_stats:
                    module_stats[module_name] = {'trainable': 0, 'total': 0}
                module_stats[module_name]['total'] += param.numel()
        
        return {
            'total_params': all_param,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / all_param * 100,
            'module_stats': module_stats
        }

# Load Dataset 
@dataclass
class PRMCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, batch):
        input_ids, rewards = zip(*batch)
        lengths = [len(ids) for ids in input_ids]
        max_len = max(lengths)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        padded = [
            torch.cat([ids, ids.new_full((max_len - len(ids),), self.tokenizer.pad_token_id)])
            for ids in input_ids
        ]
        input_ids = torch.stack(padded)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        rewards = torch.tensor(rewards, dtype=torch.float)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": rewards,
        }

def main():
    # Load Model
    model_name = "Qwen/Qwen2.5-Math-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = FTLM(base_model_name=model_name, value_head_prefix="value_head")
    model.to(device)

    stats = model.get_parameter_stats()
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Trainable parameters: {stats['trainable_params']:,}")
    print(f"Trainable ratio: {stats['trainable_ratio']:.2f}%")
    
    # Load Dataset
    with open("/home/leena/ccc_eval/mcts_prm/cmi_samples/total_gsm8k_merge_mistral.json", "r") as file:
        gsm8k_raw = json.load(file)

    # Use the new CausalLM PRM dataset
    reward_type = "cmi"
    print(f"Load Dataset with {reward_type}")
    full_ds = StepwisePRMDataset(gsm8k_raw, tokenizer, reward_type=reward_type)
    indices = list(range(len(full_ds)))
    split_idx = int(0.9 * len(full_ds)) if len(full_ds) > 1 else 1
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:] if len(full_ds) > 1 else indices[:1]

    train_ds = Subset(full_ds, train_indices)
    valid_ds = Subset(full_ds, val_indices)
    collate = PRMCollator(tokenizer)
    print("Finish Loading Dataset")

    # Load Trainer
    output_dir = f"/home/leena/ccc_eval/mcts_prm/checkpoints/pt_lm/{reward_type}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,       
        weight_decay=0.0, 
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=200, 
        save_strategy="steps",
        save_steps=600,
        save_total_limit=2, 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        seed=42,
        bf16=True,  # bf16=True
        report_to="wandb",
        run_name=f"qwen_lm_ft_gsm8k_{reward_type}",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collate
    )

    print("Start Training")
    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model")) # save finetuned model
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    adapter_dir = os.path.join(output_dir, "adapter")
    model.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    wandb.finish()


if __name__ == "__main__":
    main()