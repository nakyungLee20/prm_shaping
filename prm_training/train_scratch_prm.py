import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict, deque
import json
import wandb
import os
import random
from torch.utils.data import DataLoader
import wandb
from torch.utils.data import Subset

# Project‑level helpers 
from config import PRMConfig
from prm_trainer import PRMTrainer
from prm_dataset import StepwisePRMDataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model_name = "Qwen/Qwen2.5-Math-7B" # PRM training용 작은 모델 사용 (dataset generation과 독립적)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    cfg = PRMConfig()
    print(f"Using model: {model_name} for PRM training")

    with open("/home/leena/ccc_eval/mcts_prm/cmi_samples/total_gsm8k_merge_mistral.json", "r") as file:
        gsm8k_raw = json.load(file)
    
    full_ds = StepwisePRMDataset(gsm8k_raw, tokenizer, cfg.max_new_tokens, reward_type=cfg.reward_type)
    print(f"Full dataset size: {len(full_ds)}") 

    indices = list(range(len(full_ds)))
    # random.shuffle(indices)
    split_idx = int(0.9 * len(full_ds)) if len(full_ds) > 1 else 1
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:] if len(full_ds) > 1 else indices[:1]

    train_ds = Subset(full_ds, train_indices)
    val_ds   = Subset(full_ds, val_indices)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True,)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True,)
    print(f"Train/Val size: {len(train_ds)} / {len(val_ds)}")
    print("Finish Loading PRM Datasets!")

    trainer = PRMTrainer(cfg, model=model, tokenizer=tokenizer)
    history = trainer.fit(train_loader, val_loader)
    print("PRM Training complete. Loss history:", history)


if __name__ == "__main__":
    main()