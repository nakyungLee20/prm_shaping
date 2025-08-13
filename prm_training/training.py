import os, json, math, random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, PreTrainedModel
)
from peft import LoraConfig, get_peft_model, PeftModel
from dataset import PRMDataset, PRMPackCollator
from wrapper import PRMRewardWrapper

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_PROJECT"]="mc_prm"
os.environ["WANDB_WATCH"]="false"

# ------------------------------------------------------------------------------------
try:
    from transformers import set_seed as _hf_set_seed      # 최신 버전
except Exception:
    try:
        from transformers.trainer_utils import set_seed as _hf_set_seed  # 구버전
    except Exception:
        _hf_set_seed = None

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _hf_set_seed is not None:
        _hf_set_seed(seed)

def regression_loss(pred: torch.Tensor, target: torch.Tensor, *, kind: str = "huber", delta: float = 0.5):
    if kind == "mse":
        return F.mse_loss(pred, target, reduction="none")
    return F.huber_loss(pred, target, delta=delta, reduction="none")

# ------------------------------------------------------------------------------------
class PRMTrainer(Trainer):
    def __init__(self, *args, loss_type: str = "huber", huber_delta: float = 0.5, incorrect_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.incorrect_weight = incorrect_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # inputs: {"input_ids","attention_mask","rw_positions","targets_list","is_incorrect_list",...}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        hs = model(input_ids=input_ids, attention_mask=attention_mask)  # (B, T, H)
        B, T, H = hs.shape

        preds, tgts, wgts = [], [], []
        for b in range(B):
            pos = inputs["rw_positions"][b].to(hs.device)            # (K,)
            tar = inputs["targets_list"][b].to(hs.device).float()    # (K,)
            inc = inputs["is_incorrect_list"][b].to(hs.device).long()# (K,)
            if pos.numel() == 0: 
                continue
            vecs = hs[b, pos, :]                                     # (K, H)
            p = model.reward_head(vecs).squeeze(-1)                  # (K,)
            w = torch.where(inc == 1, torch.as_tensor(self.incorrect_weight, device=hs.device), torch.tensor(1.0, device=hs.device))
            preds.append(p)
            tgts.append(tar)
            wgts.append(w)

        if not preds:
            loss = torch.zeros([], device=hs.device, requires_grad=True)
            return (loss, {"preds": [], "tgts": []}) if return_outputs else loss

        preds = torch.cat(preds, dim=0)  # (N,)
        tgts  = torch.cat(tgts,  dim=0)  # (N,)
        wgts  = torch.cat(wgts,  dim=0)  # (N,)

        base = regression_loss(preds, tgts, kind=self.loss_type, delta=self.huber_delta)  # (N,)
        loss = (base * wgts).mean()
        outputs = {"preds": preds.detach(), "tgts": tgts.detach()}

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lr_backbone = self.args.learning_rate              # LoRA용
        lr_head     = getattr(self.args, "learning_rate_head", lr_backbone)
        lr_embed    = getattr(self.args, "learning_rate_embed", lr_backbone * 0.1)

        head_params, lora_params, embed_params = [], [], []
        emb_param_ids = set(id(p) for p in self.model.backbone.get_input_embeddings().parameters())

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if "reward_head" in n:
                head_params.append(p)
            elif pid in emb_param_ids:
                embed_params.append(p)
            else:
                lora_params.append(p)

        optim_groups = [
            {"params": lora_params,  "weight_decay": 0.0, "lr": lr_backbone},
            {"params": head_params,  "weight_decay": 0.0, "lr": lr_head},
            {"params": embed_params, "weight_decay": 0.0, "lr": lr_embed},  # ★ 임베딩은 작은 LR
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer

# ------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-Math-7B"   # or "meta-llama/Meta-Llama-3.1-8B", "mistralai/Mistral-7B-v0.3"
    rw_token: str = "<RW>"
    add_rw_token: bool = True
    max_length: int = 3500
    output_dir: str = "/home/leena/ccc_eval/mcts_prm/checkpoints/mi"
    # trainarguments
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 10
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    learning_rate_head: float = 2e-4      # reward_head 크게
    learning_rate_embed: float = 2e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    bf16: bool = True
    seed: int = 42
    # loss
    loss_type: str = "huber"  # "huber" | "mse"
    huber_delta: float = 0.5
    incorrect_weight: float = 1.0
    # data type
    reward_key: Optional[str] = "mi_loo"
    reward_source_priority: tuple = ("mi_loo","mi_shapley","mi_margin","mi_norm","ori_rewards","contributions")
    apply_norm: bool = True
    norm_mode: str = "unit"
    norm_kwargs: dict = field(default_factory=lambda: {"tau":1.5,"clip_z":3.0,"deadzone":0.2,"q_low":5,"q_high":95,"round_to":4})
    use_incorrect_mask_for_norm: bool = True
    rand_prompt_variant: bool = True
    val_ratio: float = 0.1
    # lora
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

# ------------------------------------------------------------------------------------
def main(cfg: TrainConfig):
    set_all_seeds(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1) Tokenizer & Model
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    if cfg.add_rw_token and cfg.rw_token not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [cfg.rw_token]})
        base.resize_token_embeddings(len(tok))
    
    if cfg.use_lora:
        lconf = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",)
        base = get_peft_model(base, lconf)
        base.print_trainable_parameters()

    rw_id = tok.convert_tokens_to_ids(cfg.rw_token) if cfg.add_rw_token else None
    model = PRMRewardWrapper(base, rw_token_id=rw_id, rw_token=cfg.rw_token)
    emb_module = model.backbone.get_input_embeddings()   # nn.Embedding
    emb_module.weight.requires_grad_(True)
    # print("Model Strucutre:", model)

    # 2) Dataset (pack mode)
    data_path = "/home/leena/ccc_eval/mcts_prm/cmi_samples/test_incorr.json"
    with open(data_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    full_ds  = PRMDataset(
        entries , tok,
        reward_key=cfg.reward_key,
        reward_source_priority=list(cfg.reward_source_priority),
        apply_norm=cfg.apply_norm, norm_mode=cfg.norm_mode, norm_kwargs=cfg.norm_kwargs,
        use_incorrect_mask_for_norm=cfg.use_incorrect_mask_for_norm,
        add_rw_token=cfg.add_rw_token, rw_token=cfg.rw_token,
        mode="pack", max_length=cfg.max_length, truncate_strategy="tail",
        rand_prompt_variant=cfg.rand_prompt_variant,
    )

    n = len(full_ds)
    n_val = max(1, int(n * cfg.val_ratio))
    n_train = max(1, n - n_val)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    data_collator = PRMPackCollator(pad_token_id=tok.pad_token_id, rw_token_id=tok.convert_tokens_to_ids(cfg.rw_token), strict=False,)
    print(f"Load model and {cfg.reward_key} type dataset!")

    # 3) TrainingArguments
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps if val_ds is not None else None,
        bf16=cfg.bf16,
        report_to=["wandb"],
        run_name = f"test_{cfg.reward_key}_{cfg.norm_mode}_test",
    )
    setattr(args, "learning_rate_head", cfg.learning_rate_head)
    setattr(args, "learning_rate_embed", cfg.learning_rate_embed)

    # 4) Trainer
    trainer = PRMTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        loss_type=cfg.loss_type,
        huber_delta=cfg.huber_delta,
        incorrect_weight=cfg.incorrect_weight,
    )

    # 5) Train
    print("Start Training")
    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(os.path.join(cfg.output_dir, "final_model"))
    tok.save_pretrained(os.path.join(cfg.output_dir, "final_model"))
    adapter_dir = os.path.join(cfg.output_dir, "final_model/adapter")
    wandb.finish()


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)

