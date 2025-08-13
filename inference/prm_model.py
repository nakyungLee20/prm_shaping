import torch
import torch.nn as nn
from typing import Optional
from config import PRMConfig
# from prm_training.config import PRMConfig

class ProcessRewardModel(nn.Module):
    def __init__(self, input_size: int, cfg: "PRMConfig"):
        """ 
        Args:
            input_size : CLS-embedding dim of the frozen LLM backbone
            cfg        : PRMConfig instance (hidden_size, num_layers, dropout …)
        """
        super().__init__()
        
        self.input_size = input_size
        # self.output_size = cfg.output_size
        h = cfg.hidden_size
        p_drop = cfg.dropout
        n_layers = cfg.num_layers
        act_fn     = nn.GELU()

         # ── first projection ────────────────────────────────────────────
        self.in_proj = nn.Sequential(
            nn.Linear(input_size, h),
            nn.LayerNorm(h),
            act_fn,
            nn.Dropout(p_drop),
        )

        # ── stacked residual blocks ─────────────────────────────────────
        blocks = []
        for _ in range(n_layers - 1):
            blocks.append(
                nn.Sequential(                   # pre-LN residual MLP
                    nn.LayerNorm(h),
                    nn.Linear(h, h),
                    act_fn,
                    nn.Dropout(p_drop),
                    nn.Linear(h, h),
                    nn.Dropout(p_drop),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # ── output head ────────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for blk in self.blocks:
            x = x + blk(x)          # residual connection
        return self.out_proj(x).squeeze(-1)

    def get_complexity(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
