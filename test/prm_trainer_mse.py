import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project-level helpers
from prm_dataset import StepwisePRMDataset
from prm_model import ProcessRewardModel
from config import PRMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prm_mse_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PRMTrainerMSE:
    """
    Enhanced PRM Trainer with MSE loss for step-wise reward prediction
    Supports both from-scratch training and fine-tuning
    """
    def __init__(self, cfg: PRMConfig, model, tokenizer, from_scratch: bool = False):
        self.cfg = cfg
        self.from_scratch = from_scratch
        torch.manual_seed(cfg.seed)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model and tokenizer setup
        self.tokenizer = tokenizer
        self.model = model
        
        if not from_scratch:
            # Freeze the backbone LLM for feature extraction
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            logger.info("Backbone LLM frozen for feature extraction")
        else:
            logger.info("Training from scratch - all parameters trainable")
        
        # PRM model setup
        feat_dim = self.model.config.hidden_size
        self.prm = ProcessRewardModel(feat_dim, cfg=cfg)
        self.model.to(self.device)
        self.prm.to(self.device)
        
        # Optimizer setup
        if from_scratch:
            # Train all parameters
            trainable_params = list(self.model.parameters()) + list(self.prm.parameters())
            self.opt = optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        else:
            # Train only PRM parameters
            self.opt = optim.AdamW(self.prm.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        
        # Loss function - MSE for regression
        self.crit = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = None
        if cfg.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.opt, T_max=cfg.epochs, eta_min=1e-6)
        elif cfg.lr_scheduler == "linear":
            self.total_steps = math.ceil(cfg.epochs * cfg.dataset_size / cfg.batch_size)
            def lr_lambda(step):
                if step < cfg.warmup_steps:
                    return step / max(1, cfg.warmup_steps)
                progress = (step - cfg.warmup_steps) / max(1, self.total_steps - cfg.warmup_steps)
                return max(0.0, 1.0 - progress)
            self.scheduler = LambdaLR(self.opt, lr_lambda)
        
        # Checkpoint directory
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        
        # Wandb setup
        self.wandb_run = None
        if cfg.use_wandb:
            self.wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name,
                config=vars(cfg),
                tags=["mse", "prm", "stepwise"]
            )
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = getattr(cfg, 'early_stopping_patience', 5)
        
        logger.info(f"PRM Model parameters: {self.prm.get_complexity():,}")
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.prm.parameters() if p.requires_grad):,}")

    def _encode_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone model
        Args:
            input_ids: [B, T] input token ids
        Returns:
            features: [B, feat_dim] extracted features
        """
        if self.from_scratch:
            # For from-scratch training, use the full model
            with torch.set_grad_enabled(True):
                outputs = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
                features = outputs.hidden_states[-1][:, 0, :]  # CLS token
        else:
            # For fine-tuning, freeze backbone
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
                features = outputs.hidden_states[-1][:, 0, :]  # CLS token
        
        return features.float()

    def _run_epoch(self, loader: DataLoader, train: bool, epoch_idx: int) -> Dict[str, float]:
        """
        Run one epoch of training or validation
        """
        self.prm.train(train)
        if self.from_scratch:
            self.model.train(train)
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(loader, desc=f"{'Training' if train else 'Validation'} Epoch {epoch_idx}")
        
        for step, (ids, targets) in enumerate(progress_bar):
            ids, targets = ids.to(self.device), targets.to(self.device)
            
            with torch.set_grad_enabled(train):
                # Forward pass
                features = self._encode_features(ids)
                predictions = self.prm(features).squeeze(-1)
                loss = self.crit(predictions, targets)
                
                if train:
                    # Backward pass
                    self.opt.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    if hasattr(self.cfg, 'grad_clip'):
                        torch.nn.utils.clip_grad_norm_(self.prm.parameters(), self.cfg.grad_clip)
                        if self.from_scratch:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    
                    # Gradient accumulation (optional)
                    if hasattr(self.cfg, 'grad_accum_steps') and self.cfg.grad_accum_steps > 1:
                        if (step + 1) % self.cfg.grad_accum_steps == 0:
                            self.opt.step()
                            if self.scheduler:
                                self.scheduler.step()
                    else:
                        self.opt.step()
                        if self.scheduler:
                            self.scheduler.step()
                
                # Collect metrics
                total_loss += loss.item()
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (step + 1):.4f}'
                })
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(loader)
        
        return metrics

    def _calculate_metrics(self, predictions: List[float], targets: List[float]) -> Dict[str, float]:
        """
        Calculate regression metrics
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Calculate correlation
        correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'correlation': correlation,
            'rmse': np.sqrt(mse)
        }

    def train(self, train_entries: List[Dict], val_entries: Optional[List[Dict]] = None):
        """
        Main training loop
        Args:
            train_entries: Training data entries
            val_entries: Validation data entries (optional)
        """
        logger.info(f"Starting training with {len(train_entries)} training samples")
        if val_entries:
            logger.info(f"Validation set size: {len(val_entries)}")
        
        # Create datasets
        train_dataset = StepwisePRMDataset(
            entries=train_entries,
            tokenizer=self.tokenizer,
            max_length=self.cfg.max_length,
            reward_type=self.cfg.reward_type,
            cache_encodings=True,
            preprocess=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )
        
        val_loader = None
        if val_entries:
            val_dataset = StepwisePRMDataset(
                entries=val_entries,
                tokenizer=self.tokenizer,
                max_length=self.cfg.max_length,
                reward_type=self.cfg.reward_type,
                cache_encodings=True,
                preprocess=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True
            )
        
        # Training loop
        for epoch in range(self.cfg.epochs):
            # Training
            train_metrics = self._run_epoch(train_loader, train=True, epoch_idx=epoch)
            
            # Validation
            val_metrics = None
            if val_loader:
                val_metrics = self._run_epoch(val_loader, train=False, epoch_idx=epoch)
            
            # Logging
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Checkpointing
            if val_metrics and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if epoch % 5 == 0:  # Save every 5 epochs
                    self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("Training completed!")

    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]):
        """
        Log metrics to wandb and console
        """
        log_dict = {
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/mse': train_metrics['mse'],
            'train/mae': train_metrics['mae'],
            'train/r2': train_metrics['r2'],
            'train/correlation': train_metrics['correlation'],
            'train/rmse': train_metrics['rmse']
        }
        
        if val_metrics:
            log_dict.update({
                'val/loss': val_metrics['loss'],
                'val/mse': val_metrics['mse'],
                'val/mae': val_metrics['mae'],
                'val/r2': val_metrics['r2'],
                'val/correlation': val_metrics['correlation'],
                'val/rmse': val_metrics['rmse']
            })
        
        if self.wandb_run:
            self.wandb_run.log(log_dict)
        
        # Console logging
        logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                   f"Train R²: {train_metrics['r2']:.4f}")
        if val_metrics:
            logger.info(f"          Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val R²: {val_metrics['r2']:.4f}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'prm_state_dict': self.prm.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.cfg)
        }
        
        if self.from_scratch:
            checkpoint['model_state_dict'] = self.model.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.prm.load_state_dict(checkpoint['prm_state_dict'])
        if self.from_scratch and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the trained model
        """
        self.prm.eval()
        if self.from_scratch:
            self.model.eval()
        
        with torch.no_grad():
            features = self._encode_features(input_ids)
            predictions = self.prm(features)
        
        return predictions 