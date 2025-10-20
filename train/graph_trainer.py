"""
Enhanced trainer for graph-based recommendation systems.
Handles training loops, validation, evaluation, and model checkpointing.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REC_DIR = os.path.normpath(os.path.join(_CURRENT_DIR, '..'))
sys.path.extend([_CURRENT_DIR, _REC_DIR])
from evalue.loss import bpr_loss, pairwise_hinge_loss
from evalue.metric import evaluate_all_at_k
from log.deep_learning_logger import get_logger


class GraphTrainer:
    """Enhanced trainer for graph-based recommendation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config,
        device: str = "cuda",
        logger=None
    ):
        """
        Initialize trainer.
        
        Args:
            model: The recommendation model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration object
            device: Device to use for training
            logger: Logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.logger = logger or get_logger("GraphTrainer")
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_metrics = []
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup directories
        self._setup_directories()
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.config.system.log_dir)
        
        # Early stopping
        self.patience_counter = 0
        self.best_model_state = None
    
    def _setup_optimizer(self):
        """Setup optimizer based on config."""
        if self.config.training.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.training.scheduler.lower() == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
                # verbose=True
            )
        elif self.config.training.scheduler.lower() == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.training.scheduler.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        else:
            self.scheduler = None
    
    def _setup_directories(self):
        """Setup directories for saving models and logs."""
        os.makedirs(self.config.system.save_dir, exist_ok=True)
        os.makedirs(self.config.system.log_dir, exist_ok=True)
        os.makedirs(self.config.system.results_dir, exist_ok=True)
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)
            # print(batch.keys())
            # print(input())
            self.optimizer.zero_grad()
            # try:
            # 获取必需的参数
            # Xs, embs, ks, alphas = None, None, None, None  # 实际训练时外部应适配
            outputs = self.model(batch)
            # print(outputs)
            # print(input())
            if 'loss' in outputs:
                loss = outputs["loss"]
            else:
                # 支持正负样本BPR损失
                user_emb = outputs.get('user_embeddings', outputs.get('embeddings'))
                pos_item_emb = outputs.get('pos_item_embeddings')
                neg_item_emb = outputs.get('neg_item_embeddings')
                if user_emb is not None and pos_item_emb is not None and neg_item_emb is not None:
                    # print(user_emb.shape, neg_item_emb.shape)
                    pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
                    neg_scores = torch.sum(user_emb.unsqueeze(1) * neg_item_emb, dim=-1)
                    loss = bpr_loss(pos_scores, neg_scores)
                else:
                    # Fallback: compute loss from batch data
                    loss = self._compute_loss_from_batch(batch, outputs)
            print(f"{batch_idx} train_loss",loss)
            loss.backward()
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                self.logger.log_batch_progress(batch_idx, num_batches, loss.item(), self.optimizer.param_groups[0]['lr'])
            # except Exception as e:
            # self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                   for k, v in value.items()}
            else:
                device_batch[key] = value
        return device_batch
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        all_scores = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                # print(batch.keys())
                # try:
                outputs = self.model(batch)
                
                
                # Get predictions
                if 'scores' in outputs:
                    scores = outputs["scores"]
                else:
                    # Compute scores from embeddings
                    user_ids = batch['user_ids']
                    item_ids = batch['item_ids']
                    embeddings = outputs.get('embeddings', outputs)
                    # print(outputs.keys())
                    # print(embeddings.shape, user_ids.shape, item_ids.shape)
                    scores = torch.sum(embeddings[user_ids] * embeddings[item_ids], dim=1)
                
                all_scores.append(scores.cpu())
                all_targets.append(batch['ratings'].cpu())
                
                # except Exception as e:
                # self.logger.error(f"Error in validation batch: {str(e)}")
                continue
    
        if not all_scores:
            return {}
        
        # Combine all scores and targets
        all_scores = torch.cat(all_scores, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Convert to binary targets (ratings > 0)
        binary_targets = (all_targets > 0).float()
        
        # Evaluate metrics
        metrics = {}
        for k in self.config.evaluation.k_values:
            # For evaluation, we need to reshape scores and targets properly
            # This is a simplified version - in practice, you'd need proper ranking evaluation
            batch_scores = all_scores.unsqueeze(0)  # (1, N)
            batch_targets = binary_targets.unsqueeze(0)  # (1, N)
            
            k_metrics = evaluate_all_at_k(batch_scores, batch_targets, k)
            for metric_name, value in k_metrics.items():
                metrics[f"{metric_name}@{k}"] = value.item()
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.log_model_info(
            self.config.model.model_name,
            sum(p.numel() for p in self.model.parameters()),
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        )
        start_time = time.time()
        for epoch in range(1, self.config.training.epochs + 1):
            self.current_epoch = epoch
            self.logger.log_epoch_start(epoch, self.config.training.epochs)
            train_loss = self.train_epoch()
            print(f"The {epoch} training average loss:",train_loss)
            # print(input())
            # Validate
            if epoch % self.config.training.eval_every == 0:
                val_metrics = self.validate()
                if val_metrics:
                    self.val_metrics.append(val_metrics)
                    self.logger.log_validation_results(val_metrics)
                    # Update best model
                    primary_metric = f"ndcg@{self.config.evaluation.k_values[0]}"
                    if primary_metric in val_metrics:
                        current_metric = val_metrics[primary_metric]
                        if current_metric > self.best_val_metric:
                            self.best_val_metric = current_metric
                            self.best_epoch = epoch
                            self.best_model_state = self.model.state_dict().copy()
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                    # Update scheduler
                    if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    elif self.scheduler:
                        self.scheduler.step()
                    # Log to tensorboard
                    self.writer.add_scalar('Loss/Train', train_loss, epoch)
                    for metric_name, value in val_metrics.items():
                        self.writer.add_scalar(f'Metrics/{metric_name}', value, epoch)
                    self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            # Log epoch end
            epoch_metrics = {'train_loss': train_loss}
            if epoch % self.config.training.eval_every == 0 and self.val_metrics:
                epoch_metrics.update(self.val_metrics[-1])
            self.logger.log_epoch_end(epoch, epoch_metrics)
            # Save checkpoint
            if epoch % self.config.training.save_every == 0:
                self._save_checkpoint(epoch)
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                self.logger.log_early_stopping(epoch, self.best_val_metric)
                break
        # Training completed
        total_time = time.time() - start_time
        best_metrics = {f"best_{k}": v for k, v in self.val_metrics[-1].items()} if self.val_metrics else {}
        self.logger.log_training_complete(f"{total_time/3600:.2f} hours", best_metrics)
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"Loaded best model from epoch {self.best_epoch}")
        # Final evaluation
        final_metrics = self.evaluate()
        return {
            'best_epoch': self.best_epoch,
            'best_val_metric': self.best_val_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'final_test_metrics': final_metrics,
            'training_time': total_time
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        all_scores = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = self._move_batch_to_device(batch)
                
                # try:
                outputs = self.model(batch)
                
                # Get predictions
                if 'scores' in outputs:
                    scores = outputs["scores"]
                else:
                    user_ids = batch['user_ids']
                    item_ids = batch['item_ids']
                    
                    embeddings = outputs.get('embeddings', outputs)
                    scores = torch.sum(embeddings[user_ids] * embeddings[item_ids], dim=1)
                
                all_scores.append(scores.cpu())
                all_targets.append(batch['ratings'].cpu())
                
                # except Exception as e:
                # self.logger.error(f"Error in test batch: {str(e)}")
                continue
        
        if not all_scores:
            return {}
        
        # Combine all scores and targets
        all_scores = torch.cat(all_scores, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Convert to binary targets
        binary_targets = (all_targets > 0).float()
        
        # Evaluate metrics
        metrics = {}
        for k in self.config.evaluation.k_values:
            batch_scores = all_scores.unsqueeze(0)
            batch_targets = binary_targets.unsqueeze(0)
            
            k_metrics = evaluate_all_at_k(batch_scores, batch_targets, k)
            for metric_name, value in k_metrics.items():
                metrics[f"test_{metric_name}@{k}"] = value.item()
        
        return metrics
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(
            self.config.system.save_dir,
            f"checkpoint_epoch_{epoch}.pth"
        )
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.log_checkpoint_save(epoch, checkpoint_path)
    
    def save_best_model(self, filepath: str):
        """Save the best model."""
        if self.best_model_state:
            torch.save({
                'model_state_dict': self.best_model_state,
                'config': self.config.to_dict(),
                'best_val_metric': self.best_val_metric,
                'best_epoch': self.best_epoch
            }, filepath)
            self.logger.info(f"Best model saved to {filepath}")
        else:
            self.logger.warning("No best model state found")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'writer'):
            self.writer.close()


if __name__ == "__main__":
    # Test trainer (requires model and data loaders)
    pass
