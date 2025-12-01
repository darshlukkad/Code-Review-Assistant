"""
Training loop with TensorBoard logging and early stopping.

Implements training pipeline with:
- TensorBoard logging
- Early stopping
- Model checkpointing
- Learning rate scheduling
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Optional, Dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeReviewTrainer:
    """
    Trainer class for code review models.
    
    Handles training loop, validation, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Training on device: {self.device}")
        self.model.to(self.device)
        
        # Optimizer setup
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function (BCEWithLogitsLoss is already in model)
        # We could override here if needed
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=config.get('tensorboard_dir', 'logs/tensorboard'))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Trainer initialized successfully")
    
    def _create_optimizer(self):
        """Create AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(
                self.config.get('adam_beta1', 0.9),
                self.config.get('adam_beta2', 0.999)
            ),
            eps=self.config.get('adam_epsilon', 1e-8)
        )
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with linear warmup."""
        num_training_steps = len(self.train_loader) * self.config.get('num_epochs', 15)
        warmup_steps = self.config.get('warmup_steps', 500)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log to TensorBoard
            if self.global_step % self.config.get('logging_steps', 100) == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate on validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filename: str = None):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename (default: checkpoint_epoch_{epoch}.pt)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """
        Main training loop.
        
        Trains for specified number of epochs with early stopping.
        """
        num_epochs = self.config.get('num_epochs', 15)
        early_stopping_patience = self.config.get('early_stopping_patience', 3)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train for one epoch
            train_loss = self.train_epoch()
            logger.info(f"Epoch {self.current_epoch} - Train loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            logger.info(f"Epoch {self.current_epoch} - Val loss: {val_loss:.4f}")
            
            # Log to TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_loss, self.current_epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, self.current_epoch)
            
            # Check if best model
            if val_loss < self.best_val_loss:
                logger.info(f"New best model! Val loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt')
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement} epochs")
            
            # Save periodic checkpoint
            if (self.current_epoch) % self.config.get('save_steps', 5) == 0:
                self.save_checkpoint()
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
                break
        
        logger.info("Training complete!")
        self.writer.close()


if __name__ == "__main__":
    logger.info("Trainer module - import and use with actual data loaders")
