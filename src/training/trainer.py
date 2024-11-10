# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
from tqdm import tqdm
from ..utils.metrics import calculate_metrics


class Trainer:
    def __init__(
            self,
            model,
            criterion,
            train_loader,
            val_loader,
            config,
            device='cuda'
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Early stopping
        self.best_val_auc = 0
        self.patience = config.patience
        self.patience_counter = 0

        # Initialize wandb
        wandb.init(
            project="chest-xray-classification",
            config=config,
            name=config.run_name
        )

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Get batch data
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            bbox_data = batch['bbox']

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, bbox_data)

            # Compute loss
            loss, loss_components = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions.append(outputs.detach().cpu())
            targets.append(labels.cpu())

            pbar.set_postfix({'loss': loss.item()})

        # Compute epoch metrics
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        metrics = calculate_metrics(predictions, targets)

        return total_loss / len(self.train_loader), metrics

    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            bbox_data = batch['bbox']

            outputs = self.model(images, bbox_data)
            loss, _ = self.criterion(outputs, labels)

            total_loss += loss.item()
            predictions.append(outputs.cpu())
            targets.append(labels.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        metrics = calculate_metrics(predictions, targets)

        return total_loss / len(self.val_loader), metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save latest checkpoint
        torch.save(
            checkpoint,
            f'{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt'
        )

        # Save best model
        if is_best:
            torch.save(
                checkpoint,
                f'{self.config.checkpoint_dir}/best_model.pt'
            )

    def train(self, num_epochs):
        """Complete training procedure"""
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')

            # Training
            train_loss, train_metrics = self.train_epoch()

            # Validation
            val_loss, val_metrics = self.validate()

            # Learning rate scheduling
            self.scheduler.step(val_metrics['mean_auc'])

            # Early stopping
            if val_metrics['mean_auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['mean_auc']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_metrics)

            # Log metrics
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })

            # Check early stopping
            if self.patience_counter >= self.patience:
                print('Early stopping triggered')
                break

