# src/training/trainer.py

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from typing import Dict, Tuple
from ..utils.metrics import calculate_metrics


class Trainer:
    """Training manager"""

    def __init__(
            self,
            model,
            criterion,
            train_loader,
            val_loader,
            config: Dict,
            device: str = 'cuda'
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
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        # Early stopping
        self.best_val_auc = 0
        self.patience = config['patience']
        self.patience_counter = 0

    def train_epoch(self) -> Tuple[float, Dict]:
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

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, batch)

            # Compute loss
            loss, loss_components = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
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
    def validate(self) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(images, batch)
            loss, _ = self.criterion(outputs, labels)

            total_loss += loss.item()
            predictions.append(outputs.cpu())
            targets.append(labels.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        metrics = calculate_metrics(predictions, targets)

        return total_loss / len(self.val_loader), metrics

    def save_checkpoint(
            self,
            epoch: int,
            metrics: Dict,
            is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save checkpoint
        path = f"{self.config['checkpoint_dir']}/checkpoint_epoch_{epoch}_auc_{metrics['mean_auc']:.4f}.pt"
        torch.save(checkpoint, path)

        # Save best model
        if is_best:
            best_path = f"{self.config['checkpoint_dir']}/best_model.pt"
            torch.save(checkpoint, best_path)

    def train(self, num_epochs: int):
        """Complete training procedure"""
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')

            # Training
            train_loss, train_metrics = self.train_epoch()

            # Validation
            val_loss, val_metrics = self.validate()

            # Scheduler step
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

            if self.patience_counter >= self.patience:
                print('Early stopping triggered')
                break

