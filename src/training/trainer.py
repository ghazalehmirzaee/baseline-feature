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
    def __init__(self, model, criterion, train_loader, val_loader, config, device='cuda'):
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
        self.patience = config['patience']
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move everything to device
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Prepare batch data dictionary
            batch_data = {
                'images': images,
                'labels': labels,
                'bbox': batch['bbox'],
                'image_path': batch['image_path'],
                'image_name': batch['image_name']
            }

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, batch_data)

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
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            batch_data = {
                'images': images,
                'labels': labels,
                'bbox': batch['bbox'],
                'image_path': batch['image_path'],
                'image_name': batch['image_name']
            }

            outputs = self.model(images, batch_data)
            loss, _ = self.criterion(outputs, labels)

            total_loss += loss.item()
            predictions.append(outputs.cpu())
            targets.append(labels.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        metrics = calculate_metrics(predictions, targets)

        return total_loss / len(self.val_loader), metrics

    def train(self, num_epochs):
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

            # Log metrics
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'lr': self.optimizer.param_groups[0]['lr']
            })

            if self.patience_counter >= self.patience:
                print('Early stopping triggered')
                break

