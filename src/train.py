# src/train.py
import torch
import torch.nn as nn
import wandb
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Tuple

from models.feature_graph import FeatureGraphModel
from data.dataset import get_data_loaders
from utils.metrics import MetricsCalculator
from utils.checkpointing import CheckpointManager
from utils.visualization import Visualizer
from utils.logger import setup_logger


class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger("feature_graph_training")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self._setup_wandb()
        self._setup_model()
        self._setup_training()
        self._setup_metrics()

    def _setup_wandb(self):
        """Initialize Weights & Biases."""
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            config=self.config,
            tags=self.config['wandb']['tags']
        )

    def _setup_model(self):
        """Initialize model and load baseline weights."""
        try:
            # Load baseline checkpoint
            checkpoint_path = self.config['model']['baseline_checkpoint']
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if 'model_state_dict' not in checkpoint:
                raise ValueError("Invalid checkpoint format")

            # Initialize model
            self.model = FeatureGraphModel(
                self.config,
                checkpoint['model_state_dict']
            ).to(self.device)

            self.logger.info(f"Loaded baseline weights from {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            raise

    def _setup_training(self):
        """Initialize training components."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['training']['warmup_epochs'],
            eta_min=self.config['training']['scheduler']['eta_min']
        )

        # Loss function
        self.criterion = MultiLabelLoss(self.config['training']['loss_weights'])

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=Path(self.config['training']['checkpoint_dir']),
            config=self.config
        )

        # Initialize best metrics
        self.best_metrics = {
            'mean_auc': 0.0,
            'epoch': 0
        }

        # Early stopping
        self.patience = self.config['training']['patience']
        self.patience_counter = 0

    def _setup_metrics(self):
        """Initialize metrics calculation and visualization."""
        self.metrics_calculator = MetricsCalculator(
            num_classes=self.config['dataset']['num_classes'],
            class_names=self.config['dataset']['classes']
        )

        self.visualizer = Visualizer(
            save_dir=Path(self.config['training']['checkpoint_dir']) / 'visualizations',
            class_names=self.config['dataset']['classes']
        )

    def train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (images, labels, bboxes) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.model(images, bboxes, labels)
                loss = self.criterion(predictions, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
                })

                # Log to wandb
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

                # Visualize graph periodically
                if batch_idx % 100 == 0:
                    graph_weights = self.model.get_graph_weights()
                    self.visualizer.plot_graph_weights(
                        graph_weights,
                        epoch,
                        batch_idx
                    )

        return epoch_loss / len(train_loader)

    def validate(self, val_loader, epoch: int) -> Dict:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, bboxes in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.model(images, bboxes)
                loss = self.criterion(predictions, labels)

                val_loss += loss.item()
                all_preds.append(predictions.cpu())
                all_labels.append(labels.cpu())

        # Compute metrics
        predictions = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        metrics = self.metrics_calculator.compute_all_metrics(
            predictions, labels
        )

        # Log validation metrics
        wandb.log({
            'val_loss': val_loss / len(val_loader),
            'val_mean_auc': metrics['mean_auc'],
            'epoch': epoch,
            **{f'val_{k}': v for k, v in metrics.items() if k != 'mean_auc'}
        })

        # Update best metrics
        if metrics['mean_auc'] > self.best_metrics['mean_auc']:
            self.best_metrics = {
                'mean_auc': metrics['mean_auc'],
                'epoch': epoch,
                'metrics': metrics
            }

            # Save best model
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics,
                is_best=True
            )

            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return metrics

    def train(self, train_loader, val_loader):
        """Main training loop."""
        self.logger.info("Starting training...")

        for epoch in range(self.config['training']['num_epochs']):
            # Train epoch
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Scheduler step
            self.scheduler.step()

            # Logging
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Val AUC = {val_metrics['mean_auc']:.4f}"
            )

            # Save checkpoint
            if (epoch + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=False
                )

            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch} epochs. "
                    f"Best mean AUC: {self.best_metrics['mean_auc']:.4f} "
                    f"at epoch {self.best_metrics['epoch']}"
                )
                break

        return self.best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-dir', required=True, help='Path to data directory')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Update paths
    config['dataset'].update({
        'data_dir': args.data_dir,
        'train_csv': f"{args.data_dir}/train_list.txt",
        'val_csv': f"{args.data_dir}/val_list.txt",
        'test_csv': f"{args.data_dir}/test_list.txt",
    })

    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Initialize trainer
    trainer = Trainer(config)

    # Train
    best_metrics = trainer.train(train_loader, val_loader)

    # Final evaluation
    trainer.logger.info("Training completed. Best metrics:")
    for k, v in best_metrics['metrics'].items():
        trainer.logger.info(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()

