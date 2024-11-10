# scripts/train.py
import argparse
import os
import yaml
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from pathlib import Path

from models.feature_graph import FeatureGraphModel
from models.loss import MultiComponentLoss
from utils.metrics import compute_metrics
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.logger import setup_logging
from data.datasets import get_data_loaders


# scripts/train.py (relevant section)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging("feature_graph_training")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize wandb
        wandb.init(
            project="chest-xray-classification",
            entity="mirzaeeghazal",
            config=self.config
        )

        self._setup_model()
        self._setup_training()

    def _load_checkpoint(self, checkpoint_path):
        """Helper function to load checkpoint with proper error handling."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Check checkpoint structure
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    return checkpoint['model']
                elif 'state_dict' in checkpoint:
                    return checkpoint['state_dict']
                else:
                    # If checkpoint is a dict but doesn't have expected keys,
                    # try using it directly
                    self.logger.warning("Checkpoint structure is non-standard. Attempting to use directly.")
                    return checkpoint
            else:
                # If checkpoint is not a dict, try using it directly
                self.logger.warning("Checkpoint is not a dictionary. Attempting to use directly.")
                return checkpoint

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")

    def _setup_model(self):
        try:
            checkpoint_path = self.config['model']['baseline_checkpoint']
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # We want to use the model_state_dict for our baseline model
            if 'model_state_dict' in checkpoint:
                self.logger.info("Found model_state_dict in checkpoint")
                self.baseline_model = checkpoint['model_state_dict']
            else:
                raise ValueError("Checkpoint does not contain model_state_dict")

            # Store best validation AUC for reference
            self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
            self.logger.info(f"Previous best validation AUC: {self.best_val_auc}")

            # Initialize feature graph model with the state dict
            self.logger.info("Initializing feature graph model...")
            self.model = FeatureGraphModel(self.config, self.baseline_model).to(self.device)

            # Loss and metrics
            self.logger.info("Setting up loss function...")
            self.criterion = MultiComponentLoss(self.config['training']['loss_weights'])

            # Calculate positive weights for WBCE
            pos_counts = torch.tensor(self.config['dataset']['positive_counts'])
            neg_counts = torch.tensor(self.config['dataset']['negative_counts'])
            self.pos_weights = (neg_counts / pos_counts).to(self.device)

            # Log some information about the loaded model
            if 'metrics' in checkpoint:
                self.logger.info("Previous model metrics:")
                self.logger.info(f"Mean AUC: {checkpoint['metrics']['mean_auc']:.4f}")
                self.logger.info(f"Mean AP: {checkpoint['metrics']['mean_ap']:.4f}")
                self.logger.info(f"Mean F1: {checkpoint['metrics']['mean_f1']:.4f}")

        except Exception as e:
            self.logger.error(f"Error in model setup: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            raise RuntimeError(f"Model setup failed: {str(e)}")


    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_losses = []

        with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (images, targets, bboxes) in enumerate(pbar):
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                predictions = self.model(images, bboxes)
                loss, loss_components = self.criterion(predictions, targets, self.pos_weights)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update progress bar
                epoch_losses.append(loss.item())
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{sum(epoch_losses) / len(epoch_losses):.4f}'
                })

                # Log to wandb
                wandb.log({
                    'train_loss': loss.item(),
                    'train_wbce_loss': loss_components['wbce'].item(),
                    'train_focal_loss': loss_components['focal'].item(),
                    'train_asl_loss': loss_components['asl'].item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

        return sum(epoch_losses) / len(epoch_losses)

    def validate(self, val_loader, epoch):
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets, bboxes in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(images, bboxes)
                loss, _ = self.criterion(predictions, targets, self.pos_weights)

                val_losses.append(loss.item())
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        # Compute metrics
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        metrics = compute_metrics(predictions, targets)

        # Log metrics
        wandb.log({
            'val_loss': sum(val_losses) / len(val_losses),
            'val_mean_auc': metrics['mean_auc'],
            'val_mean_ap': metrics['mean_ap'],
            'val_mean_f1': metrics['mean_f1'],
            'epoch': epoch
        })

        # Update best metrics and save checkpoint
        if metrics['mean_auc'] > self.best_metrics['mean_auc']:
            self.best_metrics = {
                'mean_auc': metrics['mean_auc'],
                'epoch': epoch,
                'metrics': metrics
            }

            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                metrics,
                Path(self.config['training']['checkpoint_dir']) / f'best_model.pt'
            )

            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return metrics

    def train(self, train_loader, val_loader):
        self.logger.info("Starting training...")

        for epoch in range(self.config['training']['num_epochs']):
            # Train epoch
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            self.logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"Val AUC = {val_metrics['mean_auc']:.4f}"
            )

            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch} epochs. "
                    f"Best mean AUC: {self.best_metrics['mean_auc']:.4f} "
                    f"at epoch {self.best_metrics['epoch']}"
                )
                break

        # Save final model
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            val_metrics,
            Path(self.config['training']['checkpoint_dir']) / f'final_model.pt'
        )

        return self.best_metrics


def main():
    parser = argparse.ArgumentParser(description='Train the multi-stage chest X-ray classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()

    # Load and validate configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    # Initialize default configuration if sections don't exist
    if 'dataset' not in config:
        config['dataset'] = {}
    if 'model' not in config:
        config['model'] = {}
    if 'training' not in config:
        config['training'] = {}

    # Update dataset paths
    config['dataset'].update({
        'train_csv': os.path.join(args.data_dir, 'labels/train_list.txt'),
        'val_csv': os.path.join(args.data_dir, 'labels/val_list.txt'),
        'test_csv': os.path.join(args.data_dir, 'labels/test_list.txt'),
        'image_dir': os.path.join(args.data_dir, 'images'),
        'bb_annotations': os.path.join(args.data_dir, 'labels/BBox_List_2017.csv')
    })

    # Add default values for positive and negative counts if not present
    if 'positive_counts' not in config['dataset']:
        config['dataset']['positive_counts'] = [1000] * 14  # Default values
    if 'negative_counts' not in config['dataset']:
        config['dataset']['negative_counts'] = [10000] * 14  # Default values

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add checkpoint directory to config
    config['training']['checkpoint_dir'] = str(output_dir / 'checkpoints')
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    # Save the updated config
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create data loaders
    try:
        train_loader, val_loader, test_loader = get_data_loaders(config)
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return

    # Initialize trainer and start training
    try:
        trainer = Trainer(config)  # Pass config directly, not config path
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"Error during training: {e}")
        return

if __name__ == '__main__':
    main()
    