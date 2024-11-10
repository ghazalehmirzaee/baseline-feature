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
from data.dataset_utils import DatasetUtils
from models.graph_modules.feature_utils import FeatureExtractor
from utils.visualization import GraphVisualizer, GraphMetrics


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

        # Initialize utilities
        self.dataset_utils = DatasetUtils()
        self.disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        self.graph_metrics = GraphMetrics()

        self._setup_model()
        self._setup_training()

    def _setup_model(self):
        try:
            checkpoint_path = self.config['model']['baseline_checkpoint']
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                self.logger.info("Found model_state_dict in checkpoint")
                self.baseline_model = checkpoint['model_state_dict']
            else:
                raise ValueError("Checkpoint does not contain model_state_dict")

            # Initialize model
            self.model = FeatureGraphModel(self.config, self.baseline_model).to(self.device)

            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor(
                self.model.backbone,
                self.model.feature_proj,
                self.config['model']['graph_hidden_dim']
            )

            # Initialize graph visualizer
            self.graph_visualizer = GraphVisualizer(self.disease_names)

            # Setup loss function
            self.criterion = MultiComponentLoss(self.config['training']['loss_weights'])

            # Calculate positive weights for WBCE
            pos_counts = torch.tensor(self.config['dataset']['positive_counts'])
            neg_counts = torch.tensor(self.config['dataset']['negative_counts'])
            self.pos_weights = (neg_counts / pos_counts).to(self.device)

            if 'metrics' in checkpoint:
                self.logger.info("Previous model metrics:")
                metrics = checkpoint['metrics']
                self.logger.info(f"Mean AUC: {float(metrics['mean_auc']):.4f}")

        except Exception as e:
            self.logger.error(f"Error in model setup: {str(e)}")
            raise RuntimeError(f"Model setup failed: {str(e)}")

    def _setup_training(self):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=int(self.config['training']['warmup_epochs']),
            T_mult=2
        )

        self.best_metrics = {
            'mean_auc': 0.0,
            'epoch': 0,
            'metrics': None
        }

        self.patience = int(self.config['training'].get('patience', 10))
        self.patience_counter = 0

        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization directory
        self.viz_dir = checkpoint_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)

    def process_batch(self, images, targets, bboxes):
        """Process a single batch of data."""
        images = images.to(self.device)
        targets = targets.to(self.device)

        # Process bounding boxes
        if bboxes is not None:
            processed_bboxes = self.dataset_utils.process_bbox_annotations(bboxes)

            # Extract features and build graph
            features_dict, areas_dict = self.feature_extractor.extract_features(
                images, processed_bboxes
            )

            # Create disease pairs
            disease_pairs = self.dataset_utils.create_disease_pairs(
                processed_bboxes['labels'],
                processed_bboxes['boxes']
            )
        else:
            processed_bboxes = None
            features_dict = {}
            areas_dict = {}
            disease_pairs = {}

        return images, targets, processed_bboxes, features_dict, areas_dict, disease_pairs

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_losses = []

        with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (images, targets, bboxes) in enumerate(pbar):
                # Process batch
                images, targets, processed_bboxes, features_dict, areas_dict, disease_pairs = \
                    self.process_batch(images, targets, bboxes)

                # Forward pass
                predictions = self.model(
                    images=images,
                    bboxes=processed_bboxes,
                    features_dict=features_dict,
                    areas_dict=areas_dict,
                    disease_pairs=disease_pairs
                )

                # Compute loss
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

                # Visualize graph weights periodically
                if batch_idx % 100 == 0:
                    graph_weights = self.model.get_graph_weights()
                    self.graph_visualizer.plot_graph_weights(
                        graph_weights,
                        str(self.viz_dir / f'graph_weights_epoch{epoch}_batch{batch_idx}.png')
                    )

                    # Log graph statistics
                    graph_stats = self.graph_metrics.compute_graph_statistics(graph_weights)
                    wandb.log({f'graph_{k}': v for k, v in graph_stats.items()})

        return sum(epoch_losses) / len(epoch_losses)

    def validate(self, val_loader, epoch):
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets, bboxes in tqdm(val_loader, desc='Validation'):
                # Process batch
                images, targets, processed_bboxes, features_dict, areas_dict, disease_pairs = \
                    self.process_batch(images, targets, bboxes)

                # Forward pass
                predictions = self.model(
                    images=images,
                    bboxes=processed_bboxes,
                    features_dict=features_dict,
                    areas_dict=areas_dict,
                    disease_pairs=disease_pairs
                )

                loss, _ = self.criterion(predictions, targets, self.pos_weights)

                val_losses.append(loss.item())
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        # Compute metrics
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        metrics = compute_metrics(predictions, targets)

        # Evaluate graph quality
        graph_weights = self.model.get_graph_weights()
        graph_quality = self.graph_metrics.evaluate_graph_quality(graph_weights, targets)

        # Log metrics
        wandb.log({
            'val_loss': sum(val_losses) / len(val_losses),
            'val_mean_auc': float(metrics['mean_auc']),
            'val_mean_ap': float(metrics['mean_ap']),
            'val_mean_f1': float(metrics['mean_f1']),
            'epoch': epoch,
            **{f'graph_quality_{k}': v for k, v in graph_quality.items()}
        })

        # Save visualization
        self.graph_visualizer.plot_graph_weights(
            graph_weights,
            str(self.viz_dir / f'graph_weights_epoch{epoch}_val.png')
        )

        # Update best metrics
        current_auc = float(metrics['mean_auc'])
        if current_auc > self.best_metrics['mean_auc']:
            self.best_metrics = {
                'mean_auc': current_auc,
                'epoch': epoch,
                'metrics': metrics
            }

            # Save checkpoint
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
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            self.scheduler.step()

            self.logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"Val AUC = {val_metrics['mean_auc']:.4f}"
            )

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

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    # Update paths and create directories
    config['dataset'].update({
        'train_csv': os.path.join(args.data_dir, 'labels/train_list.txt'),
        'val_csv': os.path.join(args.data_dir, 'labels/val_list.txt'),
        'test_csv': os.path.join(args.data_dir, 'labels/test_list.txt'),
        'image_dir': os.path.join(args.data_dir, 'images'),
        'bb_annotations': os.path.join(args.data_dir, 'labels/BBox_List_2017.csv')
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config['training']['checkpoint_dir'] = str(output_dir / 'checkpoints')
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    # Save config
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    try:
        train_loader, val_loader, test_loader = get_data_loaders(config)
        trainer = Trainer(config)
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"Error during training: {e}")
        return


if __name__ == '__main__':
    main()

    