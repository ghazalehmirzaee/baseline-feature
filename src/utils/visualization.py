# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict
import wandb
from sklearn.metrics import roc_curve, precision_recall_curve
import cv2


class Visualizer:
    """Visualization utilities for model analysis and debugging."""

    def __init__(self, save_dir: Path, class_names: List[str]):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names

        # Set up style
        plt.style.use('seaborn')
        self.colors = sns.color_palette('husl', n_colors=len(class_names))

    def plot_graph_weights(self,
                           weights: torch.Tensor,
                           epoch: int,
                           batch: Optional[int] = None):
        """Plot disease relationship graph weights."""
        plt.figure(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(
            weights.cpu().numpy(),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f'
        )

        plt.title('Disease Relationship Graph Weights')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Save plot
        name = f'graph_weights_epoch_{epoch}'
        if batch is not None:
            name += f'_batch_{batch}'

        plt.tight_layout()
        plt.savefig(self.save_dir / f'{name}.png', bbox_inches='tight', dpi=300)

        # Log to wandb
        wandb.log({
            'graph_weights': wandb.Image(plt),
            'epoch': epoch
        })

        plt.close()

    def plot_attention_maps(self,
                            image: torch.Tensor,
                            attention_weights: torch.Tensor,
                            save_path: str):
        """Visualize attention weights on the image."""
        # Convert image to numpy
        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        # Resize attention weights to image size
        attn = F.interpolate(
            attention_weights.unsqueeze(0).unsqueeze(0),
            size=img.shape[:2],
            mode='bilinear',
            align_corners=False
        )
        attn = attn.squeeze().cpu().numpy()

        # Create heatmap
        heatmap = cv2.applyColorMap(
            (attn * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Blend with original image
        output = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

        # Save visualization
        cv2.imwrite(save_path, output)

    def plot_metrics(self,
                     metrics: Dict[str, float],
                     epoch: int,
                     phase: str = 'val'):
        """Plot training/validation metrics."""
        metrics_by_class = {}
        mean_metrics = {}

        # Separate class-specific and mean metrics
        for k, v in metrics.items():
            if k.startswith('mean_'):
                mean_metrics[k] = v
            else:
                class_name = k.split('_')[0]
                metric_name = '_'.join(k.split('_')[1:])
                if class_name not in metrics_by_class:
                    metrics_by_class[class_name] = {}
                metrics_by_class[class_name][metric_name] = v

        # Plot class-specific metrics
        for metric in ['auc', 'ap', 'f1']:
            plt.figure(figsize=(12, 6))
            values = [
                metrics_by_class[c][metric]
                for c in self.class_names
            ]

            plt.bar(self.class_names, values, color=self.colors)
            plt.title(f'{phase.capitalize()} {metric.upper()} by Class')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(metric.upper())

            # Add mean line
            mean_value = mean_metrics[f'mean_{metric}']
            plt.axhline(y=mean_value, color='r', linestyle='--',
                        label=f'Mean: {mean_value:.3f}')
            plt.legend()

            plt.tight_layout()
            plt.savefig(
                self.save_dir / f'{phase}_{metric}_epoch_{epoch}.png',
                bbox_inches='tight',
                dpi=300
            )
            plt.close()

    def plot_roc_curves(self, predictions: torch.Tensor, targets: torch.Tensor, epoch: int):
        """Plot ROC curves for each class."""
        plt.figure(figsize=(12, 8))

        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        for i, (name, color) in enumerate(zip(self.class_names, self.colors)):
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
            plt.plot(fpr, tpr, color=color, label=f'{name} (AUC = {auc(fpr, tpr):.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Class')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save plot
        plt.tight_layout()
        plt.savefig(self.save_dir / f'roc_curves_epoch_{epoch}.png', bbox_inches='tight', dpi=300)

        # Log to wandb
        wandb.log({
            'roc_curves': wandb.Image(plt),
            'epoch': epoch
        })

        plt.close()

    def plot_precision_recall_curves(self, predictions: torch.Tensor, targets: torch.Tensor, epoch: int):
        """Plot Precision-Recall curves for each class."""
        plt.figure(figsize=(12, 8))

        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        for i, (name, color) in enumerate(zip(self.class_names, self.colors)):
            precision, recall, _ = precision_recall_curve(targets[:, i], predictions[:, i])
            plt.plot(recall, precision, color=color, label=f'{name}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves by Class')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save plot
        plt.tight_layout()
        plt.savefig(self.save_dir / f'precision_recall_curves_epoch_{epoch}.png', bbox_inches='tight', dpi=300)

        # Log to wandb
        wandb.log({
            'precision_recall_curves': wandb.Image(plt),
            'epoch': epoch
        })

        plt.close()

