# src/utils/metrics.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict


class MetricsCalculator:
    """Calculate and track various performance metrics."""

    def __init__(self, num_classes: int, class_names: list):
        self.num_classes = num_classes
        self.class_names = class_names

    def compute_all_metrics(self, predictions: torch.Tensor,
                            targets: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics for multi-label classification."""
        predictions = torch.sigmoid(predictions)
        metrics = {}

        # Per-class metrics
        for i, name in enumerate(self.class_names):
            pred = predictions[:, i].numpy()
            target = targets[:, i].numpy()

            # AUC-ROC
            if len(np.unique(target)) > 1:
                metrics[f'{name}_auc'] = roc_auc_score(target, pred)
            else:
                metrics[f'{name}_auc'] = 0.0

            # Average Precision
            metrics[f'{name}_ap'] = average_precision_score(target, pred)

        # Compute mean metrics
        metrics['mean_auc'] = np.mean([
            metrics[f'{name}_auc'] for name in self.class_names
        ])
        metrics['mean_ap'] = np.mean([
            metrics[f'{name}_ap'] for name in self.class_names
        ])

        return metrics

