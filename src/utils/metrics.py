# src/utils/metrics.py

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from typing import Dict


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict:
    """
    Calculate comprehensive evaluation metrics

    Args:
        predictions: Model predictions [N, num_diseases]
        targets: Ground truth labels [N, num_diseases]
        threshold: Classification threshold
    """
    predictions = predictions.numpy()
    targets = targets.numpy()

    # Convert predictions to binary using threshold
    binary_preds = (predictions > threshold).astype(np.int32)

    # Initialize metric storage
    metrics = {}
    disease_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    # Calculate per-class metrics
    for i, disease in enumerate(disease_names):
        try:
            # ROC AUC
            if targets[:, i].sum() > 0:
                auc = roc_auc_score(targets[:, i], predictions[:, i])
            else:
                auc = 0.5
            metrics[f'{disease}_auc'] = auc

            # Average Precision
            ap = average_precision_score(targets[:, i], predictions[:, i])
            metrics[f'{disease}_ap'] = ap

            # F1 Score
            f1 = f1_score(targets[:, i], binary_preds[:, i])
            metrics[f'{disease}_f1'] = f1

            # Sensitivity (Recall)
            sensitivity = recall_score(targets[:, i], binary_preds[:, i])
            metrics[f'{disease}_sensitivity'] = sensitivity

            # Specificity
            tn, fp, fn, tp = confusion_matrix(targets[:, i], binary_preds[:, i]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'{disease}_specificity'] = specificity

            # Precision
            precision = precision_score(targets[:, i], binary_preds[:, i])
            metrics[f'{disease}_precision'] = precision

        except Exception as e:
            print(f"Error calculating metrics for {disease}: {str(e)}")
            metrics[f'{disease}_auc'] = 0.5
            metrics[f'{disease}_ap'] = 0.0
            metrics[f'{disease}_f1'] = 0.0
            metrics[f'{disease}_sensitivity'] = 0.0
            metrics[f'{disease}_specificity'] = 1.0
            metrics[f'{disease}_precision'] = 0.0

    # Calculate mean metrics
    metrics['mean_auc'] = np.mean([metrics[f'{d}_auc'] for d in disease_names])
    metrics['mean_ap'] = np.mean([metrics[f'{d}_ap'] for d in disease_names])
    metrics['mean_f1'] = np.mean([metrics[f'{d}_f1'] for d in disease_names])
    metrics['mean_sensitivity'] = np.mean([metrics[f'{d}_sensitivity'] for d in disease_names])
    metrics['mean_specificity'] = np.mean([metrics[f'{d}_specificity'] for d in disease_names])
    metrics['mean_precision'] = np.mean([metrics[f'{d}_precision'] for d in disease_names])

    # Exact match ratio (all predictions correct)
    metrics['exact_match'] = (binary_preds == targets).all(axis=1).mean()

    return metrics

