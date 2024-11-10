# src/utils/metrics.py

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)


def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate various classification metrics

    Args:
        predictions: Model predictions (N, num_diseases)
        targets: Ground truth labels (N, num_diseases)
        threshold: Classification threshold
    """
    predictions = predictions.numpy()
    targets = targets.numpy()

    # Binary predictions
    binary_preds = (predictions > threshold).astype(np.int32)

    # Per-class metrics
    num_classes = predictions.shape[1]
    aucs = []
    aps = []
    f1s = []
    precisions = []
    recalls = []

    for i in range(num_classes):
        # Skip if no positive samples
        if targets[:, i].sum() == 0:
            continue

        # ROC AUC
        try:
            auc = roc_auc_score(targets[:, i], predictions[:, i])
        except:
            auc = 0.5
        aucs.append(auc)

        # Average Precision
        ap = average_precision_score(targets[:, i], predictions[:, i])
        aps.append(ap)

        # F1 Score
        f1 = f1_score(targets[:, i], binary_preds[:, i])
        f1s.append(f1)

        # Precision & Recall
        precision = precision_score(targets[:, i], binary_preds[:, i])
        recall = recall_score(targets[:, i], binary_preds[:, i])
        precisions.append(precision)
        recalls.append(recall)

    # Compute mean metrics
    metrics = {
        'mean_auc': np.mean(aucs),
        'mean_ap': np.mean(aps),
        'mean_f1': np.mean(f1s),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),

        # Per-class metrics
        'class_auc': aucs,
        'class_ap': aps,
        'class_f1': f1s,
        'class_precision': precisions,
        'class_recall': recalls,

        # Exact match ratio (all predictions correct)
        'exact_match': (binary_preds == targets).all(axis=1).mean()
    }

    return metrics

