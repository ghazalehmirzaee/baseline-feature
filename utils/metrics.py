# utils/metrics.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix


def compute_metrics(predictions, targets, threshold=0.5):
    """Compute comprehensive evaluation metrics."""
    predictions = torch.sigmoid(predictions)
    predictions_binary = (predictions > threshold).float()

    # Convert to numpy for sklearn metrics
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    predictions_binary_np = predictions_binary.numpy()

    metrics = {}
    disease_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    # Compute per-class metrics
    for i, disease in enumerate(disease_names):
        # AUC-ROC
        if len(np.unique(targets_np[:, i])) > 1:
            metrics[f'{disease}_auc'] = roc_auc_score(targets_np[:, i], predictions_np[:, i])
        else:
            metrics[f'{disease}_auc'] = 0.0

        # Average Precision
        metrics[f'{disease}_ap'] = average_precision_score(
            targets_np[:, i], predictions_np[:, i]
        )

        # F1 Score
        metrics[f'{disease}_f1'] = f1_score(
            targets_np[:, i], predictions_binary_np[:, i], zero_division=0
        )

        # Sensitivity (Recall)
        metrics[f'{disease}_sensitivity'] = recall_score(
            targets_np[:, i], predictions_binary_np[:, i], zero_division=0
        )

        # Specificity
        tn, fp, fn, tp = confusion_matrix(
            targets_np[:, i], predictions_binary_np[:, i], labels = [0,
                                                                                                             1]).ravel()
        metrics[f'{disease}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Precision
        metrics[f'{disease}_precision'] = precision_score(
            targets_np[:, i], predictions_binary_np[:, i], zero_division=0
        )

        # Compute mean metrics
        mean_metrics = {
            'mean_auc': np.mean([metrics[f'{d}_auc'] for d in disease_names]),
            'mean_ap': np.mean([metrics[f'{d}_ap'] for d in disease_names]),
            'mean_f1': np.mean([metrics[f'{d}_f1'] for d in disease_names]),
            'mean_sensitivity': np.mean([metrics[f'{d}_sensitivity'] for d in disease_names]),
            'mean_specificity': np.mean([metrics[f'{d}_specificity'] for d in disease_names]),
            'mean_precision': np.mean([metrics[f'{d}_precision'] for d in disease_names])
        }

        # Add mean metrics to output
        metrics.update(mean_metrics)

        # Compute exact match ratio
        exact_match = torch.all(predictions_binary == targets, dim=1).float().mean().item()
        metrics['exact_match'] = exact_match

        return metrics

