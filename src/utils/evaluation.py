# src/utils/evaluation.py
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, roc_auc_score,
    average_precision_score
)
from scipy import stats


class ResultsManager:
    """Comprehensive evaluation results management."""

    def __init__(self, config: Dict, save_dir: Path):
        self.config = config
        self.save_dir = save_dir
        self.class_names = config['dataset']['classes']
        self.best_metrics = {
            'mean_auc': 0.0,
            'epoch': 0
        }

    def compute_metrics_with_ci(self,
                                predictions: torch.Tensor,
                                targets: torch.Tensor,
                                alpha: float = 0.05) -> Dict:
        """
        Compute all metrics with confidence intervals using bootstrapping.

        Args:
            predictions: Model predictions [N, num_classes]
            targets: Ground truth labels [N, num_classes]
            alpha: Significance level for confidence intervals

        Returns:
            Dictionary containing all metrics and their confidence intervals
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        n_samples = len(predictions)
        n_bootstrap = 1000

        metrics = {
            'metrics': {},
            'confidence_intervals': {},
            'error_analysis': {
                'per_disease_errors': {
                    'Disease': {},
                    'Error Rate': {}
                },
                'error_cooccurrence': []
            }
        }

        # Compute basic metrics for each class
        for i, class_name in enumerate(self.class_names):
            class_pred = predictions[:, i]
            class_target = targets[:, i]

            # Basic metrics
            metrics['metrics'].update({
                f'{class_name}_auc': roc_auc_score(class_target, class_pred),
                f'{class_name}_ap': average_precision_score(class_target, class_pred),
                f'{class_name}_f1': self._compute_f1(class_target, class_pred > 0.5),
                f'{class_name}_sensitivity': self._compute_sensitivity(class_target, class_pred > 0.5),
                f'{class_name}_specificity': self._compute_specificity(class_target, class_pred > 0.5),
                f'{class_name}_precision': self._compute_precision(class_target, class_pred > 0.5)
            })

            # Bootstrap confidence intervals
            bootstrap_metrics = self._bootstrap_metrics(
                class_pred, class_target, n_bootstrap
            )

            for metric_name in ['auc', 'ap', 'f1', 'sensitivity', 'specificity', 'precision']:
                ci_key = f'{class_name}_{metric_name}_ci'
                metrics['confidence_intervals'][ci_key] = self._compute_ci(
                    bootstrap_metrics[metric_name], alpha
                )

            # Error analysis
            metrics['error_analysis']['per_disease_errors']['Disease'][str(i)] = class_name
            metrics['error_analysis']['per_disease_errors']['Error Rate'][str(i)] = float(
                np.mean(np.abs(class_pred - class_target))
            )

        # Compute mean metrics
        mean_metrics = ['auc', 'ap', 'f1', 'sensitivity', 'specificity', 'precision']
        for metric in mean_metrics:
            values = [
                metrics['metrics'][f'{c}_{metric}']
                for c in self.class_names
            ]
            metrics['metrics'][f'mean_{metric}'] = float(np.mean(values))

            # Confidence intervals for mean metrics
            bootstrap_means = self._bootstrap_mean_metric(
                predictions, targets, metric, n_bootstrap
            )
            metrics['confidence_intervals'][f'mean_{metric}_ci'] = self._compute_ci(
                bootstrap_means, alpha
            )

        # Error co-occurrence matrix
        pred_binary = predictions > 0.5
        errors = np.abs(pred_binary - targets)
        cooccurrence = np.zeros((len(self.class_names), len(self.class_names)))

        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                cooccurrence[i, j] = float(
                    np.mean(errors[:, i] * errors[:, j])
                )

        metrics['error_analysis']['error_cooccurrence'] = cooccurrence.tolist()

        # Compute exact match
        exact_match = np.mean(np.all(pred_binary == targets, axis=1))
        metrics['metrics']['exact_match'] = float(exact_match)

        # Statistical significance tests
        metrics['statistical_tests'] = self._perform_significance_tests(
            predictions, targets
        )

        return metrics

    def _bootstrap_metrics(self,
                           predictions: np.ndarray,
                           targets: np.ndarray,
                           n_bootstrap: int) -> Dict[str, List[float]]:
        """Compute bootstrap samples for all metrics."""
        n_samples = len(predictions)
        bootstrap_metrics = {
            'auc': [], 'ap': [], 'f1': [],
            'sensitivity': [], 'specificity': [], 'precision': []
        }

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_pred = predictions[indices]
            boot_target = targets[indices]

            bootstrap_metrics['auc'].append(
                roc_auc_score(boot_target, boot_pred)
            )
            bootstrap_metrics['ap'].append(
                average_precision_score(boot_target, boot_pred)
            )

            boot_pred_binary = boot_pred > 0.5
            bootstrap_metrics['f1'].append(
                self._compute_f1(boot_target, boot_pred_binary)
            )
            bootstrap_metrics['sensitivity'].append(
                self._compute_sensitivity(boot_target, boot_pred_binary)
            )
            bootstrap_metrics['specificity'].append(
                self._compute_specificity(boot_target, boot_pred_binary)
            )
            bootstrap_metrics['precision'].append(
                self._compute_precision(boot_target, boot_pred_binary)
            )

        return bootstrap_metrics

    def _compute_ci(self,
                    values: List[float],
                    alpha: float) -> List[float]:
        """Compute confidence interval."""
        return [
            float(np.percentile(values, alpha * 100 / 2)),
            float(np.percentile(values, (1 - alpha / 2) * 100))
        ]

    def _perform_significance_tests(self,
                                    predictions: np.ndarray,
                                    targets: np.ndarray) -> Dict:
        """Perform statistical significance tests."""
        tests = {}

        # McNemar's test for each class
        for i, class_name in enumerate(self.class_names):
            pred_i = predictions[:, i] > 0.5
            target_i = targets[:, i]

            # Create contingency table
            b01 = np.sum((pred_i == 0) & (target_i == 1))
            b10 = np.sum((pred_i == 1) & (target_i == 0))

            # Perform McNemar's test
            statistic = float(((b01 - b10) ** 2) / (b01 + b10)) if (b01 + b10) > 0 else 0
            p_value = float(1 - stats.chi2.cdf(statistic, df=1))

            tests[f'{class_name}_mcnemar'] = {
                'statistic': statistic,
                'p_value': p_value
            }

        return tests

    def save_results(self, results: Dict, epoch: int):
        """Save evaluation results."""
        save_path = self.save_dir / f'evaluation_results_epoch_{epoch}.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)

        # Update and save best results if needed
        if results['metrics']['mean_auc'] > self.best_metrics['mean_auc']:
            self.best_metrics = {
                'mean_auc': results['metrics']['mean_auc'],
                'epoch': epoch,
                'results': results
            }

            best_path = self.save_dir / 'best_results.json'
            with open(best_path, 'w') as f:
                json.dump(results, f, indent=4)

