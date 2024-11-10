# tests/test_metrics.py
import unittest
import torch
import numpy as np
from src.utils.metrics import MetricsCalculator


class TestMetrics(unittest.TestCase):
    """Test cases for metrics calculation."""

    def setUp(self):
        """Set up test environment."""
        self.num_classes = 14
        self.class_names = [f'class_{i}' for i in range(self.num_classes)]
        self.metrics_calculator = MetricsCalculator(
            self.num_classes,
            self.class_names
        )

    def test_auc_calculation(self):
        """Test AUC-ROC calculation."""
        predictions = torch.sigmoid(torch.randn(100, self.num_classes))
        targets = torch.randint(0, 2, (100, self.num_classes)).float()

        metrics = self.metrics_calculator.compute_all_metrics(
            predictions, targets
        )

        self.assertIn('mean_auc', metrics)
        self.assertTrue(0 <= metrics['mean_auc'] <= 1)

    def test_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        predictions = torch.ones(10, self.num_classes) * 10  # High confidence
        targets = torch.ones(10, self.num_classes)

        metrics = self.metrics_calculator.compute_all_metrics(
            predictions, targets
        )

        self.assertAlmostEqual(metrics['mean_auc'], 1.0)
        self.assertAlmostEqual(metrics['mean_ap'], 1.0)


if __name__ == '__main__':
    unittest.main()

