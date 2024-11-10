# src/models/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MultiLabelLoss(nn.Module):
    """
    Combined loss function for multi-label classification with three components:
    1. Weighted Binary Cross-Entropy (WBCE)
    2. Focal Loss (FL)
    3. Asymmetric Loss (ASL)
    """

    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights: Dictionary containing weights for each loss component
        """
        super().__init__()
        self.weights = weights

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]

        Returns:
            total_loss: Weighted sum of all loss components
            losses: Dictionary containing individual loss values
        """
        # Compute individual losses
        losses = {
            'wbce': self._weighted_bce_loss(predictions, targets),
            'focal': self._focal_loss(predictions, targets),
            'asl': self._asymmetric_loss(predictions, targets)
        }

        # Compute weighted sum
        total_loss = sum(
            self.weights[name] * loss
            for name, loss in losses.items()
        )

        return total_loss, losses

    def _weighted_bce_loss(self,
                           predictions: torch.Tensor,
                           targets: torch.Tensor,
                           pos_weight: float = 1.0) -> torch.Tensor:
        """
        Weighted binary cross-entropy loss.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            pos_weight: Weight for positive class to handle imbalance

        Returns:
            Loss value
        """
        return F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=torch.tensor([pos_weight]).to(predictions.device)
        )

    def _focal_loss(self,
                    predictions: torch.Tensor,
                    targets: torch.Tensor,
                    gamma: float = 2.0,
                    alpha: float = 0.25) -> torch.Tensor:
        """
        Focal Loss for better handling of hard examples.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            gamma: Focusing parameter
            alpha: Weighting factor

        Returns:
            Loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(predictions)

        # Compute pt
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Compute focal weight
        focal_weight = (1 - pt) ** gamma

        # Compute alpha weight
        alpha_weight = torch.where(
            targets == 1,
            alpha * torch.ones_like(targets),
            (1 - alpha) * torch.ones_like(targets)
        )

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction='none'
        )

        return (focal_weight * alpha_weight * bce_loss).mean()

    def _asymmetric_loss(self,
                         predictions: torch.Tensor,
                         targets: torch.Tensor,
                         gamma_pos: float = 1,
                         gamma_neg: float = 4) -> torch.Tensor:
        """
        Asymmetric Loss for handling positive/negative sample imbalance.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            gamma_pos: Focusing parameter for positive samples
            gamma_neg: Focusing parameter for negative samples

        Returns:
            Loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(predictions)

        # Compute weights
        pos_weights = targets * (1 - probs) ** gamma_pos
        neg_weights = (1 - targets) * probs ** gamma_neg

        # Compute loss
        loss = pos_weights * torch.log(probs + 1e-8) + \
               neg_weights * torch.log(1 - probs + 1e-8)

        return -loss.mean()

