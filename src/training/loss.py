# src/training/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MultiComponentLoss(nn.Module):
    """Combined loss for multi-label classification"""

    def __init__(self, weights: Dict[str, float], class_weights: torch.Tensor):
        super().__init__()
        self.weights = weights
        self.class_weights = class_weights

    def weighted_bce_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted binary cross entropy loss"""
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
        else:
            weights = torch.ones_like(pred)

        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        return (weights * bce_loss).mean()

    def focal_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            gamma: float = 2.0
    ) -> torch.Tensor:
        """Focal Loss"""
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def asymmetric_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            gamma_pos: float = 1,
            gamma_neg: float = 4
    ) -> torch.Tensor:
        """Asymmetric Loss"""
        # Positive samples
        pos_mask = target > 0.5
        pos_loss = F.binary_cross_entropy_with_logits(
            pred[pos_mask], target[pos_mask], reduction='none'
        ) if pos_mask.any() else torch.tensor(0.0, device=pred.device)

        if isinstance(pos_loss, torch.Tensor) and pos_mask.any():
            pt_pos = torch.exp(-pos_loss)
            pos_term = ((1 - pt_pos) ** gamma_pos * pos_loss).mean()
        else:
            pos_term = torch.tensor(0.0, device=pred.device)

        # Negative samples
        neg_mask = target <= 0.5
        neg_loss = F.binary_cross_entropy_with_logits(
            pred[neg_mask], target[neg_mask], reduction='none'
        )
        pt_neg = torch.exp(-neg_loss)
        neg_term = ((1 - pt_neg) ** gamma_neg * neg_loss).mean()

        return (pos_term + neg_term) / 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute combined loss"""
        target = target.to(pred.device)

        # Individual losses
        wbce = self.weighted_bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        asymmetric = self.asymmetric_loss(pred, target)

        # Combined loss
        total_loss = (
                self.weights['wbce'] * wbce +
                self.weights['focal'] * focal +
                self.weights['asymmetric'] * asymmetric
        )

        return total_loss, {
            'wbce': wbce.item(),
            'focal': focal.item(),
            'asymmetric': asymmetric.item()
        }


