# src/training/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiComponentLoss(nn.Module):
    def __init__(self, weights=(1.0, 1.0, 1.0), class_weights=None):
        """
        Combined loss for multi-label classification

        Args:
            weights: Weights for WBCE, FL, and ASL components
            class_weights: Per-class weights for handling imbalance
        """
        super().__init__()
        self.weights = weights
        self.class_weights = class_weights

    def weighted_bce_loss(self, pred, target):
        """Weighted Binary Cross Entropy"""
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
        else:
            weights = torch.ones_like(pred)

        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        weighted_bce = weights * bce_loss
        return weighted_bce.mean()

    def focal_loss(self, pred, target, gamma=2.0):
        """Focal Loss"""
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def asymmetric_loss(self, pred, target, gamma_pos=1, gamma_neg=4):
        """Asymmetric Loss"""
        # Positive samples
        pos_mask = target > 0.5
        pos_loss = F.binary_cross_entropy_with_logits(
            pred[pos_mask], target[pos_mask], reduction='none'
        )
        pt_pos = torch.exp(-pos_loss)
        pos_term = (1 - pt_pos) ** gamma_pos * pos_loss

        # Negative samples
        neg_mask = target <= 0.5
        neg_loss = F.binary_cross_entropy_with_logits(
            pred[neg_mask], target[neg_mask], reduction='none'
        )
        pt_neg = torch.exp(-neg_loss)
        neg_term = (1 - pt_neg) ** gamma_neg * neg_loss

        return (pos_term.mean() + neg_term.mean()) / 2

    def forward(self, pred, target):
        """
        Compute combined loss

        Args:
            pred: Model predictions (B, num_diseases)
            target: Ground truth labels (B, num_diseases)
        """
        wbce = self.weighted_bce_loss(pred, target)
        fl = self.focal_loss(pred, target)
        asl = self.asymmetric_loss(pred, target)

        total_loss = (
                self.weights[0] * wbce +
                self.weights[1] * fl +
                self.weights[2] * asl
        )

        return total_loss, {
            'wbce': wbce.item(),
            'focal': fl.item(),
            'asymmetric': asl.item()
        }

