# src/training/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiComponentLoss(nn.Module):
    def __init__(self, weights=None, class_weights=None):
        """
        Combined loss for multi-label classification

        Args:
            weights: Dictionary with keys 'wbce', 'focal', and 'asymmetric' for loss weights
            class_weights: Per-class weights for handling imbalance
        """
        super().__init__()
        # Default weights if none provided
        self.weights = weights if weights is not None else {
            'wbce': 1.0,
            'focal': 1.0,
            'asymmetric': 1.0
        }
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
        ) if pos_mask.any() else 0.0

        if isinstance(pos_loss, torch.Tensor):
            pt_pos = torch.exp(-pos_loss)
            pos_term = (1 - pt_pos) ** gamma_pos * pos_loss
            pos_term = pos_term.mean() if pos_mask.any() else 0.0
        else:
            pos_term = 0.0

        # Negative samples
        neg_mask = target <= 0.5
        neg_loss = F.binary_cross_entropy_with_logits(
            pred[neg_mask], target[neg_mask], reduction='none'
        ) if neg_mask.any() else 0.0

        if isinstance(neg_loss, torch.Tensor):
            pt_neg = torch.exp(-neg_loss)
            neg_term = (1 - pt_neg) ** gamma_neg * neg_loss
            neg_term = neg_term.mean() if neg_mask.any() else 0.0
        else:
            neg_term = 0.0

        return (pos_term + neg_term) / 2

    def forward(self, pred, target):
        """
        Compute combined loss

        Args:
            pred: Model predictions (B, num_diseases)
            target: Ground truth labels (B, num_diseases)
        """
        # Move target to same device as predictions
        target = target.to(pred.device)

        # Compute individual losses
        wbce = self.weighted_bce_loss(pred, target)
        fl = self.focal_loss(pred, target)
        asl = self.asymmetric_loss(pred, target)

        # Combine losses using weights from dictionary
        total_loss = (
                self.weights['wbce'] * wbce +
                self.weights['focal'] * fl +
                self.weights['asymmetric'] * asl
        )

        return total_loss, {
            'wbce': wbce.item(),
            'focal': fl.item(),
            'asymmetric': asl.item()
        }


