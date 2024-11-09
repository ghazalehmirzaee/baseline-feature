# models/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiComponentLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {"wbce": 1.0, "focal": 1.0, "asl": 1.0}

    def weighted_bce_loss(self, predictions, targets, pos_weights=None):
        if pos_weights is None:
            pos_weights = torch.ones_like(targets)
        return F.binary_cross_entropy_with_logits(
            predictions, targets, pos_weight=pos_weights, reduction='mean'
        )

    def focal_loss(self, predictions, targets, gamma=2.0, alpha=0.25):
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def asymmetric_loss(self, predictions, targets, gamma_neg=4, gamma_pos=1):
        with torch.no_grad():
            targets = targets.type(predictions.type())

        anti_targets = 1 - targets
        xs_pos = torch.sigmoid(predictions)
        xs_neg = 1 - xs_pos

        asymmetric_w = torch.pow(1 - xs_pos - targets, gamma_pos) * targets + \
                       torch.pow(xs_neg - anti_targets, gamma_neg) * anti_targets
        log_p = -F.logsigmoid(predictions) * targets - \
                F.logsigmoid(-predictions) * anti_targets

        return (asymmetric_w * log_p).mean()

    def forward(self, predictions, targets, pos_weights=None):
        losses = {
            "wbce": self.weighted_bce_loss(predictions, targets, pos_weights),
            "focal": self.focal_loss(predictions, targets),
            "asl": self.asymmetric_loss(predictions, targets)
        }

        total_loss = sum(self.weights[k] * v for k, v in losses.items())
        return total_loss, losses




