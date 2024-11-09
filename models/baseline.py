# models/baseline.py
import torch
import torch.nn as nn
import timm
from typing import Dict, Any


class BaselineViT(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Load pre-trained ViT
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0  # Remove classification head
        )

        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, config['model']['graph_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(config['model']['graph_hidden_dim'], 14)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features from ViT
        features = self.vit.forward_features(x)

        # Apply classification head
        logits = self.classifier(features)

        return logits, features

    def extract_features(self, x):
        """Extract features without classification."""
        return self.vit.forward_features(x)

    def freeze_backbone(self):
        """Freeze ViT backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ViT backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True

