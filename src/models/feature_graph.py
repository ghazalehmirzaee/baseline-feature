# src/models/feature_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, Tuple
import logging

from .graph_modules.graph_builder import ProgressiveGraphBuilder
from .graph_modules.feature_extractor import RegionFeatureExtractor


class FeatureGraphModel(nn.Module):
    def __init__(self, config: Dict, baseline_state_dict: Optional[Dict] = None):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Dimensions
        self.feature_dim = config['model']['feature_dim']
        self.graph_hidden_dim = config['model']['graph_hidden_dim']
        self.num_classes = config['dataset']['num_classes']

        # Initialize backbone ViT
        self.backbone = self._init_backbone(baseline_state_dict)

        # Feature projector
        self.feature_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.graph_hidden_dim),
            nn.LayerNorm(self.graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate'])
        )

        # Graph components
        self.graph_builder = ProgressiveGraphBuilder(config)
        self.region_extractor = RegionFeatureExtractor(self.backbone, self.feature_projector)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.graph_hidden_dim * 2, self.graph_hidden_dim),
            nn.LayerNorm(self.graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(self.graph_hidden_dim, self.num_classes)
        )

    def _init_backbone(self, state_dict: Optional[Dict]) -> nn.Module:
        """Initialize and load the backbone model."""
        backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )

        if state_dict is not None:
            try:
                # Remove classifier weights
                state_dict = {k: v for k, v in state_dict.items()
                              if not k.startswith('head.')}
                backbone.load_state_dict(state_dict, strict=False)
                self.logger.info("Loaded pretrained weights")
            except Exception as e:
                self.logger.error(f"Error loading weights: {e}")
                raise

        # Freeze backbone if specified
        if self.config['model'].get('freeze_backbone', True):
            for param in backbone.parameters():
                param.requires_grad = False

        return backbone

    def forward(self, images: torch.Tensor,
                bboxes: Optional[Dict] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the model."""
        batch_size = images.shape[0]
        device = images.device

        # Extract base features
        with torch.no_grad():
            base_features = self.backbone(images)  # [B, 768]
            base_features = self.feature_projector(base_features)  # [B, 256]

        # Process bounding boxes and build graph if available
        if self.training and bboxes is not None:
            try:
                # Extract region features
                region_features, regions_by_disease = self.region_extractor(
                    images, bboxes
                )

                # Build graph
                graph_weights = self.graph_builder(
                    features=region_features,
                    regions_by_disease=regions_by_disease,
                    labels=labels
                )

                # Apply graph weights to features
                graph_features = torch.matmul(base_features, graph_weights)  # [B, 256]

            except Exception as e:
                self.logger.error(f"Error in graph processing: {e}")
                graph_features = torch.zeros_like(base_features)
        else:
            graph_features = torch.zeros_like(base_features)

        # Combine features and classify
        combined_features = torch.cat([base_features, graph_features], dim=1)  # [B, 512]
        logits = self.classifier(combined_features)  # [B, num_classes]

        return logits

    def get_graph_weights(self) -> torch.Tensor:
        """Get current graph weights for visualization."""
        return self.graph_builder.get_weights()

    def save_graph_weights(self, path: str):
        """Save graph weights to disk."""
        weights = self.get_graph_weights()
        torch.save(weights, path)

    def load_graph_weights(self, path: str):
        """Load pre-computed graph weights."""
        weights = torch.load(path)
        self.graph_builder.set_weights(weights)


