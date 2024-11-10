# models/feature_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from typing import Optional, Dict, Any
from .graph_modules.graph_builder import ProgressiveGraphBuilder


class FeatureGraphModel(nn.Module):
    def __init__(self, config: Dict[str, Any], baseline_state_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize dimensions
        self.feature_dim = config['model']['feature_dim']  # 768 from ViT
        self.graph_hidden_dim = config['model']['graph_hidden_dim']  # 256
        self.num_classes = config['dataset']['num_classes']  # 14

        # Initialize the base ViT model
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )

        # Load pretrained weights
        if baseline_state_dict is not None:
            try:
                state_dict = {k: v for k, v in baseline_state_dict.items()
                              if not k.startswith('head.')}
                msg = self.backbone.load_state_dict(state_dict, strict=False)
                self.logger.info(f"Loaded baseline model weights: {msg}")
            except Exception as e:
                self.logger.error(f"Error loading baseline weights: {e}")
                raise

        # Freeze backbone if specified
        if config['model'].get('freeze_backbone', True):
            self.logger.info("Freezing backbone weights")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Feature projection
        self.feature_proj = nn.Linear(self.feature_dim, self.graph_hidden_dim)

        # Graph builder
        self.graph_builder = ProgressiveGraphBuilder(config)

        # Disease relationship statistics
        self.register_buffer('disease_counts', torch.zeros(14))
        self.register_buffer('cooccurrence_matrix', torch.zeros(14, 14))

        # Final MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.graph_hidden_dim * 2, self.graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(self.graph_hidden_dim, self.num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def extract_region_features(self, images: torch.Tensor, bboxes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from image regions defined by bounding boxes."""
        batch_size = images.shape[0]
        device = images.device
        all_features = []

        for i in range(batch_size):
            # Get boxes for current image
            boxes = bboxes['boxes'][i]
            if len(boxes) == 0:
                # If no boxes, use global features
                with torch.no_grad():
                    features = self.backbone(images[i].unsqueeze(0))
                    features = self.feature_proj(features)
                all_features.append(features)
                continue

            # Process each box
            region_features = []
            for box in boxes:
                try:
                    # Extract box coordinates
                    x1, y1, w, h = box.tolist()
                    if w <= 0 or h <= 0:
                        continue

                    # Extract and resize region
                    region = images[i:i + 1, :, int(y1):int(y1 + h), int(x1):int(x1 + w)]
                    if region.numel() == 0:
                        continue

                    # Resize region to ViT input size
                    region = F.interpolate(
                        region,
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )

                    # Extract features
                    with torch.no_grad():
                        features = self.backbone(region)
                        features = self.feature_proj(features)
                    region_features.append(features)

                except Exception as e:
                    self.logger.warning(f"Error processing box {box}: {str(e)}")
                    continue

            # Combine region features or use global features
            if region_features:
                features = torch.mean(torch.cat(region_features, dim=0), dim=0, keepdim=True)
            else:
                with torch.no_grad():
                    features = self.backbone(images[i].unsqueeze(0))
                    features = self.feature_proj(features)

            all_features.append(features)

        return torch.cat(all_features, dim=0)

    def _update_disease_statistics(self, labels: torch.Tensor):
        """Update disease occurrence and co-occurrence statistics."""
        batch_size = labels.size(0)

        # Update disease counts
        disease_counts = torch.sum(labels, dim=0)
        self.disease_counts += disease_counts

        # Update co-occurrence matrix
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                cooccur = torch.sum(labels[:, i] * labels[:, j])
                self.cooccurrence_matrix[i, j] += cooccur

    def forward(self, images: torch.Tensor, bboxes: Optional[Dict[str, list]] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the model."""
        batch_size = images.shape[0]
        device = images.device

        # Extract image features
        if bboxes is not None:
            features = self.extract_region_features(images, bboxes)
        else:
            with torch.no_grad():
                features = self.backbone(images)
                features = self.feature_proj(features)

        # Update disease statistics during training
        if self.training and labels is not None:
            self._update_disease_statistics(labels)

            # Prepare data for graph builder
            feature_dict = {i: features for i in range(self.num_classes)}
            area_dict = {i: torch.ones(1) for i in range(self.num_classes)}  # Placeholder
            sample_pairs = {}

            # Build graph
            graph_weights = self.graph_builder.build_graph(
                feature_dict, area_dict, sample_pairs
            )
        else:
            # During inference, use stored statistics
            graph_weights = torch.eye(self.num_classes, device=device)

        # Compute graph features
        graph_features = torch.matmul(features, graph_weights.to(device))

        # Concatenate and classify
        combined_features = torch.cat([features, graph_features], dim=1)
        logits = self.classifier(combined_features)

        return logits

    def get_graph_weights(self) -> torch.Tensor:
        """Get the current graph weights for visualization."""
        return self.graph_builder.get_current_weights()

    def save_graph_weights(self, path: str):
        """Save the current graph weights."""
        weights = self.get_graph_weights()
        torch.save(weights, path)

    def load_graph_weights(self, path: str):
        """Load pre-computed graph weights."""
        weights = torch.load(path)
        self.graph_builder.set_weights(weights)

