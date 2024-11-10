# models/feature_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv
import logging
from typing import Optional, Dict, Any


class FeatureGraphModel(nn.Module):
    def __init__(self, config: Dict[str, Any], baseline_state_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize dimensions
        self.feature_dim = config['model']['feature_dim']  # Should be 768 from ViT
        self.graph_hidden_dim = config['model']['graph_hidden_dim']  # Should be 256

        # Initialize the base ViT model
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=self.feature_dim  # Set to feature_dim to ensure correct output
        )

        # Load the pretrained weights if provided
        if baseline_state_dict is not None:
            try:
                state_dict = {k: v for k, v in baseline_state_dict.items()
                              if not k.startswith('head.')}
                msg = self.backbone.load_state_dict(state_dict, strict=False)
                self.logger.info(f"Loaded baseline model weights: {msg}")
            except Exception as e:
                self.logger.error(f"Error loading baseline weights: {e}")
                raise

        # Freeze the backbone if specified in config
        if config['model'].get('freeze_backbone', True):
            self.logger.info("Freezing backbone weights")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Feature extraction layers with explicit dimensions
        self.region_feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.graph_hidden_dim),
            nn.LayerNorm(self.graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate'])
        )

        # Graph neural network layers
        self.graph_layers = nn.ModuleList([
            GCNConv(
                in_channels=self.graph_hidden_dim,
                out_channels=self.graph_hidden_dim
            )
            for _ in range(config['model']['num_graph_layers'])
        ])

        # Graph attention for combining node features
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=self.graph_hidden_dim,
            num_heads=8,
            dropout=config['model']['dropout_rate']
        )

        # Feature fusion module with explicit dimensions
        fusion_input_dim = self.feature_dim + self.graph_hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.graph_hidden_dim),
            nn.LayerNorm(self.graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(self.graph_hidden_dim, config['dataset']['num_classes'])
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def extract_region_features(self, images: torch.Tensor, bboxes: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = images.shape[0]
        device = images.device
        all_region_features = []

        for i in range(batch_size):
            boxes = bboxes['boxes'][i]

            # Handle case with no valid boxes
            if len(boxes) == 0 or not isinstance(boxes, torch.Tensor):
                with torch.no_grad():
                    features = self.backbone(images[i].unsqueeze(0))  # [1, 768]
                    features = self.region_feature_proj(features)  # [1, 256]
                all_region_features.append(features)
                continue

            region_features = []
            valid_boxes = 0
            for box in boxes:
                try:
                    x1, y1, w, h = box.tolist()
                    if w <= 0 or h <= 0:
                        continue

                    x2, y2 = x1 + w, y1 + h
                    padding = min(0.1, 50 / torch.sqrt(torch.tensor(w * h)))
                    pad_x = int(w * padding)
                    pad_y = int(h * padding)

                    x1_pad = max(0, int(x1 - pad_x))
                    y1_pad = max(0, int(y1 - pad_y))
                    x2_pad = min(images.shape[3], int(x2 + pad_x))
                    y2_pad = min(images.shape[2], int(y2 + pad_y))

                    if x2_pad <= x1_pad or y2_pad <= y1_pad:
                        continue

                    region = images[i:i + 1, :, y1_pad:y2_pad, x1_pad:x2_pad]
                    if region.numel() == 0 or region.size(2) == 0 or region.size(3) == 0:
                        continue

                    region = F.interpolate(
                        region,
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )

                    with torch.no_grad():
                        features = self.backbone(region)  # [1, 768]
                        features = self.region_feature_proj(features)  # [1, 256]
                    region_features.append(features)
                    valid_boxes += 1

                except Exception as e:
                    self.logger.warning(f"Error processing box {box}: {str(e)}")
                    continue

            if valid_boxes > 0:
                region_features = torch.cat(region_features, dim=0)  # [N, 256]
                all_region_features.append(region_features)
            else:
                # Fallback to global features
                with torch.no_grad():
                    features = self.backbone(images[i].unsqueeze(0))  # [1, 768]
                    features = self.region_feature_proj(features)  # [1, 256]
                all_region_features.append(features)

        return all_region_features

    def build_graph(self, region_features: list) -> tuple:
        batch_graphs = []
        batch_features = []

        for features in region_features:
            # Ensure features are of correct shape
            if len(features.shape) != 2 or features.shape[1] != self.graph_hidden_dim:
                self.logger.warning(f"Invalid feature shape: {features.shape}")
                continue

            num_nodes = features.size(0)
            if num_nodes == 0:
                continue

            # Reshape features for attention
            features_t = features.transpose(0, 1).unsqueeze(1)  # [256, 1, N]

            # Apply attention
            attn_output, attn_weights = self.graph_attention(
                features_t, features_t, features_t
            )

            # Convert attention weights to adjacency matrix
            adj_matrix = attn_weights.squeeze(0)  # [N, N]

            batch_graphs.append(adj_matrix)
            batch_features.append(features)

        return batch_graphs, batch_features

    def forward(self, images: torch.Tensor, bboxes: Optional[Dict[str, list]] = None) -> torch.Tensor:
        # Get baseline features
        batch_features = self.backbone(images)  # [B, 768]

        if self.training and bboxes is not None and isinstance(bboxes, dict):
            try:
                # Extract region features and build graph
                region_features = self.extract_region_features(images, bboxes)

                if region_features:
                    graphs, node_features = self.build_graph(region_features)

                    graph_embeddings = []
                    for adj_matrix, features in zip(graphs, node_features):
                        x = features  # [N, 256]

                        # Apply GNN layers
                        for layer in self.graph_layers:
                            x = layer(x, adj_matrix)
                            x = F.relu(x)
                            x = F.dropout(x, p=self.config['model']['dropout_rate'], training=self.training)

                        # Aggregate node embeddings (mean pooling)
                        graph_embedding = x.mean(dim=0)  # [256]
                        graph_embeddings.append(graph_embedding)

                    if graph_embeddings:
                        graph_embeddings = torch.stack(graph_embeddings)  # [B, 256]
                    else:
                        graph_embeddings = torch.zeros(
                            batch_features.shape[0],
                            self.graph_hidden_dim,
                            device=images.device
                        )
                else:
                    graph_embeddings = torch.zeros(
                        batch_features.shape[0],
                        self.graph_hidden_dim,
                        device=images.device
                    )

            except Exception as e:
                self.logger.error(f"Error in graph processing: {str(e)}")
                graph_embeddings = torch.zeros(
                    batch_features.shape[0],
                    self.graph_hidden_dim,
                    device=images.device
                )
        else:
            graph_embeddings = torch.zeros(
                batch_features.shape[0],
                self.graph_hidden_dim,
                device=images.device
            )

        # Project backbone features if needed
        if batch_features.shape[1] != self.feature_dim:
            batch_features = self.region_feature_proj(batch_features)

        # Concatenate and fuse features
        combined_features = torch.cat([batch_features, graph_embeddings], dim=1)  # [B, 768+256]
        logits = self.fusion_mlp(combined_features)  # [B, num_classes]

        return logits

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        if hasattr(self, 'graph_attention'):
            return {
                'graph_attention': self.graph_attention.get_attention_weights()
            }
        return {}

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

