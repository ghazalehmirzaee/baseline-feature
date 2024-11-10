# models/feature_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv
import logging
from typing import Optional, Dict, Any, List, Tuple


class FeatureGraphModel(nn.Module):
    def __init__(self, config: Dict[str, Any], baseline_state_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize dimensions
        self.feature_dim = config['model']['feature_dim']  # 768 from ViT
        self.graph_hidden_dim = config['model']['graph_hidden_dim']  # 256
        self.num_classes = config['dataset']['num_classes']  # 14 for ChestX-ray14

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

        # Feature projection layers
        self.global_feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.graph_hidden_dim),
            nn.LayerNorm(self.graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate'])
        )

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

        # Graph attention mechanism
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=self.graph_hidden_dim,
            num_heads=8,
            dropout=config['model']['dropout_rate']
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.graph_hidden_dim * 2, self.graph_hidden_dim),
            nn.LayerNorm(self.graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(self.graph_hidden_dim, self.num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def extract_region_features(self, images: torch.Tensor, bboxes: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Extract features from the global image and region proposals."""
        batch_size = images.shape[0]
        device = images.device
        all_region_features = []

        for i in range(batch_size):
            # Extract global features first
            with torch.no_grad():
                global_features = self.backbone(images[i].unsqueeze(0))  # [1, 768]
                if len(global_features.shape) == 1:
                    global_features = global_features.unsqueeze(0)
                global_features = self.global_feature_proj(global_features)  # [1, 256]

            # Process each bounding box
            boxes = bboxes['boxes'][i]
            if len(boxes) == 0 or not isinstance(boxes, torch.Tensor):
                all_region_features.append(global_features)
                continue

            # Start with global features
            region_features = [global_features]

            for box in boxes:
                try:
                    # Extract and validate box coordinates
                    x1, y1, w, h = box.tolist()
                    if w <= 0 or h <= 0:
                        continue

                    # Calculate padded coordinates
                    x2, y2 = x1 + w, y1 + h
                    padding = min(0.1, 50 / torch.sqrt(torch.tensor(w * h)))
                    pad_x, pad_y = int(w * padding), int(h * padding)

                    x1_pad = max(0, int(x1 - pad_x))
                    y1_pad = max(0, int(y1 - pad_y))
                    x2_pad = min(images.shape[3], int(x2 + pad_x))
                    y2_pad = min(images.shape[2], int(y2 + pad_y))

                    if x2_pad <= x1_pad or y2_pad <= y1_pad:
                        continue

                    # Extract and resize region
                    region = images[i:i + 1, :, y1_pad:y2_pad, x1_pad:x2_pad]
                    if region.numel() == 0 or region.size(2) == 0 or region.size(3) == 0:
                        continue

                    region = F.interpolate(
                        region,
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )

                    # Extract region features
                    with torch.no_grad():
                        region_feature = self.backbone(region)  # [1, 768]
                        if len(region_feature.shape) == 1:
                            region_feature = region_feature.unsqueeze(0)
                        region_feature = self.region_feature_proj(region_feature)  # [1, 256]
                    region_features.append(region_feature)

                except Exception as e:
                    self.logger.warning(f"Error processing box {box}: {str(e)}")
                    continue

            # Combine all features including global features
            if len(region_features) > 0:
                region_features = torch.cat(region_features, dim=0)  # [N+1, 256]
                all_region_features.append(region_features)
            else:
                all_region_features.append(global_features)

        return all_region_features

    def build_graph(self, region_features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Build graph from region features using attention mechanism."""
        batch_graphs = []
        batch_features = []

        for features in region_features:
            # Validate feature dimensions
            if features.shape[1] != self.graph_hidden_dim:
                self.logger.error(f"Invalid feature dimension: {features.shape}")
                continue

            # Compute self-attention
            features_t = features.transpose(0, 1).unsqueeze(1)  # [256, 1, N]
            attn_output, attn_weights = self.graph_attention(
                features_t, features_t, features_t
            )

            # Convert attention weights to adjacency matrix
            adj_matrix = attn_weights.squeeze(0)  # [N, N]

            batch_graphs.append(adj_matrix)
            batch_features.append(features)

        return batch_graphs, batch_features

    def forward(self, images: torch.Tensor, bboxes: Optional[Dict[str, list]] = None) -> torch.Tensor:
        """Forward pass of the model."""
        batch_size = images.shape[0]
        device = images.device

        # Extract global features
        global_features = self.backbone(images)  # [B, 768]
        if len(global_features.shape) == 1:
            global_features = global_features.unsqueeze(0)
        global_features = self.global_feature_proj(global_features)  # [B, 256]

        # Process regions and build graph during training
        if self.training and bboxes is not None:
            try:
                region_features = self.extract_region_features(images, bboxes)

                if region_features:
                    graphs, node_features = self.build_graph(region_features)

                    graph_embeddings = []
                    for idx, (adj_matrix, features) in enumerate(zip(graphs, node_features)):
                        x = features

                        # Apply GNN layers
                        for layer in self.graph_layers:
                            x = layer(x, adj_matrix)
                            x = F.relu(x)
                            x = F.dropout(x, p=self.config['model']['dropout_rate'], training=self.training)

                        # Attention-weighted pooling
                        with torch.no_grad():
                            weights = F.softmax(torch.matmul(x, x.transpose(0, 1)), dim=1)
                        graph_embedding = torch.matmul(weights, x).mean(dim=0)  # [256]
                        graph_embeddings.append(graph_embedding)

                    if graph_embeddings:
                        graph_embeddings = torch.stack(graph_embeddings)  # [B, 256]
                    else:
                        graph_embeddings = torch.zeros(batch_size, self.graph_hidden_dim, device=device)
                else:
                    graph_embeddings = torch.zeros(batch_size, self.graph_hidden_dim, device=device)

            except Exception as e:
                self.logger.error(f"Error in graph processing: {str(e)}")
                graph_embeddings = torch.zeros(batch_size, self.graph_hidden_dim, device=device)
        else:
            graph_embeddings = torch.zeros(batch_size, self.graph_hidden_dim, device=device)

        # Combine features and classify
        combined_features = torch.cat([global_features, graph_embeddings], dim=1)  # [B, 512]
        logits = self.classifier(combined_features)  # [B, num_classes]

        return logits

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization."""
        if hasattr(self, 'graph_attention'):
            return {
                'graph_attention': self.graph_attention.get_attention_weights()
            }
        return {}

    def freeze_backbone(self) -> None:
        """Freeze the backbone network."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone network."""
        for param in self.backbone.parameters():
            param.requires_grad = True

