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
        self.feature_dim = config['model']['feature_dim']
        self.graph_hidden_dim = config['model']['graph_hidden_dim']

        # Initialize the base ViT model
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0  # Remove classification head
        )

        # Load the pretrained weights if provided
        if baseline_state_dict is not None:
            try:
                # Remove the head weights as we don't need them
                state_dict = {k: v for k, v in baseline_state_dict.items()
                              if not k.startswith('head.')}

                # Load the state dict
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

        # Feature extraction layers
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

        # Feature fusion module
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
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def extract_region_features(self, images: torch.Tensor, bboxes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from image regions defined by bounding boxes."""
        batch_size = images.shape[0]
        all_region_features = []

        for i in range(batch_size):
            # Get bounding boxes for this image
            boxes = bboxes['boxes'][i]
            if len(boxes) == 0:
                # If no bounding boxes, use global features
                with torch.no_grad():
                    features = self.backbone(images[i].unsqueeze(0))
                all_region_features.append(features)
                continue

            region_features = []
            for box in boxes:
                # Extract region
                x1, y1, w, h = box.tolist()
                x2, y2 = x1 + w, y1 + h

                # Add padding
                padding = min(0.1, 50 / torch.sqrt(torch.tensor(w * h)))
                pad_x = w * padding
                pad_y = h * padding

                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(images.shape[3], x2 + pad_x)
                y2 = min(images.shape[2], y2 + pad_y)

                # Crop and resize region
                region = F.interpolate(
                    images[i:i + 1, :, int(y1):int(y2), int(x1):int(x2)],
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )

                # Extract features
                with torch.no_grad():
                    features = self.backbone(region)
                region_features.append(features)

            # Combine region features
            if region_features:
                region_features = torch.cat(region_features, dim=0)
                all_region_features.append(region_features)
            else:
                # Fallback to global features if no valid regions
                with torch.no_grad():
                    features = self.backbone(images[i].unsqueeze(0))
                all_region_features.append(features)

        # Process features through projection layer
        processed_features = []
        for features in all_region_features:
            projected = self.region_feature_proj(features)
            processed_features.append(projected)

        return processed_features

    def build_graph(self, region_features: list) -> tuple:
        """Build graph from region features."""
        batch_graphs = []
        batch_features = []

        for features in region_features:
            num_nodes = features.size(0)
            if num_nodes == 0:
                continue

            # Compute adjacency matrix using attention
            features_t = features.transpose(0, 1).unsqueeze(1)  # [D, 1, N]
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
        # Get baseline features
        batch_features = self.backbone(images)  # [B, D]

        if self.training and bboxes is not None and isinstance(bboxes, dict):
            try:
                # Convert list of tensors to padded tensor
                max_boxes = max(len(boxes) for boxes in bboxes['boxes'])
                if max_boxes > 0:
                    padded_boxes = torch.zeros(len(bboxes['boxes']), max_boxes, 4, device=images.device)
                    padded_labels = torch.zeros(len(bboxes['labels']), max_boxes, device=images.device,
                                                dtype=torch.long)
                    padded_areas = torch.zeros(len(bboxes['areas']), max_boxes, device=images.device)

                    # Pad each sample's boxes
                    for i, (boxes, labels, areas) in enumerate(zip(bboxes['boxes'], bboxes['labels'], bboxes['areas'])):
                        if len(boxes) > 0:
                            padded_boxes[i, :len(boxes)] = boxes.to(images.device)
                            padded_labels[i, :len(labels)] = labels.to(images.device)
                            padded_areas[i, :len(areas)] = areas.to(images.device)

                    # Create padded bboxes dict
                    padded_bboxes = {
                        'boxes': padded_boxes,
                        'labels': padded_labels,
                        'areas': padded_areas
                    }

                    # Extract region features and build graph
                    region_features = self.extract_region_features(images, padded_bboxes)
                    graphs, node_features = self.build_graph(region_features)

                    # Process each graph
                    graph_embeddings = []
                    for adj_matrix, features in zip(graphs, node_features):
                        x = features

                        # Apply GNN layers
                        for layer in self.graph_layers:
                            x = layer(x, adj_matrix)
                            x = F.relu(x)
                            x = F.dropout(x, p=self.config['model']['dropout_rate'], training=self.training)

                        # Aggregate node embeddings
                        graph_embedding = x.mean(dim=0)  # [D]
                        graph_embeddings.append(graph_embedding)

                    if graph_embeddings:
                        graph_embeddings = torch.stack(graph_embeddings)  # [B, D]
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
            # During inference or when no BBs available
            graph_embeddings = torch.zeros(
                batch_features.shape[0],
                self.graph_hidden_dim,
                device=images.device
            )

        # Concatenate and fuse features
        combined_features = torch.cat([batch_features, graph_embeddings], dim=1)
        logits = self.fusion_mlp(combined_features)

        return logits

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization."""
        if hasattr(self, 'graph_attention'):
            return {
                'graph_attention': self.graph_attention.get_attention_weights()
            }
        return {}

    def freeze_backbone(self):
        """Freeze the backbone network."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the backbone network."""
        for param in self.backbone.parameters():
            param.requires_grad = True

