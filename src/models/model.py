# src/models/model.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import timm
from src.models.graph import GraphLayer


class GraphAugmentedViT(nn.Module):
    """
    Graph-augmented Vision Transformer for multi-label chest X-ray classification.

    Args:
        pretrained_path (str): Path to pretrained ViT model weights
        num_diseases (int): Number of disease classes
        feature_dim (int): Dimension of input features from ViT
        hidden_dim (int): Dimension of hidden layers in graph component
        graph_layers (int): Number of graph neural network layers
        dropout (float): Dropout probability
    """

    def __init__(
            self,
            pretrained_path: str,
            num_diseases: int = 14,
            feature_dim: int = 768,
            hidden_dim: int = 512,
            graph_layers: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Disease names in order
        self.diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        # Load pre-trained model
        self.load_pretrained_model(pretrained_path)


        # Graph-based refinement components
        self.graph_layers = nn.ModuleList([
            GraphLayer(
                feature_dim=feature_dim,  # Changed from in_dim to feature_dim
                hidden_dim=hidden_dim,
                num_diseases=num_diseases,
                dropout=dropout
            ) for _ in range(graph_layers)
        ])

        # Feature projections
        self.global_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.region_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Disease-specific attention
        self.disease_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Final prediction layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.classifier = nn.Linear(hidden_dim // 2, num_diseases)

        # Initialize co-occurrence tracking
        self.register_buffer('co_occurrence_matrix', torch.zeros(num_diseases, num_diseases))
        self.register_buffer('co_occurrence_count', torch.zeros(num_diseases, num_diseases))

        # Add validation
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")


    def load_pretrained_model(self, checkpoint_path: str):
        """Load pre-trained ViT model with correct architecture"""
        print(f"Loading pre-trained model from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        # Initialize ViT with timm architecture to match your pretrained model
        self.base_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=self.num_diseases
        )

        # Print model keys for debugging
        print("Keys in checkpoint:", state_dict.keys())
        print("Keys in model:", self.base_model.state_dict().keys())

        # Load weights
        try:
            # Create a new state dict with matching keys
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove any "module." prefix (from DataParallel)
                if k.startswith('module.'):
                    k = k[7:]
                # Add to new state dict
                new_state_dict[k] = v

            # Load the weights
            msg = self.base_model.load_state_dict(new_state_dict, strict=False)
            print(f"Load results: {msg}")
            print("Successfully loaded pre-trained weights")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            raise

        # Remove classification head
        self.base_model.head = nn.Identity()


    def extract_global_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global image features"""
        with torch.no_grad():
            features = self.base_model(images)
        return features

    def extract_region_features(
            self,
            images: torch.Tensor,
            bbox_data: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from image regions based on bounding boxes"""
        batch_size = images.size(0)
        device = images.device

        # Initialize storage
        region_features = torch.zeros(batch_size, self.num_diseases, self.feature_dim).to(device)
        area_matrix = torch.zeros(batch_size, self.num_diseases).to(device)

        # Process each image in batch
        for b in range(batch_size):
            try:
                bbox_info = bbox_data[b]

                # Process each disease
                for disease_idx, disease_name in enumerate(self.diseases):
                    boxes = bbox_info.get(disease_name, [])
                    if not boxes:
                        continue

                    # Process each bounding box
                    box_features = []
                    total_area = 0

                    for box in boxes:
                        try:
                            x1, y1, w, h = map(float, box)
                            x2, y2 = x1 + w, y1 + h

                            # Add padding
                            padding = min(0.1, 50 / max(1, (w * h) ** 0.5))
                            pad_w, pad_h = int(w * padding), int(h * padding)

                            # Adjust coordinates
                            x1 = max(0, int(x1 - pad_w))
                            y1 = max(0, int(y1 - pad_h))
                            x2 = min(images.size(3), int(x2 + pad_w))
                            y2 = min(images.size(2), int(y2 + pad_h))

                            if x2 > x1 and y2 > y1:
                                # Extract and process region
                                region = images[b:b + 1, :, y1:y2, x1:x2]
                                if region.numel() > 0:
                                    # Resize region
                                    region = F.interpolate(
                                        region,
                                        size=(224, 224),
                                        mode='bilinear',
                                        align_corners=False
                                    )

                                    # Extract features
                                    with torch.no_grad():
                                        features = self.base_model(region)
                                    box_features.append(features)
                                    total_area += (y2 - y1) * (x2 - x1)

                        except Exception as e:
                            print(f"Error processing box in batch {b}, disease {disease_name}: {str(e)}")
                            continue

                    if box_features:
                        features_stack = torch.stack(box_features).to(device)
                        region_features[b, disease_idx] = features_stack.mean(0)
                        area_matrix[b, disease_idx] = total_area

            except Exception as e:
                print(f"Error processing batch item {b}: {str(e)}")
                continue

        return region_features, area_matrix


    def forward(self, images: torch.Tensor, batch_data: Optional[Dict] = None) -> torch.Tensor:
        """Forward pass"""
        # Print shapes for debugging
        print(f"Input image shape: {images.shape}")

        device = images.device
        batch_size = images.size(0)

        # Extract global features
        global_features = self.extract_global_features(images)  # [B, feature_dim]
        print(f"Global features shape: {global_features.shape}")

        global_features = self.global_projection(global_features)  # [B, hidden_dim]

        # Extract region features if bbox data available
        if batch_data is not None:
            region_features, area_matrix = self.extract_region_features(images, batch_data['bbox'])
        else:
            region_features = torch.zeros(batch_size, self.num_diseases, self.feature_dim).to(device)
            area_matrix = torch.zeros(batch_size, self.num_diseases).to(device)

        print(f"Region features shape: {region_features.shape}")
        print(f"Area matrix shape: {area_matrix.shape}")

        # Project region features
        region_features = self.region_projection(region_features)  # [B, num_diseases, hidden_dim]

        # Apply graph layers
        graph_features = region_features
        for graph_layer in self.graph_layers:
            graph_features = graph_layer(
                graph_features,  # [B, num_diseases, hidden_dim]
                area_matrix,  # [B, num_diseases]
                self.co_occurrence_matrix.to(device)  # [num_diseases, num_diseases]
            )

        # Disease-specific attention
        query = global_features.unsqueeze(1)  # [B, 1, hidden_dim]
        attn_output, _ = self.disease_attention(
            query,
            graph_features,
            graph_features
        )
        attn_output = attn_output.squeeze(1)  # [B, hidden_dim]

        # Combine features
        combined_features = torch.cat([global_features, attn_output], dim=1)  # [B, hidden_dim*2]
        fused_features = self.fusion_layer(combined_features)  # [B, hidden_dim//2]

        # Final prediction
        logits = self.classifier(fused_features)  # [B, num_diseases]

        # Update co-occurrence if training
        if self.training and 'labels' in batch_data:
            self._update_co_occurrence(batch_data['labels'].to(device))

        return torch.sigmoid(logits)


    def _update_co_occurrence(self, labels: torch.Tensor):
        """Update disease co-occurrence statistics"""
        device = labels.device
        pos_samples = (labels > 0.5).float()
        batch_co_occurrence = torch.matmul(pos_samples.t(), pos_samples)
        self.co_occurrence_matrix = self.co_occurrence_matrix.to(device)
        self.co_occurrence_count = self.co_occurrence_count.to(device)
        self.co_occurrence_matrix += batch_co_occurrence
        self.co_occurrence_count += (batch_co_occurrence > 0).float()

