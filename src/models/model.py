# src/models/model.py

import torch
import torch.nn as nn
import torchvision.models as models
from .graph import ProgressiveGraphConstruction
from .fusion import FeatureGraphFusion
import os

class GraphAugmentedViT(nn.Module):
    def __init__(self, num_diseases=14, pretrained_path=None, feature_dim=768, hidden_dim=512):
        """
        Graph-augmented Vision Transformer for multi-label chest X-ray classification

        Args:
            num_diseases: Number of disease classes
            pretrained_path: Path to pretrained ViT weights
            feature_dim: Dimension of ViT features
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()

        # Store parameters as instance variables
        self.num_diseases = num_diseases
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Initialize Vision Transformer backbone
        self.vit = models.vit_b_16(pretrained=True)

        # Replace the classification head
        self.vit.heads = nn.Identity()

        # Load pretrained weights if provided
        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            # Handle potential differences in state dict keys
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.vit.load_state_dict(state_dict, strict=False)

        # Graph modules
        self.graph_constructor = ProgressiveGraphConstruction(
            num_diseases=num_diseases,
            feature_dim=feature_dim
        )

        self.fusion_module = FeatureGraphFusion(
            vit_dim=feature_dim,
            num_diseases=num_diseases,
            hidden_dim=hidden_dim
        )

        # Additional layers
        self.feature_pooling = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_diseases)
        )

        # Disease attention
        self.disease_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1)

        # Initialize co-occurrence tracking
        self.register_buffer('co_occurrence_matrix', torch.zeros(num_diseases, num_diseases))
        self.register_buffer('co_occurrence_count', torch.zeros(num_diseases, num_diseases))


    def extract_regions(self, images, batch_data):
        """Extract and process regions based on bounding box data"""
        batch_size = images.size(0)
        device = images.device

        # Initialize feature storage
        region_features = torch.zeros(batch_size, self.num_diseases, self.feature_dim, device=device)
        area_matrix = torch.zeros(batch_size, self.num_diseases, device=device)

        # Handle batch processing
        for b in range(batch_size):
            try:
                # Get bbox data for this image
                bbox_info = batch_data['bbox'][b]  # This should be a dictionary
                if not bbox_info:  # Skip if no bbox data
                    continue

                # Process each disease
                for disease_idx, disease_name in enumerate(self.diseases):
                    if disease_name in bbox_info:
                        boxes = bbox_info[disease_name]
                        if not boxes:
                            continue

                        # Process boxes for this disease
                        box_features = []
                        total_area = 0

                        for box in boxes:
                            try:
                                x1, y1, w, h = [float(coord) for coord in box]
                                x2, y2 = x1 + w, y1 + h

                                # Add padding
                                padding = min(0.1, 50 / max(1, (w * h) ** 0.5))
                                pad_w = int(w * padding)
                                pad_h = int(h * padding)

                                # Ensure coordinates are within image bounds
                                x1 = max(0, int(x1 - pad_w))
                                y1 = max(0, int(y1 - pad_h))
                                x2 = min(images.size(3), int(x2 + pad_w))
                                y2 = min(images.size(2), int(y2 + pad_h))

                                if x2 > x1 and y2 > y1:  # Valid region
                                    region = images[b:b + 1, :, y1:y2, x1:x2]
                                    region = nn.functional.interpolate(
                                        region,
                                        size=(224, 224),
                                        mode='bilinear',
                                        align_corners=False
                                    )

                                    with torch.no_grad():
                                        features = self.vit.forward_features(region)
                                    box_features.append(features.mean(1))
                                    total_area += (y2 - y1) * (x2 - x1)

                            except Exception as e:
                                print(f"Error processing box: {e}")
                                continue

                        if box_features:
                            region_features[b, disease_idx] = torch.stack(box_features).mean(0)
                            area_matrix[b, disease_idx] = total_area

            except Exception as e:
                print(f"Error processing batch item {b}: {e}")
                continue

        return region_features, area_matrix


    def forward(self, images, batch_data=None):
        """
        Forward pass

        Args:
            images: Input images (B, C, H, W)
            batch_data: Batch information including bbox data
        """
        batch_size = images.size(0)
        device = images.device

        # Get global image features
        vit_features = self.vit(images)
        pooled_features = self.feature_pooling(vit_features)

        # Extract region features if bbox_data is provided
        if batch_data is not None:
            region_features, area_matrix = self.extract_regions(images, batch_data)
        else:
            region_features = torch.zeros(batch_size, self.num_diseases, self.feature_dim, device=device)
            area_matrix = torch.zeros(batch_size, self.num_diseases, device=device)

        # Construct graph
        adjacency_matrix = self.graph_constructor(
            region_features,
            area_matrix,
            self.co_occurrence_count
        )

        # Apply graph attention
        graph_features = torch.matmul(adjacency_matrix, region_features)

        # Disease-specific attention
        attn_output, _ = self.disease_attention(
            pooled_features.unsqueeze(0),
            graph_features.transpose(0, 1),
            graph_features.transpose(0, 1)
        )
        attn_output = attn_output.squeeze(0)

        # Fuse features
        fused_features = self.fusion_module(attn_output, graph_features)

        # Final classification
        logits = self.final_classifier(fused_features)

        # If in training mode and labels are provided
        if self.training and 'labels' in batch_data:
            self.update_co_occurrence(batch_data['labels'])

        return torch.sigmoid(logits)



    def get_attention_weights(self, images, bbox_data=None):
        """
        Get attention weights for visualization

        Args:
            images: Input images
            bbox_data: Optional bounding box information
        """
        self.eval()
        with torch.no_grad():
            # Get features
            vit_features = self.vit(images)
            pooled_features = self.feature_pooling(vit_features)

            if bbox_data is not None:
                region_features, area_matrix = self.extract_regions(images, bbox_data)
            else:
                region_features = torch.zeros_like(pooled_features.unsqueeze(1).expand(-1, self.num_diseases, -1))
                area_matrix = torch.zeros(images.size(0), self.num_diseases, device=images.device)

            # Get attention weights
            _, attn_weights = self.disease_attention(
                pooled_features.unsqueeze(0),
                region_features.transpose(0, 1),
                region_features.transpose(0, 1)
            )

        return attn_weights

    @torch.no_grad()
    def get_graph_visualization(self):
        """Get graph adjacency matrix for visualization"""
        norm_co_occurrence = self.co_occurrence_matrix / (self.co_occurrence_count + 1e-8)
        return norm_co_occurrence.cpu().numpy()

