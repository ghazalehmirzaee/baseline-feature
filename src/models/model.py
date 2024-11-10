# src/models/model.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .graph import ProgressiveGraphConstruction


class GraphAugmentedViT(nn.Module):
    def __init__(self, num_diseases=14, pretrained_path=None, feature_dim=768, hidden_dim=512):
        super().__init__()

        # Store parameters
        self.num_diseases = num_diseases
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        # Initialize ViT with the correct weights
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()  # Remove classification head

        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.vit.load_state_dict(state_dict, strict=False)

        # Graph modules
        self.graph_constructor = ProgressiveGraphConstruction(
            num_diseases=num_diseases,
            feature_dim=feature_dim
        )

        # Project ViT features to hidden dimension
        self.feature_pooling = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Project graph features to hidden dimension
        self.graph_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Disease attention with correct dimensions
        self.disease_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Final fusion and classification
        self.fusion_module = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.final_classifier = nn.Linear(hidden_dim // 2, num_diseases)

        # Initialize co-occurrence tracking
        self.register_buffer('co_occurrence_matrix', torch.zeros(num_diseases, num_diseases))
        self.register_buffer('co_occurrence_count', torch.zeros(num_diseases, num_diseases))

    def to(self, device):
        """Override to() to ensure all model components move to the same device"""
        super().to(device)
        self.vit = self.vit.to(device)
        self.graph_constructor = self.graph_constructor.to(device)
        self.feature_pooling = self.feature_pooling.to(device)
        self.graph_projection = self.graph_projection.to(device)
        self.disease_attention = self.disease_attention.to(device)
        self.fusion_module = self.fusion_module.to(device)
        self.final_classifier = self.final_classifier.to(device)
        return self


    def extract_regions(self, images, batch_data):
        """Extract and process regions based on bounding box data"""
        batch_size = images.size(0)
        device = images.device

        # Initialize feature storage
        region_features = torch.zeros(batch_size, self.num_diseases, self.feature_dim).to(device)
        area_matrix = torch.zeros(batch_size, self.num_diseases).to(device)

        # Move ViT to correct device
        self.vit = self.vit.to(device)

        # Handle batch processing
        bbox_data = batch_data['bbox']  # This is now a list of dictionaries

        for b in range(batch_size):
            try:
                bbox_info = bbox_data[b]  # Get bbox info for this sample

                for disease_idx, disease_name in enumerate(self.diseases):
                    if disease_name in bbox_info and bbox_info[disease_name]:  # Check if disease has boxes
                        boxes = bbox_info[disease_name]
                        box_features = []
                        total_area = 0

                        for box in boxes:
                            try:
                                x1, y1, w, h = map(float, box)
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
                                    region = images[b:b + 1, :, y1:y2, x1:x2].to(device)
                                    if region.numel() > 0:
                                        region = F.interpolate(
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


    def forward(self, images, batch_data=None):
        """Forward pass"""
        device = images.device
        batch_size = images.size(0)

        # Move entire model to correct device
        self.to(device)

        # Get global image features from ViT
        vit_features = self.vit(images)  # [batch_size, feature_dim]

        # Project ViT features to hidden dimension
        pooled_features = self.feature_pooling(vit_features)  # [batch_size, hidden_dim]

        # Extract and process region features
        if batch_data is not None:
            region_features, area_matrix = self.extract_regions(images, batch_data)
        else:
            region_features = torch.zeros(batch_size, self.num_diseases, self.feature_dim).to(device)
            area_matrix = torch.zeros(batch_size, self.num_diseases).to(device)

        # Construct graph
        adjacency_matrix = self.graph_constructor(
            region_features,
            area_matrix,
            self.co_occurrence_count.to(device)
        )

        # Apply graph attention and project to hidden dimension
        graph_features = torch.matmul(adjacency_matrix.to(device), region_features.to(device))
        graph_features = self.graph_projection(graph_features)

        # Prepare features for attention
        query = pooled_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        key = graph_features  # [batch_size, num_diseases, hidden_dim]
        value = graph_features  # [batch_size, num_diseases, hidden_dim]

        # Apply disease attention
        attn_output, _ = self.disease_attention(query, key, value)
        attn_output = attn_output.squeeze(1)

        # Concatenate and fuse features
        fused_features = torch.cat([pooled_features, attn_output], dim=1)
        fused_features = self.fusion_module(fused_features)

        # Final classification
        logits = self.final_classifier(fused_features)

        # Update co-occurrence if in training
        if self.training and 'labels' in batch_data:
            labels = batch_data['labels'].to(device)
            self.update_co_occurrence(labels)

        return torch.sigmoid(logits)

    def update_co_occurrence(self, labels):
        """Update co-occurrence statistics"""
        device = labels.device
        pos_samples = (labels > 0.5).float()
        batch_co_occurrence = torch.matmul(pos_samples.t(), pos_samples)
        self.co_occurrence_matrix = self.co_occurrence_matrix.to(device)
        self.co_occurrence_count = self.co_occurrence_count.to(device)
        self.co_occurrence_matrix += batch_co_occurrence
        self.co_occurrence_count += (batch_co_occurrence > 0).float()


    def forward(self, images, batch_data=None):
        """Forward pass"""
        batch_size = images.size(0)
        device = images.device

        # Move model to correct device if needed
        self.to(device)

        # Get global image features from ViT
        vit_features = self.vit(images)  # [batch_size, feature_dim]

        # Project ViT features to hidden dimension
        pooled_features = self.feature_pooling(vit_features)  # [batch_size, hidden_dim]

        # Extract and process region features
        if batch_data is not None:
            region_features, area_matrix = self.extract_regions(images, batch_data)
        else:
            region_features = torch.zeros(batch_size, self.num_diseases, self.feature_dim, device=device)
            area_matrix = torch.zeros(batch_size, self.num_diseases, device=device)

        # Ensure all tensors are on the correct device
        region_features = region_features.to(device)
        area_matrix = area_matrix.to(device)
        co_occurrence_count = self.co_occurrence_count.to(device)

        # Move graph constructor to correct device
        self.graph_constructor = self.graph_constructor.to(device)

        # Construct graph
        adjacency_matrix = self.graph_constructor(
            region_features,
            area_matrix,
            co_occurrence_count
        )

        # Ensure adjacency matrix is on correct device
        adjacency_matrix = adjacency_matrix.to(device)
        region_features = region_features.to(device)

        # Apply graph attention and project to hidden dimension
        graph_features = torch.matmul(adjacency_matrix, region_features)  # [batch_size, num_diseases, feature_dim]
        graph_features = self.graph_projection(graph_features)  # [batch_size, num_diseases, hidden_dim]

        # Prepare features for attention
        query = pooled_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        key = graph_features  # [batch_size, num_diseases, hidden_dim]
        value = graph_features  # [batch_size, num_diseases, hidden_dim]

        # Move attention module to correct device
        self.disease_attention = self.disease_attention.to(device)

        # Apply disease attention
        attn_output, _ = self.disease_attention(query, key, value)  # [batch_size, 1, hidden_dim]
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_dim]

        # Concatenate and fuse features
        fused_features = torch.cat([pooled_features, attn_output], dim=1)  # [batch_size, hidden_dim * 2]
        fused_features = self.fusion_module(fused_features)  # [batch_size, hidden_dim // 2]

        # Final classification
        logits = self.final_classifier(fused_features)  # [batch_size, num_diseases]

        # Update co-occurrence if in training
        if self.training and 'labels' in batch_data:
            labels = batch_data['labels'].to(device)
            self.update_co_occurrence(labels)

        return torch.sigmoid(logits)

    def get_attention_maps(self, images, batch_data=None):
        """Get attention maps for visualization"""
        self.eval()
        with torch.no_grad():
            batch_size = images.size(0)
            device = images.device

            # Get features and process as in forward pass
            vit_features = self.vit(images)
            pooled_features = self.feature_pooling(vit_features)

            if batch_data is not None:
                region_features, area_matrix = self.extract_regions(images, batch_data)
            else:
                region_features = torch.zeros(batch_size, self.num_diseases, self.feature_dim, device=device)
                area_matrix = torch.zeros(batch_size, self.num_diseases, device=device)

            adjacency_matrix = self.graph_constructor(
                region_features.to(device),
                area_matrix.to(device),
                self.co_occurrence_count.to(device)
            )

            graph_features = torch.matmul(adjacency_matrix, region_features)
            graph_features = self.graph_projection(graph_features)

            query = pooled_features.unsqueeze(1)
            key = graph_features
            value = graph_features

            # Get attention weights
            _, attention_weights = self.disease_attention(
                query, key, value,
                need_weights=True,
                average_head_weights=True
            )

            return attention_weights  # [batch_size, 1, num_diseases]


    @torch.no_grad()
    def get_graph_visualization(self):
        """Get graph adjacency matrix for visualization"""
        norm_co_occurrence = self.co_occurrence_matrix / (self.co_occurrence_count + 1e-8)
        return norm_co_occurrence.cpu().numpy()

