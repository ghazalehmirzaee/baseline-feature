# src/models/fusion.py

import torch
import torch.nn as nn


class FeatureGraphFusion(nn.Module):
    def __init__(self, vit_dim=768, num_diseases=14, hidden_dim=512):
        super().__init__()

        self.vit_dim = vit_dim
        self.num_diseases = num_diseases

        # Feature projection layers
        self.vit_projection = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.graph_projection = nn.Sequential(
            nn.Linear(num_diseases, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Fusion layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_diseases)
        )

        # Disease-specific attention
        self.disease_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1)

    def forward(self, vit_features, graph_features):
        """
        Fuse ViT features with graph information

        Args:
            vit_features: Features from ViT (batch_size, vit_dim)
            graph_features: Features from graph (batch_size, num_diseases)

        Returns:
            Fused predictions (batch_size, num_diseases)
        """
        # Project features
        vit_proj = self.vit_projection(vit_features)
        graph_proj = self.graph_projection(graph_features)

        # Apply disease-specific attention
        attn_output, _ = self.disease_attention(
            vit_proj.unsqueeze(0),
            graph_proj.unsqueeze(0),
            graph_proj.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)

        # Concatenate features
        fused_features = torch.cat([attn_output, graph_proj], dim=1)

        # Final prediction
        predictions = self.fusion_mlp(fused_features)
        predictions = torch.sigmoid(predictions)

        return predictions

