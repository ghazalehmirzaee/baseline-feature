
# models/graph_modules/graph_fusion.py
import torch
import torch.nn as nn


class ConcatMLPFusion(nn.Module):
    def __init__(self, feature_dim, graph_dim, hidden_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features, graph_embeddings):
        combined = torch.cat([features, graph_embeddings], dim=1)
        return self.mlp(combined)

