# src/models/graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    def __init__(
            self,
            feature_dim: int,  # This should now be 512
            hidden_dim: int,  # This stays 512
            num_diseases: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_diseases = num_diseases


        # Project input features
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # Graph attention components
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)

        # Edge weighting
        self.edge_weight = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Output transformation
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,  # [batch_size, num_diseases, feature_dim]
            area_matrix: torch.Tensor,  # [batch_size, num_diseases]
            co_occurrence: torch.Tensor  # [num_diseases, num_diseases]
    ) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        # Project for attention
        q = self.query_proj(x)  # [B, N, H]
        k = self.key_proj(x)  # [B, N, H]
        v = self.value_proj(x)  # [B, N, H]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)

        # Edge weights
        edge_inputs = torch.stack([
            area_matrix.unsqueeze(-1).expand(-1, -1, self.num_diseases),
            co_occurrence.unsqueeze(0).expand(batch_size, -1, -1)
        ], dim=-1)

        edge_weights = self.edge_weight(edge_inputs).squeeze(-1)
        attn_scores = attn_scores * edge_weights

        # Attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        out = torch.matmul(attn_probs, v)
        out = self.norm1(out)

        # Final transformation with residual connection
        out = self.output_transform(out)
        out = self.norm2(out + x)  # Residual connection with input (already in correct dim)

        return out




