# src/models/graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            hidden_dim: int,
            num_diseases: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_diseases = num_diseases

        # Add input projection layer
        self.input_proj = nn.Linear(feature_dim, hidden_dim)  # Add this line

        # Rest of the initialization remains the same
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)

        self.edge_weight = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

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
            x: torch.Tensor,
            area_matrix: torch.Tensor,
            co_occurrence: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        # Reshape input if needed
        if x.shape[-1] != self.feature_dim:
            x = x.transpose(-1, -2)

        # Project input for residual connection
        x_proj = self.input_proj(x)  # Add this line

        # Rest of the forward pass remains the same
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)

        edge_inputs = torch.stack([
            area_matrix.unsqueeze(-1).expand(-1, -1, self.num_diseases),
            co_occurrence.unsqueeze(0).expand(batch_size, -1, -1)
        ], dim=-1)

        edge_weights = self.edge_weight(edge_inputs).squeeze(-1)
        attn_scores = attn_scores * edge_weights
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = self.norm1(out)
        out = self.output_transform(out)
        out = self.norm2(out + x_proj)  # Now x_proj is defined

        return out

