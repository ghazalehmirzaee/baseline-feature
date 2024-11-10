# src/models/graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    """
    Graph neural network layer for disease relationship modeling
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            num_diseases: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_diseases = num_diseases

        # Graph attention components
        self.query_proj = nn.Linear(in_dim, hidden_dim)
        self.key_proj = nn.Linear(in_dim, hidden_dim)
        self.value_proj = nn.Linear(in_dim, hidden_dim)

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

        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            area_matrix: torch.Tensor,
            co_occurrence: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [batch_size, num_diseases, in_dim]
            area_matrix: Region areas [batch_size, num_diseases]
            co_occurrence: Disease co-occurrence matrix [num_diseases, num_diseases]
        """
        batch_size = x.size(0)
        device = x.device

        # Project inputs
        q = self.query_proj(x)  # [B, N, H]
        k = self.key_proj(x)  # [B, N, H]
        v = self.value_proj(x)  # [B, N, H]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [B, N, N]

        # Compute edge weights based on area and co-occurrence
        edge_inputs = torch.stack([
            area_matrix.unsqueeze(-1).expand(-1, -1, self.num_diseases),
            co_occurrence.unsqueeze(0).expand(batch_size, -1, -1)
        ], dim=-1)  # [B, N, N, 2]

        edge_weights = self.edge_weight(edge_inputs).squeeze(-1)  # [B, N, N]

        # Apply edge weights to attention scores
        attn_scores = attn_scores * edge_weights

        # Normalize attention scores
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        out = torch.matmul(attn_probs, v)  # [B, N, H]
        out = self.norm1(out + x)  # Residual connection

        # Output transformation
        out = self.output_transform(out)
        out = self.norm2(out)

        return out


