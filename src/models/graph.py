# src/models/graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProgressiveGraphConstruction(nn.Module):
    def __init__(self, num_diseases=14, feature_dim=768):
        super().__init__()
        self.num_diseases = num_diseases
        self.feature_dim = feature_dim

        # Learnable parameters
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, 8, dropout=0.1)


    def compute_area_weights(self, areas_i, areas_j):
        """Compute area-based weights for feature relationships"""
        min_areas = torch.minimum(areas_i, areas_j)
        weights = min_areas / min_areas.sum()
        return weights

    def compute_cosine_similarity(self, features_i, features_j):
        """Compute cosine similarity between feature vectors"""
        return F.cosine_similarity(features_i, features_j, dim=-1)

    def direct_relationships(self, features, areas, co_occurrence_mask):
        """Compute direct relationships for disease pairs with sufficient samples"""
        batch_size = features.size(0)
        direct_weights = torch.zeros((self.num_diseases, self.num_diseases))

        # Project features
        features = self.feature_projection(features)

        for i in range(self.num_diseases):
            for j in range(self.num_diseases):
                if co_occurrence_mask[i, j] > 10:  # Sufficient samples
                    # Get features for disease pair
                    feat_i = features[areas[:, i] > 0]
                    feat_j = features[areas[:, j] > 0]

                    if len(feat_i) > 0 and len(feat_j) > 0:
                        # Compute area weights
                        area_weights = self.compute_area_weights(
                            areas[areas[:, i] > 0][:, i],
                            areas[areas[:, j] > 0][:, j]
                        )

                        # Compute similarity
                        sim = self.compute_cosine_similarity(feat_i, feat_j)
                        direct_weights[i, j] = (area_weights * sim).sum()

        return direct_weights

    def limited_relationships(self, features, areas, co_occurrence_mask, direct_weights):
        """Compute relationships for disease pairs with limited samples"""
        limited_weights = torch.zeros_like(direct_weights)

        for i in range(self.num_diseases):
            for j in range(self.num_diseases):
                if 3 <= co_occurrence_mask[i, j] <= 10:
                    # Compute beta based on areas and sample count
                    beta = (co_occurrence_mask[i, j] / 10.0) * (
                            torch.min(areas[:, i].sum(), areas[:, j].sum()) /
                            torch.max(areas[:, i].sum(), areas[:, j].sum())
                    )

                    # Get similar disease pairs
                    similar_weights = []
                    for k in range(self.num_diseases):
                        for l in range(self.num_diseases):
                            if direct_weights[k, l] > 0:
                                similar_weights.append(direct_weights[k, l])

                    if similar_weights:
                        avg_weight = torch.stack(similar_weights).mean()
                        limited_weights[i, j] = beta * direct_weights[i, j] + (1 - beta) * avg_weight

        return limited_weights

    def estimate_missing_relationships(self, direct_weights, limited_weights, co_occurrence_mask):
        """Estimate relationships for disease pairs without samples"""
        estimated_weights = torch.zeros_like(direct_weights)

        for i in range(self.num_diseases):
            for j in range(self.num_diseases):
                if co_occurrence_mask[i, j] < 3:
                    # Find neighbors with direct relationships
                    neighbors_i = (direct_weights[i] > 0).nonzero().squeeze()
                    neighbors_j = (direct_weights[j] > 0).nonzero().squeeze()

                    if len(neighbors_i) > 0 and len(neighbors_j) > 0:
                        weights = []
                        confidences = []

                        for ni in neighbors_i:
                            for nj in neighbors_j:
                                w = direct_weights[ni, nj]
                                if w > 0:
                                    # Compute confidence based on sample count
                                    conf = 1.0 if co_occurrence_mask[ni, nj] > 15 else \
                                        0.8 if co_occurrence_mask[ni, nj] > 10 else \
                                            0.6 if co_occurrence_mask[ni, nj] > 5 else 0.4

                                    weights.append(w * conf)
                                    confidences.append(conf)

                        if weights:
                            weights = torch.stack(weights)
                            confidences = torch.tensor(confidences)
                            estimated_weights[i, j] = self.gamma * (
                                    (weights * confidences).sum() / confidences.sum()
                            )

        return estimated_weights

    def forward(self, region_features, area_matrix, co_occurrence_count):
        device = region_features.device
        # Ensure inputs are on the same device
        region_features = region_features.to(device)
        area_matrix = area_matrix.to(device)
        co_occurrence_count = co_occurrence_count.to(device)

        # Project features
        region_features = self.feature_projection(region_features.float())

        # Compute similarity matrix
        batch_size = region_features.size(0)
        adjacency_matrix = torch.zeros(batch_size, self.num_diseases, self.num_diseases, device=device)

        for b in range(batch_size):
            # Compute pairwise similarities
            features = region_features[b]  # [num_diseases, feature_dim]
            areas = area_matrix[b]  # [num_diseases]

            # Normalize features
            norm_features = F.normalize(features, p=2, dim=1)

            # Compute similarity
            sim_matrix = torch.matmul(norm_features, norm_features.t())

            # Apply area weights
            area_weights = torch.outer(areas, areas)
            area_weights = area_weights / (area_weights.max() + 1e-8)

            # Combine with co-occurrence information
            co_occurrence_weights = co_occurrence_count / (co_occurrence_count.sum() + 1e-8)

            # Final adjacency matrix
            adjacency_matrix[b] = sim_matrix * area_weights * co_occurrence_weights

        # Normalize adjacency matrix
        adjacency_matrix = F.softmax(adjacency_matrix, dim=-1)

        return adjacency_matrix

