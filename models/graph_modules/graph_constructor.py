# models/graph_modules/graph_constructor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveGraphConstructor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.direct_threshold = config.direct_threshold
        self.limited_threshold = config.limited_threshold
        self.confidence_gamma = config.confidence_gamma

    def compute_direct_relationships(self, features, areas, sample_pairs):
        """Compute relationships for pairs with sufficient samples."""
        weights = {}

        for (i, j), samples in sample_pairs.items():
            if len(samples) > self.direct_threshold:
                # Compute area-based weights
                pair_areas = torch.stack([areas[idx] for idx in samples])
                min_areas = torch.min(pair_areas, dim=1)[0]
                alpha = min_areas / min_areas.sum()

                # Compute similarities
                feat_i = features[samples, i]
                feat_j = features[samples, j]
                similarities = F.cosine_similarity(feat_i, feat_j)

                # Weighted sum
                weights[(i, j)] = (similarities * alpha).sum().item()

        return weights

    def compute_limited_relationships(self, features, areas, sample_pairs, direct_weights):
        """Compute relationships for pairs with limited samples."""
        weights = {}

        for (i, j), samples in sample_pairs.items():
            if self.limited_threshold <= len(samples) <= self.direct_threshold:
                # Compute beta
                beta = (len(samples) / self.direct_threshold) * (
                        min(areas[i].mean(), areas[j].mean()) /
                        max(areas[i].mean(), areas[j].mean())
                )

                # Compute direct component
                feat_i = features[samples, i]
                feat_j = features[samples, j]
                direct_sim = F.cosine_similarity(feat_i, feat_j).mean().item()

                # Find similar pairs
                similar_pairs = self._find_similar_pairs(i, j, direct_weights)
                avg_sim = sum(direct_weights[p] for p in similar_pairs) / len(similar_pairs)

                weights[(i, j)] = beta * direct_sim + (1 - beta) * avg_sim

        return weights

    def compute_estimated_relationships(self, direct_weights, limited_weights):
        """Estimate relationships for pairs without sufficient samples."""
        weights = {}
        all_weights = {**direct_weights, **limited_weights}

        for i in range(14):
            for j in range(i + 1, 14):
                if (i, j) not in all_weights:
                    # Find neighboring relationships
                    i_neighbors = self._get_neighbors(i, all_weights)
                    j_neighbors = self._get_neighbors(j, all_weights)

                    if i_neighbors and j_neighbors:
                        numerator = 0
                        denominator = 0

                        for ni in i_neighbors:
                            for nj in j_neighbors:
                                if (ni, nj) in all_weights:
                                    conf = self._compute_confidence(ni, nj, all_weights)
                                    numerator += all_weights[(ni, nj)] * conf
                                    denominator += conf

                        if denominator > 0:
                            weights[(i, j)] = self.confidence_gamma * (numerator / denominator)

        return weights

    def _find_similar_pairs(self, i, j, direct_weights):
        """Find similar disease pairs based on connection patterns."""
        similar_pairs = []
        i_connections = set(k for (k, l) in direct_weights.keys() if k == i or l == i)
        j_connections = set(k for (k, l) in direct_weights.keys() if k == j or l == j)

        for pair in direct_weights.keys():
            if pair[0] in i_connections and pair[1] in j_connections:
                similar_pairs.append(pair)

        return similar_pairs

    def _get_neighbors(self, node, weights):
        """Get neighboring nodes with direct connections."""
        neighbors = set()
        for (i, j) in weights.keys():
            if i == node:
                neighbors.add(j)
            elif j == node:
                neighbors.add(i)
        return neighbors

    def _compute_confidence(self, i, j, weights):
        """Compute confidence score based on sample count."""
        pair = (min(i, j), max(i, j))
        if pair not in weights:
            return 0.4  # minimum confidence

        weight = weights[pair]
        if weight > 0.8:
            return 1.0
        elif weight > 0.6:
            return 0.8
        elif weight > 0.4:
            return 0.6
        else:
            return 0.4

    def forward(self, features, areas, sample_pairs):
        """Construct the complete graph using the progressive framework."""
        # Step 1: Direct relationships
        direct_weights = self.compute_direct_relationships(features, areas, sample_pairs)

        # Step 2: Limited sample relationships
        limited_weights = self.compute_limited_relationships(
            features, areas, sample_pairs, direct_weights
        )

        # Step 3: Estimated relationships
        estimated_weights = self.compute_estimated_relationships(
            direct_weights, limited_weights
        )

        # Combine all weights into adjacency matrix
        adj_matrix = torch.zeros(14, 14)
        for (i, j), w in {**direct_weights, **limited_weights, **estimated_weights}.items():
            adj_matrix[i, j] = adj_matrix[j, i] = w

        return adj_matrix

