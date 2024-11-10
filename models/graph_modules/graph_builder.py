# models/graph_modules/graph_builder.py

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any


class ProgressiveGraphBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.direct_threshold = 10
        self.limited_threshold = 3
        self.gamma = 0.8
        self.num_classes = config['dataset']['num_classes']
        self.current_weights = None

    def compute_direct_relationships(self, features: Dict[int, torch.Tensor],
                                     areas: Dict[int, torch.Tensor],
                                     sample_pairs: Dict[Tuple[int, int], List[int]]) -> Dict[Tuple[int, int], float]:
        """Compute relationships for disease pairs with sufficient samples."""
        weights = {}
        for (i, j), samples in sample_pairs.items():
            if len(samples) > self.direct_threshold:
                # Area-based weighting
                pair_areas = torch.stack([areas[k] for k in samples])
                min_areas = torch.min(pair_areas, dim=1)[0]
                alpha = min_areas / min_areas.sum()

                # Feature similarity
                feat_i = features[i][samples]
                feat_j = features[j][samples]
                similarities = F.cosine_similarity(feat_i, feat_j)

                weights[(i, j)] = (similarities * alpha).sum().item()
        return weights

    def compute_limited_relationships(self, features: Dict[int, torch.Tensor],
                                      areas: Dict[int, torch.Tensor],
                                      sample_pairs: Dict[Tuple[int, int], List[int]],
                                      direct_weights: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """Compute relationships for disease pairs with limited samples."""
        weights = {}
        for (i, j), samples in sample_pairs.items():
            if self.limited_threshold <= len(samples) <= self.direct_threshold:
                # Compute beta
                beta = (len(samples) / self.direct_threshold) * (
                        min(areas[i].mean(), areas[j].mean()) /
                        max(areas[i].mean(), areas[j].mean())
                )

                # Direct component
                feat_i = features[i][samples]
                feat_j = features[j][samples]
                direct_sim = F.cosine_similarity(feat_i, feat_j).mean().item()

                # Find similar pairs
                similar_pairs = self._find_similar_pairs(i, j, direct_weights)
                if similar_pairs:
                    avg_sim = sum(direct_weights[p] for p in similar_pairs) / len(similar_pairs)
                    weights[(i, j)] = beta * direct_sim + (1 - beta) * avg_sim
                else:
                    weights[(i, j)] = direct_sim

        return weights

    def compute_estimated_relationships(self, direct_weights: Dict[Tuple[int, int], float],
                                        limited_weights: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """Estimate relationships for disease pairs without direct samples."""
        weights = {}
        all_weights = {**direct_weights, **limited_weights}

        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                if (i, j) not in all_weights:
                    neighbors_i = self._get_neighbors(i, all_weights)
                    neighbors_j = self._get_neighbors(j, all_weights)

                    numerator = 0.0
                    denominator = 0.0
                    for ni in neighbors_i:
                        for nj in neighbors_j:
                            if (ni, nj) in all_weights:
                                conf = self._compute_confidence(ni, nj, all_weights)
                                numerator += all_weights[(ni, nj)] * conf
                                denominator += conf

                    if denominator > 0:
                        weights[(i, j)] = self.gamma * (numerator / denominator)

        return weights

    def _find_similar_pairs(self, i: int, j: int, weights: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
        """Find similar disease pairs based on connection patterns."""
        similar_pairs = []
        for (p1, p2) in weights.keys():
            if p1 == i or p2 == i or p1 == j or p2 == j:
                similar_pairs.append((p1, p2))
        return similar_pairs

    def _get_neighbors(self, node: int, weights: Dict[Tuple[int, int], float]) -> List[int]:
        """Get neighboring nodes with direct connections."""
        neighbors = set()
        for (i, j) in weights.keys():
            if i == node:
                neighbors.add(j)
            elif j == node:
                neighbors.add(i)
        return list(neighbors)

    def _compute_confidence(self, i: int, j: int, weights: Dict[Tuple[int, int], float]) -> float:
        """Compute confidence score for a relationship."""
        weight = weights.get((min(i, j), max(i, j)), 0.0)
        if weight > 0.8:
            return 1.0
        elif weight > 0.6:
            return 0.8
        elif weight > 0.4:
            return 0.6
        else:
            return 0.4

    def build_graph(self, features: Dict[int, torch.Tensor], areas: Dict[int, torch.Tensor],
                    sample_pairs: Dict[Tuple[int, int], List[int]]) -> torch.Tensor:
        """Build the complete graph using the progressive framework."""
        # Compute relationships at each tier
        direct_weights = self.compute_direct_relationships(features, areas, sample_pairs)
        limited_weights = self.compute_limited_relationships(features, areas, sample_pairs, direct_weights)
        estimated_weights = self.compute_estimated_relationships(direct_weights, limited_weights)

        # Combine all weights into adjacency matrix
        adj_matrix = torch.zeros(self.num_classes, self.num_classes)
        for (i, j), w in {**direct_weights, **limited_weights, **estimated_weights}.items():
            adj_matrix[i, j] = adj_matrix[j, i] = w

        # Normalize matrix
        adj_matrix = F.normalize(adj_matrix, p=1, dim=1)

        # Store current weights
        self.current_weights = adj_matrix

        return adj_matrix

    def get_current_weights(self) -> torch.Tensor:
        """Get the current graph weights."""
        return self.current_weights if self.current_weights is not None else torch.eye(self.num_classes)

    def set_weights(self, weights: torch.Tensor):
        """Set pre-computed graph weights."""
        if weights.shape == (self.num_classes, self.num_classes):
            self.current_weights = weights
        else:
            raise ValueError(f"Invalid weights shape: {weights.shape}")

