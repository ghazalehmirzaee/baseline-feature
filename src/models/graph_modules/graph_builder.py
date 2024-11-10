# src/models/graph_modules/graph_builder.py
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class ProgressiveGraphBuilder:
    """Progressive graph construction following the three-tier approach."""

    def __init__(self, config: Dict):
        self.config = config
        self.direct_threshold = config['training']['direct_threshold']
        self.limited_threshold = config['training']['limited_threshold']
        self.gamma = config['training']['confidence_gamma']
        self.num_classes = config['dataset']['num_classes']

        # Initialize graph weights
        self.weights = torch.eye(self.num_classes)

    def __call__(self, features: Dict[int, torch.Tensor],
                 regions_by_disease: Dict[int, List[int]],
                 labels: torch.Tensor) -> torch.Tensor:
        """Build the graph using the progressive framework."""
        # Compute direct relationships
        direct_weights = self._compute_direct_relationships(
            features, regions_by_disease
        )

        # Compute limited sample relationships
        limited_weights = self._compute_limited_relationships(
            features, regions_by_disease, direct_weights
        )

        # Estimate missing relationships
        estimated_weights = self._estimate_relationships(
            direct_weights, limited_weights
        )

        # Combine all weights
        graph_weights = torch.zeros(
            (self.num_classes, self.num_classes),
            device=labels.device
        )

        # Fill in weights based on relationship type
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if (i, j) in direct_weights:
                    graph_weights[i, j] = direct_weights[(i, j)]
                elif (i, j) in limited_weights:
                    graph_weights[i, j] = limited_weights[(i, j)]
                elif (i, j) in estimated_weights:
                    graph_weights[i, j] = estimated_weights[(i, j)]

        # Normalize weights
        graph_weights = F.normalize(graph_weights, p=1, dim=1)
        self.weights = graph_weights

        return graph_weights

    def _compute_direct_relationships(self,
                                      features: Dict[int, torch.Tensor],
                                      regions_by_disease: Dict[int, List[int]]) -> Dict[Tuple[int, int], float]:
        """Compute relationships for disease pairs with sufficient samples."""
        weights = {}

        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                # Get common samples
                if i not in regions_by_disease or j not in regions_by_disease:
                    continue

                common_samples = set(regions_by_disease[i]) & set(regions_by_disease[j])
                if len(common_samples) > self.direct_threshold:
                    # Compute similarity
                    feat_i = features[i]
                    feat_j = features[j]
                    sim = F.cosine_similarity(feat_i.mean(0), feat_j.mean(0))
                    weights[(i, j)] = weights[(j, i)] = sim.item()

        return weights

    def _compute_limited_relationships(self,
                                       features: Dict[int, torch.Tensor],
                                       regions_by_disease: Dict[int, List[int]],
                                       direct_weights: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """Compute relationships for disease pairs with limited samples."""
        weights = {}

        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                if i not in regions_by_disease or j not in regions_by_disease:
                    continue

                common_samples = set(regions_by_disease[i]) & set(regions_by_disease[j])
                if self.limited_threshold <= len(common_samples) <= self.direct_threshold:
                    # Compute beta
                    beta = (len(common_samples) / self.direct_threshold)

                    # Direct component
                    feat_i = features[i]
                    feat_j = features[j]
                    direct_sim = F.cosine_similarity(feat_i.mean(0), feat_j.mean(0))

                    # Average component
                    avg_sim = self._compute_average_similarity(i, j, direct_weights)

                    # Combine
                    weight = beta * direct_sim.item() + (1 - beta) * avg_sim
                    weights[(i, j)] = weights[(j, i)] = weight

        return weights

    def _estimate_relationships(self,
                                direct_weights: Dict[Tuple[int, int], float],
                                limited_weights: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """Estimate relationships for disease pairs without sufficient samples."""
        weights = {}
        all_weights = {**direct_weights, **limited_weights}

        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                if (i, j) not in all_weights:
                    # Find neighbors
                    i_neighbors = self._get_neighbors(i, all_weights)
                    j_neighbors = self._get_neighbors(j, all_weights)

                    if i_neighbors and j_neighbors:
                        # Compute transitive relationships
                        relations = []
                        for ni in i_neighbors:
                            for nj in j_neighbors:
                                if (ni, nj) in all_weights:
                                    relations.append(all_weights[(ni, nj)])

                        if relations:
                            weight = self.gamma * np.mean(relations)
                            weights[(i, j)] = weights[(j, i)] = weight

        return weights

    def _compute_average_similarity(self,
                                    i: int,
                                    j: int,
                                    weights: Dict[Tuple[int, int], float]) -> float:
        """Compute average similarity from similar disease pairs."""
        similar_weights = []
        for (di, dj), w in weights.items():
            if di == i or di == j or dj == i or dj == j:
                similar_weights.append(w)
        return np.mean(similar_weights) if similar_weights else 0.0

    def _get_neighbors(self,
                       node: int,
                       weights: Dict[Tuple[int, int], float]) -> List[int]:
        """Get neighboring nodes with direct connections."""
        neighbors = set()
        for (i, j) in weights.keys():
            if i == node:
                neighbors.add(j)
            elif j == node:
                neighbors.add(i)
        return list(neighbors)

    def get_weights(self) -> torch.Tensor:
        """Get current graph weights."""
        return self.weights

    def set_weights(self, weights: torch.Tensor):
        """Set pre-computed weights."""
        self.weights = weights

