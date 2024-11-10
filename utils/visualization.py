# utils/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict


class GraphVisualizer:
    def __init__(self, disease_names: List[str]):
        self.disease_names = disease_names

    def plot_graph_weights(self, weights: torch.Tensor, save_path: str = None):
        """Plot graph weights as a heatmap."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            weights.cpu().numpy(),
            xticklabels=self.disease_names,
            yticklabels=self.disease_names,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f'
        )
        plt.title('Disease Relationship Graph Weights')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_relationship_distribution(self, weights: torch.Tensor, save_path: str = None):
        """Plot distribution of relationship weights."""
        weights_flat = weights[torch.triu_indices(14, 14, offset=1)].cpu().numpy()

        plt.figure(figsize=(10, 6))
        sns.histplot(weights_flat, bins=30)
        plt.title('Distribution of Disease Relationship Weights')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class GraphMetrics:
    @staticmethod
    def compute_graph_statistics(weights: torch.Tensor) -> Dict[str, float]:
        """Compute various statistics about the graph structure."""
        # Convert to numpy for calculations
        W = weights.cpu().numpy()

        stats = {
            'mean_weight': float(np.mean(W)),
            'std_weight': float(np.std(W)),
            'max_weight': float(np.max(W)),
            'min_weight': float(np.min(W)),
            'sparsity': float(np.sum(W == 0) / W.size),
            'symmetry_error': float(np.mean(np.abs(W - W.T))),
        }

        return stats

    @staticmethod
    def evaluate_graph_quality(weights: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate graph quality using label correlations."""
        # Compute label correlations
        label_corr = torch.corrcoef(labels.T)

        # Compare with graph weights
        mse = torch.mean((weights - label_corr) ** 2).item()
        mae = torch.mean(torch.abs(weights - label_corr)).item()

        return {
            'weight_label_mse': mse,
            'weight_label_mae': mae,
        }

