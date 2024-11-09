# models/feature_graph.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from .graph_modules.feature_extractor import RegionFeatureExtractor
from .graph_modules.graph_constructor import ProgressiveGraphConstructor
from .graph_modules.graph_fusion import ConcatMLPFusion


class FeatureGraphModel(nn.Module):
    def __init__(self, config, baseline_model):
        super().__init__()
        self.config = config

        # Initialize components
        self.feature_extractor = RegionFeatureExtractor(baseline_model)
        self.graph_constructor = ProgressiveGraphConstructor(config.graph_construction)

        # Graph neural network layers
        self.graph_layers = nn.ModuleList([
            GCNConv(config.model.feature_dim, config.model.graph_hidden_dim),
            GCNConv(config.model.graph_hidden_dim, config.model.graph_hidden_dim)
        ])

        # Fusion module
        self.fusion = ConcatMLPFusion(
            feature_dim=config.model.feature_dim,
            graph_dim=config.model.graph_hidden_dim,
            hidden_dim=config.model.graph_hidden_dim,
            num_classes=14
        )

        self.dropout = nn.Dropout(config.model.dropout_rate)

    def forward(self, images, bboxes=None):
        # Get baseline features
        baseline_features = self.feature_extractor.vit(images)

        if self.training and bboxes is not None:
            # Extract region features and construct graph
            region_features, areas = self.feature_extractor.extract_region_features(images, bboxes)
            adj_matrix = self.graph_constructor(region_features, areas, self._get_sample_pairs(bboxes))

            # Apply GNN layers
            x = region_features
            for layer in self.graph_layers:
                x = layer(x, adj_matrix)
                x = torch.relu(x)
                x = self.dropout(x)

            graph_embeddings = x
        else:
            # During inference or when no BBs available, use learned average embeddings
            graph_embeddings = self.graph_layers[-1].weight.mean(0).expand(len(images), -1)

        # Fuse features and get predictions
        predictions = self.fusion(baseline_features, graph_embeddings)
        return predictions

    def _get_sample_pairs(self, bboxes):
        """Create dictionary of disease pairs and their sample indices."""
        pairs = {}
        for idx, bb in enumerate(bboxes):
            diseases = bb["labels"]
            for i in range(len(diseases)):
                for j in range(i + 1, len(diseases)):
                    pair = (min(diseases[i], diseases[j]), max(diseases[i], diseases[j]))
                    if pair not in pairs:
                        pairs[pair] = []
                    pairs[pair].append(idx)
        return pairs

