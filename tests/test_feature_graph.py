# tests/test_feature_graph.py
import unittest
import torch
import numpy as np
from src.models.feature_graph import FeatureGraphModel
from src.models.graph_modules.graph_builder import ProgressiveGraphBuilder
from src.models.graph_modules.feature_extractor import RegionFeatureExtractor


class TestFeatureGraphModel(unittest.TestCase):
    """Test cases for Feature Graph Model."""

    def setUp(self):
        """Set up test environment."""
        self.config = {
            'model': {
                'feature_dim': 768,
                'graph_hidden_dim': 256,
                'num_graph_layers': 2,
                'dropout_rate': 0.1,
                'freeze_backbone': True
            },
            'dataset': {
                'num_classes': 14,
                'image_size': 224
            },
            'training': {
                'direct_threshold': 10,
                'limited_threshold': 3,
                'confidence_gamma': 0.8
            }
        }

        self.model = FeatureGraphModel(self.config)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.backbone)
        self.assertIsNotNone(self.model.feature_projector)
        self.assertIsNotNone(self.model.classifier)

    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        bboxes = {
            'boxes': [torch.tensor([[10, 10, 50, 50]])] * batch_size,
            'labels': [torch.tensor([0])] * batch_size,
            'areas': [torch.tensor([2500.0])] * batch_size
        }

        output = self.model(images, bboxes)
        self.assertEqual(output.shape, (batch_size, 14))

    def test_feature_extraction(self):
        """Test feature extraction from regions."""
        images = torch.randn(2, 3, 224, 224)
        boxes = torch.tensor([[10, 10, 50, 50]])

        features = self.model.region_extractor._process_region(
            images[0], boxes[0], torch.tensor([2500.0])
        )

        self.assertEqual(features.shape[1], self.config['model']['graph_hidden_dim'])


class TestGraphBuilder(unittest.TestCase):
    """Test cases for Graph Builder."""

    def setUp(self):
        """Set up test environment."""
        self.config = {
            'training': {
                'direct_threshold': 10,
                'limited_threshold': 3,
                'confidence_gamma': 0.8
            },
            'dataset': {
                'num_classes': 14
            }
        }

        self.graph_builder = ProgressiveGraphBuilder(self.config)

    def test_direct_relationships(self):
        """Test computation of direct relationships."""
        features = {
            0: torch.randn(15, 256),
            1: torch.randn(15, 256)
        }
        regions = {
            0: list(range(15)),
            1: list(range(15))
        }

        weights = self.graph_builder._compute_direct_relationships(
            features, regions
        )

        self.assertIn((0, 1), weights)
        self.assertTrue(0 <= weights[(0, 1)] <= 1)

    def test_weight_normalization(self):
        """Test graph weight normalization."""
        weights = self.graph_builder(
            features={0: torch.randn(5, 256)},
            regions_by_disease={0: list(range(5))},
            labels=torch.zeros(5, 14)
        )

        self.assertEqual(weights.shape, (14, 14))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(14)))

