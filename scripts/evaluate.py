# scripts/evaluate.py
import os

import torch
import json
import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from data.datasets import get_data_loaders
from models.feature_graph import FeatureGraphModel
from utils.metrics import compute_metrics
from utils.visualization import VisualizationManager
from utils.logger import setup_logging


class Evaluator:
    def __init__(self, config_path, checkpoint_path, output_dir):
        self.config = self._load_config(config_path)
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = setup_logging("evaluation")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.visualizer = VisualizationManager(self.output_dir)

        self._setup_model()

    def _load_config(self, config_path):
        with open(config_path) as f:
            return json.load(f)

    def _setup_model(self):
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)

        # Initialize model
        self.model = FeatureGraphModel(self.config, None).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def evaluate(self, test_loader):
        """Perform comprehensive evaluation on test set."""
        self.logger.info("Starting evaluation...")

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets, bboxes in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                predictions = self.model(images, bboxes)

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        # Compute metrics
        metrics = compute_metrics(predictions, targets)

        # Save metrics
        metrics_path = self.output_dir / 'evaluation_results.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Generate visualizations
        self.logger.info("Generating visualizations...")
        self.visualizer.plot_roc_curves(predictions, targets)
        self.visualizer.plot_pr_curves(predictions, targets)
        self.visualizer.plot_confusion_matrices(predictions, targets)

        # Plot graph attention if available
        if hasattr(self.model, 'graph_constructor') and \
                hasattr(self.model.graph_constructor, 'last_adj_matrix'):
            self.visualizer.plot_graph_attention(
                self.model.graph_constructor.last_adj_matrix.cpu().numpy()
            )

        # Analyze performance on rare vs common diseases
        self._analyze_disease_groups(metrics)

        return metrics

    def _analyze_disease_groups(self, metrics):
        """Analyze performance across different disease frequency groups."""
        disease_groups = {
            'high_freq': ['Infiltration', 'Effusion', 'Atelectasis'],
            'medium_freq': ['Mass', 'Nodule', 'Pneumothorax', 'Consolidation',
                            'Pleural_Thickening', 'Cardiomegaly'],
            'low_freq': ['Edema', 'Emphysema', 'Fibrosis', 'Pneumonia', 'Hernia']
        }

        group_metrics = {}
        for group_name, diseases in disease_groups.items():
            group_metrics[group_name] = {
                'mean_auc': np.mean([metrics[f'{d}_auc'] for d in diseases]),
                'mean_ap': np.mean([metrics[f'{d}_ap'] for d in diseases]),
                'mean_f1': np.mean([metrics[f'{d}_f1'] for d in diseases])
            }

        # Save group analysis
        analysis_path = self.output_dir / 'disease_group_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(group_metrics, f, indent=4)

        return group_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update config with data paths
    config['dataset'].update({
        'test_csv': os.path.join(args.data_dir, 'test_list.txt'),
        'image_dir': os.path.join(args.data_dir, 'images'),
        'bb_annotations': os.path.join(args.data_dir, 'BBox_List_2017.csv')
    })

    # Create test data loader
    _, _, test_loader = get_data_loaders(config)

    # Initialize evaluator and run evaluation
    evaluator = Evaluator(args.config, args.checkpoint, args.output_dir)
    metrics = evaluator.evaluate(test_loader)

    print("\nEvaluation Results:")
    print(f"Mean AUC: {metrics['mean_auc']:.4f}")
    print(f"Mean AP: {metrics['mean_ap']:.4f}")
    print(f"Mean F1: {metrics['mean_f1']:.4f}")
    print(f"Exact Match: {metrics['exact_match']:.4f}")

