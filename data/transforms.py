# data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TransformFactory:
    @staticmethod
    def get_transforms(phase: str, config: dict):
        """Get appropriate transforms for each phase."""
        if phase == 'train':
            return A.Compose([
                A.RandomResizedCrop(
                    height=config['dataset']['image_size'],
                    width=config['dataset']['image_size'],
                    scale=(0.8, 1.0)
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.2
                ),
                A.Normalize(
                    mean=config['dataset']['mean'],
                    std=config['dataset']['std']
                ),
                ToTensorV2(),
            ])
        else:  # val or test
            return A.Compose([
                A.Resize(
                    height=int(config['dataset']['image_size'] * 1.14),
                    width=int(config['dataset']['image_size'] * 1.14)
                ),
                A.CenterCrop(
                    height=config['dataset']['image_size'],
                    width=config['dataset']['image_size']
                ),
                A.Normalize(
                    mean=config['dataset']['mean'],
                    std=config['dataset']['std']
                ),
                ToTensorV2(),
            ])


# data/utils.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging


class DatasetUtils:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_and_validate_data(self, phase: str) -> pd.DataFrame:
        """Load and validate dataset files."""
        csv_path = Path(self.config['dataset'][f'{phase}_csv'])
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find {phase} CSV file: {csv_path}")

        df = pd.read_csv(csv_path)
        required_columns = ['Image_Index', 'Finding Labels']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df

    def compute_class_weights(self, df: pd.DataFrame) -> torch.Tensor:
        """Compute class weights for handling imbalance."""
        findings = df['Finding Labels'].str.split('|')
        counts = {disease: 0 for disease in self.config['dataset']['classes']}

        for finding_list in findings:
            for finding in finding_list:
                if finding in counts:
                    counts[finding] += 1

        total = len(df)
        weights = torch.tensor([
            total / (counts[disease] + 1)  # Add 1 for smoothing
            for disease in self.config['dataset']['classes']
        ])

        return weights

    def validate_bounding_boxes(self, bb_df: pd.DataFrame) -> None:
        """Validate bounding box annotations."""
        required_columns = ['Image_Index', 'Finding Label', 'Bbox_X', 'Bbox_Y', 'Bbox_W', 'Bbox_H']
        missing_columns = [col for col in required_columns if col not in bb_df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns in BB annotations: {missing_columns}")

        # Validate BB coordinates
        invalid_boxes = bb_df[
            (bb_df['Bbox_X'] < 0) |
            (bb_df['Bbox_Y'] < 0) |
            (bb_df['Bbox_W'] <= 0) |
            (bb_df['Bbox_H'] <= 0)
            ]

        if len(invalid_boxes) > 0:
            self.logger.warning(f"Found {len(invalid_boxes)} invalid bounding boxes")
            self.logger.warning(invalid_boxes)

    def get_split_indices(self, df: pd.DataFrame, val_split: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices for train-val split."""
        # Use deterministic split for reproducibility
        np.random.seed(42)
        indices = np.arange(len(df))
        np.random.shuffle(indices)

        split_idx = int(len(indices) * (1 - val_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        return train_indices, val_indices

    def analyze_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset statistics."""
        stats = {
            'total_samples': len(df),
            'class_distribution': {},
            'multi_label_stats': {
                'avg_labels_per_sample': 0,
                'max_labels_per_sample': 0
            }
        }

        # Compute class distribution
        findings = df['Finding Labels'].str.split('|')
        for disease in self.config['dataset']['classes']:
            count = sum(1 for f in findings if disease in f)
            stats['class_distribution'][disease] = {
                'count': count,
                'percentage': count / len(df) * 100
            }

        # Compute multi-label statistics
        label_counts = [len(f) for f in findings]
        stats['multi_label_stats'].update({
            'avg_labels_per_sample': np.mean(label_counts),
            'max_labels_per_sample': np.max(label_counts),
            'label_count_distribution': {
                i: label_counts.count(i)
                for i in range(1, max(label_counts) + 1)
            }
        })

        return stats

