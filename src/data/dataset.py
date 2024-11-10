# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import logging
from typing import Dict, Tuple, Optional


class ChestXrayDataset(Dataset):
    """Dataset class for NIH ChestX-ray14 dataset with bounding box annotations."""

    def __init__(self,
                 csv_path: str,
                 image_dir: str,
                 bb_file: Optional[str] = None,
                 transform=None,
                 phase: str = 'train'):
        """
        Args:
            csv_path: Path to label CSV file
            image_dir: Directory containing images
            bb_file: Path to bounding box annotation file
            transform: Image transform pipeline
            phase: train/val/test
        """
        self.logger = logging.getLogger(__name__)
        self.phase = phase
        self.transform = transform

        # Load label data
        self.df = self._load_labels(csv_path)
        self.image_dir = Path(image_dir)

        # Load bounding box annotations if provided
        self.bb_data = self._load_bb_annotations(bb_file) if bb_file else None

        # Disease names
        self.disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def _load_labels(self, csv_path: str) -> pd.DataFrame:
        """Load and process label data."""
        try:
            df = pd.read_csv(csv_path, delimiter=' ', header=None)
            # Rename columns
            df.columns = ['Image_Index'] + [f'label_{i}' for i in range(14)]
            self.logger.info(f"Loaded {len(df)} samples from {csv_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            raise

    def _load_bb_annotations(self, bb_file: str) -> Dict:
        """Load and process bounding box annotations."""
        try:
            bb_df = pd.read_csv(bb_file)
            # Group by image for faster access
            bb_data = {}
            for _, row in bb_df.iterrows():
                img_id = row['Image Index']
                if img_id not in bb_data:
                    bb_data[img_id] = []
                bb_data[img_id].append({
                    'label': row['Finding Label'],
                    'x': float(row['Bbox_x']),
                    'y': float(row['Bbox_y']),
                    'w': float(row['Bbox_w']),
                    'h': float(row['Bbox_h'])
                })
            self.logger.info(f"Loaded {len(bb_df)} bounding box annotations")
            return bb_data
        except Exception as e:
            self.logger.warning(f"Error loading bounding box annotations: {e}")
            return None

    def _process_bboxes(self, image_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Process bounding boxes for an image."""
        if not self.bb_data or image_id not in self.bb_data:
            return None

        boxes = []
        labels = []
        areas = []

        for bb in self.bb_data[image_id]:
            try:
                # Get disease index
                if bb['label'] not in self.disease_names:
                    continue
                disease_idx = self.disease_names.index(bb['label'])

                # Get box coordinates
                x, y, w, h = bb['x'], bb['y'], bb['w'], bb['h']

                boxes.append([x, y, w, h])
                labels.append(disease_idx)
                areas.append(w * h)

            except Exception as e:
                self.logger.warning(f"Error processing box for {image_id}: {e}")
                continue

        if not boxes:
            return None

        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'areas': torch.tensor(areas, dtype=torch.float32)
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """Get a sample from the dataset."""
        try:
            # Get image path and labels
            row = self.df.iloc[idx]
            image_id = row['Image_Index']
            image_path = self.image_dir / image_id

            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get labels
            labels = torch.tensor([row[f'label_{i}'] for i in range(14)], dtype=torch.float32)

            # Apply transforms
            if self.transform:
                image = self.transform(image=image)['image']

            # Get bounding boxes
            bboxes = self._process_bboxes(image_id)

            return image, labels, bboxes

        except Exception as e:
            self.logger.error(f"Error processing sample {idx}: {e}")
            raise


def get_data_loaders(config: Dict) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create data loaders for train/val/test sets."""
    from .transforms import get_transforms

    # Create datasets with appropriate transforms
    train_dataset = ChestXrayDataset(
        csv_path=config['dataset']['train_csv'],
        image_dir=config['dataset']['image_dir'],
        bb_file=config['dataset']['bb_annotations'],
        transform=get_transforms('train', config),
        phase='train'
    )

    val_dataset = ChestXrayDataset(
        csv_path=config['dataset']['val_csv'],
        image_dir=config['dataset']['image_dir'],
        bb_file=config['dataset']['bb_annotations'],
        transform=get_transforms('val', config),
        phase='val'
    )

    test_dataset = ChestXrayDataset(
        csv_path=config['dataset']['test_csv'],
        image_dir=config['dataset']['image_dir'],
        bb_file=config['dataset']['bb_annotations'],
        transform=get_transforms('test', config),
        phase='test'
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

