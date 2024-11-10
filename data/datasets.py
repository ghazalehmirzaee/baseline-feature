# data/datasets.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import logging
from torch.utils.data._utils.collate import default_collate


def custom_collate_fn(batch):
    """Custom collate function to handle None values in bounding boxes."""
    # Separate images, labels, and bboxes
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    bboxes = [item[2] for item in batch]

    # Collate images and labels normally
    images = default_collate(images)
    labels = default_collate(labels)

    # Handle bboxes specially since they might be None
    if all(bbox is None for bbox in bboxes):
        bboxes = None
    else:
        # Replace None with empty tensors
        bboxes = [bbox if bbox is not None else {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.long),
            'areas': torch.zeros((0,), dtype=torch.float32)
        } for bbox in bboxes]

        # Collate bboxes
        collated_bboxes = {
            'boxes': default_collate([b['boxes'] for b in bboxes]),
            'labels': default_collate([b['labels'] for b in bboxes]),
            'areas': default_collate([b['areas'] for b in bboxes])
        }
        bboxes = collated_bboxes

    return images, labels, bboxes


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, bb_file=None, transform=None, phase='train'):
        self.logger = logging.getLogger(__name__)

        # Load CSV file
        try:
            # Read space-delimited CSV with custom column names
            self.df = pd.read_csv(
                csv_path,
                delimiter=' ',  # Space-delimited
                header=None,  # No header in the file
                names=['Image_Index'] + [f'label_{i}' for i in range(14)]  # Column names
            )
            self.logger.info(f"Loaded dataset from {csv_path} with {len(self.df)} samples")
        except Exception as e:
            self.logger.error(f"Error loading CSV file {csv_path}: {e}")
            raise

        self.image_dir = Path(image_dir)
        self.transform = transform
        self.phase = phase

        # Load bounding box annotations if provided
        self.bb_data = None
        if bb_file is not None and Path(bb_file).exists():
            try:
                # Read bbox data
                self.bb_data = pd.read_csv(bb_file, header=0)
                self.logger.info(f"Loaded {len(self.bb_data)} bounding box annotations")
            except Exception as e:
                self.logger.warning(f"Error loading bounding box annotations from {bb_file}: {e}")

        # Disease names
        self.disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            # Get image ID and labels
            row = self.df.iloc[idx]
            image_id = row['Image_Index']

            # Load image
            image_path = self.image_dir / image_id
            if not image_path.exists():
                self.logger.error(f"Image not found: {image_path}")
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                raise ValueError(f"Failed to load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get labels (all columns except Image_Index)
            labels = torch.tensor(
                [row[f'label_{i}'] for i in range(14)],
                dtype=torch.float32
            )

            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            # Get bounding boxes if available
            bboxes = self._get_bounding_boxes(image_id) if self.bb_data is not None else None

            return image, labels, bboxes

        except Exception as e:
            self.logger.error(f"Error processing sample {idx} ({image_id}): {str(e)}")
            raise

    def _get_bounding_boxes(self, image_id):
        """Get bounding box annotations for an image."""
        try:
            # Get all bounding boxes for this image
            image_bbs = self.bb_data[self.bb_data['Image Index'] == image_id]
            if len(image_bbs) == 0:
                return None

            bboxes = []
            labels = []
            areas = []

            for _, row in image_bbs.iterrows():
                try:
                    # Get coordinates
                    x = float(row['Bbox [x'].strip('['))  # Remove '[' from x
                    y = float(row['y'])
                    w = float(row['w'])
                    h = float(row['h'].strip(']'))  # Remove ']' from h

                    # Get disease label
                    disease = row['Finding Label']
                    if disease in self.disease_names:
                        disease_idx = self.disease_names.index(disease)
                        bboxes.append([x, y, w, h])
                        labels.append(disease_idx)
                        areas.append(w * h)

                except Exception as e:
                    self.logger.warning(f"Error processing bounding box for {image_id}: {str(e)}")
                    continue

            if not bboxes:
                return None

            return {
                'boxes': torch.tensor(bboxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long),
                'areas': torch.tensor(areas, dtype=torch.float32)
            }

        except Exception as e:
            self.logger.warning(f"Error getting bounding boxes for {image_id}: {str(e)}")
            return None


def get_data_loaders(config):
    """Create data loaders with custom collate function."""
    dataset_config = config['dataset']

    # Create datasets
    train_dataset = ChestXrayDataset(
        csv_path=dataset_config['train_csv'],
        image_dir=dataset_config['image_dir'],
        bb_file=dataset_config['bb_annotations'],
        phase='train'
    )

    val_dataset = ChestXrayDataset(
        csv_path=dataset_config['val_csv'],
        image_dir=dataset_config['image_dir'],
        bb_file=dataset_config['bb_annotations'],
        phase='val'
    )

    test_dataset = ChestXrayDataset(
        csv_path=dataset_config['test_csv'],
        image_dir=dataset_config['image_dir'],
        bb_file=dataset_config['bb_annotations'],
        phase='test'
    )

    # Create data loaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader

