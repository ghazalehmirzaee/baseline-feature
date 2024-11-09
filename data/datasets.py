# data/datasets.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, bb_file=None, transform=None, phase='train'):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform or self._get_default_transforms(phase)
        self.phase = phase

        # Load bounding box annotations if provided
        self.bb_data = None
        if bb_file is not None:
            self.bb_data = pd.read_csv(bb_file)

        # Disease names
        self.disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def _get_default_transforms(self, phase):
        if phase == 'train':
            return A.Compose([
                A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def _get_bounding_boxes(self, image_id):
        """Get bounding box annotations for an image."""
        if self.bb_data is None:
            return None

        image_bbs = self.bb_data[self.bb_data['Image_Index'] == image_id]
        if len(image_bbs) == 0:
            return None

        bboxes = []
        labels = []
        areas = []

        for _, row in image_bbs.iterrows():
            x, y, w, h = row['Bbox_X'], row['Bbox_Y'], row['Bbox_W'], row['Bbox_H']
            disease_idx = self.disease_names.index(row['Finding Label'])

            bboxes.append([x, y, w, h])
            labels.append(disease_idx)
            areas.append(w * h)

        return {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'areas': torch.tensor(areas, dtype=torch.float32)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['Image_Index']

        # Load image
        image_path = self.image_dir / image_id
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get labels
        labels = torch.zeros(len(self.disease_names), dtype=torch.float32)
        findings = row['Finding Labels'].split('|')
        for finding in findings:
            if finding in self.disease_names:
                labels[self.disease_names.index(finding)] = 1

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        # Get bounding boxes if available
        bboxes = self._get_bounding_boxes(image_id)

        return image, labels, bboxes


def get_data_loaders(config):
    """Create data loaders for train, validation, and test sets."""
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

