# src/data/dataset.py

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple


class ChestXrayDataset(Dataset):
    """Dataset class for chest X-ray images"""

    def __init__(
            self,
            image_paths: List[str],
            labels: np.ndarray,
            bbox_data: Optional[Dict] = None,
            transform=None
    ):
        self.image_paths = image_paths
        self.labels = torch.FloatTensor(labels)
        self.bbox_data = bbox_data or {}
        self.transform = transform

        self.diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        labels = self.labels[idx]

        # Get bounding box data
        img_name = os.path.basename(image_path)
        bbox_info = {}

        if img_name in self.bbox_data:
            for disease in self.diseases:
                bbox_info[disease] = self.bbox_data[img_name].get(disease, [])
        else:
            bbox_info = {disease: [] for disease in self.diseases}

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'labels': labels,
            'bbox': bbox_info,
            'image_path': image_path,
            'image_name': img_name
        }


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function to handle variable size bbox data"""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'bbox': [item['bbox'] for item in batch],
        'image_path': [item['image_path'] for item in batch],
        'image_name': [item['image_name'] for item in batch]
    }


