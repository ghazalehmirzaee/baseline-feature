# src/data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

# src/data/dataset.py

class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, bbox_data=None, transform=None):
        self.image_paths = image_paths
        self.labels = torch.FloatTensor(labels)
        self.bbox_data = bbox_data or {}  # Initialize as empty dict if None
        self.transform = transform

        self.diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        labels = self.labels[idx]

        # Get bounding box if available
        img_name = os.path.basename(image_path)
        bbox = self.bbox_data.get(img_name, {})  # Return empty dict if no bbox

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'labels': labels,
            'bbox': bbox,
            'path': image_path
        }

        return sample

