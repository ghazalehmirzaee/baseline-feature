# src/data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, bbox_data=None, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (np.ndarray): Labels for each image (N x 14)
            bbox_data (dict): Dictionary mapping image paths to bounding box data
            transform: Image transformations
        """
        self.image_paths = image_paths
        self.labels = torch.FloatTensor(labels)
        self.bbox_data = bbox_data
        self.transform = transform

        # Disease names for reference
        self.diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        labels = self.labels[idx]

        # Get bounding box if available
        bbox = None
        if self.bbox_data is not None:
            img_name = self.image_paths[idx].split('/')[-1]
            bbox = self.bbox_data.get(img_name, None)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'labels': labels,
            'bbox': bbox,
            'path': self.image_paths[idx]
        }

        return sample

