# src/data/dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, bbox_data=None, transform=None):
        self.image_paths = image_paths
        self.labels = torch.FloatTensor(labels)
        self.bbox_data = bbox_data or {}
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
        bbox_info = {}
        if img_name in self.bbox_data:
            for disease in self.diseases:
                if disease in self.bbox_data[img_name]:
                    bbox_info[disease] = self.bbox_data[img_name][disease]
                else:
                    bbox_info[disease] = []
        else:
            # Initialize empty lists for all diseases if no bbox data
            bbox_info = {disease: [] for disease in self.diseases}

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'labels': labels,
            'bbox': bbox_info,
            'image_path': image_path,
            'image_name': img_name
        }

