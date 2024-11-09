# models/graph_modules/feature_extractor.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class RegionFeatureExtractor(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        self.vit.requires_grad_(False)  # Freeze ViT weights

    def extract_region_features(self, images, bboxes, padding_factor=0.1):
        """Extract features from image regions defined by bounding boxes."""
        features = []
        areas = []

        for img, bbox in zip(images, bboxes):
            # Apply adaptive padding
            area = bbox[2] * bbox[3]
            padding = min(0.1, 50 / torch.sqrt(torch.tensor(area)))

            # Crop and resize region
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            # Add padding
            pad_x = w * padding
            pad_y = h * padding
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img.shape[2], x2 + pad_x)
            y2 = min(img.shape[1], y2 + pad_y)

            region = TF.crop(img, y1, x1, y2 - y1, x2 - x1)
            region = TF.resize(region, (224, 224))

            # Extract features
            with torch.no_grad():
                region_features = self.vit(region.unsqueeze(0))

            features.append(region_features)
            areas.append(area)

        return torch.cat(features, dim=0), torch.tensor(areas)


