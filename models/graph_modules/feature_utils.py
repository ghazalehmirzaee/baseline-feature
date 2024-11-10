# models/graph_modules/feature_utils.py
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


class FeatureExtractor:
    def __init__(self, backbone, feature_proj, graph_hidden_dim):
        self.backbone = backbone
        self.feature_proj = feature_proj
        self.graph_hidden_dim = graph_hidden_dim

    def extract_features(self, images: torch.Tensor, bboxes: Dict[str, torch.Tensor]) -> Tuple[
        Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Extract features and areas for each disease from the images.

        Args:
            images: Batch of images
            bboxes: Bounding box information

        Returns:
            Tuple of (features_dict, areas_dict)
        """
        features_dict = {}
        areas_dict = {}

        # Process each disease
        unique_labels = torch.unique(bboxes['labels'])
        for disease_idx in unique_labels:
            # Get boxes for this disease
            disease_mask = bboxes['labels'] == disease_idx
            disease_boxes = bboxes['boxes'][disease_mask]

            if len(disease_boxes) == 0:
                continue

            disease_features = []
            disease_areas = []

            # Process each box
            for box in disease_boxes:
                try:
                    # Extract region
                    x, y, w, h = box
                    area = w * h

                    # Skip invalid boxes
                    if w <= 0 or h <= 0:
                        continue

                    # Extract region
                    region = self._extract_region(images, x, y, w, h)
                    if region is None:
                        continue

                    # Get features
                    with torch.no_grad():
                        features = self.backbone(region)
                        features = self.feature_proj(features)

                    disease_features.append(features)
                    disease_areas.append(area)

                except Exception as e:
                    print(f"Error processing box for disease {disease_idx}: {e}")
                    continue

            if disease_features:
                features_dict[disease_idx.item()] = torch.cat(disease_features, dim=0)
                areas_dict[disease_idx.item()] = torch.tensor(disease_areas)

        return features_dict, areas_dict

    def _extract_region(self, images: torch.Tensor, x: float, y: float, w: float, h: float) -> torch.Tensor:
        """Extract and preprocess image region."""
        try:
            # Convert to integers for indexing
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            # Extract region
            region = images[:, :, y1:y2, x1:x2]

            # Verify region is valid
            if region.numel() == 0 or region.size(2) == 0 or region.size(3) == 0:
                return None

            # Resize to ViT input size
            region = F.interpolate(
                region,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

            return region

        except Exception as e:
            print(f"Error extracting region: {e}")
            return None

