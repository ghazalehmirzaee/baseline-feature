# src/models/graph_modules/feature_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class RegionFeatureExtractor:
    """Extract and process region features from images using ViT backbone."""

    def __init__(self, backbone: nn.Module, projector: nn.Module):
        self.backbone = backbone
        self.projector = projector
        self.logger = logging.getLogger(__name__)

    def __call__(self,
                 images: torch.Tensor,
                 bboxes: Dict[str, torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[int]]]:
        """
        Extract features from image regions and organize by disease.

        Args:
            images: Batch of images [B, C, H, W]
            bboxes: Dictionary containing bounding box information

        Returns:
            features_by_disease: Dictionary mapping disease ID to tensor of features
            regions_by_disease: Dictionary mapping disease ID to list of region indices
        """
        batch_size = images.shape[0]
        device = images.device

        features_by_disease = {}  # Store features for each disease class
        regions_by_disease = {}  # Store region indices for each disease class

        for i in range(batch_size):
            try:
                boxes = bboxes['boxes'][i]
                labels = bboxes['labels'][i]
                areas = bboxes['areas'][i]

                if len(boxes) == 0:
                    continue

                # Process each box
                for box_idx, (box, label, area) in enumerate(zip(boxes, labels, areas)):
                    # Extract region features
                    region_features = self._process_region(
                        images[i],
                        box,
                        area
                    )

                    if region_features is None:
                        continue

                    # Store by disease class
                    label = label.item()
                    if label not in features_by_disease:
                        features_by_disease[label] = []
                        regions_by_disease[label] = []

                    features_by_disease[label].append(region_features)
                    regions_by_disease[label].append(box_idx)

            except Exception as e:
                self.logger.warning(f"Error processing batch item {i}: {str(e)}")
                continue

        # Stack features for each disease
        for label in features_by_disease:
            try:
                features_by_disease[label] = torch.stack(features_by_disease[label])
            except Exception as e:
                self.logger.error(f"Error stacking features for disease {label}: {str(e)}")
                del features_by_disease[label]
                del regions_by_disease[label]

        return features_by_disease, regions_by_disease

    def _process_region(self,
                        image: torch.Tensor,
                        box: torch.Tensor,
                        area: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract and process features from a single image region.

        Args:
            image: Single image tensor [C, H, W]
            box: Bounding box coordinates [x, y, w, h]
            area: Box area for adaptive padding

        Returns:
            Region features tensor or None if extraction fails
        """
        try:
            # Get coordinates
            x, y, w, h = box

            # Compute adaptive padding
            padding = min(0.1, 50 / torch.sqrt(area))
            pad_x = int(w * padding)
            pad_y = int(h * padding)

            # Apply padding with bounds checking
            x1 = max(0, int(x - pad_x))
            y1 = max(0, int(y - pad_y))
            x2 = min(image.shape[2], int(x + w + pad_x))
            y2 = min(image.shape[1], int(y + h + pad_y))

            # Validate region size
            if x2 <= x1 or y2 <= y1:
                return None

            # Extract region
            region = image[:, y1:y2, x1:x2].unsqueeze(0)

            # Verify region is not empty
            if region.numel() == 0 or region.size(2) == 0 or region.size(3) == 0:
                return None

            # Resize to ViT input size
            region = F.interpolate(
                region,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

            # Extract features
            with torch.no_grad():
                features = self.backbone(region)  # [1, 768]

                # Ensure correct shape
                if len(features.shape) == 1:
                    features = features.unsqueeze(0)

                # Project features
                features = self.projector(features)  # [1, 256]

                # Verify feature dimensions
                if features.shape[1] != self.projector[-3].out_features:  # Access Linear layer output dim
                    raise ValueError(
                        f"Invalid feature dimension: {features.shape[1]}, "
                        f"expected {self.projector[-3].out_features}"
                    )

            return features

        except Exception as e:
            self.logger.warning(f"Error extracting region features: {str(e)}")
            return None

    def _verify_box(self, box: torch.Tensor, image_shape: Tuple[int, int]) -> bool:
        """
        Verify bounding box coordinates are valid.

        Args:
            box: Bounding box coordinates [x, y, w, h]
            image_shape: Image dimensions (H, W)

        Returns:
            Boolean indicating if box is valid
        """
        x, y, w, h = box

        # Check coordinates are within image bounds
        if x < 0 or y < 0 or x + w > image_shape[1] or y + h > image_shape[0]:
            return False

        # Check box has non-zero area
        if w <= 0 or h <= 0:
            return False

        return True

    def save_region_visualization(self,
                                  image: torch.Tensor,
                                  box: torch.Tensor,
                                  save_path: Path):
        """
        Save visualization of extracted region for debugging.

        Args:
            image: Image tensor [C, H, W]
            box: Bounding box coordinates
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt

        # Convert image for visualization
        img_np = image.permute(1, 2, 0).cpu().numpy()

        # Plot image and box
        plt.figure(figsize=(10, 10))
        plt.imshow(img_np)

        x, y, w, h = box.tolist()
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red')
        plt.gca().add_patch(rect)

        plt.axis('off')
        plt.savefig(save_path)
        plt.close()

        