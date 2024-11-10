# data/dataset_utils.py
import torch
import numpy as np
from typing import Dict, List, Tuple


class DatasetUtils:
    @staticmethod
    def process_bbox_annotations(bbox_data: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """
        Process bounding box annotations into standardized format.

        Args:
            bbox_data: Dictionary containing bounding box annotations

        Returns:
            Processed bounding box information
        """
        # Map disease names to indices
        disease_map = {
            'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3,
            'Mass': 4, 'Nodule': 5, 'Pneumonia': 6, 'Pneumothorax': 7,
            'Consolidation': 8, 'Edema': 9, 'Emphysema': 10, 'Fibrosis': 11,
            'Pleural_Thickening': 12, 'Hernia': 13
        }

        boxes = []
        labels = []
        areas = []
        image_ids = []

        for idx, ann in enumerate(bbox_data):
            try:
                # Extract coordinates
                x = float(ann['x'])
                y = float(ann['y'])
                w = float(ann['width'])
                h = float(ann['height'])

                # Get disease label
                disease = ann['label']
                if disease not in disease_map:
                    continue

                disease_idx = disease_map[disease]

                # Store information
                boxes.append([x, y, w, h])
                labels.append(disease_idx)
                areas.append(w * h)
                image_ids.append(ann['image_id'])

            except Exception as e:
                print(f"Error processing annotation {idx}: {e}")
                continue

        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'areas': torch.tensor(areas, dtype=torch.float32),
            'image_ids': image_ids
        }

    @staticmethod
    def create_disease_pairs(labels: torch.Tensor, boxes: torch.Tensor) -> Dict[Tuple[int, int], List[int]]:
        """
        Create disease pairs from labels and bounding boxes.

        Args:
            labels: Disease labels tensor
            boxes: Bounding boxes tensor

        Returns:
            Dictionary mapping disease pairs to sample indices
        """
        pairs = {}
        num_samples = len(labels)

        for i in range(num_samples):
            curr_label = labels[i]
            curr_box = boxes[i]

            # Find other samples with overlapping boxes
            for j in range(i + 1, num_samples):
                other_label = labels[j]
                other_box = boxes[j]

                # Skip if same disease
                if curr_label == other_label:
                    continue

                # Check for spatial overlap
                iou = DatasetUtils.compute_iou(curr_box, other_box)
                if iou > 0:
                    pair = tuple(sorted([int(curr_label), int(other_label)]))
                    if pair not in pairs:
                        pairs[pair] = []
                    pairs[pair].append(i)

        return pairs

    @staticmethod
    def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to x1,y1,x2,y2 format
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Find intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

