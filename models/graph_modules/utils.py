# models/graph_modules/utils.py
import torch
import numpy as np
from typing import Dict, Tuple, List


class GraphUtils:
    @staticmethod
    def prepare_sample_pairs(labels: torch.Tensor) -> Dict[Tuple[int, int], List[int]]:
        """
        Prepare sample pairs for graph construction from labels.

        Args:
            labels: Binary label matrix of shape [N, num_classes]

        Returns:
            Dictionary mapping disease pairs to lists of sample indices
        """
        num_samples, num_classes = labels.shape
        sample_pairs = {}

        # Convert to numpy for easier processing
        labels_np = labels.cpu().numpy()

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                # Find samples where both diseases occur
                cooccur_samples = np.where((labels_np[:, i] == 1) & (labels_np[:, j] == 1))[0]

                if len(cooccur_samples) > 0:
                    sample_pairs[(i, j)] = cooccur_samples.tolist()

        return sample_pairs

    @staticmethod
    def compute_area_weights(bboxes: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Compute area weights for each disease's bounding boxes.

        Args:
            bboxes: Dictionary containing bounding box information

        Returns:
            Dictionary mapping disease indices to area tensors
        """
        area_weights = {}

        # Extract box coordinates and labels
        boxes = bboxes['boxes']
        labels = bboxes['labels']

        for i in range(14):  # For each disease
            # Find boxes for this disease
            disease_mask = labels == i
            if disease_mask.any():
                # Compute areas for these boxes
                disease_boxes = boxes[disease_mask]
                areas = disease_boxes[:, 2] * disease_boxes[:, 3]  # width * height
                area_weights[i] = areas
            else:
                # If no boxes for this disease, use unit weight
                area_weights[i] = torch.ones(1)

        return area_weights

    @staticmethod
    def normalize_adjacency_matrix(adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Normalize adjacency matrix using symmetric normalization.

        Args:
            adj_matrix: Square adjacency matrix

        Returns:
            Normalized adjacency matrix
        """
        # Add self-loops
        adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0])

        # Compute degree matrix
        deg = torch.sum(adj_matrix, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        # Symmetric normalization
        norm_adj = torch.mm(torch.mm(
            torch.diag(deg_inv_sqrt),
            adj_matrix
        ), torch.diag(deg_inv_sqrt))

        return norm_adj

    