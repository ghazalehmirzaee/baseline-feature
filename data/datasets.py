# data/datasets.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, bb_file=None, transform=None, phase='train'):
        self.logger = logging.getLogger(__name__)

        # Load CSV file
        try:
            self.df = pd.read_csv(
                csv_path,
                delimiter=' ',  # Space-delimited
                header=None,  # No header in the file
                names=['Image_Index'] + [f'label_{i}' for i in range(14)]  # Column names
            )
            self.logger.info(f"Loaded dataset from {csv_path} with {len(self.df)} samples")
        except Exception as e:
            self.logger.error(f"Error loading CSV file {csv_path}: {e}")
            raise

        self.image_dir = Path(image_dir)
        self.transform = transform or self._get_default_transforms(phase)
        self.phase = phase

        # Load bounding box annotations if provided
        self.bb_data = None
        if bb_file is not None and Path(bb_file).exists():
            try:
                self.bb_data = pd.read_csv(bb_file)
                self.logger.info(f"Loaded {len(self.bb_data)} bounding box annotations")
            except Exception as e:
                self.logger.warning(f"Error loading bounding box annotations from {bb_file}: {e}")

        self.disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def _get_default_transforms(self, phase):
        """Get default transforms for the dataset."""
        if phase == 'train':
            return A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:  # val or test
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])

    def _get_bounding_boxes(self, image_id):
        """Get bounding box annotations for an image."""
        if self.bb_data is None:
            return None

        try:
            image_bbs = self.bb_data[self.bb_data['Image Index'] == image_id]
            if len(image_bbs) == 0:
                return None

            bboxes = []
            labels = []
            areas = []

            for _, row in image_bbs.iterrows():
                try:
                    # Parse coordinates
                    if 'Bbox [x,y,w,h]' in row:
                        coords = row['Bbox [x,y,w,h]'].strip('[]').split(',')
                        x, y, w, h = map(float, coords)
                    else:
                        # If coordinates are in separate columns
                        x = float(row.get('x', row.get('Bbox_x', 0)))
                        y = float(row.get('y', row.get('Bbox_y', 0)))
                        w = float(row.get('w', row.get('Bbox_w', 0)))
                        h = float(row.get('h', row.get('Bbox_h', 0)))

                    # Get disease label
                    disease = row['Finding Label']
                    if disease in self.disease_names:
                        disease_idx = self.disease_names.index(disease)

                        # Scale bounding box coordinates to match resized image
                        scale_x = 224.0 / image_bbs['width'].iloc[0] if 'width' in image_bbs else 1
                        scale_y = 224.0 / image_bbs['height'].iloc[0] if 'height' in image_bbs else 1

                        x_scaled = x * scale_x
                        y_scaled = y * scale_y
                        w_scaled = w * scale_x
                        h_scaled = h * scale_y

                        bboxes.append([x_scaled, y_scaled, w_scaled, h_scaled])
                        labels.append(disease_idx)
                        areas.append(w_scaled * h_scaled)

                except Exception as e:
                    self.logger.warning(f"Error processing bounding box row for {image_id}: {str(e)}")
                    continue

            if not bboxes:
                return None

            return {
                'boxes': torch.tensor(bboxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long),
                'areas': torch.tensor(areas, dtype=torch.float32)
            }

        except Exception as e:
            self.logger.warning(f"Error getting bounding boxes for {image_id}: {str(e)}")
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            # Get image ID and labels
            row = self.df.iloc[idx]
            image_id = row['Image_Index']

            # Load image
            image_path = self.image_dir / image_id
            if not image_path.exists():
                self.logger.error(f"Image not found: {image_path}")
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                raise ValueError(f"Failed to load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get labels
            labels = torch.tensor(
                [row[f'label_{i}'] for i in range(14)],
                dtype=torch.float32
            )

            # Get original image dimensions for bbox scaling
            orig_h, orig_w = image.shape[:2]

            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            # Get bounding boxes if available
            bboxes = self._get_bounding_boxes(image_id)
            if bboxes is not None:
                # Scale bounding boxes to match transformed image size
                scale_x = 224.0 / orig_w
                scale_y = 224.0 / orig_h
                bboxes['boxes'][:, [0, 2]] *= scale_x
                bboxes['boxes'][:, [1, 3]] *= scale_y

            return image, labels, bboxes

        except Exception as e:
            self.logger.error(f"Error processing sample {idx} ({image_id}): {str(e)}")
            raise


def get_data_loaders(config):
    """Create data loaders."""
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
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader

