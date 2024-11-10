# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(phase: str, config: dict):
    """Get transforms for each phase."""
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(
                height=config['dataset']['image_size'],
                width=config['dataset']['image_size'],
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.2
            ),
            A.Normalize(
                mean=config['dataset']['mean'],
                std=config['dataset']['std']
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(
                height=config['dataset']['image_size'],
                width=config['dataset']['image_size']
            ),
            A.Normalize(
                mean=config['dataset']['mean'],
                std=config['dataset']['std']
            ),
            ToTensorV2()
        ])

