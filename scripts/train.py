# scripts/train.py

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model import GraphAugmentedViT
from src.training.loss import MultiComponentLoss
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()


def load_bbox_data(bbox_file):
    """Load bounding box annotations"""
    bbox_data = {}
    if os.path.exists(bbox_file):
        bbox_df = pd.read_csv(bbox_file)
        for _, row in bbox_df.iterrows():
            img_name = row['Image Index']
            if img_name not in bbox_data:
                bbox_data[img_name] = {}
            label = row['Finding Label']
            if label not in bbox_data[img_name]:
                bbox_data[img_name][label] = []
            bbox_data[img_name][label].append([
                row['Bbox_x'], row['Bbox_y'],
                row['Bbox_w'], row['Bbox_h']
            ])
    return bbox_data


def load_dataset_from_txt(image_dir, label_file):
    """Load dataset from txt file"""
    images = []
    labels = []

    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 15:  # Image name + 14 labels
            img_path = os.path.join(image_dir, parts[0])
            if os.path.exists(img_path):
                images.append(img_path)
                # Convert label strings to integers
                label_vector = [int(x) for x in parts[1:15]]
                labels.append(label_vector)

    return images, np.array(labels)


def compute_class_weights(labels):
    """Compute class weights based on label distribution"""
    num_samples = len(labels)
    pos_counts = np.sum(labels, axis=0)
    neg_counts = num_samples - pos_counts

    disease_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    # Compute weights
    pos_ratios = pos_counts / neg_counts
    min_ratio = np.min(pos_ratios)
    weights = min_ratio / pos_ratios

    print("\nDisease Distribution Analysis:")
    print("-" * 80)
    print(f"{'Disease':20} {'Count':>8} {'Percentage':>12} {'Pos/Neg Ratio':>15} {'Weight':>8}")
    print("-" * 80)

    for i, disease in enumerate(disease_names):
        count = pos_counts[i]
        percentage = (count / num_samples) * 100
        ratio = pos_ratios[i]
        weight = weights[i]
        print(f"{disease:20} {int(count):8d} {percentage:11.2f}% {ratio:14.3f} {weight:8.2f}")

    print("-" * 80)
    print(f"Total samples: {num_samples}")

    return torch.FloatTensor(weights)


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_images, train_labels = load_dataset_from_txt(
        config['train_paths'],
        config['train_labels']
    )

    val_images, val_labels = load_dataset_from_txt(
        config['val_paths'],
        config['val_labels']
    )

    # Load bounding box data
    print("Loading bounding box annotations...")
    bbox_data = load_bbox_data(config['bbox_data_path'])

    # Compute class weights
    print("Computing class weights...")
    class_weights = compute_class_weights(train_labels)
    print("Class weights:", class_weights.numpy())

    # Create datasets
    train_dataset = ChestXrayDataset(
        image_paths=train_images,
        labels=train_labels,
        bbox_data=bbox_data,
        transform=get_train_transforms()
    )

    val_dataset = ChestXrayDataset(
        image_paths=val_images,
        labels=val_labels,
        bbox_data=bbox_data,
        transform=get_val_transforms()
    )

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    # Create model
    print("Creating model...")
    model = GraphAugmentedViT(
        num_diseases=14,
        pretrained_path=config['model']['pretrained_path']
    )

    # Create loss function
    criterion = MultiComponentLoss(
        weights=config['loss_weights'],
        class_weights=class_weights
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device
    )

    # Start training
    print("Starting training...")
    trainer.train(num_epochs=config['training']['num_epochs'])


if __name__ == '__main__':
    main()
