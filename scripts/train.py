# scripts/train.py

import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import wandb
from src.data.dataset import ChestXrayDataset, custom_collate_fn
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model import GraphAugmentedViT
from src.training.loss import MultiComponentLoss
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()


def load_dataset_from_txt(image_dir: str, label_file: str):
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
                label_vector = [int(x) for x in parts[1:15]]
                labels.append(label_vector)

    return images, np.array(labels)


def load_bbox_data(bbox_file: str):
    """Load bounding box annotations"""
    bbox_data = {}
    if os.path.exists(bbox_file):
        with open(bbox_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                img_name = parts[0]
                disease = parts[1]
                bbox = list(map(float, parts[2:6]))  # x, y, w, h

                if img_name not in bbox_data:
                    bbox_data[img_name] = {}
                if disease not in bbox_data[img_name]:
                    bbox_data[img_name][disease] = []
                bbox_data[img_name][disease].append(bbox)

    return bbox_data


def compute_class_weights(labels):
    """Compute class weights based on label distribution"""
    num_samples = len(labels)
    pos_counts = np.sum(labels, axis=0)
    neg_counts = num_samples - pos_counts

    # Compute weights using positive/negative ratio
    ratios = pos_counts / neg_counts
    weights = np.min(ratios) / ratios

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

    # Print class weights for verification
    print("Class weights:", class_weights)

    # Create datasets
    train_dataset = ChestXrayDataset(
        train_images,
        train_labels,
        bbox_data,
        transform=get_train_transforms()
    )

    val_dataset = ChestXrayDataset(
        val_images,
        val_labels,
        bbox_data,
        transform=get_val_transforms()
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # Initialize model
    print("Creating model...")
    model = GraphAugmentedViT(
        pretrained_path=config['model']['pretrained_path'],
        num_diseases=14,
        feature_dim=config['model']['feature_dim'],
        hidden_dim=config['model']['hidden_dim']
    )

    # Create loss function
    criterion = MultiComponentLoss(
        weights=config['loss_weights'],
        class_weights=class_weights
    )

    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        config=config,
        name=config['wandb']['run_name']
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
