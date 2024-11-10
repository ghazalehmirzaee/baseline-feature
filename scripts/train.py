# scripts/train.py

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.dataset import ChestXrayDataset
from src.models.model import GraphAugmentedViT
from src.training.loss import MultiComponentLoss
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    train_dataset = ChestXrayDataset(
        image_paths=config['train_paths'],
        labels=config['train_labels'],
        bbox_data=config['bbox_data'],
        transform=config['train_transform']
    )

    val_dataset = ChestXrayDataset(
        image_paths=config['val_paths'],
        labels=config['val_labels'],
        bbox_data=config['bbox_data'],
        transform=config['val_transform']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Create model
    model = GraphAugmentedViT(
        num_diseases=14,
        pretrained_path=config['pretrained_path']
    )

    # Create loss function
    criterion = MultiComponentLoss(
        weights=config['loss_weights'],
        class_weights=config['class_weights']
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Start training
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == '__main__':
    main()

    