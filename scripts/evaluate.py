# scripts/evaluate.py

import torch
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.dataset import ChestXrayDataset
from src.models.model import GraphAugmentedViT
from src.utils.metrics import calculate_metrics
from src.utils.visualization import visualize_attention, plot_roc_curves
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    attention_maps = []

    for batch in tqdm(dataloader):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        bbox_data = batch['bbox']

        # Get predictions and attention weights
        predictions = model(images, bbox_data)
        attention = model.get_attention_weights(images, bbox_data)

        all_preds.append(predictions.cpu())
        all_targets.append(labels.cpu())
        attention_maps.append(attention.cpu())

    predictions = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    attention_maps = torch.cat(attention_maps)

    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)

    return predictions, targets, attention_maps, metrics


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    test_dataset = ChestXrayDataset(
        image_paths=config['test_paths'],
        labels=config['test_labels'],
        bbox_data=config['bbox_data'],
        transform=config['test_transform']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Load model
    model = GraphAugmentedViT(num_diseases=14)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    predictions, targets, attention_maps, metrics = evaluate(model, test_loader, device)

    # Save metrics
    with open(f'{args.output_dir}/evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Generate visualizations
    visualize_attention(
        images=test_dataset.get_images()[:10],
        attention_maps=attention_maps[:10],
        save_dir=f'{args.output_dir}/attention_maps'
    )

    plot_roc_curves(
        predictions.numpy(),
        targets.numpy(),
        test_dataset.diseases,
        save_path=f'{args.output_dir}/roc_curves.png'
    )


if __name__ == '__main__':
    main()

