# utils/checkpointing.py
import torch
from pathlib import Path
import json


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint with comprehensive information."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_config': model.config if hasattr(model, 'config') else None,
        'graph_adjacency': model.graph_constructor.last_adj_matrix.cpu().numpy().tolist()
        if hasattr(model, 'graph_constructor') else None
    }

    # Save checkpoint
    torch.save(checkpoint, filepath)

    # Save metrics separately for easy access
    metrics_path = Path(filepath).parent / f'metrics_epoch_{epoch}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint and return relevant information."""
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics'],
        'model_config': checkpoint['model_config'],
        'graph_adjacency': checkpoint['graph_adjacency']
    }


