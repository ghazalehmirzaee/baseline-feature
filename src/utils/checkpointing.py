# src/utils/checkpointing.py
import torch
from pathlib import Path
import json
import logging
from typing import Dict, Optional
import shutil
import time


class CheckpointManager:
    """Comprehensive checkpoint management with extensive state saving."""

    def __init__(self, config: Dict, save_dir: Path):
        self.config = config
        self.save_dir = save_dir
        self.checkpoints_dir = save_dir / 'checkpoints'
        self.best_model_dir = save_dir / 'best_model'
        self.results_dir = save_dir / 'results'

        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.best_metric = 0.0

    def save_checkpoint(self,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler._LRScheduler,
                        epoch: int,
                        iteration: int,
                        metrics: Dict,
                        is_best: bool = False):
        """
        Save comprehensive checkpoint.

        Args:
            model: Model instance
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            epoch: Current epoch
            iteration: Current iteration
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        try:
            # Prepare checkpoint
            checkpoint = {
                # Model states
                'model_state_dict': model.state_dict(),
                'backbone_config': model.backbone.config if hasattr(model.backbone, 'config') else None,
                'graph_weights': model.get_graph_weights() if hasattr(model, 'get_graph_weights') else None,
                'fusion_state': model.classifier.state_dict() if hasattr(model, 'classifier') else None,

                # Training states
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'iteration': iteration,
                'best_metric': self.best_metric,

                # Metadata
                'config': self.config,
                'metrics': metrics,

                # Timestamp
                'timestamp': time.strftime('%Y%m%d-%H%M%S')
            }

            # Regular checkpoint
            checkpoint_path = self.checkpoints_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)

            # Save metrics separately
            metrics_path = self.results_dir / f'metrics_epoch_{epoch}.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            # If best model so far
            if is_best:
                self.best_metric = metrics['mean_auc']
                best_path = self.best_model_dir / 'best_model.pt'
                shutil.copy(checkpoint_path, best_path)

                # Save best metrics
                best_metrics_path = self.best_model_dir / 'best_metrics.json'
                with open(best_metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)

                self.logger.info(
                    f"Saved best model at epoch {epoch} "
                    f"with mean AUC: {self.best_metric:.4f}"
                )

            # Keep only recent checkpoints
            self._cleanup_old_checkpoints()

            return True

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            return False

    def load_checkpoint(self,
                        model: torch.nn.Module,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                        path: Optional[str] = None) -> Dict:
        """
        Load checkpoint with full state recreation.

        Args:
            model: Model instance
            optimizer: Optional optimizer instance
            scheduler: Optional scheduler instance
            path: Optional specific checkpoint path

        Returns:
            Dictionary containing loaded checkpoint info
        """
        try:
            # Load best model by default
            if path is None:
                path = self.best_model_dir / 'best_model.pt'

            checkpoint = torch.load(path, map_location=next(model.parameters()).device)

            # Load model states
            model.load_state_dict(checkpoint['model_state_dict'])

            # Load graph weights if available
            if checkpoint['graph_weights'] is not None and hasattr(model, 'set_graph_weights'):
                model.set_graph_weights(checkpoint['graph_weights'])

            # Load training states
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.best_metric = checkpoint['best_metric']

            self.logger.info(
                f"Loaded checkpoint from epoch {checkpoint['epoch']} "
                f"with mean AUC: {checkpoint['metrics']['mean_auc']:.4f}"
            )

            return checkpoint

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise

    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """Keep only the most recent checkpoints."""
        checkpoints = sorted(self.checkpoints_dir.glob('checkpoint_*.pt'))

        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()

