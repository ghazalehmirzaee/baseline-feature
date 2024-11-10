# src/utils/logger.py
import logging
from pathlib import Path
import sys
from typing import Optional
import time
from datetime import datetime


class Logger:
    """Custom logger with file and console output."""

    def __init__(self,
                 name: str,
                 log_dir: str = "logs",
                 log_level: int = logging.INFO,
                 file_prefix: Optional[str] = None):
        """
        Initialize logger with both file and console handlers.

        Args:
            name: Logger name
            log_dir: Directory to store log files
            log_level: Logging level
            file_prefix: Optional prefix for log file name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Create log directory
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{file_prefix}_" if file_prefix else ""
        file_name += f"{name}_{timestamp}.log"

        # File handler
        fh = logging.FileHandler(log_dir / file_name)
        fh.setLevel(log_level)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add formatter to handlers
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers to logger
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        self.log_file = log_dir / file_name

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def get_log_file(self) -> Path:
        """Get path to current log file."""
        return self.log_file


class TrainingLogger(Logger):
    """Extended logger specifically for training progress."""

    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        super().__init__(
            name=experiment_name,
            log_dir=log_dir,
            file_prefix="training"
        )
        self.start_time = None
        self.epoch_start_time = None

    def start_training(self, config: dict):
        """Log training start with configuration."""
        self.start_time = time.time()
        self.info("=== Starting Training ===")
        self.info(f"Configuration:\n{config}")

    def end_training(self, best_metrics: dict):
        """Log training completion with final metrics."""
        duration = time.time() - self.start_time
        hours = duration // 3600
        minutes = (duration % 3600) // 60

        self.info("=== Training Completed ===")
        self.info(f"Total training time: {int(hours)}h {int(minutes)}m")
        self.info("Best metrics:")
        for k, v in best_metrics.items():
            self.info(f"  {k}: {v:.4f}")

    def start_epoch(self, epoch: int):
        """Log epoch start."""
        self.epoch_start_time = time.time()
        self.info(f"\nEpoch {epoch}")

    def end_epoch(self, epoch: int, metrics: dict):
        """Log epoch completion with metrics."""
        duration = time.time() - self.epoch_start_time
        self.info(f"Epoch {epoch} completed in {duration:.2f}s")
        self.info("Metrics:")
        for k, v in metrics.items():
            self.info(f"  {k}: {v:.4f}")

    def log_batch(self, epoch: int, batch: int, total_batches: int, loss: float):
        """Log batch progress."""
        if batch % 100 == 0:
            self.info(
                f"Epoch {epoch} [{batch}/{total_batches}] "
                f"Loss: {loss:.4f}"
            )


class GraphLogger(Logger):
    """Specialized logger for graph-related operations."""

    def __init__(self, log_dir: str = "logs"):
        super().__init__(
            name="graph_construction",
            log_dir=log_dir,
            file_prefix="graph"
        )

    def log_graph_construction(self,
                               direct_pairs: int,
                               limited_pairs: int,
                               estimated_pairs: int):
        """Log graph construction statistics."""
        self.info("Graph Construction Summary:")
        self.info(f"  Direct relationships: {direct_pairs}")
        self.info(f"  Limited sample relationships: {limited_pairs}")
        self.info(f"  Estimated relationships: {estimated_pairs}")

    def log_relationship_strength(self,
                                  disease1: str,
                                  disease2: str,
                                  strength: float,
                                  relationship_type: str):
        """Log individual relationship strength."""
        self.debug(
            f"Relationship ({relationship_type}): "
            f"{disease1} - {disease2} = {strength:.4f}"
        )


def setup_logger(name: str,
                 log_dir: str = "logs",
                 log_level: int = logging.INFO) -> Logger:
    """Helper function to create a logger instance."""
    return Logger(name, log_dir, log_level)


def setup_training_logger(experiment_name: str,
                          log_dir: str = "logs") -> TrainingLogger:
    """Helper function to create a training logger instance."""
    return TrainingLogger(experiment_name, log_dir)


def setup_graph_logger(log_dir: str = "logs") -> GraphLogger:
    """Helper function to create a graph logger instance."""
    return GraphLogger(log_dir)

