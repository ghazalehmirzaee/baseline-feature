# utils/gpu_utils.py
import torch
import torch.cuda as cuda
from typing import Optional, Dict
import logging


class GPUManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_specs = self._get_gpu_specs()

    def _get_gpu_specs(self) -> Dict[str, Dict]:
        """Get specifications for available GPUs."""
        gpu_specs = {}

        if not torch.cuda.is_available():
            self.logger.warning("CUDA is not available")
            return gpu_specs

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_specs[i] = {
                'name': props.name,
                'memory': props.total_memory,
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count
            }

        return gpu_specs

    def optimize_batch_size(self, model: torch.nn.Module, input_shape: tuple,
                            target_gpu_utilization: float = 0.8) -> int:
        """Determine optimal batch size for the model and GPU."""
        if not self.gpu_specs:
            return 32  # Default batch size for CPU

        device = torch.device('cuda')
        model = model.to(device)

        # Start with a small batch size
        batch_size = 1
        max_batch_size = 1024

        while batch_size < max_batch_size:
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape).to(device)

                # Try forward pass
                with torch.no_grad():
                    model(dummy_input)

                # Check memory utilization
                memory_allocated = torch.cuda.memory_allocated(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                current_utilization = memory_allocated / total_memory

                if current_utilization >= target_gpu_utilization:
                    break

                batch_size *= 2
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = batch_size // 2
                    break
                else:
                    raise e

        # Add safety margin
        batch_size = int(batch_size * 0.9)
        return max(1, batch_size)

    def optimize_model_parallel(self, model: torch.nn.Module,
                                num_gpus: Optional[int] = None) -> torch.nn.Module:
        """Optimize model for multi-GPU training."""
        if not torch.cuda.is_available():
            return model

        available_gpus = torch.cuda.device_count()
        num_gpus = min(num_gpus or available_gpus, available_gpus)

        if num_gpus > 1:
            # Check if model is already parallelized
            if not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

        return model

    def get_optimal_device(self) -> torch.device:
        """Get the optimal GPU device based on memory and compute capability."""
        if not self.gpu_specs:
            return torch.device('cpu')

        # Score each GPU based on specs
        device_scores = {}
        for device_id, specs in self.gpu_specs.items():
            score = (
                    specs['memory'] / (1024 ** 3) +  # Memory in GB
                    float(specs['compute_capability']) * 10 +  # Weight compute capability
                    specs['multi_processor_count'] / 10  # Consider SM count
            )
            device_scores[device_id] = score

        # Get device with highest score
        optimal_device = max(device_scores.items(), key=lambda x: x[1])[0]
        return torch.device(f'cuda:{optimal_device}')

    def optimize_training_settings(self, model: torch.nn.Module,
                                   input_shape: tuple) -> Dict:
        """Get optimized training settings for the current GPU setup."""
        device = self.get_optimal_device()

        # Base settings
        settings = {
            'device': device,
            'batch_size': 32,
            'mixed_precision': False,
            'num_workers': 4,
            'pin_memory': True
        }

        if device.type == 'cuda':
            # Optimize batch size
            settings['batch_size'] = self.optimize_batch_size(model, input_shape)

            # Enable mixed precision for newer GPUs
            gpu_name = self.gpu_specs[device.index]['name'].lower()
            if any(gpu in gpu_name for gpu in ['a100', 'a40', 'a30']):
                settings['mixed_precision'] = True

            # Adjust number of workers based on CPU cores and GPU count
            import multiprocessing
            settings['num_workers'] = min(
                multiprocessing.cpu_count(),
                settings['batch_size']
            )

        return settings


def apply_gpu_optimizations(model: torch.nn.Module,
                            input_shape: tuple,
                            num_gpus: Optional[int] = None) -> tuple:
    """Apply all GPU optimizations and return optimized model and settings."""
    gpu_manager = GPUManager()

    # Get optimal training settings
    settings = gpu_manager.optimize_training_settings(model, input_shape)

    # Optimize model for multi-GPU if available
    model = gpu_manager.optimize_model_parallel(model, num_gpus)

    # Move model to optimal device
    model = model.to(settings['device'])

    return model, settings

