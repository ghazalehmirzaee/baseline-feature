# scripts/inspect_checkpoint.py
import torch
import os
import warnings

warnings.filterwarnings('ignore')


def inspect_checkpoint(filepath):
    """Inspect PyTorch checkpoint with simple error handling."""
    print(f"\nInspecting checkpoint: {filepath}")
    print("-" * 50)

    # Check file existence and size
    if not os.path.exists(filepath):
        print(f"Error: File does not exist at {filepath}")
        return

    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")

    try:
        # Try loading with simple settings
        checkpoint = torch.load(
            filepath,
            map_location='cpu',
            weights_only=False  # Changed to False to load full checkpoint
        )

        print("\nSuccessfully loaded checkpoint")
        print(f"Type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print("\nKeys in checkpoint:")
            for key in checkpoint.keys():
                print(f"- {key}")
                if isinstance(checkpoint[key], dict):
                    print(f"  Subkeys in {key}:")
                    for subkey in checkpoint[key].keys():
                        print(f"    - {subkey}")
                elif isinstance(checkpoint[key], torch.nn.Module):
                    print(f"  {key} is a nn.Module")
                elif isinstance(checkpoint[key], torch.Tensor):
                    print(f"  {key} is a Tensor with shape: {checkpoint[key].shape}")

        return checkpoint

    except Exception as e:
        print(f"\nError loading checkpoint: {str(e)}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint)

