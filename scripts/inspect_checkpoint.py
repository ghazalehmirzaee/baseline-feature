# scripts/inspect_checkpoint.py
import torch
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def inspect_checkpoint(filepath):
    """Inspect PyTorch checkpoint with robust error handling."""
    print(f"\nInspecting checkpoint: {filepath}")
    print("-" * 50)

    # Check file existence and size
    if not os.path.exists(filepath):
        print(f"Error: File does not exist at {filepath}")
        return

    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")

    if file_size == 0:
        print("Error: File is empty")
        return

    try:
        # Try loading with extra settings for compatibility
        checkpoint = torch.load(
            filepath,
            map_location='cpu',
            pickle_module=torch.serialization.pickle,
            weights_only=True  # Try loading only weights to avoid pickle issues
        )

        print("\nSuccessfully loaded checkpoint")
        print(f"Type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print("\nKeys in checkpoint:")
            for key in checkpoint.keys():
                print(f"- {key}")

            # If state_dict exists, show its structure
            if 'state_dict' in checkpoint:
                print("\nFirst 5 state_dict keys:")
                for i, key in enumerate(list(checkpoint['state_dict'].keys())[:5]):
                    print(f"- {key}")

            # If model exists, show its structure
            elif 'model' in checkpoint:
                if isinstance(checkpoint['model'], dict):
                    print("\nFirst 5 model keys:")
                    for i, key in enumerate(list(checkpoint['model'].keys())[:5]):
                        print(f"- {key}")
                else:
                    print("\nModel object type:", type(checkpoint['model']))

        print("\nCheckpoint loaded successfully!")
        return checkpoint

    except Exception as e:
        print(f"\nError loading checkpoint: {str(e)}")
        print("\nTrying alternative loading method...")

        try:
            # Try loading with pickle5 if available
            import pickle5 as pickle
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)

            print("\nSuccessfully loaded with pickle5")
            print(f"Type: {type(checkpoint)}")
            return checkpoint

        except Exception as e2:
            print(f"Alternative loading also failed: {str(e2)}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    args = parser.parse_args()

    # Set up Python path to include necessary directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

    # Install required packages if missing
    try:
        import numpy as np
    except ImportError:
        print("NumPy not found. Installing...")
        os.system('pip install numpy --upgrade')
        import numpy as np

    try:
        import pickle5
    except ImportError:
        print("pickle5 not found. Installing...")
        os.system('pip install pickle5')

    inspect_checkpoint(args.checkpoint)

    