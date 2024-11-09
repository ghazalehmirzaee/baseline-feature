# scripts/compute_disease_counts.py
import pandas as pd
import numpy as np


def compute_disease_counts(csv_path):
    """
    Compute positive and negative counts for each disease in the dataset.

    Args:
        csv_path: Path to the training CSV file containing image labels

    Returns:
        positive_counts: List of positive sample counts for each disease
        negative_counts: List of negative sample counts for each disease
    """
    # List of all diseases in order
    diseases = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    # Read the CSV file without headers and specify whitespace as the separator
    df = pd.read_csv(csv_path, header=None, delim_whitespace=True)

    # Assign column names to DataFrame
    df.columns = ['Image'] + diseases

    # Initialize lists for positive and negative counts
    positive_counts = []
    negative_counts = []
    total_samples = len(df)

    print("\nDisease Statistics:")
    print("-" * 60)
    print(f"{'Disease':<20} {'Positive':<10} {'Negative':<10} {'Positive %':<10}")
    print("-" * 60)

    # Calculate positive and negative counts for each disease
    for disease in diseases:
        positive = df[disease].sum()  # Count of positive samples for the disease
        negative = total_samples - positive  # Count of negative samples

        positive_counts.append(positive)
        negative_counts.append(negative)

        # Print statistics
        print(f"{disease:<20} {positive:<10} {negative:<10} {positive / total_samples * 100:>8.2f}%")

    print("-" * 60)
    print(f"Total samples: {total_samples}")

    return positive_counts, negative_counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()

    train_csv = f"{args.data_dir}/train_list.txt"
    pos_counts, neg_counts = compute_disease_counts(train_csv)

    print("\nConfig format:")
    print("\npositive_counts:", pos_counts)
    print("\nnegative_counts:", neg_counts)

