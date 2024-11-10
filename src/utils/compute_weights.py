
import os
import numpy as np


def analyze_disease_distribution(label_file):
    """Analyze disease distribution and compute weights"""
    disease_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    # Initialize counters
    disease_counts = {disease: 0 for disease in disease_names}
    total_images = 0

    # Read and process label file
    print("Reading label file...")
    with open(label_file, 'r') as f:
        lines = f.readlines()

    total_images = len(lines)
    print(f"Found {total_images} images")

    # Count disease occurrences
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:  # Image name + 14 labels
            continue

        # Parse binary labels (skip image name)
        labels = [int(x) for x in parts[1:15]]

        # Count positive cases
        for i, label in enumerate(labels):
            if label == 1:
                disease_counts[disease_names[i]] += 1

    # Print disease counts
    print("\nInitial disease counts:")
    for disease, count in disease_counts.items():
        print(f"{disease}: {count}")

    # Compute weights using median frequency balancing
    frequencies = np.array([max(1, disease_counts[disease]) for disease in disease_names])
    neg_frequencies = np.array([total_images - freq for freq in frequencies])

    # Use the more imbalanced side for each class
    ratios = np.maximum(frequencies / neg_frequencies, neg_frequencies / frequencies)
    weights = ratios / np.min(ratios)  # Normalize so minimum weight is 1.0

    # Print statistics
    print("\nDisease Distribution Analysis:")
    print("-" * 80)
    print(f"{'Disease':20} {'Count':>8} {'Percentage':>12} {'Pos/Neg Ratio':>15} {'Weight':>8}")
    print("-" * 80)

    for i, disease in enumerate(disease_names):
        count = disease_counts[disease]
        percentage = (count / total_images) * 100
        pos_neg_ratio = frequencies[i] / neg_frequencies[i]
        weight = weights[i]
        print(f"{disease:20} {count:8d} {percentage:11.2f}% {pos_neg_ratio:14.3f} {weight:8.2f}")

    print("-" * 80)
    print(f"Total Images: {total_images}")

    return dict(zip(disease_names, weights))


if __name__ == "__main__":
    label_file = "/home/ghazal/Documents/Datasets/ChestX-ray14/labels/train_list.txt"

    print(f"Processing file: {label_file}")
    print(f"File exists: {os.path.exists(label_file)}")

    weights = analyze_disease_distribution(label_file)

    # Print weights in YAML format
    print("\nClass weights in YAML format:")
    print("class_weights:")
    for disease, weight in weights.items():
        print(f"  {disease}: {weight:.2f}")

    # Also save to a file
    output_file = "class_weights.yaml"
    print(f"\nSaving weights to {output_file}")
    with open(output_file, 'w') as f:
        f.write("class_weights:\n")
        for disease, weight in weights.items():
            f.write(f"  {disease}: {weight:.2f}\n")
