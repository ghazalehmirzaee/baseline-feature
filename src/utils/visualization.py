# src/utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import torch
import cv2
import os


def visualize_attention(images, attention_maps, save_dir):
    """Visualize attention maps overlaid on images"""
    os.makedirs(save_dir, exist_ok=True)

    for i, (image, attention) in enumerate(zip(images, attention_maps)):
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        # Attention map
        plt.subplot(1, 3, 2)
        sns.heatmap(attention.mean(0), cmap='viridis')
        plt.title('Attention Map')

        # Overlay
        plt.subplot(1, 3, 3)
        attention_resized = cv2.resize(
            attention.mean(0).numpy(),
            (image.shape[1], image.shape[0])
        )
        plt.imshow(image)
        plt.imshow(attention_resized, alpha=0.5, cmap='viridis')
        plt.title('Overlay')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/attention_map_{i}.png')
        plt.close()


def plot_roc_curves(predictions, targets, class_names, save_path):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(15, 10))

    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            label=f'{class_names[i]} (AUC = {roc_auc:.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def visualize_graph(adjacency_matrix, class_names, save_path):
    """Visualize disease relationship graph"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        adjacency_matrix,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='YlOrRd'
    )
    plt.title('Disease Relationship Graph')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

