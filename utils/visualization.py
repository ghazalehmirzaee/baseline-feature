# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import wandb


class VisualizationManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def plot_roc_curves(self, predictions, targets):
        """Plot ROC curves for each disease."""
        plt.figure(figsize=(15, 10))

        for i, disease in enumerate(self.disease_names):
            if len(np.unique(targets[:, i])) > 1:
                fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
                plt.plot(fpr, tpr, label=f'{disease} (AUC: {auc(fpr, tpr):.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Diseases')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save to wandb
        wandb.log({"ROC_Curves": wandb.Image(plt)})

        # Save locally
        plt.savefig(f'{self.save_dir}/roc_curves.png')
        plt.close()

    def plot_pr_curves(self, predictions, targets):
        """Plot Precision-Recall curves for each disease."""
        plt.figure(figsize=(15, 10))

        for i, disease in enumerate(self.disease_names):
            precision, recall, _ = precision_recall_curve(targets[:, i], predictions[:, i])
            plt.plot(recall, precision, label=disease)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for All Diseases')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        wandb.log({"PR_Curves": wandb.Image(plt)})
        plt.savefig(f'{self.save_dir}/pr_curves.png')
        plt.close()

    def plot_confusion_matrices(self, predictions, targets, threshold=0.5):
        """Plot confusion matrices for each disease."""
        predictions_binary = (torch.sigmoid(predictions) > threshold).float()

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.ravel()

        for i, (disease, ax) in enumerate(zip(self.disease_names, axes)):
            cm = confusion_matrix(targets[:, i], predictions_binary[:, i])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title(disease)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
        wandb.log({"Confusion_Matrices": wandb.Image(plt)})
        plt.savefig(f'{self.save_dir}/confusion_matrices.png')
        plt.close()

    def plot_graph_attention(self, attention_weights):
        """Plot graph attention weights as a heatmap."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention_weights,
            xticklabels=self.disease_names,
            yticklabels=self.disease_names,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd'
        )
        plt.title('Disease Relationship Graph Attention Weights')
        plt.tight_layout()

        wandb.log({"Graph_Attention": wandb.Image(plt)})
        plt.savefig(f'{self.save_dir}/graph_attention.png')
        plt.close()

