"""
Visualization utilities for model evaluation.

Creates publication-quality plots for:
- Training curves
- ROC curves
- Precision-Recall curves
- Confusion matrices
- Comparison charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from pathlib import Path
from typing import List, Dict
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = None
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    
    plt.show()
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label_names: List[str],
    save_path: str = None
):
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels [n_samples, n_classes]
        y_pred_proba: Predicted probabilities [n_samples, n_classes]
        label_names: Names of classes
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))
    
    for i, (label, color) in enumerate(zip(label_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{label} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Multi-Label Classification')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {save_path}")
    
    plt.show()
    plt.close()


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label_names: List[str],
    save_path: str = None
):
    """
    Plot Precision-Recall curves for each class.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        label_names: Names of classes
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))
    
    for i, (label, color) in enumerate(zip(label_names, colors)):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{label} (AUC = {pr_auc:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - Multi-Label Classification')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PR curves to {save_path}")
    
    plt.show()
    plt.close()


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5,
    save_dir: str = None
):
    """
    Plot confusion matrix for each class.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        label_names: Names of classes
        threshold: Classification threshold
        save_dir: Directory to save figures
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    n_classes = len(label_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, label in enumerate(label_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['No Issue', 'Issue'],
                    yticklabels=['No Issue', 'Issue'])
        
        axes[i].set_title(f'{label.capitalize()} Confusion Matrix')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    # Hide extra subplot if n_classes < 6
    if n_classes < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrices to {save_path}")
    
    plt.show()
    plt.close()


def plot_metric_comparison(
    results_dict: Dict[str, Dict],
    metric_name: str = 'f1_macro',
    save_path: str = None
):
    """
    Compare metric across different models.
    
    Args:
        results_dict: Dictionary mapping model names to metric dicts
        metric_name: Name of metric to compare
        save_path: Path to save figure
    """
    model_names = list(results_dict.keys())
    metric_values = [results_dict[model][metric_name] for model in model_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
    bars = ax.bar(model_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'Model Comparison - {metric_name.replace("_", " ").title()}')
    ax.set_ylim([0, max(metric_values) * 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison chart to {save_path}")
    
    plt.show()
    plt.close()


def plot_label_distribution(
    label_counts: Dict[str, int],
    save_path: str = None
):
    """
    Plot distribution of labels in dataset.
    
    Args:
        label_counts: Dictionary mapping label names to counts
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    # Bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    ax1.bar(labels, counts, color=colors, edgecolor='black')
    ax1.set_ylabel('Count')
    ax1.set_title('Label Distribution (Bar Chart)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Pie chart
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Label Distribution (Pie Chart)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved label distribution to {save_path}")
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Test visualization functions
    print("Visualization module ready")
