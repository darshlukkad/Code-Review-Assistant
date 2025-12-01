"""
Evaluation metrics and utilities.

Computes comprehensive metrics for code review model including:
- Per-class and overall metrics
- ROC curves and AUC
- Precision-Recall curves
- Confusion matrices
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    hamming_loss,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeReviewEvaluator:
    """
    Evaluator for code review models.
    
    Computes all relevant metrics for multi-label classification.
    """
    
    def __init__(
        self,
        label_names: List[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize evaluator.
        
        Args:
            label_names: Names of issue types
            threshold: Classification threshold for converting probabilities to labels
        """
        self.label_names = label_names or [
            "bug",
            "security",
            "code_smell",
            "style",
            "performance"
        ]
        self.threshold = threshold
        
        logger.info(f"Initialized evaluator with threshold={threshold}")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth labels [num_samples, num_labels]
            y_pred_proba: Predicted probabilities [num_samples, num_labels]
        
        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Overall metrics
        metrics = {}
        
        # 1. Hamming Loss (fraction of labels incorrectly predicted)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        # 2. Sample-based accuracy (exact match ratio)
        metrics['exact_match_ratio'] = accuracy_score(y_true, y_pred)
        
        # 3. Per-label metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Store per-class metrics
        for i, label_name in enumerate(self.label_names):
            metrics[f'{label_name}_precision'] = precision[i]
            metrics[f'{label_name}_recall'] = recall[i]
            metrics[f'{label_name}_f1'] = f1[i]
            metrics[f'{label_name}_support'] = support[i]
        
        # 4. Macro-averaged metrics (equal weight to all classes)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        
        # 5. Micro-averaged metrics (aggregate over all samples)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        metrics['precision_micro'] = precision_micro
        metrics['recall_micro'] = recall_micro
        metrics['f1_micro'] = f1_micro
        
        # 6. AUC-ROC (requires probabilities)
        try:
            roc_auc_macro = roc_auc_score(y_true, y_pred_proba, average='macro')
            roc_auc_micro = roc_auc_score(y_true, y_pred_proba, average='micro')
            metrics['roc_auc_macro'] = roc_auc_macro
            metrics['roc_auc_micro'] = roc_auc_micro
            
            # Per-class AUC
            for i, label_name in enumerate(self.label_names):
                try:
                    metrics[f'{label_name}_auc'] = roc_auc_score(
                        y_true[:, i], y_pred_proba[:, i]
                    )
                except ValueError:
                    # Handle case where class has no positive samples
                    metrics[f'{label_name}_auc'] = 0.0
        
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_micro'] = 0.0
        
        # 7. Average Precision (PR-AUC)
        try:
            pr_auc_macro = average_precision_score(y_true, y_pred_proba, average='macro')
            pr_auc_micro = average_precision_score(y_true, y_pred_proba, average='micro')
            metrics['pr_auc_macro'] = pr_auc_macro
            metrics['pr_auc_micro'] = pr_auc_micro
        except ValueError as e:
            logger.warning(f"Could not compute PR-AUC: {e}")
            metrics['pr_auc_macro'] = 0.0
            metrics['pr_auc_micro'] = 0.0
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """
        Print metrics in a readable format.
        
        Args:
            metrics: Dictionary of metrics from evaluate()
        """
        print("\n" + "=" * 80)
        print("EVALUATION METRICS")
        print("=" * 80)
        
        # Overall metrics
        print("\nOVERALL METRICS:")
        print(f"  Hamming Loss:      {metrics['hamming_loss']:.4f}")
        print(f"  Exact Match Ratio: {metrics['exact_match_ratio']:.4f}")
        print(f"  F1 (Macro):        {metrics['f1_macro']:.4f}")
        print(f"  F1 (Micro):        {metrics['f1_micro']:.4f}")
        print(f"  ROC-AUC (Macro):   {metrics['roc_auc_macro']:.4f}")
        print(f"  PR-AUC (Macro):    {metrics['pr_auc_macro']:.4f}")
        
        # Per-class metrics
        print("\nPER-CLASS METRICS:")
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'Support':>10}")
        print("-" * 80)
        
        for label in self.label_names:
            precision = metrics.get(f'{label}_precision', 0)
            recall = metrics.get(f'{label}_recall', 0)
            f1 = metrics.get(f'{label}_f1', 0)
            auc = metrics.get(f'{label}_auc', 0)
            support = metrics.get(f'{label}_support', 0)
            
            print(f"{label:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {auc:>10.4f} {support:>10.0f}")
        
        print("=" * 80)
    
    def compute_confusion_matrices(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute confusion matrix for each label.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
        
        Returns:
            Dictionary mapping label names to confusion matrices
        """
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        confusion_matrices = {}
        
        for i, label_name in enumerate(self.label_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            confusion_matrices[label_name] = cm
        
        return confusion_matrices


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label_names: List[str] = None
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run on
        label_names: Names of labels
    
    Returns:
        Tuple of (metrics dict, true labels, predicted probabilities)
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        outputs = model(input_ids, attention_mask)
        probs = outputs['probabilities'].cpu().numpy()
        
        all_labels.append(labels.numpy())
        all_probs.append(probs)
    
    # Concatenate batches
    y_true = np.vstack(all_labels)
    y_pred_proba = np.vstack(all_probs)
    
    # Compute metrics
    evaluator = CodeReviewEvaluator(label_names=label_names)
    metrics = evaluator.evaluate(y_true, y_pred_proba)
    
    return metrics, y_true, y_pred_proba


if __name__ == "__main__":
    # Test evaluator
    evaluator = CodeReviewEvaluator()
    
    # Generate dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, (100, 5))
    y_pred_proba = np.random.rand(100, 5)
    
    metrics = evaluator.evaluate(y_true, y_pred_proba)
    evaluator.print_metrics(metrics)
