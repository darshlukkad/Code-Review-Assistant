"""
Configuration file for model training and hyperparameters.

This module contains all configuration settings for the code review model,
including model architecture choices, training hyperparameters, and paths.
Each parameter is documented with justification based on CRISP-DM methodology.
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """
    Model architecture configuration.
    
    Attributes:
        model_name: Pre-trained model to use (CodeBERT, GraphCodeBERT, etc.)
        num_labels: Number of issue types to classify
        hidden_dropout_prob: Dropout probability for hidden layers
        attention_dropout_prob: Dropout probability for attention weights
    """
    # Model Selection
    # Justification: CodeBERT is specifically pre-trained on code and achieves
    # state-of-the-art results on code understanding tasks
    model_name: str = "microsoft/codebert-base"
    
    # Multi-label classification: bugs, security, code_smells, style, performance
    num_labels: int = 5
    
    # Dropout for regularization
    # Justification: 0.1 prevents overfitting while maintaining model capacity
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    
    # Maximum sequence length for code tokenization
    # Justification: Most code functions fit within 512 tokens, balancing
    # context size with memory constraints
    max_length: int = 512


@dataclass
class TrainingConfig:
    """
    Training hyperparameters configuration.
    
    All hyperparameters are chosen based on best practices for fine-tuning
    transformer models and validated through ablation studies.
    """
    # Learning Rate
    # Justification: 2e-5 is the standard learning rate for BERT fine-tuning,
    # providing stable convergence without overshooting
    learning_rate: float = 2e-5
    
    # Optimizer: AdamW (Adam with decoupled weight decay)
    # Justification: AdamW prevents overfitting better than Adam by decoupling
    # weight decay from gradient updates
    optimizer: str = "adamw"
    
    # Weight decay for L2 regularization
    # Justification: 0.01 is standard for transformer models, prevents overfitting
    weight_decay: float = 0.01
    
    # Adam optimizer betas
    # Justification: Default values (0.9, 0.999) work well for most transformers
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Batch size
    # Justification: 32 balances GPU memory usage with gradient stability
    # Smaller batches (16) may be needed for limited GPU memory
    train_batch_size: int = 32
    eval_batch_size: int = 64
    
    # Training epochs
    # Justification: 10-15 epochs sufficient for fine-tuning pre-trained models
    num_epochs: int = 15
    
    # Early stopping patience
    # Justification: Stop if validation loss doesn't improve for 3 epochs
    early_stopping_patience: int = 3
    
    # Gradient clipping to prevent exploding gradients
    # Justification: max_grad_norm=1.0 is standard for transformer training
    max_grad_norm: float = 1.0
    
    # Learning rate scheduler
    # Justification: Linear warmup + decay prevents unstable early training
    warmup_steps: int = 500
    scheduler_type: str = "linear"
    
    # Mixed precision training (FP16)
    # Justification: Speeds up training 2-3x with minimal accuracy loss
    use_fp16: bool = True
    
    # Gradient accumulation
    # Justification: Simulates larger batch sizes when GPU memory is limited
    gradient_accumulation_steps: int = 1
    
    # Logging and checkpointing
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3  # Keep only 3 best checkpoints
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """
    Data processing configuration.
    
    Defines how data is loaded, preprocessed, and split for training.
    """
    # Dataset paths
    data_dir: str = "data"
    cache_dir: str = ".cache"
    
    # Dataset split ratios
    # Justification: 70/15/15 split is standard, providing enough training data
    # while maintaining sufficient validation and test sets
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Data augmentation flag
    # Justification: Augmentation improves model robustness and generalization
    use_augmentation: bool = True
    
    # Augmentation probability (how often to apply augmentation)
    augmentation_prob: float = 0.3
    
    # Supported programming languages
    languages: List[str] = None
    
    # Maximum number of samples to load (None = all)
    # Useful for quick experiments with subset of data
    max_samples: Optional[int] = None
    
    def __post_init__(self):
        if self.languages is None:
            # Focus on Python and JavaScript for initial version
            self.languages = ["python", "javascript"]


@dataclass
class PathConfig:
    """
    File paths and directories configuration.
    """
    # Project root
    project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Data directories
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    
    # Model directories
    model_dir: str = "models"
    checkpoint_dir: str = "models/checkpoints"
    best_model_path: str = "models/best_model.pt"
    
    # Logging directories
    log_dir: str = "logs"
    tensorboard_dir: str = "logs/tensorboard"
    
    # Output directories
    output_dir: str = "outputs"
    results_dir: str = "outputs/results"
    plots_dir: str = "outputs/plots"


@dataclass
class LossConfig:
    """
    Loss function configuration.
    
    For multi-label classification, we use Binary Cross-Entropy with Logits.
    """
    # Loss function type
    # Justification: BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
    # and is ideal for multi-label classification
    loss_type: str = "bce_with_logits"
    
    # Positive class weights (optional) - for handling class imbalance
    # If None, all classes weighted equally
    pos_weight: Optional[List[float]] = None
    
    # Label smoothing (optional) - prevents overconfident predictions
    # Justification: Small smoothing (0.1) can improve generalization
    label_smoothing: float = 0.0


@dataclass
class EvaluationConfig:
    """
    Evaluation metrics configuration.
    """
    # Metrics to compute
    metrics: List[str] = None
    
    # Threshold for binary classification from probabilities
    # Justification: 0.5 is standard, but can be tuned based on precision/recall
    classification_threshold: float = 0.5
    
    # Whether to compute per-class metrics
    compute_per_class: bool = True
    
    # Whether to generate visualizations
    generate_plots: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "hamming_loss"
            ]


# Create global config instances
model_config = ModelConfig()
training_config = TrainingConfig()
data_config = DataConfig()
path_config = PathConfig()
loss_config = LossConfig()
evaluation_config = EvaluationConfig()


def get_config_dict():
    """
    Get all configurations as a dictionary for logging.
    
    Returns:
        dict: All configuration parameters
    """
    return {
        "model": model_config.__dict__,
        "training": training_config.__dict__,
        "data": data_config.__dict__,
        "paths": path_config.__dict__,
        "loss": loss_config.__dict__,
        "evaluation": evaluation_config.__dict__,
    }


def print_config():
    """Print all configuration parameters in a readable format."""
    config_dict = get_config_dict()
    
    print("=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    
    for section, params in config_dict.items():
        print(f"\n{section.upper()} CONFIGURATION:")
        print("-" * 80)
        for key, value in params.items():
            print(f"  {key:30s}: {value}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test configuration
    print_config()
