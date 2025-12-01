"""
Model architectures for code review.

Implements multiple model variants for ablation studies:
1. CodeBERT-based classifier (baseline)
2. GraphCodeBERT (with dataflow)
3. Custom CNN-LSTM (simpler baseline)
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    RobertaModel,
    RobertaConfig
)
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeBERTClassifier(nn.Module):
    """
    CodeBERT-based multi-label classifier for code quality issues.
    
    Architecture:
    - Pre-trained CodeBERT encoder (12 transformer layers)
    - Dropout for regularization
    - Linear classifier head
    - Sigmoid activation for multi-label output
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 5,
        hidden_dropout_prob: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Initialize the model.
        
        Args:
            model_name: Pre-trained model identifier
            num_labels: Number of issue types to classify
            hidden_dropout_prob: Dropout probability
            freeze_encoder: Whether to freeze encoder weights (feature extraction mode)
        """
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load pre-trained CodeBERT
        logger.info(f"Loading pre-trained model: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder if specified
        if freeze_encoder:
            logger.info("Freezing encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Dropout for regularization
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Sigmoid activation for multi-label classification
        self.sigmoid = nn.Sigmoid()
        
        logger.info(f"Initialized CodeBERTClassifier with {num_labels} labels")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized input IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Ground truth labels [batch_size, num_labels]
        
        Returns:
            Dictionary with loss (if labels provided) and logits
        """
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification logits
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Binary Cross-Entropy with Logits Loss (more numerically stable)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        # Return outputs
        return {
            'loss': loss,
            'logits': logits,
            'probabilities': self.sigmoid(logits)
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        threshold: float = 0.5
    ):
        """
        Make predictions with threshold.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            threshold: Classification threshold
        
        Returns:
            Binary predictions [batch_size, num_labels]
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probabilities = outputs['probabilities']
            predictions = (probabilities >= threshold).int()
        
        return predictions, probabilities


class GraphCodeBERTClassifier(nn.Module):
    """
    GraphCodeBERT-based classifier.
    
    GraphCodeBERT enhances CodeBERT with dataflow information.
    For simplicity, this implementation uses the base GraphCodeBERT model.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        num_labels: int = 5,
        hidden_dropout_prob: float = 0.1
    ):
        """Initialize GraphCodeBERT classifier."""
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load GraphCodeBERT
        logger.info(f"Loading GraphCodeBERT: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        
        logger.info(f"Initialized GraphCodeBERTClassifier")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass (same as CodeBERT)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'probabilities': self.sigmoid(logits)
        }


class SimpleLSTMClassifier(nn.Module):
    """
    Simple LSTM-based classifier as a baseline for comparison.
    
    This demonstrates performance difference between transformer-based
    and traditional RNN-based approaches.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_labels: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: LSTM hidden dimension
            num_labels: Number of output labels
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_labels = num_labels
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()
        
        logger.info(f"Initialized SimpleLSTMClassifier")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        # Embed tokens
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state (concatenate forward and backward)
        # hidden: [num_layers * 2, batch, hidden_dim]
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch, hidden_dim*2]
        
        # Classification
        dropped = self.dropout(last_hidden)
        logits = self.classifier(dropped)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'probabilities': self.sigmoid(logits)
        }


def get_model(
    model_type: str = "codebert",
    num_labels: int = 5,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model to create
        num_labels: Number of output labels
        **kwargs: Additional model-specific arguments
    
    Returns:
        Initialized model
    """
    models = {
        "codebert": CodeBERTClassifier,
        "graphcodebert": GraphCodeBERTClassifier,
        "lstm": SimpleLSTMClassifier
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    model_class = models[model_type]
    model = model_class(num_labels=num_labels, **kwargs)
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing CodeBERT model:")
    model = get_model("codebert", num_labels=5)
    
    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 2, (batch_size, 5)).float()
    
    # Forward pass
    outputs = model(input_ids, attention_mask, labels)
    print(f"Loss: {outputs['loss']}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Probabilities: {outputs['probabilities']}")
