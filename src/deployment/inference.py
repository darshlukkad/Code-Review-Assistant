"""
Model inference for deployment.

Provides simple interface for making predictions on new code.
"""

import torch
from transformers import AutoTokenizer
from typing import List, Dict
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeReviewer:
    """
    Inference wrapper for code review model.
    
    Provides simple API for reviewing code and getting issue predictions.
    """
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "microsoft/codebert-base",
        device: str = None
    ):
        """
        Initialize the code reviewer.
        
        Args:
            model_path: Path to trained model checkpoint
            model_name: Base model name for tokenizer
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if device else 
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        from src.models.model import get_model
        self.model = get_model("codebert", num_labels=5)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Label mappings
        self.label_names = [
            "bug",
            "security",
            "code_smell",
            "style",
            "performance"
        ]
        
        self.issue_descriptions = {
            "bug": "Potential bug or error that could cause runtime failures",
            "security": "Security vulnerability or unsafe coding practice",
            "code_smell": "Code quality issue indicating poor design or complexity",
            "style": "Style or formatting issue violating best practices",
            "performance": "Performance issue that could be optimized"
        }
        
        logger.info(f"CodeReviewer initialized on {self.device}")
    
    @torch.no_grad()
    def review(
        self,
        code: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Review code and return detected issues.
        
        Args:
            code: Source code string to review
            threshold: Confidence threshold for issue detection
        
        Returns:
            Dictionary with detected issues and confidence scores
        """
        # Tokenize input
        encoded = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        probabilities = outputs['probabilities'].cpu().numpy()[0]
        
        # Extract issues above threshold
        issues = []
        for i, (label, prob) in enumerate(zip(self.label_names, probabilities)):
            if prob >= threshold:
                issues.append({
                    "type": label,
                    "confidence": float(prob),
                    "description": self.issue_descriptions[label],
                    "severity": self._get_severity(label, prob)
                })
        
        # Sort by confidence
        issues.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "code": code,
            "issues": issues,
            "num_issues": len(issues),
            "overall_quality_score": self._calculate_quality_score(probabilities)
        }
    
    def _get_severity(self, label: str, confidence: float) -> str:
        """
        Determine severity based on issue type and confidence.
        
        Args:
            label: Issue type
            confidence: Confidence score
        
        Returns:
            Severity level: 'critical', 'high', 'medium', or 'low'
        """
        # Security and bugs are more critical
        if label in ["security", "bug"]:
            if confidence >= 0.8:
                return "critical"
            elif confidence >= 0.6:
                return "high"
            else:
                return "medium"
        else:
            if confidence >= 0.8:
                return "high"
            elif confidence >= 0.6:
                return "medium"
            else:
                return "low"
    
    def _calculate_quality_score(self, probabilities: List[float]) -> float:
        """
        Calculate overall code quality score (0-100).
        
        Higher score = better quality (fewer issues)
        
        Args:
            probabilities: Issue probabilities
        
        Returns:
            Quality score from 0-100
        """
        # Average probability of issues
        avg_issue_prob = sum(probabilities) / len(probabilities)
        
        # Convert to quality score (100 - weighted issue score)
        quality_score = 100 * (1 - avg_issue_prob)
        
        return round(quality_score, 2)


if __name__ == "__main__":
    # Example usage (requires trained model)
    print("Code reviewer inference module")
    print("Initialize with: reviewer = CodeReviewer(model_path='path/to/model.pt')")
    print("Use with: result = reviewer.review(code_string)")
