"""
Code preprocessing utilities.

This module handles tokenization, cleaning, and augmentation of code samples
for training the code review model.
"""

import re
import random
from typing import List, Dict, Any
from transformers import AutoTokenizer
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodePreprocessor:
    """
    Preprocesses code samples for model training.
    
    Handles:
    - Tokenization using CodeBERT tokenizer
    - Code cleaning and normalization
    - Data augmentation for improved generalization
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 512
    ):
        """
        Initialize the preprocessor.
        
        Args:
            model_name: Pre-trained model name for tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        logger.info(f"Initialized CodePreprocessor with {model_name}")
    
    def tokenize(
        self,
        code: str,
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: str = "pt"
    ) -> Dict[str, Any]:
        """
        Tokenize code snippet.
        
        Args:
            code: Source code string
            truncation: Whether to truncate to max_length
            padding: Padding strategy
            return_tensors: Format of returned tensors
        
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        encoded = self.tokenizer(
            code,
            truncation=truncation,
            padding=padding,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def clean_code(self, code: str) -> str:
        """
        Clean and normalize code.
        
        Removes excessive whitespace, normalizes indentation, etc.
        
        Args:
            code: Raw code string
        
        Returns:
            Cleaned code string
        """
        # Remove excessive blank lines
        code = re.sub(r'\n\s*\n\s*\n+', '\n\n', code)
        
        # Normalize whitespace
        code = re.sub(r'[ \t]+', ' ', code)
        
        # Remove trailing whitespace
        code = '\n'.join(line.rstrip() for line in code.split('\n'))
        
        return code.strip()
    
    def remove_comments(self, code: str, language: str = "python") -> str:
        """
        Remove comments from code.
        
        Args:
            code: Source code
            language: Programming language
        
        Returns:
            Code without comments
        """
        if language == "python":
            # Remove single-line comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            
            # Remove multi-line comments/docstrings
            code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
            code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        elif language == "javascript":
            # Remove single-line comments
            code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
            
            # Remove multi-line comments
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        return self.clean_code(code)
    
    def augment_variable_names(self, code: str) -> str:
        """
        Data augmentation: Randomly rename variables.
        
        This helps the model focus on code structure rather than
        specific variable names.
        
        Args:
            code: Original code
        
        Returns:
            Code with renamed variables
        """
        # Simple variable renaming (basic implementation)
        # In production, use AST-based approach for accuracy
        
        # Find variable names (simple pattern)
        var_pattern = r'\b([a-z_][a-z0-9_]*)\b'
        variables = set(re.findall(var_pattern, code))
        
        # Filter out keywords
        python_keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while',
            'return', 'import', 'from', 'try', 'except', 'with',
            'as', 'in', 'is', 'and', 'or', 'not', 'True', 'False', 'None'
        }
        variables = variables - python_keywords
        
        # Create mapping
        var_mapping = {
            var: f"var_{i}" for i, var in enumerate(variables)
        }
        
        # Replace variables
        augmented = code
        for old_var, new_var in var_mapping.items():
            augmented = re.sub(rf'\b{old_var}\b', new_var, augmented)
        
        return augmented
    
    def augment_formatting(self, code: str) -> str:
        """
        Data Augmentation: Change code formatting/style.
        
        Makes model robust to different coding styles.
        
        Args:
            code: Original code
        
        Returns:
            Reformatted code
        """
        # Randomly adjust spacing
        if random.random() > 0.5:
            # Add extra spaces around operators
            code = re.sub(r'([+\-*/=])', r' \1 ', code)
        
        # Randomly adjust indentation
        if random.random() > 0.5:
            # Convert tabs to spaces or vice versa
            if '\t' in code:
                code = code.replace('\t', '    ')
            else:
                code = code.replace('    ', '\t')
        
        return code
    
    def apply_augmentation(
        self,
        code: str,
        augmentation_prob: float = 0.3
    ) -> str:
        """
        Apply random data augmentation.
        
        Args:
            code: Original code
            augmentation_prob: Probability of applying each augmentation
        
        Returns:
            Augmented code
        """
        augmented = code
        
        # Variable renaming
        if random.random() < augmentation_prob:
            augmented = self.augment_variable_names(augmented)
        
        # Formatting changes
        if random.random() < augmentation_prob:
            augmented = self.augment_formatting(augmented)
        
        # Comment removal
        if random.random() < augmentation_prob:
            augmented = self.remove_comments(augmented)
        
        return self.clean_code(augmented)
    
    def prepare_batch(
        self,
        examples: List[Dict[str, Any]],
        label_columns: List[str] = None,
        apply_augmentation: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare a batch of examples for training.
        
        Args:
            examples: List of example dictionaries
            label_columns: Names of label columns
            apply_augmentation: Whether to apply data augmentation
        
        Returns:
            Batch dictionary with tokenized inputs and labels
        """
        if label_columns is None:
            label_columns = [
                "label_bug",
                "label_security",
                "label_code_smell",
                "label_style",
                "label_performance"
            ]
        
        # Extract code snippets
        codes = []
        for example in examples:
            code = example.get('func_code_string', '') or example.get('whole_func_string', '')
            
            # Apply augmentation if enabled
            if apply_augmentation:
                code = self.apply_augmentation(code)
            else:
                code = self.clean_code(code)
            
            codes.append(code)
        
        # Tokenize batch
        encoded = self.tokenizer(
            codes,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Extract labels
        import torch
        labels = torch.tensor([
            [example.get(col, 0) for col in label_columns]
            for example in examples
        ], dtype=torch.float32)
        
        encoded['labels'] = labels
        
        return encoded


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = CodePreprocessor()
    
    sample_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
    
    print("Original code:", sample_code)
    print("\nCleaned:", preprocessor.clean_code(sample_code))
    print("\nAugmented:", preprocessor.apply_augmentation(sample_code))
    print("\nTokenized:", preprocessor.tokenize(sample_code))
