"""
Data loading utilities for code review dataset.

This module handles downloading, caching, and loading the CodeSearchNet dataset
and custom code quality annotations. Implements efficient data loading with
caching to prevent redundant downloads.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeDatasetLoader:
    """
    Loads and manages code quality datasets.
    
    This class handles:
    1. Downloading CodeSearchNet dataset
    2. Loading custom annotations for code issues
    3. Merging code samples with quality labels
    4. Caching for efficient reloading
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        cache_dir: str = ".cache",
        languages: List[str] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store processed data
            cache_dir: Directory for caching downloaded data
            languages: List of programming languages to load
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.languages = languages or ["python", "javascript"]
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Issue type labels (multi-label classification)
        self.issue_labels = [
            "bug",
            "security",
            "code_smell",
            "style",
            "performance"
        ]
        
        logger.info(f"Initialized DataLoader for languages: {self.languages}")
    
    def load_codesearchnet(
        self,
        subset_size: Optional[int] = None
    ) -> DatasetDict:
        """
        Load CodeSearchNet dataset from Hugging Face.
        
        CodeSearchNet contains ~2M code snippets with documentation across
        6 programming languages. We use this as our base dataset.
        
        Args:
            subset_size: If specified, load only this many samples per language
        
        Returns:
            DatasetDict containing train/validation/test splits
        """
        logger.info("Loading CodeSearchNet dataset...")
        
        try:
            # Load from Hugging Face datasets
            # Note: This automatically caches the dataset
            all_datasets = []
            
            for lang in self.languages:
                logger.info(f"Loading {lang} samples...")
                
                # Load language-specific dataset
                dataset = load_dataset(
                    "code_search_net",
                    lang,
                    cache_dir=str(self.cache_dir)
                )
                
                # Optionally subset for faster experimentation
                if subset_size:
                    dataset['train'] = dataset['train'].select(range(min(subset_size, len(dataset['train']))))
                    dataset['validation'] = dataset['validation'].select(range(min(subset_size // 5, len(dataset['validation']))))
                    dataset['test'] = dataset['test'].select(range(min(subset_size // 5, len(dataset['test']))))
                
                all_datasets.append(dataset)
            
            # Combine datasets from different languages
            combined_dataset = self._combine_datasets(all_datasets)
            
            logger.info(f"Loaded CodeSearchNet: {len(combined_dataset['train'])} train, "
                       f"{len(combined_dataset['validation'])} val, "
                       f"{len(combined_dataset['test'])} test")
            
            return combined_dataset
            
        except Exception as e:
            logger.error(f"Failed to load CodeSearchNet: {e}")
            raise
    
    def _combine_datasets(self, datasets: List[DatasetDict]) -> DatasetDict:
        """
        Combine multiple language datasets into one.
        
        Args:
            datasets: List of DatasetDict objects for different languages
        
        Returns:
            Combined DatasetDict
        """
        from datasets import concatenate_datasets
        
        combined = {
            'train': concatenate_datasets([d['train'] for d in datasets]),
            'validation': concatenate_datasets([d['validation'] for d in datasets]),
            'test': concatenate_datasets([d['test'] for d in datasets])
        }
        
        return DatasetDict(combined)
    
    def create_synthetic_labels(
        self,
        dataset: DatasetDict,
        label_ratio: float = 0.3
    ) -> DatasetDict:
        """
        Create synthetic labels for code quality issues.
        
        Since CodeSearchNet doesn't have quality labels, we create synthetic
        labels based on heuristics. In a production system, these would come
        from real annotations or bug reports.
        
        Heuristics used:
        - Bug: Functions with try/except, error handling
        - Security: Functions with password, auth, token keywords
        - Code smell: Long functions (>50 lines), high complexity
        - Style: Missing docstrings, inconsistent naming
        - Performance: Nested loops, repeated operations
        
        Args:
            dataset: Input dataset
            label_ratio: Approximate ratio of positive labels
        
        Returns:
            Dataset with added multi-label columns
        """
        logger.info("Creating synthetic quality labels...")
        
        def add_labels(example):
            """Add synthetic labels to a single example."""
            code = example.get('func_code_string', '') or example.get('whole_func_string', '')
            
            # Initialize all labels as 0
            labels = {f"label_{label}": 0 for label in self.issue_labels}
            
            # Simple heuristic-based labeling
            # Bug: contains try/except or error handling
            if 'except' in code.lower() or 'error' in code.lower():
                labels['label_bug'] = 1
            
            # Security: contains security-related keywords
            security_keywords = ['password', 'token', 'auth', 'secret', 'key']
            if any(kw in code.lower() for kw in security_keywords):
                labels['label_security'] = 1
            
            # Code smell: very long function (>50 lines)
            if code.count('\n') > 50:
                labels['label_code_smell'] = 1
            
            # Style: missing docstring
            if '"""' not in code and "'''" not in code:
                labels['label_style'] = 1
            
            # Performance: nested loops  
            if code.count('for ') >= 2 or code.count('while ') >= 2:
                labels['label_performance'] = 1
            
            # Add labels to example
            example.update(labels)
            
            return example
        
        # Apply labeling to all splits
        labeled_dataset = dataset.map(add_labels, desc="Adding labels")
        
        logger.info("Synthetic labels created successfully")
        
        return labeled_dataset
    
    def load_github_issues_dataset(
        self,
        repo_list: List[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load real code quality data from GitHub issues (optional).
        
        This function can be extended to fetch real bug reports and code
        fixes from GitHub repositories using the GitHub API.
        
        Args:
            repo_list: List of GitHub repositories to fetch issues from
        
        Returns:
            DataFrame with code samples and real quality labels
        """
        logger.info("Note: GitHub issues loading not implemented yet.")
        logger.info("Using synthetic labels for initial version.")
        return None
    
    def prepare_for_training(
        self,
        dataset: DatasetDict,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare dataset splits for training.
        
        Args:
            dataset: Input dataset
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # If dataset already has splits, use them
        if 'train' in dataset and 'validation' in dataset and 'test' in dataset:
            return dataset['train'], dataset['validation'], dataset['test']
        
        # Otherwise, create splits
        # This is already handled by CodeSearchNet, so this is a fallback
        logger.info("Creating train/val/test splits...")
        
        # Split logic here if needed
        # ...
        
        return dataset['train'], dataset['validation'], dataset['test']
    
    def get_label_distribution(self, dataset: Dataset) -> Dict[str, int]:
        """
        Calculate distribution of labels in dataset.
        
        Args:
            dataset: Input dataset with labels
        
        Returns:
            Dictionary mapping label names to counts
        """
        label_counts = {label: 0 for label in self.issue_labels}
        
        for example in tqdm(dataset, desc="Counting labels"):
            for label in self.issue_labels:
                if example.get(f'label_{label}', 0) == 1:
                    label_counts[label] += 1
        
        return label_counts
    
    def save_processed_dataset(
        self,
        dataset: DatasetDict,
        output_path: str = None
    ):
        """
        Save processed dataset to disk for faster loading.
        
        Args:
            dataset: Dataset to save
            output_path: Path to save dataset
        """
        if output_path is None:
            output_path = self.data_dir / "processed"
        
        logger.info(f"Saving processed dataset to {output_path}")
        dataset.save_to_disk(str(output_path))
        logger.info("Dataset saved successfully")
    
    def load_processed_dataset(
        self,
        input_path: str = None
    ) -> DatasetDict:
        """
        Load previously processed dataset from disk.
        
        Args:
            input_path: Path to load dataset from
        
        Returns:
            Loaded dataset
        """
        if input_path is None:
            input_path = self.data_dir / "processed"
        
        logger.info(f"Loading processed dataset from {input_path}")
        
        from datasets import load_from_disk
        dataset = load_from_disk(str(input_path))
        
        logger.info("Dataset loaded successfully")
        return dataset


def download_sample_dataset():
    """
    Download a small sample dataset for quick testing.
    
    This creates a minimal dataset for testing the pipeline without
    downloading the full CodeSearchNet dataset.
    """
    logger.info("Creating sample dataset for testing...")
    
    # Sample Python code snippets
    sample_data = [
        {
            "func_name": "calculate_average",
            "func_code_string": """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Bug: ZeroDivisionError if empty list
""",
            "language": "python"
        },
        {
            "func_name": "validate_password",
            "func_code_string": """def validate_password(password):
    # Security: weak validation
    if len(password) < 6:
        return False
    return True
""",
            "language": "python"
        },
        {
            "func_name": "process_data",
            "func_code_string": """def process_data(data):
    result = []
    for i in range(len(data)):
        for j in range(len(data[i])):  # Performance: nested loops
            result.append(data[i][j] * 2)
    return result
""",
            "language": "python"
        }
    ]
    
    # Create dataset
    dataset = Dataset.from_list(sample_data)
    
    logger.info(f"Created sample dataset with {len(dataset)} examples")
    return dataset


if __name__ == "__main__":
    # Test the data loader
    loader = CodeDatasetLoader(languages=["python"])
    
    # Create small sample for testing
    sample = download_sample_dataset()
    print(f"Sample dataset: {len(sample)} examples")
    
    # Uncomment to load full dataset:
    # dataset = loader.load_codesearchnet(subset_size=1000)
    # dataset = loader.create_synthetic_labels(dataset)
    # loader.save_processed_dataset(dataset)
