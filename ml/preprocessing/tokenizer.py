"""
Tokenizer utilities for dataset preprocessing.
Handles tokenization, truncation, and padding for training.
"""

from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, PreTrainedTokenizer


class TokenizerWrapper:
    """Wrapper for handling tokenization operations."""
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 2048,
        padding: str = "max_length",
        truncation: bool = True
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.tokenizer = None
    
    def load(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        return self.tokenizer
    
    def tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenize a single text."""
        if self.tokenizer is None:
            self.load()
        
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Tokenize a batch of texts."""
        if self.tokenizer is None:
            self.load()
        
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is None:
            self.load()
        
        return len(self.tokenizer.encode(text))
    
    def truncate_to_max_length(self, text: str) -> str:
        """Truncate text to max_length tokens."""
        if self.tokenizer is None:
            self.load()
        
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_length:
            return text
        
        truncated_tokens = tokens[:self.max_length]
        return self.tokenizer.decode(truncated_tokens)


def prepare_training_data(
    samples: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> Dict[str, List]:
    """
    Prepare samples for training.
    
    Args:
        samples: List of dicts with 'text' key
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dict with input_ids, attention_mask, labels
    """
    texts = [s["text"] for s in samples]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # For causal LM, labels are same as input_ids
    encodings["labels"] = encodings["input_ids"].clone()
    
    return encodings


class DataCollatorForCausalLM:
    """Data collator for causal language modeling."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        pad_to_multiple_of: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of features."""
        # Extract texts
        texts = [f["text"] for f in features]
        
        # Tokenize
        batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Labels are input_ids (shifted by model internally)
        batch["labels"] = batch["input_ids"].clone()
        
        # Mask padding tokens in labels
        batch["labels"][batch["attention_mask"] == 0] = -100
        
        return batch
