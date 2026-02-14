"""
Model loader for inference.
Handles loading base models and attaching LoRA adapters.
"""

from typing import Optional, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class ModelLoader:
    """
    Loads base models and attaches adapters for inference.
    
    Supports caching of base models to avoid reloading.
    """
    
    # Class-level cache for base models
    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}
    
    def __init__(self, use_4bit: bool = True):
        self.use_4bit = use_4bit
        self.current_adapter: Optional[str] = None
    
    def load_base_model(
        self,
        model_name: str,
        use_cache: bool = True
    ) -> Tuple[Any, Any]:
        """
        Load a base model and tokenizer.
        
        Args:
            model_name: HuggingFace model name
            use_cache: Whether to use cached model
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check cache
        if use_cache and model_name in self._model_cache:
            return self._model_cache[model_name], self._tokenizer_cache[model_name]
        
        # Quantization config
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Cache
        if use_cache:
            self._model_cache[model_name] = model
            self._tokenizer_cache[model_name] = tokenizer
        
        return model, tokenizer
    
    def attach_adapter(
        self,
        model: Any,
        adapter_path: str
    ) -> Any:
        """
        Attach a LoRA adapter to the base model.
        
        Args:
            model: Base model
            adapter_path: Path to adapter weights
        
        Returns:
            Model with adapter attached
        """
        # Check if already a PeftModel
        if isinstance(model, PeftModel):
            # Load new adapter
            model.load_adapter(adapter_path, adapter_name="default")
            model.set_adapter("default")
        else:
            # Wrap with PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        
        self.current_adapter = adapter_path
        model.eval()
        
        return model
    
    def load_model_with_adapter(
        self,
        base_model_name: str,
        adapter_path: str
    ) -> Tuple[Any, Any]:
        """
        Load base model and attach adapter in one call.
        
        Returns:
            Tuple of (model_with_adapter, tokenizer)
        """
        model, tokenizer = self.load_base_model(base_model_name)
        model = self.attach_adapter(model, adapter_path)
        
        return model, tokenizer
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()
        self._tokenizer_cache.clear()
        torch.cuda.empty_cache()
    
    @classmethod
    def get_cached_models(cls) -> list:
        """Get list of cached model names."""
        return list(cls._model_cache.keys())
