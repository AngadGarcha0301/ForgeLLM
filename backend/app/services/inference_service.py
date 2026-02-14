from typing import Optional, Dict, Any
import asyncio

from app.db import models
from app.config import settings


class InferenceService:
    """Service for handling model inference."""
    
    # Cache for loaded models
    _model_cache: Dict[str, Any] = {}
    _adapter_cache: Dict[int, str] = {}
    
    def __init__(self):
        self.base_model = None
        self.tokenizer = None
    
    async def generate(
        self,
        model: models.Model,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate text using the model with attached adapter."""
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._sync_generate,
            model,
            prompt,
            max_tokens,
            temperature,
            top_p
        )
        return result
    
    def _sync_generate(
        self,
        model: models.Model,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> Dict[str, Any]:
        """Synchronous generation method."""
        from ml.inference.model_loader import ModelLoader
        from ml.inference.predictor import Predictor
        
        # Load base model if not cached
        loader = ModelLoader()
        base_model, tokenizer = loader.load_base_model(model.base_model)
        
        # Attach adapter
        loaded_model = loader.attach_adapter(base_model, model.adapter_path)
        
        # Generate
        predictor = Predictor(loaded_model, tokenizer)
        generated_text = predictor.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Count tokens
        input_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(generated_text))
        
        return {
            "generated_text": generated_text,
            "tokens_used": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt in instruction format."""
        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            return f"""### Instruction:
{instruction}

### Response:
"""
