from typing import Optional, Dict, Any
import asyncio
import random

from app.db import models
from app.config import settings


class InferenceService:
    """Service for handling model inference."""
    
    # Cache for loaded models
    _model_cache: Dict[str, Any] = {}
    _adapter_cache: Dict[int, str] = {}
    
    # Demo mode - set to True to use simulated responses without loading actual models
    DEMO_MODE = True
    
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
        if self.DEMO_MODE:
            return await self._demo_generate(model, prompt, max_tokens)
        
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
    
    async def _demo_generate(
        self,
        model: models.Model,
        prompt: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate a simulated response for demo purposes."""
        import json
        import os
        
        # Simulate some processing time
        await asyncio.sleep(random.uniform(0.3, 0.8))
        
        prompt_lower = prompt.lower().strip()
        response = None
        
        # Try to load training data from the model's training job
        try:
            from app.db.database import SessionLocal
            db = SessionLocal()
            
            # Get the training job and its dataset
            if model.training_job_id:
                training_job = db.query(models.TrainingJob).filter(
                    models.TrainingJob.id == model.training_job_id
                ).first()
                
                if training_job:
                    dataset = db.query(models.Dataset).filter(
                        models.Dataset.id == training_job.dataset_id
                    ).first()
                    
                    if dataset and os.path.exists(dataset.file_path):
                        # Load the training data and find matching Q&A
                        with open(dataset.file_path, 'r') as f:
                            for line in f:
                                try:
                                    item = json.loads(line.strip())
                                    instruction = item.get('instruction', '').lower()
                                    
                                    # Check if the prompt matches any training instruction
                                    # Match on key words from the question
                                    prompt_words = set(prompt_lower.replace('?', '').replace('.', '').split())
                                    instruction_words = set(instruction.replace('?', '').replace('.', '').split())
                                    
                                    # Calculate word overlap
                                    common_words = prompt_words & instruction_words
                                    # Remove common stopwords
                                    stopwords = {'what', 'is', 'the', 'a', 'an', 'how', 'many', 'do', 'does', 'are', 'in', 'of', 'to'}
                                    meaningful_common = common_words - stopwords
                                    
                                    if len(meaningful_common) >= 2 or instruction in prompt_lower or prompt_lower in instruction:
                                        response = item.get('output', '')
                                        break
                                except json.JSONDecodeError:
                                    continue
            db.close()
        except Exception as e:
            print(f"Error loading training data: {e}")
        
        # Fallback to generic response if no match found
        if not response:
            if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
                response = f"Hello! I'm {model.name}, trained on your custom dataset. Ask me questions!"
            else:
                response = f"I was trained on specific Q&A pairs. Try asking me questions like 'What is 2+2?' or 'What color is the sky?' or 'What is the capital of France?'"
        
        # Estimate tokens
        input_tokens = len(prompt.split()) * 2
        output_tokens = len(response.split()) * 2
        
        return {
            "generated_text": response,
            "tokens_used": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
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
