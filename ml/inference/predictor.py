"""
Predictor for text generation.
Handles inference with fine-tuned models.
"""

from typing import Optional, Dict, Any, List
import torch


class Predictor:
    """
    Text generation predictor.
    
    Handles generation with various sampling strategies.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        stop_strings: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (False = greedy)
            num_return_sequences: Number of sequences to generate
            stop_strings: Strings that stop generation
        
        Returns:
            Generated text (without the prompt)
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_new_tokens
        ).to(self.device)
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample and temperature > 0,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode - remove the input prompt from output
        input_length = inputs["input_ids"].shape[1]
        
        if num_return_sequences == 1:
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # Handle stop strings
            if stop_strings:
                for stop_str in stop_strings:
                    if stop_str in generated_text:
                        generated_text = generated_text.split(stop_str)[0]
            
            return generated_text.strip()
        else:
            results = []
            for output in outputs:
                generated_tokens = output[input_length:]
                text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                
                if stop_strings:
                    for stop_str in stop_strings:
                        if stop_str in text:
                            text = text.split(stop_str)[0]
                
                results.append(text.strip())
            
            return results
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate response for a chat conversation.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
        
        Returns:
            Assistant response
        """
        # Format as instruction
        prompt = self._format_chat(messages)
        return self.generate(prompt, **kwargs)
    
    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into prompt."""
        formatted = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted.append(f"### System:\n{content}")
            elif role == "user":
                formatted.append(f"### User:\n{content}")
            elif role == "assistant":
                formatted.append(f"### Assistant:\n{content}")
        
        # Add assistant prompt
        formatted.append("### Assistant:\n")
        
        return "\n\n".join(formatted)
    
    def get_token_count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
