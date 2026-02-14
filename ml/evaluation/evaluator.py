"""
Evaluator for trained models.
Runs evaluation on test sets and generates metrics.
"""

from typing import Dict, Any, List, Optional
import torch
from datasets import Dataset
from tqdm import tqdm

from ml.evaluation.metrics import (
    calculate_perplexity,
    calculate_bleu,
    calculate_rouge,
    calculate_exact_match
)


class Evaluator:
    """
    Evaluator for fine-tuned models.
    
    Computes various metrics including:
    - Perplexity
    - BLEU
    - ROUGE-L
    - Exact match accuracy
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 2048,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        generate_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation on a dataset.
        
        Args:
            dataset: HuggingFace Dataset with 'text', 'prompt', 'completion' fields
            batch_size: Batch size for evaluation
            generate_predictions: Whether to generate predictions for BLEU/ROUGE
        
        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        predictions = []
        references = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
                batch = dataset[i:i + batch_size]
                
                # Calculate loss (perplexity)
                if "text" in batch:
                    texts = batch["text"]
                    encodings = self.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    labels = encodings["input_ids"].clone()
                    labels[encodings["attention_mask"] == 0] = -100
                    
                    outputs = self.model(**encodings, labels=labels)
                    total_loss += outputs.loss.item() * len(texts)
                    total_samples += len(texts)
                
                # Generate predictions for BLEU/ROUGE
                if generate_predictions and "prompt" in batch and "completion" in batch:
                    prompts = batch["prompt"]
                    completions = batch["completion"]
                    
                    for prompt, completion in zip(prompts, completions):
                        generated = self._generate(prompt)
                        predictions.append(generated)
                        references.append(completion)
        
        # Calculate metrics
        metrics = {}
        
        # Perplexity
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            metrics["eval_loss"] = avg_loss
            metrics["perplexity"] = calculate_perplexity(avg_loss)
        
        # BLEU and ROUGE
        if predictions and references:
            bleu_scores = calculate_bleu(predictions, references)
            rouge_scores = calculate_rouge(predictions, references)
            exact_match = calculate_exact_match(predictions, references)
            
            metrics.update(bleu_scores)
            metrics.update(rouge_scores)
            metrics["exact_match"] = exact_match
        
        return metrics
    
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7
    ) -> str:
        """Generate text for a single prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def evaluate_samples(
        self,
        samples: List[Dict[str, str]],
        num_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate and evaluate a few samples for inspection.
        
        Returns list of dicts with prompt, expected, generated.
        """
        results = []
        
        for sample in samples[:num_samples]:
            prompt = sample.get("prompt", "")
            expected = sample.get("completion", "")
            
            generated = self._generate(prompt)
            
            results.append({
                "prompt": prompt,
                "expected": expected,
                "generated": generated,
                "rouge_l": calculate_rouge([generated], [expected])["rouge_l"]
            })
        
        return results
