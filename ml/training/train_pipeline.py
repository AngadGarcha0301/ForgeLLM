"""
Training Pipeline - Orchestrates the full training workflow.
"""

import os
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from datasets import Dataset

from ml.preprocessing.formatter import prepare_dataset
from ml.training.trainer import LoRATrainer, TrainingConfig
from ml.evaluation.evaluator import Evaluator


class TrainingPipeline:
    """
    Full training pipeline orchestrator.
    
    Handles:
    1. Dataset loading and formatting
    2. Tokenization
    3. Training with LoRA
    4. Evaluation
    5. Saving adapter
    """
    
    def __init__(
        self,
        base_model: str,
        dataset_path: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None
    ):
        self.base_model = base_model
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _update_progress(self, status: str, progress: float, **kwargs):
        """Update progress via callback."""
        if self.progress_callback:
            self.progress_callback(status=status, progress=progress, **kwargs)
    
    def run(
        self,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_steps: int = -1,
        eval_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run the full training pipeline.
        
        Returns:
            Dict with training results and metrics
        """
        start_time = datetime.now()
        self._update_progress("starting", 0)
        
        # Step 1: Load and prepare dataset
        self._update_progress("loading_dataset", 5)
        samples = prepare_dataset(self.dataset_path)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(samples)
        
        # Split into train/eval
        if eval_split > 0 and len(dataset) > 10:
            split = dataset.train_test_split(test_size=eval_split, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
        
        self._update_progress("dataset_ready", 10, 
                              sample_count=len(train_dataset))
        
        # Step 2: Configure training
        training_config = TrainingConfig(
            output_dir=self.output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps
        )
        
        # Step 3: Initialize trainer
        self._update_progress("loading_model", 15)
        
        trainer = LoRATrainer(
            base_model_name=self.base_model,
            training_config=training_config,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            progress_callback=self.progress_callback
        )
        
        # Step 4: Train
        self._update_progress("training", 20)
        
        train_results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        self._update_progress("training_complete", 85)
        
        # Step 5: Evaluate
        metrics = {
            "train_loss": train_results["train_loss"],
            "train_runtime": train_results["train_runtime"],
            "samples_per_second": train_results["train_samples_per_second"],
            "total_samples": len(train_dataset),
        }
        
        if eval_dataset:
            self._update_progress("evaluating", 90)
            evaluator = Evaluator(trainer.model, trainer.tokenizer)
            eval_metrics = evaluator.evaluate(eval_dataset)
            metrics.update(eval_metrics)
        
        # Step 6: Save metadata
        self._update_progress("saving", 95)
        
        adapter_path = train_results["adapter_path"]
        
        # Save training metadata
        metadata = {
            "base_model": self.base_model,
            "created_at": datetime.now().isoformat(),
            "training_config": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout
            },
            "dataset": {
                "path": self.dataset_path,
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset) if eval_dataset else 0
            },
            "metrics": metrics,
            "training_time_seconds": (datetime.now() - start_time).total_seconds()
        }
        
        with open(os.path.join(adapter_path, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        self._update_progress("completed", 100)
        
        return {
            "adapter_path": adapter_path,
            "metrics": metrics,
            "metadata": metadata
        }


def run_training_pipeline(
    job_id: int,
    base_model: str,
    dataset_path: str,
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Convenience function to run training pipeline.
    Used by Celery workers.
    """
    pipeline = TrainingPipeline(
        base_model=base_model,
        dataset_path=dataset_path,
        output_dir=output_dir,
        progress_callback=progress_callback
    )
    
    return pipeline.run(
        num_epochs=config.get("num_epochs", 3),
        batch_size=config.get("batch_size", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        max_steps=config.get("max_steps", -1)
    )
