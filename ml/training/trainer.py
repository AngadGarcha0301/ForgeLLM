"""
Trainer module for LoRA fine-tuning.
Handles model loading, training loop, and checkpointing.
"""

import os
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset

from ml.training.lora_config import get_lora_config


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001


class LoRATrainer:
    """Trainer for LoRA fine-tuning."""
    
    def __init__(
        self,
        base_model_name: str,
        training_config: TrainingConfig,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit: bool = True,
        progress_callback: Optional[Callable] = None
    ):
        self.base_model_name = base_model_name
        self.training_config = training_config
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.progress_callback = progress_callback
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model(self) -> None:
        """Load and prepare the base model for training."""
        # Quantization config for 4-bit loading
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Prepare for k-bit training
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Get LoRA config
        lora_config = get_lora_config(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Run training."""
        if self.model is None:
            self.load_model()
        
        # Ensure output dir exists
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_steps=self.training_config.warmup_steps,
            max_steps=self.training_config.max_steps,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            max_grad_norm=self.training_config.max_grad_norm,
            weight_decay=self.training_config.weight_decay,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            report_to="none",  # Disable wandb/tensorboard for now
            remove_unused_columns=False,
        )
        
        # Data collator
        def data_collator(features):
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )
            batch["labels"] = batch["input_ids"].clone()
            return batch
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        train_result = self.trainer.train()
        
        # Save adapter
        adapter_path = os.path.join(self.training_config.output_dir, "adapter")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "adapter_path": adapter_path
        }
    
    def save_adapter(self, path: str) -> None:
        """Save the trained adapter."""
        if self.model is not None:
            self.model.save_pretrained(path)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(path)
