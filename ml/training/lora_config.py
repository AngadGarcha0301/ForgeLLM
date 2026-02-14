"""
LoRA Configuration for fine-tuning.
Defines parameter-efficient fine-tuning configurations.
"""

from peft import LoraConfig, TaskType
from typing import Optional, List


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """
    Create a LoRA configuration for fine-tuning.
    
    Args:
        r: Rank of the low-rank matrices
        lora_alpha: Scaling factor for LoRA
        target_modules: Which modules to apply LoRA to
        lora_dropout: Dropout probability for LoRA layers
        bias: Bias type ('none', 'all', 'lora_only')
        task_type: Task type for the model
    
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Default for most decoder models (Llama, Mistral, etc.)
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else task_type
    )


def get_qlora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05
) -> LoraConfig:
    """
    Create a QLoRA configuration (4-bit quantized LoRA).
    
    This is the same LoRA config but meant to be used with
    a 4-bit quantized base model.
    """
    return get_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout
    )


# Preset configurations for different model sizes
LORA_CONFIGS = {
    "small": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1
    },
    "medium": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    },
    "large": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05
    },
    "xlarge": {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05
    }
}


def get_preset_config(preset: str = "medium") -> LoraConfig:
    """Get a preset LoRA configuration."""
    if preset not in LORA_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(LORA_CONFIGS.keys())}")
    
    return get_lora_config(**LORA_CONFIGS[preset])


# Model-specific target modules
TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}


def get_target_modules(model_type: str) -> List[str]:
    """Get target modules for a specific model type."""
    return TARGET_MODULES.get(model_type.lower(), TARGET_MODULES["llama"])
