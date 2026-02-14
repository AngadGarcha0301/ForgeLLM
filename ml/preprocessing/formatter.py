"""
Dataset formatter for instruction fine-tuning.
Converts raw data into instruction format compatible with LLM training.
"""

from typing import Dict, Any, List, Optional
import json


class DatasetFormatter:
    """Formats datasets for instruction fine-tuning."""
    
    # Default prompt template
    DEFAULT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
    
    # Template without input
    NO_INPUT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{output}"""
    
    def __init__(
        self,
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        template: Optional[str] = None
    ):
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.template = template
    
    def format_sample(self, sample: Dict[str, Any]) -> str:
        """Format a single sample into instruction format."""
        instruction = sample.get(self.instruction_key, "")
        input_text = sample.get(self.input_key, "")
        output = sample.get(self.output_key, "")
        
        if self.template:
            return self.template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        
        # Use default templates
        if input_text:
            return self.DEFAULT_TEMPLATE.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        else:
            return self.NO_INPUT_TEMPLATE.format(
                instruction=instruction,
                output=output
            )
    
    def format_dataset(self, samples: List[Dict[str, Any]]) -> List[str]:
        """Format entire dataset."""
        return [self.format_sample(sample) for sample in samples]
    
    def format_for_training(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Format sample for training with separate prompt and completion."""
        instruction = sample.get(self.instruction_key, "")
        input_text = sample.get(self.input_key, "")
        output = sample.get(self.output_key, "")
        
        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
"""
        
        return {
            "prompt": prompt,
            "completion": output,
            "text": prompt + output
        }


def load_json_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file."""
    samples = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def prepare_dataset(
    file_path: str,
    formatter: Optional[DatasetFormatter] = None
) -> List[Dict[str, str]]:
    """Load and format dataset for training."""
    formatter = formatter or DatasetFormatter()
    
    if file_path.endswith(".json"):
        samples = load_json_dataset(file_path)
    elif file_path.endswith(".jsonl"):
        samples = load_jsonl_dataset(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return [formatter.format_for_training(sample) for sample in samples]
