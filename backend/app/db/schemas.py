from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============== User Schemas ==============

class UserBase(BaseModel):
    email: EmailStr
    username: str


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


# ============== Workspace Schemas ==============

class WorkspaceBase(BaseModel):
    name: str
    description: Optional[str] = None


class WorkspaceCreate(WorkspaceBase):
    pass


class WorkspaceResponse(WorkspaceBase):
    id: int
    owner_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============== Dataset Schemas ==============

class DatasetBase(BaseModel):
    name: str


class DatasetCreate(DatasetBase):
    workspace_id: int


class DatasetResponse(DatasetBase):
    id: int
    workspace_id: int
    file_path: str
    file_size: Optional[int]
    token_count: Optional[int]
    sample_count: Optional[int]
    format: Optional[str]
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============== Training Job Schemas ==============

class TrainingConfig(BaseModel):
    """Training configuration options."""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_steps: Optional[int] = None


class TrainingJobCreate(BaseModel):
    workspace_id: int
    dataset_id: int
    name: Optional[str] = None
    base_model: str = "mistralai/Mistral-7B-v0.1"
    config: Optional[TrainingConfig] = None


class TrainingJobResponse(BaseModel):
    id: int
    workspace_id: int
    dataset_id: int
    name: Optional[str]
    base_model: str
    status: str
    progress: float
    current_step: int
    total_steps: Optional[int]
    lora_r: Optional[int] = 16
    lora_alpha: Optional[int] = 32
    num_epochs: Optional[int] = 3
    learning_rate: Optional[float] = 0.0002
    metrics: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


# ============== Model Schemas ==============

class ModelBase(BaseModel):
    name: str
    description: Optional[str] = None


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ModelResponse(ModelBase):
    id: int
    workspace_id: int
    training_job_id: Optional[int]
    adapter_path: str
    base_model: str
    metrics: Optional[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============== Inference Schemas ==============

class InferenceRequest(BaseModel):
    model_id: int
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


class InferenceResponse(BaseModel):
    model_id: int
    prompt: str
    generated_text: str
    tokens_used: int


class BatchInferenceRequest(BaseModel):
    model_id: int
    prompts: List[str]
    max_tokens: int = 256
    temperature: float = 0.7


class BatchInferenceResponse(BaseModel):
    model_id: int
    results: List[Dict[str, Any]]
