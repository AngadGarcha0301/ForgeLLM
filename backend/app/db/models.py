from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    workspaces = relationship("Workspace", back_populates="owner")


class Workspace(Base):
    __tablename__ = "workspaces"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="workspaces")
    datasets = relationship("Dataset", back_populates="workspace", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="workspace", cascade="all, delete-orphan")
    models = relationship("Model", back_populates="workspace", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)  # in bytes
    token_count = Column(Integer, nullable=True)
    sample_count = Column(Integer, nullable=True)
    format = Column(String(50), nullable=True)  # json, jsonl, csv
    status = Column(String(50), default="uploaded")  # uploaded, processing, ready, error
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    workspace = relationship("Workspace", back_populates="datasets")
    training_jobs = relationship("TrainingJob", back_populates="dataset")


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    
    # Job configuration
    name = Column(String(255), nullable=True)
    base_model = Column(String(255), nullable=False)
    
    # LoRA config
    lora_r = Column(Integer, default=16)
    lora_alpha = Column(Integer, default=32)
    lora_dropout = Column(Float, default=0.05)
    
    # Training config
    learning_rate = Column(Float, default=2e-4)
    num_epochs = Column(Integer, default=3)
    batch_size = Column(Integer, default=4)
    max_steps = Column(Integer, nullable=True)
    
    # Status
    status = Column(String(50), default="pending")  # pending, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)  # 0-100
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, nullable=True)
    
    # Results
    model_path = Column(String(500), nullable=True)
    metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Celery task ID
    celery_task_id = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="training_jobs")
    dataset = relationship("Dataset", back_populates="training_jobs")
    model = relationship("Model", back_populates="training_job", uselist=False)


class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=True)
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    adapter_path = Column(String(500), nullable=False)
    base_model = Column(String(255), nullable=False)
    
    # Metrics from training
    metrics = Column(JSON, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    workspace = relationship("Workspace", back_populates="models")
    training_job = relationship("TrainingJob", back_populates="model")
