import os
from typing import Optional, List
from sqlalchemy.orm import Session

from app.config import settings
from app.db import models


class ModelRegistry:
    """Manages model adapters and their metadata."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def register_model(
        self,
        workspace_id: int,
        name: str,
        adapter_path: str,
        base_model: str,
        training_job_id: int,
        metrics: Optional[dict] = None
    ) -> models.Model:
        """Register a new trained model adapter."""
        model = models.Model(
            workspace_id=workspace_id,
            name=name,
            adapter_path=adapter_path,
            base_model=base_model,
            training_job_id=training_job_id,
            metrics=metrics or {}
        )
        
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        
        return model
    
    def get_model(self, model_id: int) -> Optional[models.Model]:
        """Get a model by ID."""
        return self.db.query(models.Model).filter(models.Model.id == model_id).first()
    
    def get_workspace_models(self, workspace_id: int) -> List[models.Model]:
        """Get all models for a workspace."""
        return self.db.query(models.Model).filter(
            models.Model.workspace_id == workspace_id
        ).all()
    
    def delete_model(self, model_id: int) -> bool:
        """Delete a model and its adapter files."""
        model = self.get_model(model_id)
        if not model:
            return False
        
        # Delete adapter files
        if os.path.exists(model.adapter_path):
            import shutil
            shutil.rmtree(model.adapter_path)
        
        self.db.delete(model)
        self.db.commit()
        
        return True
    
    def model_exists(self, adapter_path: str) -> bool:
        """Check if adapter files exist."""
        return os.path.exists(adapter_path)
