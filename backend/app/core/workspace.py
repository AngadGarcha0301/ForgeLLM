import os
from typing import Optional
from sqlalchemy.orm import Session

from app.config import settings
from app.db import models


class WorkspaceManager:
    """Manages workspace directories and resources."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_workspace_path(self, workspace_id: int) -> str:
        """Get the storage path for a workspace."""
        return os.path.join(settings.MODELS_DIR, f"workspace_{workspace_id}")
    
    def get_adapter_path(self, workspace_id: int, model_name: str) -> str:
        """Get the path for a model adapter."""
        workspace_path = self.get_workspace_path(workspace_id)
        return os.path.join(workspace_path, "adapters", model_name)
    
    def get_dataset_path(self, workspace_id: int, filename: str) -> str:
        """Get the path for a dataset file."""
        return os.path.join(settings.UPLOAD_DIR, f"workspace_{workspace_id}", filename)
    
    def create_workspace_dirs(self, workspace_id: int) -> None:
        """Create all necessary directories for a workspace."""
        workspace_path = self.get_workspace_path(workspace_id)
        
        # Create directories
        os.makedirs(os.path.join(workspace_path, "adapters"), exist_ok=True)
        os.makedirs(os.path.join(workspace_path, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(workspace_path, "logs"), exist_ok=True)
        
        # Dataset directory
        dataset_path = os.path.join(settings.UPLOAD_DIR, f"workspace_{workspace_id}")
        os.makedirs(dataset_path, exist_ok=True)
    
    def delete_workspace_dirs(self, workspace_id: int) -> None:
        """Delete all directories for a workspace."""
        import shutil
        
        workspace_path = self.get_workspace_path(workspace_id)
        if os.path.exists(workspace_path):
            shutil.rmtree(workspace_path)
        
        dataset_path = os.path.join(settings.UPLOAD_DIR, f"workspace_{workspace_id}")
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
    
    def get_workspace_stats(self, workspace_id: int) -> dict:
        """Get storage statistics for a workspace."""
        workspace_path = self.get_workspace_path(workspace_id)
        
        total_size = 0
        file_count = 0
        
        if os.path.exists(workspace_path):
            for dirpath, dirnames, filenames in os.walk(workspace_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
                    file_count += 1
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count
        }
