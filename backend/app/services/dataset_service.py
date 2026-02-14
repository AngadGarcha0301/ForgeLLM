import os
import json
import aiofiles
from typing import Optional
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.db import models
from app.core.workspace import WorkspaceManager


class DatasetService:
    """Service for handling dataset operations."""
    
    ALLOWED_EXTENSIONS = {".json", ".jsonl", ".csv"}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    def __init__(self, db: Session):
        self.db = db
        self.workspace_manager = WorkspaceManager(db)
    
    async def upload_dataset(
        self,
        file: UploadFile,
        workspace_id: int
    ) -> models.Dataset:
        """Upload and process a dataset file."""
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
        
        # Create workspace directories
        self.workspace_manager.create_workspace_dirs(workspace_id)
        
        # Generate file path
        file_path = self.workspace_manager.get_dataset_path(workspace_id, file.filename)
        
        # Save file
        content = await file.read()
        file_size = len(content)
        
        if file_size > self.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {self.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        
        # Parse and validate dataset
        sample_count, token_count = await self._process_dataset(file_path, file_ext)
        
        # Create database entry
        dataset = models.Dataset(
            workspace_id=workspace_id,
            name=file.filename,
            file_path=file_path,
            file_size=file_size,
            token_count=token_count,
            sample_count=sample_count,
            format=file_ext.lstrip("."),
            status="ready"
        )
        
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        
        return dataset
    
    async def _process_dataset(self, file_path: str, file_ext: str) -> tuple:
        """Process dataset and return (sample_count, estimated_token_count)."""
        sample_count = 0
        total_chars = 0
        
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
        
        if file_ext == ".json":
            data = json.loads(content)
            if isinstance(data, list):
                sample_count = len(data)
                for sample in data:
                    total_chars += len(str(sample))
        
        elif file_ext == ".jsonl":
            lines = content.strip().split("\n")
            sample_count = len(lines)
            for line in lines:
                total_chars += len(line)
        
        elif file_ext == ".csv":
            lines = content.strip().split("\n")
            sample_count = len(lines) - 1  # Exclude header
            total_chars = len(content)
        
        # Rough token estimation (4 chars per token)
        estimated_tokens = total_chars // 4
        
        return sample_count, estimated_tokens
    
    def delete_dataset(self, dataset: models.Dataset) -> None:
        """Delete a dataset and its file."""
        # Delete file
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # Delete database entry
        self.db.delete(dataset)
        self.db.commit()
    
    def get_dataset_samples(self, dataset: models.Dataset, limit: int = 5) -> list:
        """Get sample entries from a dataset."""
        samples = []
        
        with open(dataset.file_path, "r") as f:
            if dataset.format == "json":
                data = json.load(f)
                samples = data[:limit] if isinstance(data, list) else [data]
            
            elif dataset.format == "jsonl":
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    samples.append(json.loads(line))
        
        return samples
