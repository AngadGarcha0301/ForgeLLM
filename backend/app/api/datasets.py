from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List

from app.dependencies import get_db, get_current_user
from app.db import models, schemas
from app.services.dataset_service import DatasetService

router = APIRouter()


@router.post("/upload", response_model=schemas.DatasetResponse)
async def upload_dataset(
    workspace_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Upload a dataset file for fine-tuning.
    
    Supports: JSON, JSONL, CSV formats.
    Expected format: {"instruction": "", "input": "", "output": ""}
    """
    # Verify workspace ownership
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    dataset_service = DatasetService(db)
    dataset = await dataset_service.upload_dataset(
        file=file,
        workspace_id=workspace_id
    )
    
    return dataset


@router.get("/{dataset_id}", response_model=schemas.DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get dataset details."""
    dataset = db.query(models.Dataset).join(models.Workspace).filter(
        models.Dataset.id == dataset_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return dataset


@router.get("/", response_model=List[schemas.DatasetResponse])
async def list_datasets(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """List all datasets in a workspace."""
    # Verify workspace ownership
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    datasets = db.query(models.Dataset).filter(
        models.Dataset.workspace_id == workspace_id
    ).all()
    
    return datasets


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Delete a dataset."""
    dataset = db.query(models.Dataset).join(models.Workspace).filter(
        models.Dataset.id == dataset_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    dataset_service = DatasetService(db)
    dataset_service.delete_dataset(dataset)
    
    return {"message": "Dataset deleted successfully"}
