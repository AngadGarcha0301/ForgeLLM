from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.dependencies import get_db, get_current_user
from app.db import models, schemas

router = APIRouter()


@router.get("/", response_model=List[schemas.ModelResponse])
async def list_models(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """List all trained models (adapters) in a workspace."""
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
    
    trained_models = db.query(models.Model).filter(
        models.Model.workspace_id == workspace_id
    ).all()
    
    return trained_models


@router.get("/{model_id}", response_model=schemas.ModelResponse)
async def get_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get model details."""
    model = db.query(models.Model).join(models.Workspace).filter(
        models.Model.id == model_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    return model


@router.delete("/{model_id}")
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Delete a trained model (adapter)."""
    model = db.query(models.Model).join(models.Workspace).filter(
        models.Model.id == model_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    # TODO: Delete adapter files from storage
    db.delete(model)
    db.commit()
    
    return {"message": "Model deleted successfully"}


@router.patch("/{model_id}", response_model=schemas.ModelResponse)
async def update_model(
    model_id: int,
    model_update: schemas.ModelUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Update model metadata."""
    model = db.query(models.Model).join(models.Workspace).filter(
        models.Model.id == model_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    if model_update.name:
        model.name = model_update.name
    if model_update.description:
        model.description = model_update.description
    
    db.commit()
    db.refresh(model)
    
    return model
