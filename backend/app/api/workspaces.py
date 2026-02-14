from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.dependencies import get_db, get_current_user
from app.db import models, schemas

router = APIRouter()


@router.get("/", response_model=List[schemas.WorkspaceResponse])
async def list_workspaces(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """List all workspaces for the current user."""
    workspaces = db.query(models.Workspace).filter(
        models.Workspace.owner_id == current_user.id
    ).all()
    
    return workspaces


@router.post("/", response_model=schemas.WorkspaceResponse)
async def create_workspace(
    workspace_data: schemas.WorkspaceCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Create a new workspace."""
    workspace = models.Workspace(
        name=workspace_data.name,
        description=workspace_data.description,
        owner_id=current_user.id
    )
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    
    return workspace


@router.get("/{workspace_id}", response_model=schemas.WorkspaceResponse)
async def get_workspace(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get workspace details."""
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    return workspace


@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Delete a workspace."""
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    db.delete(workspace)
    db.commit()
    
    return {"message": "Workspace deleted successfully"}
