from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import threading
import time

from app.dependencies import get_db, get_current_user
from app.db import models, schemas
from app.db.database import SessionLocal
from app.services.training_service import TrainingService

router = APIRouter()


def simulate_training_thread(job_id: int):
    """Simulate training progress for demo purposes - runs in separate thread."""
    from datetime import datetime, timezone
    
    db = SessionLocal()
    try:
        job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if not job:
            return
        
        # Update status to running
        job.status = "running"
        job.total_steps = 100
        job.current_step = 0
        db.commit()
        
        # Simulate training progress
        for step in range(1, 101):
            time.sleep(0.3)  # Simulate work - 30 seconds total
            
            # Re-fetch job to avoid stale data
            db.refresh(job)
            job.current_step = step
            job.progress = float(step)
            job.metrics = {"loss": round(2.0 - (step * 0.018), 4), "step": step}
            db.commit()
        
        # Mark as completed
        job.status = "completed"
        job.progress = 100.0
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        
        # Create a Model record so it appears in Models/Inference pages
        trained_model = models.Model(
            name=job.name,
            base_model=job.base_model,
            workspace_id=job.workspace_id,
            adapter_path=f"data/models/workspace_{job.workspace_id}/adapters/{job.name}",
            training_job_id=job.id,
            metrics=job.metrics
        )
        db.add(trained_model)
        db.commit()
        print(f"✅ Training complete! Model '{job.name}' created (ID: {trained_model.id})")
        
    except Exception as e:
        print(f"Training error: {e}")
        job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()


def start_training_simulation(job_id: int):
    """Start training simulation in a background thread."""
    thread = threading.Thread(target=simulate_training_thread, args=(job_id,))
    thread.daemon = True
    thread.start()


@router.post("/start", response_model=schemas.TrainingJobResponse)
async def start_training(
    training_request: schemas.TrainingJobCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Start a new training job.
    
    This creates an async job that will be processed by a worker.
    """
    # Verify workspace ownership
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == training_request.workspace_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Verify dataset exists
    dataset = db.query(models.Dataset).filter(
        models.Dataset.id == training_request.dataset_id,
        models.Dataset.workspace_id == training_request.workspace_id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    training_service = TrainingService(db)
    job = training_service.create_training_job(training_request)
    
    # Start simulated training in background thread
    start_training_simulation(job.id)
    
    return job


@router.get("/{job_id}", response_model=schemas.TrainingJobResponse)
async def get_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get training job status and details."""
    job = db.query(models.TrainingJob).join(models.Workspace).filter(
        models.TrainingJob.id == job_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )
    
    return job


@router.get("/", response_model=List[schemas.TrainingJobResponse])
async def list_training_jobs(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """List all training jobs in a workspace."""
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
    
    jobs = db.query(models.TrainingJob).filter(
        models.TrainingJob.workspace_id == workspace_id
    ).order_by(models.TrainingJob.created_at.desc()).all()
    
    return jobs


@router.post("/{job_id}/cancel")
async def cancel_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Cancel a running training job."""
    job = db.query(models.TrainingJob).join(models.Workspace).filter(
        models.TrainingJob.id == job_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )
    
    if job.status not in ["pending", "running"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job cannot be cancelled"
        )
    
    training_service = TrainingService(db)
    training_service.cancel_job(job)
    
    return {"message": "Training job cancelled"}
