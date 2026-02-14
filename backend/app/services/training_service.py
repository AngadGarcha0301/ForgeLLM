from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime

from app.db import models, schemas
from app.config import settings
from app.core.workspace import WorkspaceManager


class TrainingService:
    """Service for handling training job operations."""
    
    def __init__(self, db: Session):
        self.db = db
        self.workspace_manager = WorkspaceManager(db)
    
    def create_training_job(
        self,
        request: schemas.TrainingJobCreate
    ) -> models.TrainingJob:
        """Create a new training job and queue it for processing."""
        # Get config or use defaults
        config = request.config or schemas.TrainingConfig()
        
        # Create job entry
        job = models.TrainingJob(
            workspace_id=request.workspace_id,
            dataset_id=request.dataset_id,
            name=request.name or f"training_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            base_model=request.base_model,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            learning_rate=config.learning_rate,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            status="pending"
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        
        # Queue the job for processing
        self._queue_training_job(job.id)
        
        return job
    
    def _queue_training_job(self, job_id: int) -> str:
        """Queue a training job for Celery worker processing."""
        from workers.tasks import run_training_job
        
        # Send to Celery
        task = run_training_job.delay(job_id)
        
        # Update job with task ID
        job = self.db.query(models.TrainingJob).filter(
            models.TrainingJob.id == job_id
        ).first()
        
        if job:
            job.celery_task_id = task.id
            self.db.commit()
        
        return task.id
    
    def get_job_status(self, job_id: int) -> Optional[dict]:
        """Get the current status of a training job."""
        job = self.db.query(models.TrainingJob).filter(
            models.TrainingJob.id == job_id
        ).first()
        
        if not job:
            return None
        
        return {
            "id": job.id,
            "status": job.status,
            "progress": job.progress,
            "current_step": job.current_step,
            "total_steps": job.total_steps,
            "metrics": job.metrics,
            "error_message": job.error_message
        }
    
    def update_job_progress(
        self,
        job_id: int,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        metrics: Optional[dict] = None
    ) -> None:
        """Update training job progress."""
        job = self.db.query(models.TrainingJob).filter(
            models.TrainingJob.id == job_id
        ).first()
        
        if job:
            if status:
                job.status = status
                if status == "running" and not job.started_at:
                    job.started_at = datetime.now()
                elif status in ["completed", "failed"]:
                    job.completed_at = datetime.now()
            
            if progress is not None:
                job.progress = progress
            if current_step is not None:
                job.current_step = current_step
            if total_steps is not None:
                job.total_steps = total_steps
            if metrics:
                job.metrics = metrics
            
            self.db.commit()
    
    def complete_job(
        self,
        job_id: int,
        model_path: str,
        metrics: dict
    ) -> models.Model:
        """Mark job as complete and register the trained model."""
        job = self.db.query(models.TrainingJob).filter(
            models.TrainingJob.id == job_id
        ).first()
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Update job
        job.status = "completed"
        job.progress = 100.0
        job.model_path = model_path
        job.metrics = metrics
        job.completed_at = datetime.now()
        
        # Create model entry
        model = models.Model(
            workspace_id=job.workspace_id,
            training_job_id=job.id,
            name=job.name,
            adapter_path=model_path,
            base_model=job.base_model,
            metrics=metrics
        )
        
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        
        return model
    
    def fail_job(self, job_id: int, error_message: str) -> None:
        """Mark a job as failed."""
        job = self.db.query(models.TrainingJob).filter(
            models.TrainingJob.id == job_id
        ).first()
        
        if job:
            job.status = "failed"
            job.error_message = error_message
            job.completed_at = datetime.now()
            self.db.commit()
    
    def cancel_job(self, job: models.TrainingJob) -> None:
        """Cancel a running or pending job."""
        # Revoke Celery task if exists
        if job.celery_task_id:
            from workers.celery_app import celery
            celery.control.revoke(job.celery_task_id, terminate=True)
        
        job.status = "cancelled"
        job.completed_at = datetime.now()
        self.db.commit()
