"""
Celery tasks for async job processing.
"""

import os
import sys
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.celery_app import celery
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@celery.task(bind=True, max_retries=3)
def run_training_job(self, job_id: int):
    """
    Run a training job asynchronously.
    
    This task:
    1. Loads job details from database
    2. Runs the training pipeline
    3. Updates job status
    4. Registers the trained model
    """
    from sqlalchemy.orm import Session
    from backend.app.db.database import SessionLocal
    from backend.app.db import models
    from backend.app.services.training_service import TrainingService
    from backend.app.config import settings
    from ml.training.train_pipeline import run_training_pipeline
    
    logger.info(f"Starting training job {job_id}")
    
    db: Session = SessionLocal()
    
    try:
        # Get job from database
        job = db.query(models.TrainingJob).filter(
            models.TrainingJob.id == job_id
        ).first()
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Get dataset
        dataset = db.query(models.Dataset).filter(
            models.Dataset.id == job.dataset_id
        ).first()
        
        if not dataset:
            raise ValueError(f"Dataset {job.dataset_id} not found")
        
        # Update job status
        training_service = TrainingService(db)
        training_service.update_job_progress(job_id, status="running", progress=5)
        
        # Prepare output directory
        output_dir = os.path.join(
            settings.MODELS_DIR,
            f"workspace_{job.workspace_id}",
            f"job_{job_id}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Progress callback
        def progress_callback(status: str, progress: float, **kwargs):
            training_service.update_job_progress(
                job_id,
                progress=progress,
                **kwargs
            )
        
        # Training config
        config = {
            "num_epochs": job.num_epochs,
            "batch_size": job.batch_size,
            "learning_rate": job.learning_rate,
            "lora_r": job.lora_r,
            "lora_alpha": job.lora_alpha,
            "lora_dropout": job.lora_dropout,
            "max_steps": job.max_steps or -1
        }
        
        # Run training
        result = run_training_pipeline(
            job_id=job_id,
            base_model=job.base_model,
            dataset_path=dataset.file_path,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback
        )
        
        # Complete job and register model
        model = training_service.complete_job(
            job_id=job_id,
            model_path=result["adapter_path"],
            metrics=result["metrics"]
        )
        
        logger.info(f"Training job {job_id} completed. Model ID: {model.id}")
        
        return {
            "job_id": job_id,
            "model_id": model.id,
            "adapter_path": result["adapter_path"],
            "metrics": result["metrics"]
        }
    
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {str(e)}")
        
        # Update job status
        training_service = TrainingService(db)
        training_service.fail_job(job_id, str(e))
        
        # Retry on certain errors
        if "CUDA out of memory" in str(e):
            raise self.retry(exc=e, countdown=60)
        
        raise
    
    finally:
        db.close()


@celery.task
def process_dataset(dataset_id: int):
    """
    Process a newly uploaded dataset.
    
    - Validate format
    - Count tokens
    - Extract statistics
    """
    from backend.app.db.database import SessionLocal
    from backend.app.db import models
    from ml.preprocessing.formatter import prepare_dataset
    from ml.preprocessing.tokenizer import TokenizerWrapper
    from backend.app.config import settings
    
    logger.info(f"Processing dataset {dataset_id}")
    
    db = SessionLocal()
    
    try:
        dataset = db.query(models.Dataset).filter(
            models.Dataset.id == dataset_id
        ).first()
        
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Update status
        dataset.status = "processing"
        db.commit()
        
        # Process
        samples = prepare_dataset(dataset.file_path)
        
        # Count tokens
        tokenizer = TokenizerWrapper(settings.BASE_MODEL)
        tokenizer.load()
        
        total_tokens = 0
        for sample in samples:
            total_tokens += tokenizer.count_tokens(sample["text"])
        
        # Update dataset
        dataset.sample_count = len(samples)
        dataset.token_count = total_tokens
        dataset.status = "ready"
        db.commit()
        
        logger.info(f"Dataset {dataset_id} processed: {len(samples)} samples, {total_tokens} tokens")
        
        return {
            "dataset_id": dataset_id,
            "sample_count": len(samples),
            "token_count": total_tokens
        }
    
    except Exception as e:
        logger.error(f"Dataset processing failed: {str(e)}")
        dataset.status = "error"
        db.commit()
        raise
    
    finally:
        db.close()


@celery.task
def cleanup_old_jobs():
    """
    Periodic task to clean up old completed jobs.
    """
    from datetime import datetime, timedelta
    from backend.app.db.database import SessionLocal
    from backend.app.db import models
    import shutil
    
    logger.info("Running job cleanup")
    
    db = SessionLocal()
    
    try:
        # Find jobs older than 30 days
        cutoff = datetime.utcnow() - timedelta(days=30)
        
        old_jobs = db.query(models.TrainingJob).filter(
            models.TrainingJob.completed_at < cutoff,
            models.TrainingJob.status.in_(["completed", "failed", "cancelled"])
        ).all()
        
        for job in old_jobs:
            # Don't delete if model is still active
            if job.model and job.model.is_active:
                continue
            
            # Delete checkpoints directory
            if job.model_path:
                checkpoint_dir = os.path.dirname(job.model_path)
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
            
            logger.info(f"Cleaned up job {job.id}")
        
        db.commit()
        
    finally:
        db.close()
