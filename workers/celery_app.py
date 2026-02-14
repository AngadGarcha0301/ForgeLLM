"""
Celery application configuration.
"""

from celery import Celery

# Create Celery app
celery = Celery(
    "forgellm",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["workers.tasks"]
)

# Celery configuration
celery.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=1,  # One task at a time (GPU constraint)
    
    # Result backend settings
    result_expires=86400,  # 24 hours
    
    # Task time limits
    task_soft_time_limit=3600 * 6,  # 6 hours soft limit
    task_time_limit=3600 * 8,  # 8 hours hard limit
)


if __name__ == "__main__":
    celery.start()
