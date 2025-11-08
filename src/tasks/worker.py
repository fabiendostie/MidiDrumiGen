"""Celery worker configuration."""

import logging
import os
from celery import Celery
from celery.signals import worker_ready, worker_shutdown, task_prerun, task_postrun

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_BACKEND = os.getenv("REDIS_BACKEND", "redis://localhost:6379/1")

# Create Celery app
celery_app = Celery(
    'drum_generator',
    broker=REDIS_URL,
    backend=REDIS_BACKEND,
    include=['src.tasks.tasks']  # Auto-discover tasks
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # Soft limit at 4 minutes
    worker_prefetch_multiplier=1,  # Fair task distribution
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevent memory leaks)
    result_expires=3600,  # Results expire after 1 hour
    task_acks_late=True,  # Acknowledge tasks after completion
    task_reject_on_worker_lost=True,  # Reject if worker crashes
)

# Task routes
celery_app.conf.task_routes = {
    'src.tasks.tasks.generate_pattern': {'queue': 'gpu_generation'},
    'src.tasks.tasks.tokenize_midi': {'queue': 'midi_processing'},
    'src.tasks.tasks.train_model': {'queue': 'heavy_tasks'},
}


# Signal handlers
@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Log when worker is ready."""
    logger.info("✓ Celery worker ready and waiting for tasks")
    logger.info(f"  Broker: {REDIS_URL}")
    logger.info(f"  Backend: {REDIS_BACKEND}")


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    """Log when worker is shutting down."""
    logger.info("Celery worker shutting down...")


@task_prerun.connect
def on_task_prerun(sender=None, task_id=None, task=None, **kwargs):
    """Log before task execution."""
    logger.info(f"→ Starting task: {task.name} [{task_id}]")


@task_postrun.connect
def on_task_postrun(sender=None, task_id=None, task=None, state=None, **kwargs):
    """Log after task execution."""
    logger.info(f"← Completed task: {task.name} [{task_id}] - {state}")

