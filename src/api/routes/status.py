"""Task status API routes."""

import logging

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, status

from src.api.models import TaskStatusResponse
from src.tasks.worker import celery_app

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get task status",
    description="Get status and result of a generation task",
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get status of a pattern generation task.

    **Task States:**
    - `PENDING`: Task queued, waiting to start
    - `PROGRESS`: Task running, check progress field
    - `SUCCESS`: Task completed, result available
    - `FAILURE`: Task failed, error message available

    **Example:**
    ```
    GET /api/v1/status/abc-123
    ```

    **Response (completed):**
    ```json
    {
      "task_id": "abc-123",
      "status": "completed",
      "progress": 100,
      "result": {
        "midi_file": "output/patterns/abc-123_j_dilla_4bars.mid",
        "duration_seconds": 1.234,
        "style": "J Dilla"
      }
    }
    ```
    """
    try:
        # Get task result
        task = AsyncResult(task_id, app=celery_app)

        logger.debug(f"Checking status for task {task_id}: {task.state}")

        # Map Celery state to response
        if task.state == "PENDING":
            return TaskStatusResponse(task_id=task_id, status="pending", progress=0)

        elif task.state == "PROGRESS":
            # Get custom progress metadata
            progress_data = task.info or {}
            return TaskStatusResponse(
                task_id=task_id, status="processing", progress=progress_data.get("progress", 0)
            )

        elif task.state == "SUCCESS":
            return TaskStatusResponse(
                task_id=task_id, status="completed", progress=100, result=task.result
            )

        elif task.state == "FAILURE":
            error_msg = str(task.info) if task.info else "Unknown error"
            return TaskStatusResponse(task_id=task_id, status="failed", error=error_msg)

        else:
            # Unknown state
            return TaskStatusResponse(task_id=task_id, status=task.state.lower(), progress=None)

    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task status: {str(e)}",
        ) from e
