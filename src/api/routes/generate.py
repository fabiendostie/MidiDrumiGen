"""Pattern generation API routes."""

import logging
from fastapi import APIRouter, HTTPException, status
from src.api.models import PatternGenerationRequest, TaskResponse
from src.tasks.tasks import generate_pattern_task
from src.models.styles import validate_tempo_for_style, StyleNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/generate",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate drum pattern",
    description="Queue a drum pattern generation task. Returns immediately with task ID."
)
async def generate_pattern(request: PatternGenerationRequest) -> TaskResponse:
    """
    Generate drum pattern with specified producer style.

    This endpoint queues a generation task and returns immediately.
    Use the task_id to check status via GET /status/{task_id}.

    **Workflow:**
    1. Validate request parameters
    2. Queue Celery task for async processing
    3. Return task ID
    4. Client polls /status/{task_id} for completion

    **Example:**
    ```json
    POST /api/v1/generate
    {
      "producer_style": "J Dilla",
      "bars": 4,
      "tempo": 95,
      "humanize": true
    }
    ```

    **Response:**
    ```json
    {
      "task_id": "abc-123",
      "status": "queued",
      "message": "Pattern generation queued successfully"
    }
    ```
    """
    try:
        logger.info(f"Pattern generation request: {request.producer_style}, {request.bars} bars @ {request.tempo} BPM")

        # Validate tempo for style (warning only)
        try:
            validate_tempo_for_style(request.producer_style.value, request.tempo, warn_only=True)
        except Exception as e:
            logger.warning(f"Tempo validation warning: {e}")

        # Queue Celery task
        task = generate_pattern_task.delay(
            producer_style=request.producer_style.value,
            bars=request.bars,
            tempo=request.tempo,
            time_signature=request.time_signature,
            humanize=request.humanize,
            pattern_type=request.pattern_type,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        logger.info(f"Task queued: {task.id}")

        return TaskResponse(
            task_id=task.id,
            status="queued",
            message=f"Pattern generation queued successfully for {request.producer_style.value}"
        )

    except StyleNotFoundError as e:
        logger.error(f"Invalid style: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid producer style: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Failed to queue generation task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue generation task: {str(e)}"
        )
