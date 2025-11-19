"""Pattern generation API routes."""

import logging

from fastapi import APIRouter, HTTPException, status

from src.api.models import PatternGenerationRequest, TaskResponse
from src.research.producer_agent import ProducerResearchAgent
from src.tasks.tasks import generate_pattern_task

logger = logging.getLogger(__name__)

# Initialize producer research agent (singleton)
_research_agent = None


def get_research_agent() -> ProducerResearchAgent:
    """Get or create the producer research agent singleton."""
    global _research_agent
    if _research_agent is None:
        logger.info("Initializing ProducerResearchAgent")
        _research_agent = ProducerResearchAgent()
    return _research_agent

router = APIRouter()


@router.post(
    "/generate",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate drum pattern",
    description="Queue a drum pattern generation task. Supports ANY producer name via dynamic research."
)
async def generate_pattern(request: PatternGenerationRequest) -> TaskResponse:
    """
    Generate drum pattern with specified producer style (dynamic or legacy).

    This endpoint accepts ANY producer name and automatically researches
    their style characteristics using the ProducerResearchAgent.

    **Workflow:**
    1. Extract producer name (from producer_name or legacy producer_style)
    2. Research producer style (cached or fresh, ~3 seconds first time, <100ms cached)
    3. Queue Celery task with style profile
    4. Return task ID
    5. Client polls /status/{task_id} for completion

    **New Dynamic API Example:**
    ```json
    POST /api/v1/generate
    {
      "producer_name": "Timbaland",
      "bars": 4,
      "tempo": 100,
      "humanize": true
    }
    ```

    **Legacy API Example (backward compatible):**
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
      "message": "Pattern generation queued for Timbaland (medium quality)"
    }
    ```
    """
    try:
        # Get producer name (dynamic or legacy)
        try:
            producer_name = request.get_producer_name()
        except ValueError as e:
            logger.error(f"Missing producer name: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'producer_name' or 'producer_style' must be provided"
            ) from e

        logger.info(
            f"Pattern generation request: {producer_name}, "
            f"{request.bars} bars @ {request.tempo} BPM"
        )

        # Research producer style (async)
        agent = get_research_agent()
        logger.debug(f"Researching producer: {producer_name}")

        try:
            style_profile = await agent.research_producer(producer_name)
            research_quality = style_profile.get('research_quality', 'unknown')
            cached = style_profile.get('cached', False)

            logger.info(
                f"Producer research complete: {producer_name} "
                f"(quality: {research_quality}, cached: {cached})"
            )

        except Exception as e:
            logger.error(f"Producer research failed for {producer_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to research producer '{producer_name}': {str(e)}"
            ) from e

        # Validate tempo suggestion (warning only)
        tempo_range = style_profile.get('style_params', {}).get('tempo_range', [60, 200])
        if tempo_range:
            min_tempo, max_tempo = tempo_range
            if not (min_tempo <= request.tempo <= max_tempo):
                logger.warning(
                    f"Tempo {request.tempo} outside typical range "
                    f"[{min_tempo}-{max_tempo}] for {producer_name}"
                )

        # Queue Celery task with style profile
        task = generate_pattern_task.delay(
            producer_name=producer_name,
            style_profile=style_profile,  # Pass complete profile to task
            bars=request.bars,
            tempo=request.tempo,
            time_signature=request.time_signature,
            humanize=request.humanize,
            pattern_type=request.pattern_type,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        logger.info(f"Task queued: {task.id} for {producer_name}")

        return TaskResponse(
            task_id=task.id,
            status="queued",
            message=f"Pattern generation queued for {producer_name} ({research_quality} quality)"
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is

    except Exception as e:
        logger.error(f"Failed to queue generation task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue generation task: {str(e)}"
        ) from e
