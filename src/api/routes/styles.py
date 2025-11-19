"""Producer styles API routes."""

import logging

from fastapi import APIRouter, HTTPException, status

from src.api.models import StyleInfo, StylesListResponse
from src.models.styles import get_all_styles_info

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/styles",
    response_model=StylesListResponse,
    summary="List available styles",
    description="Get catalog of all available producer styles"
)
async def list_styles() -> StylesListResponse:
    """
    List all available producer styles with parameters.

    Returns complete information about each style including:
    - Model ID and description
    - Preferred tempo range
    - Humanization parameters (swing, timing, ghost notes, etc.)

    **Example:**
    ```
    GET /api/v1/styles
    ```

    **Response:**
    ```json
    {
      "count": 3,
      "styles": [
        {
          "name": "J Dilla",
          "model_id": "j_dilla_v1",
          "description": "Signature swing and soulful groove",
          "preferred_tempo_range": [85, 95],
          "humanization": {
            "swing": 62.0,
            "micro_timing_ms": 20.0,
            "ghost_note_prob": 0.4,
            "velocity_variation": 0.15
          }
        }
      ]
    }
    ```
    """
    try:
        styles_dict = get_all_styles_info()

        styles_list = [
            StyleInfo(
                name=name,
                model_id=info['model_id'],
                description=info['description'],
                preferred_tempo_range=info['preferred_tempo_range'],
                humanization=info['humanization']
            )
            for name, info in styles_dict.items()
        ]

        logger.info(f"Returning {len(styles_list)} available styles")

        return StylesListResponse(
            styles=styles_list,
            count=len(styles_list)
        )

    except Exception as e:
        logger.error(f"Failed to retrieve styles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve styles: {str(e)}"
        ) from e
