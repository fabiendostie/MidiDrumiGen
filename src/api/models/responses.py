"""Pydantic response models for API responses."""

from pydantic import BaseModel, Field
from typing import Optional, List, Tuple


class TaskResponse(BaseModel):
    """Response model for task creation."""

    task_id: str = Field(..., description="Celery task ID")
    status: str = Field(..., description="Task status (queued, processing, completed, failed)")
    message: str = Field(..., description="Human-readable status message")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123-def456-ghi789",
                "status": "queued",
                "message": "Pattern generation queued successfully"
            }
        }


class TaskStatusResponse(BaseModel):
    """Response model for task status."""

    task_id: str
    status: str
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    result: Optional[dict] = Field(None, description="Task result (if completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    created_at: Optional[str] = Field(None, description="Task creation timestamp")
    completed_at: Optional[str] = Field(None, description="Task completion timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123-def456-ghi789",
                "status": "completed",
                "progress": 100,
                "result": {
                    "midi_file": "output/patterns/abc123_j_dilla_4bars.mid",
                    "duration_seconds": 1.234,
                    "tokens_generated": 256,
                    "style": "J Dilla",
                    "bars": 4,
                    "tempo": 95
                }
            }
        }


class StyleInfo(BaseModel):
    """Information about a producer style."""

    name: str
    model_id: str
    description: str
    preferred_tempo_range: Tuple[int, int]
    humanization: dict

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class StylesListResponse(BaseModel):
    """Response model for styles list."""

    styles: List[StyleInfo]
    count: int

    class Config:
        json_schema_extra = {
            "example": {
                "count": 3,
                "styles": [
                    {
                        "name": "J Dilla",
                        "model_id": "j_dilla_v1",
                        "description": "Signature swing and soulful groove",
                        "preferred_tempo_range": [85, 95],
                        "humanization": {"swing": 62.0}
                    }
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    path: Optional[str] = Field(None, description="Request path")
