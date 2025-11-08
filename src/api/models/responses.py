"""Pydantic response models for API responses."""

from pydantic import BaseModel
from typing import Optional


class TaskResponse(BaseModel):
    """Response model for task submission."""
    
    task_id: str
    status: str
    estimated_time: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc-123-def-456",
                "status": "queued",
                "estimated_time": 2.0
            }
        }


class StatusResponse(BaseModel):
    """Response model for task status."""
    
    task_id: str
    status: str  # queued, processing, complete, failed
    progress: Optional[float] = None
    midi_path: Optional[str] = None
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc-123-def-456",
                "status": "complete",
                "progress": 100.0,
                "midi_path": "patterns/jdilla_001.mid"
            }
        }


class StylesResponse(BaseModel):
    """Response model for available styles."""
    
    styles: list[str]
    count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "styles": ["J Dilla", "Metro Boomin", "Questlove"],
                "count": 3
            }
        }

