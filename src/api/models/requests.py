"""Pydantic request models for API validation."""

from pydantic import BaseModel, Field
from typing import Tuple, Optional


class PatternRequest(BaseModel):
    """Request model for pattern generation."""
    
    producer_style: str = Field(..., example="J Dilla", description="Producer style name")
    bars: int = Field(4, ge=1, le=32, description="Number of bars to generate")
    time_signature: Tuple[int, int] = Field((4, 4), description="Time signature (numerator, denominator)")
    tempo: int = Field(120, ge=40, le=300, description="Tempo in BPM")
    humanize: bool = Field(True, description="Apply humanization to pattern")
    pattern_type: Optional[str] = Field(None, description="Pattern type (intro, verse, chorus, etc.)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "producer_style": "J Dilla",
                "bars": 4,
                "time_signature": [4, 4],
                "tempo": 95,
                "humanize": True,
                "pattern_type": "intro"
            }
        }


class BatchPatternRequest(BaseModel):
    """Request model for batch pattern generation."""
    
    requests: list[PatternRequest] = Field(..., description="List of pattern generation requests")
    
    class Config:
        json_schema_extra = {
            "example": {
                "requests": [
                    {
                        "producer_style": "J Dilla",
                        "bars": 4,
                        "tempo": 95
                    },
                    {
                        "producer_style": "Metro Boomin",
                        "bars": 8,
                        "tempo": 140
                    }
                ]
            }
        }

