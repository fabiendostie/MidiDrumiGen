"""Pydantic request models for API validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Tuple, Optional
from enum import Enum


class ProducerStyle(str, Enum):
    """Available producer styles."""
    J_DILLA = "J Dilla"
    METRO_BOOMIN = "Metro Boomin"
    QUESTLOVE = "Questlove"


class PatternGenerationRequest(BaseModel):
    """Request model for pattern generation."""

    producer_style: ProducerStyle = Field(
        ...,
        description="Producer style to emulate"
    )
    bars: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of bars to generate (1-32)"
    )
    tempo: int = Field(
        default=120,
        ge=60,
        le=200,
        description="Tempo in BPM (60-200)"
    )
    time_signature: Tuple[int, int] = Field(
        default=(4, 4),
        description="Time signature as (numerator, denominator)"
    )
    humanize: bool = Field(
        default=True,
        description="Apply humanization (timing, velocity, ghost notes)"
    )
    pattern_type: Optional[str] = Field(
        default="verse",
        description="Pattern type (intro, verse, chorus, bridge, outro)"
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (0.1-2.0, lower = more deterministic)"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Top-k sampling parameter"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )

    @field_validator('time_signature')
    @classmethod
    def validate_time_signature(cls, v):
        """Validate time signature."""
        numerator, denominator = v
        if denominator not in [2, 4, 8, 16]:
            raise ValueError("Denominator must be 2, 4, 8, or 16")
        if numerator < 1 or numerator > 16:
            raise ValueError("Numerator must be between 1 and 16")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "producer_style": "J Dilla",
                "bars": 4,
                "tempo": 95,
                "time_signature": [4, 4],
                "humanize": True,
                "pattern_type": "verse",
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.9
            }
        }
