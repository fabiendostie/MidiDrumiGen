"""Unit tests for API Pydantic models (Phase 4)."""

import pytest
from pydantic import ValidationError
from src.api.models import (
    PatternGenerationRequest,
    ProducerStyle,
    TaskResponse,
    TaskStatusResponse,
    StyleInfo,
    StylesListResponse,
    ErrorResponse,
)


class TestPatternGenerationRequest:
    """Tests for PatternGenerationRequest model."""

    def test_valid_request_with_defaults(self):
        """Test creating request with minimal required fields."""
        request = PatternGenerationRequest(producer_style=ProducerStyle.J_DILLA)

        assert request.producer_style == ProducerStyle.J_DILLA
        assert request.bars == 4
        assert request.tempo == 120
        assert request.time_signature == (4, 4)
        assert request.humanize is True
        assert request.pattern_type == "verse"
        assert request.temperature == 1.0
        assert request.top_k == 50
        assert request.top_p == 0.9

    def test_valid_request_all_fields(self):
        """Test creating request with all fields specified."""
        request = PatternGenerationRequest(
            producer_style=ProducerStyle.METRO_BOOMIN,
            bars=8,
            tempo=140,
            time_signature=(3, 4),
            humanize=False,
            pattern_type="chorus",
            temperature=0.8,
            top_k=40,
            top_p=0.95,
        )

        assert request.producer_style == ProducerStyle.METRO_BOOMIN
        assert request.bars == 8
        assert request.tempo == 140
        assert request.time_signature == (3, 4)
        assert request.humanize is False
        assert request.pattern_type == "chorus"
        assert request.temperature == 0.8
        assert request.top_k == 40
        assert request.top_p == 0.95

    def test_invalid_bars_too_low(self):
        """Test validation fails when bars < 1."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                bars=0
            )
        assert "bars" in str(exc_info.value)

    def test_invalid_bars_too_high(self):
        """Test validation fails when bars > 32."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                bars=33
            )
        assert "bars" in str(exc_info.value)

    def test_invalid_tempo_too_low(self):
        """Test validation fails when tempo < 60."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                tempo=59
            )
        assert "tempo" in str(exc_info.value)

    def test_invalid_tempo_too_high(self):
        """Test validation fails when tempo > 200."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                tempo=201
            )
        assert "tempo" in str(exc_info.value)

    def test_invalid_time_signature_denominator(self):
        """Test validation fails with invalid time signature denominator."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                time_signature=(4, 3)  # Invalid denominator
            )
        assert "Denominator must be 2, 4, 8, or 16" in str(exc_info.value)

    def test_invalid_time_signature_numerator_too_low(self):
        """Test validation fails when numerator < 1."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                time_signature=(0, 4)
            )
        assert "Numerator must be between 1 and 16" in str(exc_info.value)

    def test_invalid_time_signature_numerator_too_high(self):
        """Test validation fails when numerator > 16."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                time_signature=(17, 4)
            )
        assert "Numerator must be between 1 and 16" in str(exc_info.value)

    def test_invalid_temperature_too_low(self):
        """Test validation fails when temperature < 0.1."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                temperature=0.05
            )
        assert "temperature" in str(exc_info.value)

    def test_invalid_temperature_too_high(self):
        """Test validation fails when temperature > 2.0."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                temperature=2.1
            )
        assert "temperature" in str(exc_info.value)

    def test_invalid_top_k_negative(self):
        """Test validation fails when top_k < 0."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                top_k=-1
            )
        assert "top_k" in str(exc_info.value)

    def test_invalid_top_k_too_high(self):
        """Test validation fails when top_k > 100."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                top_k=101
            )
        assert "top_k" in str(exc_info.value)

    def test_invalid_top_p_too_low(self):
        """Test validation fails when top_p < 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                top_p=-0.1
            )
        assert "top_p" in str(exc_info.value)

    def test_invalid_top_p_too_high(self):
        """Test validation fails when top_p > 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            PatternGenerationRequest(
                producer_style=ProducerStyle.J_DILLA,
                top_p=1.1
            )
        assert "top_p" in str(exc_info.value)


class TestTaskResponse:
    """Tests for TaskResponse model."""

    def test_valid_task_response(self):
        """Test creating valid task response."""
        response = TaskResponse(
            task_id="abc-123",
            status="queued",
            message="Task queued successfully"
        )

        assert response.task_id == "abc-123"
        assert response.status == "queued"
        assert response.message == "Task queued successfully"


class TestTaskStatusResponse:
    """Tests for TaskStatusResponse model."""

    def test_pending_task_status(self):
        """Test task status for pending task."""
        response = TaskStatusResponse(
            task_id="abc-123",
            status="pending",
            progress=0
        )

        assert response.task_id == "abc-123"
        assert response.status == "pending"
        assert response.progress == 0
        assert response.result is None
        assert response.error is None

    def test_processing_task_status(self):
        """Test task status for processing task."""
        response = TaskStatusResponse(
            task_id="abc-123",
            status="processing",
            progress=50
        )

        assert response.task_id == "abc-123"
        assert response.status == "processing"
        assert response.progress == 50

    def test_completed_task_status(self):
        """Test task status for completed task."""
        result = {
            "midi_file": "output/patterns/test.mid",
            "duration_seconds": 1.5,
            "tokens_generated": 256
        }
        response = TaskStatusResponse(
            task_id="abc-123",
            status="completed",
            progress=100,
            result=result
        )

        assert response.task_id == "abc-123"
        assert response.status == "completed"
        assert response.progress == 100
        assert response.result == result

    def test_failed_task_status(self):
        """Test task status for failed task."""
        response = TaskStatusResponse(
            task_id="abc-123",
            status="failed",
            error="Model loading failed"
        )

        assert response.task_id == "abc-123"
        assert response.status == "failed"
        assert response.error == "Model loading failed"

    def test_invalid_progress_too_low(self):
        """Test validation fails when progress < 0."""
        with pytest.raises(ValidationError) as exc_info:
            TaskStatusResponse(
                task_id="abc-123",
                status="processing",
                progress=-1
            )
        assert "progress" in str(exc_info.value)

    def test_invalid_progress_too_high(self):
        """Test validation fails when progress > 100."""
        with pytest.raises(ValidationError) as exc_info:
            TaskStatusResponse(
                task_id="abc-123",
                status="processing",
                progress=101
            )
        assert "progress" in str(exc_info.value)


class TestStyleInfo:
    """Tests for StyleInfo model."""

    def test_valid_style_info(self):
        """Test creating valid style info."""
        humanization = {
            "swing": 62.0,
            "micro_timing_ms": 20.0,
            "ghost_note_prob": 0.4,
            "velocity_variation": 0.15
        }
        style = StyleInfo(
            name="J Dilla",
            model_id="j_dilla_v1",
            description="Signature swing and soulful groove",
            preferred_tempo_range=(85, 95),
            humanization=humanization
        )

        assert style.name == "J Dilla"
        assert style.model_id == "j_dilla_v1"
        assert style.description == "Signature swing and soulful groove"
        assert style.preferred_tempo_range == (85, 95)
        assert style.humanization == humanization


class TestStylesListResponse:
    """Tests for StylesListResponse model."""

    def test_valid_styles_list(self):
        """Test creating valid styles list response."""
        styles = [
            StyleInfo(
                name="J Dilla",
                model_id="j_dilla_v1",
                description="Test",
                preferred_tempo_range=(85, 95),
                humanization={}
            )
        ]
        response = StylesListResponse(
            styles=styles,
            count=1
        )

        assert len(response.styles) == 1
        assert response.count == 1
        assert response.styles[0].name == "J Dilla"


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_valid_error_response(self):
        """Test creating valid error response."""
        response = ErrorResponse(
            error="ValidationError",
            message="Invalid input",
            details={"field": "tempo"},
            path="/api/v1/generate"
        )

        assert response.error == "ValidationError"
        assert response.message == "Invalid input"
        assert response.details == {"field": "tempo"}
        assert response.path == "/api/v1/generate"
