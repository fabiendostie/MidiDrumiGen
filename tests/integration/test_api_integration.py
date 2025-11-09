"""Integration tests for Phase 4 API workflow.

These tests require:
- API server running (uvicorn src.api.main:app)
- Redis running (docker start redis)
- Celery worker running (celery -A src.tasks.worker worker -Q gpu_generation --pool=solo)

Run with: pytest tests/integration/test_api_integration.py -v -s
"""

import pytest
import requests
import time
from pathlib import Path
from typing import Dict, Any

# Base URL for API
API_BASE = "http://localhost:8000"

# Timeout for task completion (seconds)
TASK_TIMEOUT = 30


class TestAPIHealthAndConnectivity:
    """Test basic API health and connectivity."""

    def test_api_root_accessible(self):
        """Test that API root endpoint is accessible."""
        response = requests.get(f"{API_BASE}/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "Drum Pattern Generator" in data["name"]

    def test_health_endpoint_shows_connected(self):
        """Test that health endpoint shows Redis connected."""
        response = requests.get(f"{API_BASE}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["redis"] == "connected", (
            "Redis is not connected. Start Redis with: docker start redis"
        )

    def test_swagger_ui_accessible(self):
        """Test that Swagger UI documentation is accessible."""
        response = requests.get(f"{API_BASE}/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_redoc_accessible(self):
        """Test that ReDoc documentation is accessible."""
        response = requests.get(f"{API_BASE}/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()


class TestStylesEndpoint:
    """Test the styles catalog endpoint."""

    def test_get_styles_returns_list(self):
        """Test that GET /api/v1/styles returns style list."""
        response = requests.get(f"{API_BASE}/api/v1/styles")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert "styles" in data
        assert data["count"] > 0
        assert len(data["styles"]) == data["count"]

    def test_styles_have_required_fields(self):
        """Test that each style has all required fields."""
        response = requests.get(f"{API_BASE}/api/v1/styles")
        data = response.json()

        required_fields = ["name", "model_id", "description", "preferred_tempo_range", "humanization"]

        for style in data["styles"]:
            for field in required_fields:
                assert field in style, f"Style missing field: {field}"

            # Verify humanization parameters
            assert "swing" in style["humanization"]
            assert "micro_timing_ms" in style["humanization"]
            assert "ghost_note_prob" in style["humanization"]
            assert "velocity_variation" in style["humanization"]

    def test_known_styles_present(self):
        """Test that all expected producer styles are present."""
        response = requests.get(f"{API_BASE}/api/v1/styles")
        data = response.json()

        style_names = [style["name"] for style in data["styles"]]
        expected_styles = ["J Dilla", "Metro Boomin", "Questlove"]

        for expected in expected_styles:
            assert expected in style_names, f"Expected style '{expected}' not found"


class TestPatternGenerationWorkflow:
    """Test complete pattern generation workflow end-to-end."""

    def _poll_task_status(self, task_id: str, timeout: int = TASK_TIMEOUT) -> Dict[str, Any]:
        """Poll task status until completion or timeout.

        Args:
            task_id: Celery task ID
            timeout: Maximum seconds to wait

        Returns:
            Final task status response

        Raises:
            TimeoutError: If task doesn't complete within timeout
        """
        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout:
            response = requests.get(f"{API_BASE}/api/v1/status/{task_id}")
            assert response.status_code == 200

            status_data = response.json()
            last_status = status_data

            if status_data["status"] == "completed":
                return status_data
            elif status_data["status"] == "failed":
                error = status_data.get("error", "Unknown error")
                pytest.fail(f"Task failed: {error}")

            # Log progress
            progress = status_data.get("progress", 0)
            print(f"Task {task_id}: {status_data['status']} ({progress}%)")

            time.sleep(1)

        # Timeout reached
        raise TimeoutError(
            f"Task {task_id} did not complete within {timeout} seconds. "
            f"Last status: {last_status}"
        )

    def test_generate_j_dilla_pattern(self):
        """Test generating a J Dilla style pattern end-to-end."""
        # Submit generation request
        request_data = {
            "producer_style": "J Dilla",
            "bars": 2,
            "tempo": 95,
            "time_signature": [4, 4],
            "humanize": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )

        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"

        task_id = data["task_id"]
        print(f"\nTask queued: {task_id}")

        # Poll until completion
        final_status = self._poll_task_status(task_id)

        # Verify result
        assert final_status["status"] == "completed"
        assert final_status["progress"] == 100
        assert "result" in final_status

        result = final_status["result"]
        assert "midi_file" in result
        assert "duration_seconds" in result
        assert "tokens_generated" in result
        assert result["style"] == "J Dilla"
        assert result["bars"] == 2
        assert result["tempo"] == 95

        # Verify MIDI file was created
        midi_path = Path(result["midi_file"])
        assert midi_path.exists(), f"MIDI file not found: {midi_path}"
        assert midi_path.suffix == ".mid"

        print(f"✓ Generated MIDI: {result['midi_file']}")
        print(f"✓ Duration: {result['duration_seconds']:.2f}s")
        print(f"✓ Tokens: {result['tokens_generated']}")

    def test_generate_metro_boomin_pattern(self):
        """Test generating a Metro Boomin style pattern."""
        request_data = {
            "producer_style": "Metro Boomin",
            "bars": 2,
            "tempo": 140,
            "humanize": True
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )

        assert response.status_code == 202
        task_id = response.json()["task_id"]

        final_status = self._poll_task_status(task_id)
        assert final_status["status"] == "completed"
        assert final_status["result"]["style"] == "Metro Boomin"

        # Verify file exists
        midi_path = Path(final_status["result"]["midi_file"])
        assert midi_path.exists()

    def test_generate_questlove_pattern(self):
        """Test generating a Questlove style pattern."""
        request_data = {
            "producer_style": "Questlove",
            "bars": 2,
            "tempo": 110,
            "humanize": True
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )

        assert response.status_code == 202
        task_id = response.json()["task_id"]

        final_status = self._poll_task_status(task_id)
        assert final_status["status"] == "completed"
        assert final_status["result"]["style"] == "Questlove"

        # Verify file exists
        midi_path = Path(final_status["result"]["midi_file"])
        assert midi_path.exists()

    def test_generate_without_humanization(self):
        """Test generating pattern without humanization."""
        request_data = {
            "producer_style": "J Dilla",
            "bars": 1,
            "tempo": 95,
            "humanize": False
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )

        assert response.status_code == 202
        task_id = response.json()["task_id"]

        final_status = self._poll_task_status(task_id)
        assert final_status["status"] == "completed"

        # Verify file exists
        midi_path = Path(final_status["result"]["midi_file"])
        assert midi_path.exists()

    def test_generate_with_custom_sampling_params(self):
        """Test generating pattern with custom sampling parameters."""
        request_data = {
            "producer_style": "Metro Boomin",
            "bars": 2,
            "tempo": 140,
            "temperature": 1.2,
            "top_k": 40,
            "top_p": 0.85
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )

        assert response.status_code == 202
        task_id = response.json()["task_id"]

        final_status = self._poll_task_status(task_id)
        assert final_status["status"] == "completed"

        # Verify file exists
        midi_path = Path(final_status["result"]["midi_file"])
        assert midi_path.exists()

    def test_generate_with_different_bar_counts(self):
        """Test generating patterns with various bar counts."""
        bar_counts = [1, 4, 8]

        for bars in bar_counts:
            request_data = {
                "producer_style": "J Dilla",
                "bars": bars,
                "tempo": 95,
                "humanize": False
            }

            response = requests.post(
                f"{API_BASE}/api/v1/generate",
                json=request_data
            )

            assert response.status_code == 202
            task_id = response.json()["task_id"]

            final_status = self._poll_task_status(task_id)
            assert final_status["status"] == "completed"
            assert final_status["result"]["bars"] == bars

            # Verify file exists
            midi_path = Path(final_status["result"]["midi_file"])
            assert midi_path.exists()

            print(f"✓ Generated {bars} bars successfully")


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_producer_style(self):
        """Test that invalid producer style is rejected."""
        request_data = {
            "producer_style": "Invalid Producer",
            "bars": 4,
            "tempo": 120
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_bars_out_of_range(self):
        """Test that bars outside valid range are rejected."""
        # Too few bars
        request_data = {
            "producer_style": "J Dilla",
            "bars": 0,
            "tempo": 95
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )
        assert response.status_code == 422

        # Too many bars
        request_data["bars"] = 33
        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )
        assert response.status_code == 422

    def test_tempo_out_of_range(self):
        """Test that tempo outside valid range is rejected."""
        # Too slow
        request_data = {
            "producer_style": "J Dilla",
            "bars": 4,
            "tempo": 59
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )
        assert response.status_code == 422

        # Too fast
        request_data["tempo"] = 201
        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )
        assert response.status_code == 422

    def test_invalid_time_signature(self):
        """Test that invalid time signature is rejected."""
        request_data = {
            "producer_style": "J Dilla",
            "bars": 4,
            "tempo": 95,
            "time_signature": [4, 3]  # Invalid denominator
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )
        assert response.status_code == 422

    def test_nonexistent_task_id(self):
        """Test querying status of nonexistent task."""
        response = requests.get(f"{API_BASE}/api/v1/status/nonexistent-task-id")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "pending"  # Celery returns PENDING for unknown tasks

    def test_temperature_out_of_range(self):
        """Test that temperature outside valid range is rejected."""
        # Too low
        request_data = {
            "producer_style": "J Dilla",
            "bars": 4,
            "tempo": 95,
            "temperature": 0.05
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )
        assert response.status_code == 422

        # Too high
        request_data["temperature"] = 2.5
        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )
        assert response.status_code == 422


class TestConcurrentGeneration:
    """Test concurrent task generation."""

    def test_submit_multiple_tasks_concurrently(self):
        """Test submitting multiple generation tasks at once."""
        tasks = []

        # Submit 3 tasks
        for i in range(3):
            request_data = {
                "producer_style": "J Dilla",
                "bars": 1,
                "tempo": 95,
                "humanize": False
            }

            response = requests.post(
                f"{API_BASE}/api/v1/generate",
                json=request_data
            )

            assert response.status_code == 202
            tasks.append(response.json()["task_id"])

        print(f"\nSubmitted {len(tasks)} concurrent tasks")

        # Wait for all to complete
        for i, task_id in enumerate(tasks):
            start = time.time()

            while time.time() - start < TASK_TIMEOUT:
                response = requests.get(f"{API_BASE}/api/v1/status/{task_id}")
                status = response.json()

                if status["status"] == "completed":
                    print(f"✓ Task {i+1} completed")
                    break
                elif status["status"] == "failed":
                    pytest.fail(f"Task {i+1} failed: {status.get('error')}")

                time.sleep(1)
            else:
                pytest.fail(f"Task {i+1} timed out")

        print("✓ All concurrent tasks completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
