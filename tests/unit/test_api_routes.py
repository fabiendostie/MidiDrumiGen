"""Unit tests for API routes (Phase 4)."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_200(self):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_name(self):
        """Test root endpoint contains API name."""
        response = client.get("/")
        data = response.json()
        assert "name" in data
        assert "Drum Pattern Generator" in data["name"]

    def test_root_contains_version(self):
        """Test root endpoint contains version."""
        response = client.get("/")
        data = response.json()
        assert "version" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status(self):
        """Test health endpoint contains status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_contains_redis_status(self):
        """Test health endpoint contains Redis status."""
        response = client.get("/health")
        data = response.json()
        assert "redis" in data


class TestStylesEndpoint:
    """Tests for styles list endpoint."""

    def test_styles_returns_200(self):
        """Test styles endpoint returns 200."""
        response = client.get("/api/v1/styles")
        assert response.status_code == 200

    def test_styles_returns_list(self):
        """Test styles endpoint returns list of styles."""
        response = client.get("/api/v1/styles")
        data = response.json()
        assert "styles" in data
        assert isinstance(data["styles"], list)

    def test_styles_returns_count(self):
        """Test styles endpoint returns count."""
        response = client.get("/api/v1/styles")
        data = response.json()
        assert "count" in data
        assert data["count"] == len(data["styles"])

    def test_styles_contain_required_fields(self):
        """Test each style contains required fields."""
        response = client.get("/api/v1/styles")
        data = response.json()

        for style in data["styles"]:
            assert "name" in style
            assert "model_id" in style
            assert "description" in style
            assert "preferred_tempo_range" in style
            assert "humanization" in style

    def test_styles_include_j_dilla(self):
        """Test styles include J Dilla."""
        response = client.get("/api/v1/styles")
        data = response.json()
        style_names = [s["name"] for s in data["styles"]]
        assert "J Dilla" in style_names


class TestGenerateEndpoint:
    """Tests for pattern generation endpoint."""

    @patch('src.api.routes.generate.generate_pattern_task')
    def test_generate_returns_202(self, mock_task):
        """Test generate endpoint returns 202 Accepted."""
        # Mock Celery task
        mock_task.delay.return_value = Mock(id="test-task-id")

        request_data = {
            "producer_style": "J Dilla",
            "bars": 4,
            "tempo": 95
        }

        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 202

    @patch('src.api.routes.generate.generate_pattern_task')
    def test_generate_returns_task_id(self, mock_task):
        """Test generate endpoint returns task ID."""
        mock_task.delay.return_value = Mock(id="test-task-id")

        request_data = {
            "producer_style": "J Dilla",
            "bars": 4,
            "tempo": 95
        }

        response = client.post("/api/v1/generate", json=request_data)
        data = response.json()

        assert "task_id" in data
        assert data["task_id"] == "test-task-id"

    @patch('src.api.routes.generate.generate_pattern_task')
    def test_generate_returns_status(self, mock_task):
        """Test generate endpoint returns status."""
        mock_task.delay.return_value = Mock(id="test-task-id")

        request_data = {
            "producer_style": "J Dilla",
            "bars": 4,
            "tempo": 95
        }

        response = client.post("/api/v1/generate", json=request_data)
        data = response.json()

        assert "status" in data
        assert data["status"] == "queued"

    @patch('src.api.routes.generate.generate_pattern_task')
    def test_generate_calls_celery_task(self, mock_task):
        """Test generate endpoint calls Celery task with correct params."""
        mock_task.delay.return_value = Mock(id="test-task-id")

        request_data = {
            "producer_style": "J Dilla",
            "bars": 4,
            "tempo": 95,
            "time_signature": [4, 4],
            "humanize": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9
        }

        client.post("/api/v1/generate", json=request_data)

        mock_task.delay.assert_called_once()
        call_kwargs = mock_task.delay.call_args[1]
        assert call_kwargs["producer_style"] == "J Dilla"
        assert call_kwargs["bars"] == 4
        assert call_kwargs["tempo"] == 95

    def test_generate_validates_bars(self):
        """Test generate endpoint validates bars parameter."""
        request_data = {
            "producer_style": "J Dilla",
            "bars": 0,  # Invalid
            "tempo": 95
        }

        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_generate_validates_tempo(self):
        """Test generate endpoint validates tempo parameter."""
        request_data = {
            "producer_style": "J Dilla",
            "bars": 4,
            "tempo": 300  # Invalid
        }

        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_generate_validates_producer_style(self):
        """Test generate endpoint validates producer style."""
        request_data = {
            "producer_style": "Invalid Style",
            "bars": 4,
            "tempo": 95
        }

        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 422  # Validation error


class TestStatusEndpoint:
    """Tests for task status endpoint."""

    @patch('src.api.routes.status.AsyncResult')
    def test_status_returns_200(self, mock_result):
        """Test status endpoint returns 200."""
        mock_task = Mock()
        mock_task.state = 'PENDING'
        mock_task.info = None
        mock_result.return_value = mock_task

        response = client.get("/api/v1/status/test-task-id")
        assert response.status_code == 200

    @patch('src.api.routes.status.AsyncResult')
    def test_status_pending_task(self, mock_result):
        """Test status endpoint for pending task."""
        mock_task = Mock()
        mock_task.state = 'PENDING'
        mock_task.info = None
        mock_result.return_value = mock_task

        response = client.get("/api/v1/status/test-task-id")
        data = response.json()

        assert data["status"] == "pending"
        assert data["progress"] == 0

    @patch('src.api.routes.status.AsyncResult')
    def test_status_progress_task(self, mock_result):
        """Test status endpoint for task in progress."""
        mock_task = Mock()
        mock_task.state = 'PROGRESS'
        mock_task.info = {'progress': 50}
        mock_result.return_value = mock_task

        response = client.get("/api/v1/status/test-task-id")
        data = response.json()

        assert data["status"] == "processing"
        assert data["progress"] == 50

    @patch('src.api.routes.status.AsyncResult')
    def test_status_success_task(self, mock_result):
        """Test status endpoint for successful task."""
        mock_task = Mock()
        mock_task.state = 'SUCCESS'
        mock_task.result = {
            "midi_file": "output/test.mid",
            "duration_seconds": 1.5
        }
        mock_result.return_value = mock_task

        response = client.get("/api/v1/status/test-task-id")
        data = response.json()

        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert data["result"]["midi_file"] == "output/test.mid"

    @patch('src.api.routes.status.AsyncResult')
    def test_status_failure_task(self, mock_result):
        """Test status endpoint for failed task."""
        mock_task = Mock()
        mock_task.state = 'FAILURE'
        mock_task.info = Exception("Model loading failed")
        mock_result.return_value = mock_task

        response = client.get("/api/v1/status/test-task-id")
        data = response.json()

        assert data["status"] == "failed"
        assert "error" in data
