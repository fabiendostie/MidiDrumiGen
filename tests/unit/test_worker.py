"""Unit tests for Celery worker configuration."""

import os
from unittest.mock import MagicMock, patch

from src.tasks.worker import REDIS_BACKEND, REDIS_URL, celery_app


class TestCeleryConfiguration:
    """Tests for Celery configuration."""

    def test_celery_app_exists(self):
        """Test that celery_app is defined."""
        assert celery_app is not None

    def test_celery_app_name(self):
        """Test that Celery app has correct name."""
        assert celery_app.main == "drum_generator"

    def test_celery_broker_url(self):
        """Test that broker URL is set."""
        assert celery_app.conf.broker_url == REDIS_URL

    def test_celery_result_backend(self):
        """Test that result backend is set."""
        assert celery_app.conf.result_backend == REDIS_BACKEND

    def test_task_serializer(self):
        """Test that task serializer is JSON."""
        assert celery_app.conf.task_serializer == "json"

    def test_result_serializer(self):
        """Test that result serializer is JSON."""
        assert celery_app.conf.result_serializer == "json"

    def test_accept_content(self):
        """Test that only JSON content is accepted."""
        assert "json" in celery_app.conf.accept_content

    def test_timezone_utc(self):
        """Test that timezone is UTC."""
        assert celery_app.conf.timezone == "UTC"

    def test_enable_utc(self):
        """Test that UTC is enabled."""
        assert celery_app.conf.enable_utc is True

    def test_task_track_started(self):
        """Test that task tracking is enabled."""
        assert celery_app.conf.task_track_started is True

    def test_task_time_limit(self):
        """Test task time limit."""
        assert celery_app.conf.task_time_limit == 300  # 5 minutes

    def test_task_soft_time_limit(self):
        """Test task soft time limit."""
        assert celery_app.conf.task_soft_time_limit == 240  # 4 minutes

    def test_worker_prefetch_multiplier(self):
        """Test worker prefetch multiplier."""
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_worker_max_tasks_per_child(self):
        """Test worker max tasks per child."""
        assert celery_app.conf.worker_max_tasks_per_child == 50

    def test_result_expires(self):
        """Test result expiration time."""
        assert celery_app.conf.result_expires == 3600  # 1 hour

    def test_task_acks_late(self):
        """Test that late acknowledgment is enabled."""
        assert celery_app.conf.task_acks_late is True

    def test_task_reject_on_worker_lost(self):
        """Test that tasks are rejected if worker crashes."""
        assert celery_app.conf.task_reject_on_worker_lost is True


class TestTaskRoutes:
    """Tests for task routing configuration."""

    def test_task_routes_configured(self):
        """Test that task routes are configured."""
        assert celery_app.conf.task_routes is not None

    def test_generate_pattern_route(self):
        """Test routing for generate_pattern task."""
        routes = celery_app.conf.task_routes
        assert "src.tasks.tasks.generate_pattern" in routes
        assert routes["src.tasks.tasks.generate_pattern"]["queue"] == "gpu_generation"

    def test_tokenize_midi_route(self):
        """Test routing for tokenize_midi task."""
        routes = celery_app.conf.task_routes
        assert "src.tasks.tasks.tokenize_midi" in routes
        assert routes["src.tasks.tasks.tokenize_midi"]["queue"] == "midi_processing"

    def test_train_model_route(self):
        """Test routing for train_model task."""
        routes = celery_app.conf.task_routes
        assert "src.tasks.tasks.train_model" in routes
        assert routes["src.tasks.tasks.train_model"]["queue"] == "heavy_tasks"


class TestRedisConfiguration:
    """Tests for Redis configuration."""

    def test_redis_url_default(self):
        """Test default Redis URL."""
        # Reset environment
        with patch.dict(os.environ, {}, clear=True):
            from importlib import reload

            import src.tasks.worker as worker_module

            reload(worker_module)

            # Should use default
            assert "redis://localhost:6379" in worker_module.REDIS_URL

    @patch.dict(os.environ, {"REDIS_URL": "redis://custom:6379/0"})
    def test_redis_url_from_env(self):
        """Test Redis URL from environment variable."""
        from importlib import reload

        import src.tasks.worker as worker_module

        reload(worker_module)

        assert worker_module.REDIS_URL == "redis://custom:6379/0"

    def test_redis_backend_default(self):
        """Test default Redis backend."""
        with patch.dict(os.environ, {}, clear=True):
            from importlib import reload

            import src.tasks.worker as worker_module

            reload(worker_module)

            assert "redis://localhost:6379" in worker_module.REDIS_BACKEND


class TestSignalHandlers:
    """Tests for Celery signal handlers."""

    @patch("src.tasks.worker.logger")
    def test_worker_ready_signal(self, mock_logger):
        """Test worker_ready signal handler."""
        from src.tasks.worker import on_worker_ready

        on_worker_ready(sender=None)

        # Should log ready message
        assert mock_logger.info.called
        call_args = [str(call) for call in mock_logger.info.call_args_list]
        assert any("ready" in str(arg).lower() for arg in call_args)

    @patch("src.tasks.worker.logger")
    def test_worker_shutdown_signal(self, mock_logger):
        """Test worker_shutdown signal handler."""
        from src.tasks.worker import on_worker_shutdown

        on_worker_shutdown(sender=None)

        # Should log shutdown message
        assert mock_logger.info.called
        call_args = [str(call) for call in mock_logger.info.call_args_list]
        assert any("shutting down" in str(arg).lower() for arg in call_args)

    @patch("src.tasks.worker.logger")
    def test_task_prerun_signal(self, mock_logger):
        """Test task_prerun signal handler."""
        from src.tasks.worker import on_task_prerun

        mock_task = MagicMock()
        mock_task.name = "test_task"

        on_task_prerun(sender=None, task_id="test-123", task=mock_task)

        # Should log task start
        assert mock_logger.info.called
        call_args = [str(call) for call in mock_logger.info.call_args_list]
        assert any("starting task" in str(arg).lower() for arg in call_args)

    @patch("src.tasks.worker.logger")
    def test_task_postrun_signal(self, mock_logger):
        """Test task_postrun signal handler."""
        from src.tasks.worker import on_task_postrun

        mock_task = MagicMock()
        mock_task.name = "test_task"

        on_task_postrun(sender=None, task_id="test-123", task=mock_task, state="SUCCESS")

        # Should log task completion
        assert mock_logger.info.called
        call_args = [str(call) for call in mock_logger.info.call_args_list]
        assert any("completed task" in str(arg).lower() for arg in call_args)


class TestCeleryApp:
    """Tests for Celery app functionality."""

    def test_celery_app_can_send_task(self):
        """Test that Celery app can create task signatures."""
        # This doesn't actually send the task, just creates the signature
        signature = celery_app.signature("src.tasks.tasks.generate_pattern")
        assert signature is not None

    def test_celery_app_has_tasks_loaded(self):
        """Test that tasks are loaded in the app."""
        # Check that tasks module is included
        assert "src.tasks.tasks" in celery_app.conf.include

    def test_celery_app_control(self):
        """Test that Celery app has control interface."""
        assert hasattr(celery_app, "control")
        assert celery_app.control is not None

    def test_celery_app_backend(self):
        """Test that Celery app has backend configured."""
        assert hasattr(celery_app, "backend")


class TestCeleryIntegration:
    """Integration tests for Celery configuration."""

    def test_task_signature_creation(self):
        """Test creating task signatures for all configured tasks."""
        task_names = [
            "src.tasks.tasks.generate_pattern",
            "src.tasks.tasks.tokenize_midi",
            "src.tasks.tasks.train_model",
        ]

        for task_name in task_names:
            signature = celery_app.signature(task_name)
            assert signature is not None

    def test_task_routes_match_task_names(self):
        """Test that all routed tasks exist."""
        routes = celery_app.conf.task_routes

        for task_name in routes:
            # Should be able to create signature
            signature = celery_app.signature(task_name)
            assert signature is not None

    def test_serialization_config_consistency(self):
        """Test that serialization config is consistent."""
        # Task and result serializers should match
        assert celery_app.conf.task_serializer == celery_app.conf.result_serializer

        # Accepted content should include the serializer
        assert celery_app.conf.task_serializer in celery_app.conf.accept_content


class TestCeleryEdgeCases:
    """Tests for edge cases in Celery configuration."""

    def test_task_time_limits_reasonable(self):
        """Test that time limits are reasonable."""
        assert celery_app.conf.task_time_limit > 0
        assert celery_app.conf.task_soft_time_limit > 0
        assert celery_app.conf.task_soft_time_limit < celery_app.conf.task_time_limit

    def test_result_expiration_reasonable(self):
        """Test that result expiration is reasonable."""
        assert celery_app.conf.result_expires > 0
        # Should expire within a day
        assert celery_app.conf.result_expires <= 86400

    def test_worker_prefetch_multiplier_reasonable(self):
        """Test that prefetch multiplier is reasonable."""
        assert celery_app.conf.worker_prefetch_multiplier >= 1
        assert celery_app.conf.worker_prefetch_multiplier <= 10

    def test_max_tasks_per_child_prevents_memory_leak(self):
        """Test that max tasks per child helps prevent memory leaks."""
        # Should restart worker periodically
        assert celery_app.conf.worker_max_tasks_per_child > 0
        assert celery_app.conf.worker_max_tasks_per_child <= 1000


# Tests for Celery tasks
class TestGeneratePatternTask:
    """Tests for generate_pattern Celery task."""

    def test_generate_pattern_task_exists(self):
        """Test that generate_pattern task is registered."""
        from src.tasks.tasks import generate_pattern

        assert generate_pattern is not None
        assert hasattr(generate_pattern, "delay")

    @patch("src.tasks.tasks.torch")
    @patch("src.tasks.tasks.logger")
    def test_generate_pattern_detects_device(self, mock_logger, mock_torch):
        """Test that generate_pattern detects available device."""
        from src.tasks.tasks import generate_pattern

        mock_torch.cuda.is_available.return_value = True
        mock_torch.device = MagicMock()

        params = {"bars": 4, "tempo": 120}
        generate_pattern(params)

        # Should detect CUDA
        mock_torch.cuda.is_available.assert_called()

    @patch("src.tasks.tasks.torch")
    def test_generate_pattern_handles_cpu(self, mock_torch):
        """Test that generate_pattern works on CPU."""
        from src.tasks.tasks import generate_pattern

        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = MagicMock()

        params = {"bars": 4, "tempo": 120}
        result = generate_pattern(params)

        assert result is not None
        assert "status" in result

    def test_generate_pattern_task_has_max_retries(self):
        """Test that generate_pattern has retry configuration."""
        from src.tasks.tasks import generate_pattern

        # Should have max_retries configured
        assert hasattr(generate_pattern, "max_retries")
        assert generate_pattern.max_retries == 3


class TestTokenizeMidiTask:
    """Tests for tokenize_midi Celery task."""

    def test_tokenize_midi_task_exists(self):
        """Test that tokenize_midi task is registered."""
        from src.tasks.tasks import tokenize_midi

        assert tokenize_midi is not None
        assert hasattr(tokenize_midi, "delay")

    def test_tokenize_midi_returns_dict(self):
        """Test that tokenize_midi returns a dictionary."""
        from src.tasks.tasks import tokenize_midi

        result = tokenize_midi("test.mid")

        assert isinstance(result, dict)
        assert "status" in result


class TestTrainModelTask:
    """Tests for train_model Celery task."""

    def test_train_model_task_exists(self):
        """Test that train_model task is registered."""
        from src.tasks.tasks import train_model

        assert train_model is not None
        assert hasattr(train_model, "delay")

    def test_train_model_returns_dict(self):
        """Test that train_model returns a dictionary."""
        from src.tasks.tasks import train_model

        result = train_model("config.yaml")

        assert isinstance(result, dict)
        assert "status" in result
