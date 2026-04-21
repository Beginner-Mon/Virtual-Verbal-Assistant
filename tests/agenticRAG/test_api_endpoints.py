"""
Unit tests for agenticRAG API endpoints.

Target: main_api.py (FastAPI app on port 8080)
Tests route registration and endpoint behavior with mocked downstream services.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient
from fastapi import HTTPException

from conftest import make_answer_response


# ============================================================================
# Route Registration
# ============================================================================

class TestRouteRegistration:
    """Verify all expected routes are registered on the app."""

    def test_expected_routes_are_registered(self, main_api_client):
        from main_api import app

        route_paths = {route.path for route in app.routes}
        expected = {
            "/answer",
            "/answer/status/{request_id}",
            "/query",
            "/process_query",
            "/tasks/{task_id}",
            "/health",
            "/info",
            "/audio/{filename}",
            "/sessions",
            "/sessions/{user_id}",
            "/sessions/{user_id}/{session_id}",
            "/sessions/{user_id}/{session_id}/summarize",
        }
        missing = expected - route_paths
        assert not missing, f"Missing routes: {missing}"


# ============================================================================
# Health Endpoint
# ============================================================================

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_healthy_when_all_services_ok(self, main_api_client, mock_check_services_health):
        mock_check_services_health.return_value = {"agenticrag": "ok", "dart": "ok"}

        resp = main_api_client.get("/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "healthy"
        assert data["services"]["agenticrag"] == "ok"
        assert data["services"]["dart"] == "ok"

    def test_degraded_when_service_down(self, main_api_client, mock_check_services_health):
        mock_check_services_health.return_value = {
            "agenticrag": "ok",
            "dart": "unreachable (ConnectError)",
        }

        resp = main_api_client.get("/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "degraded"


# ============================================================================
# Info Endpoint
# ============================================================================

class TestInfoEndpoint:
    """Tests for GET /info."""

    def test_returns_service_metadata(self, main_api_client):
        resp = main_api_client.get("/info")
        assert resp.status_code == 200

        data = resp.json()
        assert data["service"] == "Unified Multi-Service Pipeline"
        assert "version" in data
        assert "upstream_services" in data
        assert "endpoints" in data
        assert "agenticrag" in data["upstream_services"]
        assert "dart" in data["upstream_services"]
        assert "tts" in data["upstream_services"]

    def test_includes_config_flags(self, main_api_client):
        resp = main_api_client.get("/info")
        data = resp.json()
        assert "async_enrichment" in data
        assert "include_debug" in data


# ============================================================================
# Answer Endpoint
# ============================================================================

class TestAnswerEndpoint:
    """Tests for POST /answer."""

    def test_success_returns_answer_response(self, main_api_client):
        from schemas.main_api import AnswerResponse

        with patch(
            "routers.answer.get_answer_impl",
            new_callable=AsyncMock,
            return_value=AnswerResponse(**make_answer_response()),
        ):
            resp = main_api_client.post("/answer", json={
                "query": "What helps with back pain?",
                "user_id": "test_user",
            })
            assert resp.status_code == 200

            data = resp.json()
            assert data["request_id"] == "test_abc123"
            assert data["status"] == "completed"
            assert data["text_answer"] == "Try gentle neck stretches."
            assert data["language"] == "en"
            assert isinstance(data["exercises"], list)

    def test_default_user_id(self, main_api_client):
        from schemas.main_api import AnswerResponse

        mock_impl = AsyncMock(return_value=AnswerResponse(**make_answer_response()))

        with patch("routers.answer.get_answer_impl", mock_impl):
            resp = main_api_client.post("/answer", json={"query": "test"})
            assert resp.status_code == 200

            call_args = mock_impl.call_args
            request_arg = call_args[0][0]
            assert request_arg.user_id == "default"

    def test_missing_query_returns_422(self, main_api_client):
        resp = main_api_client.post("/answer", json={})
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, main_api_client):
        resp = main_api_client.post("/answer")
        assert resp.status_code == 422

    def test_with_session_id(self, main_api_client):
        from schemas.main_api import AnswerResponse

        mock_impl = AsyncMock(return_value=AnswerResponse(**make_answer_response()))

        with patch("routers.answer.get_answer_impl", mock_impl):
            resp = main_api_client.post("/answer", json={
                "query": "followup question",
                "session_id": "sess_abc",
            })
            assert resp.status_code == 200

            call_args = mock_impl.call_args
            request_arg = call_args[0][0]
            assert request_arg.session_id == "sess_abc"

    def test_with_motion_format_npz(self, main_api_client):
        from schemas.main_api import AnswerResponse

        mock_impl = AsyncMock(return_value=AnswerResponse(**make_answer_response()))

        with patch("routers.answer.get_answer_impl", mock_impl):
            resp = main_api_client.post("/answer", json={
                "query": "show me a squat",
                "motion_format": "npz",
            })
            assert resp.status_code == 200

            call_args = mock_impl.call_args
            request_arg = call_args[0][0]
            assert request_arg.motion_format == "npz"

    def test_response_includes_generation_time(self, main_api_client):
        from schemas.main_api import AnswerResponse

        with patch(
            "routers.answer.get_answer_impl",
            new_callable=AsyncMock,
            return_value=AnswerResponse(**make_answer_response(generation_time_ms=456.7)),
        ):
            resp = main_api_client.post("/answer", json={"query": "test"})
            data = resp.json()
            assert data["generation_time_ms"] == 456.7


# ============================================================================
# Answer Status Endpoint
# ============================================================================

class TestAnswerStatusEndpoint:
    """Tests for GET /answer/status/{request_id}."""

    def test_not_found_returns_404(self, main_api_client):
        with patch(
            "routers.answer.get_answer_status_impl",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404, detail="Unknown request_id: nonexistent"),
        ):
            resp = main_api_client.get("/answer/status/nonexistent")
            assert resp.status_code == 404

    def test_found_returns_answer_response(self, main_api_client):
        from schemas.main_api import AnswerResponse

        with patch(
            "routers.answer.get_answer_status_impl",
            new_callable=AsyncMock,
            return_value=AnswerResponse(
                **make_answer_response(
                    request_id="req_poll",
                    status="processing",
                    progress_stage="motion_generation",
                    pending_services=["dart"],
                )
            ),
        ):
            resp = main_api_client.get("/answer/status/req_poll")
            assert resp.status_code == 200

            data = resp.json()
            assert data["request_id"] == "req_poll"
            assert data["status"] == "processing"
            assert data["progress_stage"] == "motion_generation"
            assert "dart" in data["pending_services"]


# ============================================================================
# Audio Proxy Endpoint
# ============================================================================

class TestAudioProxyEndpoint:
    """Tests for GET /audio/{filename}."""

    def test_proxy_returns_audio_stream(self, main_api_client):
        # Create a proper async iterator for aiter_bytes
        async def fake_aiter_bytes():
            yield b"\xff\xfb\x90\x00"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_bytes = fake_aiter_bytes

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client_instance = MagicMock()
        mock_client_instance.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client_instance):
            resp = main_api_client.get("/audio/test_audio.mp3")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/mpeg"
