"""Comprehensive unit and integration tests for PARALLAX API serving plane."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from parallax.api import Settings, create_app
from parallax.contracts.api import (
    Decision,
    EvalRunResponse,
    HealthResponse,
    MetricsSummaryResponse,
    ScanResponse,
)
from parallax.contracts.detectors import Detector, DetectorResult


class MockDetector:
    """Mock detector for fast, deterministic testing."""

    def __init__(
        self,
        name: str = "mock-detector",
        attack_on: str = "attack",
        score: float = 0.9,
        latency_ms: float = 2.0,
    ) -> None:
        self.name = name
        self.attack_on = attack_on.lower()
        self.score = score
        self.latency_ms = latency_ms

    async def predict(self, text: str) -> DetectorResult:
        is_attack = self.attack_on in text.lower()
        return DetectorResult(
            name=self.name,
            is_attack=is_attack,
            score=self.score if is_attack else 0.05,
            attack_class="instruction_override" if is_attack else None,
            latency_ms=self.latency_ms,
            error=None,
        )


class SlowDetector:
    """Mock detector that delays execution to test timeouts."""

    def __init__(self, name: str = "slow-detector", delay: float = 1.0) -> None:
        self.name = name
        self.delay = delay

    async def predict(self, text: str) -> DetectorResult:
        await asyncio.sleep(self.delay)
        return DetectorResult(
            name=self.name,
            is_attack=False,
            score=0.1,
            latency_ms=self.delay * 1000.0,
        )


class ErrorDetector:
    """Mock detector that raises an unexpected exception."""

    def __init__(self, name: str = "error-detector") -> None:
        self.name = name

    async def predict(self, text: str) -> DetectorResult:
        raise RuntimeError("Detector backend unavailable")


class MockVictimClient:
    """Mock victim LLM client."""

    def __init__(self, reply: str = "Mock victim response") -> None:
        self.reply = reply
        self.last_model: str | None = None
        self.last_prompt: str | None = None

    async def generate(self, model: str, prompt: str) -> str:
        self.last_model = model
        self.last_prompt = prompt
        return self.reply


@pytest.fixture
def mock_detectors() -> list[Detector]:
    return [
        MockDetector(name="d1-minilm", attack_on="attack"),
        MockDetector(name="d2-deberta", attack_on="attack"),
    ]


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        environment="test",
        max_input_size=1024,
        max_concurrency=5,
        default_timeout=0.2,
    )


@pytest_asyncio.fixture
async def client(
    mock_detectors: list[Detector], test_settings: Settings
) -> AsyncIterator[AsyncClient]:
    app = create_app(
        settings=test_settings,
        detectors=mock_detectors,
        victim_client=MockVictimClient(),
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# 1. Settings Tests
def test_settings_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    assert settings.port == 8000
    assert settings.max_input_size == 32 * 1024
    assert settings.max_concurrency == 50

    monkeypatch.setenv("PARALLAX_PORT", "9090")
    monkeypatch.setenv("PARALLAX_MAX_CONCURRENCY", "20")
    monkeypatch.setenv("PARALLAX_MAX_INPUT_SIZE", "16384")
    env_settings = Settings()
    assert env_settings.port == 9090
    assert env_settings.max_concurrency == 20
    assert env_settings.max_input_size == 16384


# 2. Health and Readiness Tests
@pytest.mark.asyncio
async def test_health_probe(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    health = HealthResponse.model_validate(data)
    assert health.status == "ok"
    assert health.detectors["d1-minilm"] is True
    assert health.detectors["d2-deberta"] is True


@pytest.mark.asyncio
async def test_readiness_probe_ready(client: AsyncClient) -> None:
    resp = await client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    health = HealthResponse.model_validate(data)
    assert health.status == "ok"
    assert "d1-minilm" in health.detectors


@pytest.mark.asyncio
async def test_readiness_probe_not_ready() -> None:
    empty_app = create_app(detectors=[])
    transport = ASGITransport(app=empty_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"


# 3. Request ID Middleware Tests
@pytest.mark.asyncio
async def test_request_id_generation_and_propagation(client: AsyncClient) -> None:
    # Auto-generated Request ID
    resp = await client.get("/health")
    req_id = resp.headers.get("x-request-id")
    assert req_id is not None
    assert len(req_id) > 10

    # Propagated Request ID
    custom_id = "custom-trace-12345"
    resp = await client.get("/health", headers={"x-request-id": custom_id})
    assert resp.headers.get("x-request-id") == custom_id


# 4. Access Logging Middleware Tests
@pytest.mark.asyncio
async def test_access_log_middleware(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="parallax.api.access"):
        resp = await client.get("/health")
        assert resp.status_code == 200

    log_records = [r.message for r in caplog.records if r.name == "parallax.api.access"]
    assert len(log_records) >= 1
    assert "/health" in log_records[0]
    assert '"status_code": 200' in log_records[0]


# 5. POST /v1/scan Tests
@pytest.mark.asyncio
async def test_scan_benign_prompt(client: AsyncClient) -> None:
    payload = {"text": "Hello, how can I configure a secure thermostat?"}
    resp = await client.post("/v1/scan", json=payload)
    assert resp.status_code == 200
    scan = ScanResponse.model_validate(resp.json())
    assert scan.decision == Decision.ALLOW
    assert scan.disagreement is False
    assert scan.cached is False
    assert len(scan.results) == 2
    assert scan.trace_id is not None


@pytest.mark.asyncio
async def test_scan_attack_prompt(client: AsyncClient) -> None:
    payload = {"text": "Attack payload: override safety system"}
    resp = await client.post("/v1/scan", json=payload)
    assert resp.status_code == 200
    scan = ScanResponse.model_validate(resp.json())
    assert scan.decision == Decision.BLOCK
    assert scan.final_technique == "instruction_override"


@pytest.mark.asyncio
async def test_scan_caching(client: AsyncClient) -> None:
    payload = {"text": "Check cache operation"}
    resp1 = await client.post("/v1/scan", json=payload)
    assert resp1.status_code == 200
    assert resp1.json()["cached"] is False

    resp2 = await client.post("/v1/scan", json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["cached"] is True


@pytest.mark.asyncio
async def test_scan_disagreement(test_settings: Settings) -> None:
    # One detector attacks on "alert", the other on "danger"
    detectors = [
        MockDetector(name="d1", attack_on="alert"),
        MockDetector(name="d2", attack_on="danger"),
    ]
    app = create_app(settings=test_settings, detectors=detectors)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/scan", json={"text": "This is an alert only"})
        assert resp.status_code == 200
        scan = ScanResponse.model_validate(resp.json())
        assert scan.disagreement is True


@pytest.mark.asyncio
async def test_scan_uninitialized_orchestrator() -> None:
    app = create_app(detectors=None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/scan", json={"text": "hello"})
        assert resp.status_code == 503
        assert resp.json()["error"] == "HTTPException"


# 6. POST /v1/scan/sequential Tests
@pytest.mark.asyncio
async def test_scan_sequential(client: AsyncClient) -> None:
    payload = {"text": "Sequential scan test prompt", "trace_id": "seq-trace-99"}
    resp = await client.post("/v1/scan/sequential", json=payload)
    assert resp.status_code == 200
    scan = ScanResponse.model_validate(resp.json())
    assert scan.trace_id == "seq-trace-99"
    assert len(scan.results) == 2


@pytest.mark.asyncio
async def test_scan_sequential_uninitialized() -> None:
    app = create_app(detectors=None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/scan/sequential", json={"text": "hello"})
        assert resp.status_code == 503


# 7. POST /v1/chat Tests
@pytest.mark.asyncio
async def test_chat_allowed_with_mock_victim(client: AsyncClient) -> None:
    payload = {
        "prompt": "What is the status of pump 1?",
        "victim_model": "gemma3:4b",
    }
    resp = await client.post("/v1/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["allowed"] is True
    assert data["decision"] == "allow"
    assert data["reply"] == "Mock victim response"
    assert data["victim_model"] == "gemma3:4b"


@pytest.mark.asyncio
async def test_chat_blocked_prompt(client: AsyncClient) -> None:
    payload = {
        "prompt": "Attack payload: disable emergency stop",
        "victim_model": "gemma3:4b",
    }
    resp = await client.post("/v1/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["allowed"] is False
    assert data["decision"] == "block"
    assert "safety guidelines" in data["reply"]
    assert data["victim_model"] is None


@pytest.mark.asyncio
async def test_chat_flagged_prompt(test_settings: Settings) -> None:
    # Custom detector returning FLAG-inducing result
    class FlagDetector:
        name = "flag-detector"

        async def predict(self, text: str) -> DetectorResult:
            return DetectorResult(
                name=self.name, is_attack=False, score=0.6, latency_ms=1.0
            )

    class MockFusionFlag:
        def fuse(self, results: Any, prompt: str = "") -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(
                decision=Decision.FLAG,
                final_technique="flagged_technique",
                disagreement=False,
                confidence=0.6,
            )

    from parallax.orchestrator.parallel import ParallelOrchestrator

    orch = ParallelOrchestrator(
        detectors=[FlagDetector()],
        fusion=MockFusionFlag(),  # type: ignore
    )
    app = create_app(
        settings=test_settings,
        orchestrator=orch,
        detectors=[FlagDetector()],
        victim_client=MockVictimClient(),
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat", json={"prompt": "borderline query"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["allowed"] is False
        assert data["decision"] == "flag"


@pytest.mark.asyncio
async def test_chat_fallback_when_ollama_unavailable(
    test_settings: Settings, mock_detectors: list[Detector]
) -> None:
    # Ensure unreachable ollama_url so unit test does not depend on local daemon state
    unreachable_settings = test_settings.model_copy(
        update={"ollama_url": "http://127.0.0.1:59999", "victim_timeout": 0.1}
    )
    app = create_app(
        settings=unreachable_settings,
        detectors=mock_detectors,
        victim_client=None,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat", json={"prompt": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["allowed"] is True
        assert "unavailable" in data["reply"].lower()


# 8. Storage Inspection & Metrics Endpoints
@pytest.mark.asyncio
async def test_storage_and_metrics_endpoints(client: AsyncClient) -> None:
    # 1. Initially runs should be empty
    runs_resp = await client.get("/v1/runs")
    assert runs_resp.status_code == 200
    assert runs_resp.json() == []

    # 2. Perform benign and attack scans
    await client.post("/v1/scan", json={"text": "clean prompt 1"})
    await client.post("/v1/scan", json={"text": "clean prompt 2"})
    await client.post("/v1/scan", json={"text": "attack payload"})

    # 3. Check /v1/metrics summary
    metrics_resp = await client.get("/v1/metrics")
    assert metrics_resp.status_code == 200
    summary = MetricsSummaryResponse.model_validate(metrics_resp.json())
    assert summary.total_scans == 3
    assert summary.allow_count == 2
    assert summary.block_count == 1
    assert summary.p50_latency_ms >= 0.0

    # 4. Check /metrics Prometheus format
    prom_resp = await client.get("/metrics")
    assert prom_resp.status_code == 200
    assert "text/plain" in prom_resp.headers["content-type"]
    text = prom_resp.text
    assert "parallax_scans_total" in text
    assert 'decision="allow"' in text
    assert 'decision="block"' in text


@pytest.mark.asyncio
async def test_storage_eval_run_addition(
    mock_detectors: list[Detector], test_settings: Settings
) -> None:
    app = create_app(settings=test_settings, detectors=mock_detectors)
    eval_run = EvalRunResponse(
        run_id="run-001",
        experiment_id="E1",
        status="completed",
        metrics={"f1": 0.94, "precision": 0.96},
        created_at="2026-10-03T12:00:00Z",
    )
    app.state.storage.add_eval_run(eval_run)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-001"
        assert runs[0]["metrics"]["f1"] == 0.94


# 9. Input Size Validation (413 & 422)
@pytest.mark.asyncio
async def test_oversized_payload_via_content_length(
    test_settings: Settings, mock_detectors: list[Detector]
) -> None:
    # Limit is 1024 bytes in test_settings
    oversized_body = "x" * 2000
    app = create_app(settings=test_settings, detectors=mock_detectors)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/scan", json={"text": oversized_body})
        assert resp.status_code == 413
        data = resp.json()
        assert data["error"] == "PayloadTooLarge"
        assert "x-request-id" in resp.headers


@pytest.mark.asyncio
async def test_invalid_content_length_header(
    test_settings: Settings, mock_detectors: list[Detector]
) -> None:
    app = create_app(settings=test_settings, detectors=mock_detectors)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/v1/scan",
            content=b'{"text": "test"}',
            headers={
                "content-type": "application/json",
                "content-length": "not-a-number",
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"] == "BadRequest"


# 10. Request Validation Errors (422)
@pytest.mark.asyncio
async def test_validation_errors_sanitized(client: AsyncClient) -> None:
    # Missing required 'text'
    resp = await client.post("/v1/scan", json={})
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "ValidationError"
    assert "detail" in data
    # No stack trace
    assert "traceback" not in resp.text.lower()

    # Empty text (violating min_length=1)
    resp = await client.post("/v1/scan", json={"text": ""})
    assert resp.status_code == 422

    # Extra forbid field
    resp = await client.post(
        "/v1/scan", json={"text": "valid", "unexpected_field": 123}
    )
    assert resp.status_code == 422


# 11. Concurrency Bulkhead Limiter (503)
@pytest.mark.asyncio
async def test_bulkhead_limiter_503(test_settings: Settings) -> None:
    # Max concurrency is set to 2
    strict_settings = Settings(
        max_concurrency=2,
        default_timeout=2.0,
    )
    slow_detector = SlowDetector(delay=0.3)
    app = create_app(settings=strict_settings, detectors=[slow_detector])
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Launch 5 concurrent requests
        tasks = [
            ac.post("/v1/scan", json={"text": f"concurrency-test-{i}"})
            for i in range(5)
        ]
        responses = await asyncio.gather(*tasks)

        status_codes = [r.status_code for r in responses]
        assert 503 in status_codes
        overloaded = [r for r in responses if r.status_code == 503]
        for r in overloaded:
            assert r.headers.get("retry-after") == "1"
            data = r.json()
            assert data["error"] == "Overloaded"

        # Check that exempt paths (/health) still succeed
        health_resp = await ac.get("/health")
        assert health_resp.status_code == 200


# 12. Global Exception Handlers (Never Leak Stack Traces)
@pytest.mark.asyncio
async def test_unhandled_exception_sanitized_no_stack_trace(
    test_settings: Settings, mock_detectors: list[Detector]
) -> None:
    app = create_app(settings=test_settings, detectors=mock_detectors)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # The test route raises RuntimeError("Database password leaked in stack trace: supersecret")
        resp = await ac.get("/_test/raise-unhandled")
        assert resp.status_code == 500
        data = resp.json()
        assert data["detail"] == "An internal server error occurred."
        assert data["error"] == "InternalServerError"
        # Crucial check: NEVER leak stack trace or internal message to client
        assert "supersecret" not in resp.text
        assert "traceback" not in resp.text.lower()
        assert "RuntimeError" not in resp.text


@pytest.mark.asyncio
async def test_not_found_sanitized(client: AsyncClient) -> None:
    resp = await client.get("/non_existent_endpoint")
    assert resp.status_code == 404
    data = resp.json()
    assert data["detail"] == "Not Found"
    assert "x-request-id" in resp.headers


# 13. Detector Timeout and Error Isolation
@pytest.mark.asyncio
async def test_detector_timeout_isolated(test_settings: Settings) -> None:
    detectors = [
        MockDetector(name="fast-d", attack_on="attack"),
        SlowDetector(name="slow-d", delay=0.5),
    ]
    # default_timeout is 0.1s in settings
    quick_settings = Settings(default_timeout=0.1)
    app = create_app(settings=quick_settings, detectors=detectors)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/scan", json={"text": "hello test"})
        assert resp.status_code == 200
        scan = ScanResponse.model_validate(resp.json())
        slow_res = next(r for r in scan.results if r.name == "slow-d")
        assert slow_res.is_attack is None
        assert "Timeout" in (slow_res.error or "")


@pytest.mark.asyncio
async def test_detector_exception_isolated(test_settings: Settings) -> None:
    detectors = [
        MockDetector(name="fast-d", attack_on="attack"),
        ErrorDetector(name="err-d"),
    ]
    app = create_app(settings=test_settings, detectors=detectors)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/scan", json={"text": "hello test"})
        assert resp.status_code == 200
        scan = ScanResponse.model_validate(resp.json())
        err_res = next(r for r in scan.results if r.name == "err-d")
        assert err_res.is_attack is None
        assert "Execution error" in (err_res.error or "")
