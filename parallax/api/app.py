"""FastAPI application factory and API endpoints."""

import logging
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from parallax.api.metrics import ApiMetrics
from parallax.api.middlewares import (
    AccessLogMiddleware,
    BulkheadMiddleware,
    GlobalExceptionMiddleware,
    PayloadSizeLimitMiddleware,
    RequestIdMiddleware,
)
from parallax.api.settings import Settings, get_settings
from parallax.api.storage import ApiStorage
from parallax.contracts.api import (
    ChatRequest,
    ChatResponse,
    Decision,
    EvalRunResponse,
    HealthResponse,
    MetricsSummaryResponse,
    ScanMode,
    ScanRequest,
    ScanResponse,
    SequentialScanRequest,
)
from parallax.contracts.detectors import Detector
from parallax.dashboard.app import dashboard_router
from parallax.orchestrator.cache import LRUCache, ScanCache
from parallax.orchestrator.fusion import AnyFlagFusion
from parallax.orchestrator.parallel import ParallelOrchestrator
from parallax.orchestrator.sequential import SequentialOrchestrator

logger = logging.getLogger("parallax.api")


def create_app(
    settings: Settings | None = None,
    detectors: Sequence[Detector] | None = None,
    orchestrator: ParallelOrchestrator | None = None,
    sequential_orchestrator: SequentialOrchestrator | None = None,
    cache: ScanCache | None = None,
    storage: Any | None = None,
    victim_client: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application serving plane.

    Supports full dependency injection for testing and flexible deployment.
    """
    app_settings = settings or get_settings()

    # Configure cache
    if cache is not None:
        scan_cache = cache
    elif app_settings.cache_enabled:
        scan_cache = LRUCache(max_size=app_settings.cache_capacity)
    else:
        scan_cache = None

    # Configure storage
    api_storage = storage or ApiStorage(database_url=app_settings.database_url)

    # Configure detectors and orchestrators
    active_detectors = list(detectors) if detectors is not None else []

    parallel_orch = orchestrator
    if parallel_orch is None and detectors is not None:
        parallel_orch = ParallelOrchestrator(
            detectors=active_detectors,
            fusion=AnyFlagFusion(),
            cache=scan_cache,
            default_timeout=app_settings.default_timeout,
        )

    seq_orch = sequential_orchestrator
    if seq_orch is None and detectors is not None:
        seq_orch = SequentialOrchestrator(
            detectors=active_detectors,
            fusion=AnyFlagFusion(),
            default_timeout=app_settings.default_timeout,
        )

    metrics_collector = ApiMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup phase
        logger.info("Initializing PARALLAX API serving plane...")
        yield
        # Shutdown phase
        logger.info("Shutting down PARALLAX API serving plane...")

    app = FastAPI(
        title="PARALLAX API",
        description="Parallel Adversarial-prompt Recognition and Layered Aggregation for LLM eXploits",
        version=app_settings.api_version,
        lifespan=lifespan,
    )

    # Attach application state
    app.state.settings = app_settings
    app.state.detectors = active_detectors
    app.state.orchestrator = parallel_orch
    app.state.sequential_orchestrator = seq_orch
    app.state.cache = scan_cache
    app.state.storage = api_storage
    app.state.metrics = metrics_collector
    app.state.victim_client = victim_client

    # Register Middlewares
    app.add_middleware(
        PayloadSizeLimitMiddleware,
        max_input_size=app_settings.max_input_size,
    )
    app.add_middleware(
        BulkheadMiddleware,
        max_concurrency=app_settings.max_concurrency,
        exempt_paths={"/health", "/ready", "/metrics"},
    )
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(GlobalExceptionMiddleware, metrics=metrics_collector)
    app.add_middleware(RequestIdMiddleware)

    # Global Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": exc.errors(),
                "error": "ValidationError",
                "request_id": req_id,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        headers = dict(exc.headers or {})
        return JSONResponse(
            status_code=exc.status_code,
            headers=headers,
            content={
                "detail": exc.detail,
                "error": exc.__class__.__name__,
                "request_id": req_id,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        app.state.metrics.record_error()
        logger.exception("Unhandled server error processing request: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred.",
                "error": "InternalServerError",
                "request_id": req_id,
            },
        )

    # API Endpoints
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Liveness probe verifying that the API service is active."""
        detectors_dict = {d.name: True for d in app.state.detectors}
        return HealthResponse(
            status="ok",
            detectors=detectors_dict,
            version=app.state.settings.api_version,
        )

    @app.get("/ready", response_model=HealthResponse)
    async def readiness_check() -> HealthResponse | JSONResponse:
        """Readiness probe verifying that detectors are loaded and operational."""
        if not app.state.detectors or app.state.orchestrator is None:
            resp = HealthResponse(
                status="degraded",
                detectors={},
                version=app.state.settings.api_version,
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=resp.model_dump(),
            )

        detectors_dict = {d.name: True for d in app.state.detectors}
        return HealthResponse(
            status="ok",
            detectors=detectors_dict,
            version=app.state.settings.api_version,
        )

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        """Prometheus text-format metrics exporter."""
        metrics_text = app.state.metrics.render_prometheus()
        return Response(
            content=metrics_text,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/v1/scan", response_model=ScanResponse)
    async def scan_prompt(request: Request, body: ScanRequest) -> ScanResponse:
        """Perform fast or deep parallel multi-model scan of input prompt."""
        orch = app.state.orchestrator
        if orch is None:
            raise StarletteHTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Detection orchestrator not initialized",
            )

        trace_id = body.trace_id or getattr(request.state, "request_id", None)
        scan_result = await orch.scan(
            text=body.text,
            mode=body.mode,
            trace_id=trace_id,
        )

        app.state.metrics.record_scan(scan_result)
        app.state.storage.record_scan(scan_result, body.text)
        return cast(ScanResponse, scan_result)

    @app.post("/v1/scan/sequential", response_model=ScanResponse)
    async def scan_sequential(
        request: Request, body: SequentialScanRequest
    ) -> ScanResponse:
        """Execute sequential scan for baseline latency comparison (Claim C2)."""
        seq_orch = app.state.sequential_orchestrator
        if seq_orch is None:
            raise StarletteHTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Sequential orchestrator not initialized",
            )

        trace_id = body.trace_id or getattr(request.state, "request_id", None)
        scan_result = await seq_orch.scan(
            text=body.text,
            mode=ScanMode.BENCH_SEQ,
            trace_id=trace_id,
        )

        app.state.metrics.record_scan(scan_result)
        app.state.storage.record_scan(scan_result, body.text)
        return cast(ScanResponse, scan_result)

    @app.post("/v1/chat", response_model=ChatResponse)
    async def guarded_chat(request: Request, body: ChatRequest) -> ChatResponse:
        """Guarded chat: scan prompt, forwarding to victim LLM only if allowed."""
        orch = app.state.orchestrator
        if orch is None:
            raise StarletteHTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Detection orchestrator not initialized",
            )

        trace_id = getattr(request.state, "request_id", None)
        scan_result = await orch.scan(
            text=body.prompt,
            mode=body.mode,
            trace_id=trace_id,
        )

        app.state.metrics.record_scan(scan_result)
        app.state.storage.record_scan(scan_result, body.prompt)

        # Check guard decision
        if scan_result.decision == Decision.BLOCK:
            return ChatResponse(
                allowed=False,
                decision=Decision.BLOCK,
                scan=scan_result,
                reply="I cannot fulfill this request because it violates safety guidelines.",
                victim_model=None,
            )

        if (
            scan_result.decision == Decision.FLAG
            and not app.state.settings.allow_flagged
        ):
            return ChatResponse(
                allowed=False,
                decision=Decision.FLAG,
                scan=scan_result,
                reply="This prompt was flagged by safety guardrails and cannot be forwarded.",
                victim_model=None,
            )

        # Allowed: forward to victim LLM
        reply_text = None
        victim_model = body.victim_model

        if app.state.victim_client is not None:
            # Use injected victim client
            try:
                reply_text = await app.state.victim_client.generate(
                    model=victim_model, prompt=body.prompt
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Injected victim client call failed: %s", exc)
                reply_text = "Victim model response unavailable."
        else:
            # Forward to Ollama service
            try:
                async with httpx.AsyncClient(
                    timeout=app.state.settings.victim_timeout
                ) as client:
                    resp = await client.post(
                        f"{app.state.settings.ollama_url}/api/generate",
                        json={
                            "model": victim_model,
                            "prompt": body.prompt,
                            "stream": False,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    reply_text = data.get("response", "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Victim model Ollama call failed: %s", exc)
                reply_text = "Victim model service currently unavailable."

        return ChatResponse(
            allowed=True,
            decision=scan_result.decision,
            scan=scan_result,
            reply=reply_text,
            victim_model=victim_model,
        )

    @app.get("/v1/runs", response_model=list[EvalRunResponse])
    async def list_eval_runs() -> list[EvalRunResponse]:
        """Storage inspection: list evaluation runs for dashboard."""
        return cast(list[EvalRunResponse], app.state.storage.get_eval_runs())

    @app.get("/v1/metrics", response_model=MetricsSummaryResponse)
    async def get_metrics_summary() -> MetricsSummaryResponse:
        """Storage inspection: aggregated scan metrics and latency percentiles."""
        return cast(MetricsSummaryResponse, app.state.metrics.get_metrics_summary())

    # Mount Terminal SPA Dashboard
    app.include_router(dashboard_router)

    @app.get("/", include_in_schema=False)
    async def root_dashboard_redirect() -> Response:
        """Redirect root path to interactive dark terminal dashboard."""
        return RedirectResponse(
            url="/dashboard", status_code=status.HTTP_307_TEMPORARY_REDIRECT
        )

    # Route for testing unhandled exception handling
    if app_settings.environment != "production":

        @app.get("/_test/raise-unhandled")
        async def trigger_test_unhandled_error() -> None:
            raise RuntimeError("Database password leaked in stack trace: supersecret")

    return app
