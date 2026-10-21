"""API request and response schemas contract."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Decision(str, Enum):
    """Decision produced by the fusion layer."""

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


class ScanMode(str, Enum):
    """Execution mode for detector orchestration."""

    FAST = "fast"
    DEEP = "deep"
    BENCH_SEQ = "bench-seq"


class DetectorResultView(BaseModel):
    """Pydantic view of a single detector's output."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Detector identifier")
    is_attack: bool | None = Field(
        None, description="True if attack, False if benign, None if abstained"
    )
    score: float | None = Field(
        None, description="Continuous score in [0.0, 1.0] if available"
    )
    attack_class: str | None = Field(
        None, description="Predicted technique class if available"
    )
    latency_ms: float = Field(..., description="Execution latency in milliseconds")
    error: str | None = Field(None, description="Error message if failed or timed out")


class ScanRequest(BaseModel):
    """Payload for POST /v1/scan."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="Prompt text to scan")
    mode: ScanMode = Field(
        default=ScanMode.FAST,
        description="Scanning mode: fast (D1-D4), deep (+D5), or bench-seq",
    )
    trace_id: str | None = Field(
        default=None, description="Optional request tracking identifier"
    )


class ScanResponse(BaseModel):
    """Response payload for POST /v1/scan and /v1/scan/sequential."""

    model_config = ConfigDict(extra="forbid")

    decision: Decision = Field(..., description="Aggregated fusion verdict")
    final_technique: str | None = Field(
        default=None,
        description="Multi-class technique attribution (from D1 or D5 when flagged)",
    )
    disagreement: bool = Field(
        default=False,
        description="True if individual detector predictions split/disagree",
    )
    confidence: float | None = Field(
        default=None, description="Aggregated confidence or probability"
    )
    results: list[DetectorResultView] = Field(
        default_factory=list, description="Per-detector prediction results"
    )
    latency_ms: float = Field(
        ..., description="End-to-end scan latency in milliseconds"
    )
    cached: bool = Field(
        default=False, description="True if result was served from cache"
    )
    trace_id: str | None = Field(default=None, description="Request tracking identifier")


class SequentialScanRequest(BaseModel):
    """Payload for POST /v1/scan/sequential (benchmark path)."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="Prompt text to scan sequentially")
    trace_id: str | None = Field(default=None, description="Optional trace identifier")


class ChatRequest(BaseModel):
    """Payload for POST /v1/chat (guarded chat forwarding)."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, description="User prompt to forward")
    victim_model: str = Field(
        default="gemma3:4b", description="Target victim LLM in Ollama"
    )
    mode: ScanMode = Field(
        default=ScanMode.FAST, description="Scan mode prior to forwarding"
    )
    session_id: str | None = Field(
        default=None, description="Optional conversation session ID"
    )


class ChatResponse(BaseModel):
    """Response payload for POST /v1/chat."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool = Field(
        ..., description="True if scan passed and prompt reached victim LLM"
    )
    decision: Decision = Field(..., description="Scan decision")
    scan: ScanResponse = Field(..., description="Detailed scan verdict and detector outputs")
    reply: str | None = Field(
        default=None, description="Victim LLM response if allowed, or refusal message"
    )
    victim_model: str | None = Field(
        default=None, description="Victim model that generated reply"
    )


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Overall service status ('ok', 'degraded', 'error')")
    detectors: dict[str, bool] = Field(
        default_factory=dict, description="Readiness status of each detector"
    )
    version: str = Field(default="0.1.0", description="Service version")


class EvalRunResponse(BaseModel):
    """Response payload for GET /v1/runs."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="Unique evaluation run identifier")
    experiment_id: str = Field(..., description="Experiment ID (e.g. 'E1', 'E6')")
    status: str = Field(..., description="Run status ('completed', 'failed', 'running')")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Aggregated metric values"
    )
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class MetricsSummaryResponse(BaseModel):
    """Response payload for GET /v1/metrics."""

    model_config = ConfigDict(extra="forbid")

    total_scans: int = Field(default=0, description="Total scanned prompts")
    allow_count: int = Field(default=0, description="Total allowed requests")
    flag_count: int = Field(default=0, description="Total flagged requests")
    block_count: int = Field(default=0, description="Total blocked requests")
    disagreement_rate: float = Field(
        default=0.0, description="Fraction of scans with detector disagreement"
    )
    p50_latency_ms: float = Field(
        default=0.0, description="Median end-to-end scan latency"
    )
    p95_latency_ms: float = Field(
        default=0.0, description="95th percentile scan latency"
    )
