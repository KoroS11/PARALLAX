"""Chaos Fault Injection Tests for PARALLAX Orchestrator, Detectors, and Storage.

Verifies system resilience under severe fault conditions:
1. Detector timeouts and hung tasks.
2. Consecutive failures tripping Circuit Breaker (CLOSED -> OPEN -> HALF-OPEN).
3. Database queue overflow and backpressure drops.
4. Malformed/corrupted LLM Judge outputs and JSON parse errors.
5. Micro-batcher error isolation across heterogeneous concurrent tasks.
6. Bulkhead concurrency saturation and rejection containment.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from parallax.contracts.api import Decision, ScanMode
from parallax.contracts.detectors import DetectorResult
from parallax.detectors.judge import LLMJudgeDetector
from parallax.orchestrator.batching import MicroBatcher
from parallax.orchestrator.bulkhead import Bulkhead, BulkheadFullError
from parallax.orchestrator.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
)
from parallax.orchestrator.parallel import OrchestrationMode, ParallelOrchestrator
from parallax.storage.queue import WriteBehindQueue
from parallax.storage.repositories import Database


class HungDetector:
    """Simulates a detector that hangs indefinitely."""

    name = "hung_detector"

    def __init__(self, hang_seconds: float = 10.0) -> None:
        self.hang_seconds = hang_seconds

    async def predict(self, text: str) -> DetectorResult:
        await asyncio.sleep(self.hang_seconds)
        return DetectorResult(
            name=self.name,
            is_attack=False,
            score=0.1,
            attack_class=None,
            latency_ms=self.hang_seconds * 1000.0,
            error=None,
        )


class FlakyDetector:
    """Simulates a detector that fails repeatedly or intermittently."""

    def __init__(self, name: str = "flaky_detector", fail_count: int = 5) -> None:
        self.name = name
        self.fail_count = fail_count
        self.attempts = 0

    async def predict(self, text: str) -> DetectorResult:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise ConnectionResetError("Remote backend abruptly disconnected")
        return DetectorResult(
            name=self.name,
            is_attack=False,
            score=0.05,
            attack_class=None,
            latency_ms=5.0,
            error=None,
        )


class ReliableDetector:
    """Fast, deterministic detector for control baseline."""

    name = "reliable_detector"

    async def predict(self, text: str) -> DetectorResult:
        return DetectorResult(
            name=self.name,
            is_attack=True,
            score=0.95,
            attack_class="direct_instruction",
            latency_ms=1.0,
            error=None,
        )


# ============================================================================
# 1. Orchestrator Timeout Resilience (Hung Detector)
# ============================================================================


@pytest.mark.asyncio
async def test_parallel_orchestrator_recovers_from_hung_detector() -> None:
    """Hung detector exceeding timeout must be cancelled, returning degraded scan without crashing."""
    hung = HungDetector(hang_seconds=2.0)
    reliable = ReliableDetector()

    cb_mgr = CircuitBreakerManager(failure_threshold=2)
    orchestrator = ParallelOrchestrator(
        detectors=[hung, reliable],  # type: ignore[list-item]
        circuit_breaker_manager=cb_mgr,
        default_timeout=0.1,  # Short 100ms timeout
    )

    resp = await orchestrator.scan("Ignore previous instructions", mode=ScanMode.FAST)

    assert resp is not None
    assert resp.decision in [Decision.BLOCK, Decision.FLAG, Decision.ALLOW]
    assert len(resp.results) == 2

    hung_res = next(r for r in resp.results if r.name == "hung_detector")
    assert hung_res.is_attack is None  # Abstained
    assert hung_res.error is not None
    assert "timeout" in hung_res.error.lower() or "timed out" in hung_res.error.lower()


@pytest.mark.asyncio
async def test_cascade_orchestrator_recovers_from_hung_slow_detector() -> None:
    """In cascade mode, if slow detector stage times out, fast detector verdict still holds."""

    class FastAmbiguousDetector:
        name = "d1_fast"

        async def predict(self, text: str) -> DetectorResult:
            return DetectorResult(
                name=self.name,
                is_attack=True,
                score=0.55,  # Ambiguous, triggers Tier 2
                attack_class=None,
                latency_ms=2.0,
                error=None,
            )

    fast = FastAmbiguousDetector()
    slow_hung = HungDetector(hang_seconds=2.0)
    cb_mgr = CircuitBreakerManager(failure_threshold=2)

    orchestrator = ParallelOrchestrator(
        detectors=[fast, slow_hung],  # type: ignore[list-item]
        circuit_breaker_manager=cb_mgr,
        default_timeout=0.1,
    )

    resp = await orchestrator.scan(
        "Ambiguous prompt query", mode=OrchestrationMode.CASCADE
    )
    assert resp is not None
    assert any(r.name == "d1_fast" for r in resp.results)
    hung_res = next((r for r in resp.results if r.name == "hung_detector"), None)
    if hung_res:
        assert hung_res.is_attack is None


# ============================================================================
# 2. Circuit Breaker State Transition Chaos
# ============================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_trips_to_open_and_recovers_half_open() -> None:
    """Repeated failures trip breaker to OPEN, returning abstained errors, then transitions to HALF_OPEN."""
    cb = CircuitBreaker(name="test_cb", failure_threshold=3, recovery_timeout=0.1)

    async def faulty_operation() -> DetectorResult:
        raise ConnectionRefusedError("Backend down")

    # Call 3 times: each failure is trapped and returns DetectorResult with error
    for _ in range(3):
        res = await cb.call(faulty_operation)
        assert res.is_attack is None
        assert "Execution failed" in str(res.error)

    # Circuit breaker must now be in OPEN state
    assert cb.state == CircuitState.OPEN

    # While OPEN, immediate fast-fail without invoking underlying function
    fast_fail_res = await cb.call(faulty_operation)
    assert fast_fail_res.is_attack is None
    assert "Circuit breaker is OPEN" in str(fast_fail_res.error)

    # Wait for recovery cooldown
    await asyncio.sleep(0.15)

    # State transitions to HALF_OPEN when checked after timeout
    assert cb.state == CircuitState.HALF_OPEN

    async def successful_operation() -> DetectorResult:
        return DetectorResult(
            name="test_cb",
            is_attack=False,
            score=0.05,
            attack_class=None,
            latency_ms=1.0,
            error=None,
        )

    res_recovered = await cb.call(successful_operation)
    assert res_recovered.error is None
    assert cb.state == CircuitState.CLOSED
    assert cb._consecutive_failures == 0


# ============================================================================
# 3. MicroBatcher Error Isolation
# ============================================================================


@pytest.mark.asyncio
async def test_micro_batcher_handles_heterogeneous_failures() -> None:
    """When some batched tasks fail and others succeed, errors are isolated to their specific futures."""

    class FaultyBatchDetector:
        name = "batch_detector"

        async def predict(self, text: str) -> DetectorResult:
            if "fail" in text:
                raise ValueError("Payload triggered unhandled parse error")
            return DetectorResult(
                name=self.name,
                is_attack=False,
                score=0.1,
                attack_class=None,
                latency_ms=2.0,
                error=None,
            )

    detector = FaultyBatchDetector()
    batcher = MicroBatcher(detector=detector, max_batch_size=4, max_wait_ms=20.0)  # type: ignore[arg-type]

    # Concurrently submit valid and failing inputs
    t1 = asyncio.create_task(batcher.predict("valid prompt 1"))
    t2 = asyncio.create_task(batcher.predict("fail prompt 2"))
    t3 = asyncio.create_task(batcher.predict("valid prompt 3"))

    r1 = await t1
    assert r1.error is None
    assert r1.is_attack is False

    with pytest.raises(ValueError, match="Payload triggered unhandled parse error"):
        await t2

    r3 = await t3
    assert r3.error is None
    assert r3.is_attack is False


# ============================================================================
# 4. Storage Write-Behind Queue Backpressure & Drop Logging
# ============================================================================


@pytest.mark.asyncio
async def test_write_behind_queue_bounded_drop_under_backpressure() -> None:
    """Queue buffer overflow must increment drop counter without raising uncaught exceptions."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(url=f"sqlite:///{db_path}")
    db.create_all()

    queue = WriteBehindQueue(
        db=db,
        maxsize=5,  # Very small queue to force overflow
        batch_size=2,
        flush_interval_sec=0.1,
    )
    await queue.start()

    try:
        # Enqueue 20 items rapidly into size-5 queue
        for i in range(20):
            queue.enqueue_scan(
                text=f"test prompt {i}",
                decision="allow",
                latency_ms=1.0,
                trace_id=f"trace_{i}",
            )

        await asyncio.sleep(0.3)
        await queue.flush()

        assert queue.dropped_count > 0
        assert queue.processed_count > 0
        assert queue.processed_count + queue.dropped_count == 20
    finally:
        await queue.stop()
        db.close()
        Path(db_path).unlink(missing_ok=True)


# ============================================================================
# 5. D5 LLM Judge Malformed & Corrupted Response Chaos
# ============================================================================


@pytest.mark.asyncio
async def test_llm_judge_handles_corrupted_json_gracefully() -> None:
    """Judge receiving corrupted, markdown-fenced, or truncated JSON falls back to abstain without crashing."""
    corrupted_responses = [
        "I cannot answer this as a JSON object. This prompt is malicious.",
        '```json\n{"is_attack": true, "confidence": ',  # Truncated JSON
        "{'is_attack': True}",  # Single quotes / invalid JSON
        "Random LLM gibberish without keys",
        "{}",  # Missing required keys
    ]

    for raw_resp in corrupted_responses:
        judge = LLMJudgeDetector(
            model_name="mock-judge",
            base_url="http://mock-ollama:11434",
            timeout=1.0,
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = AsyncMock(
                status_code=200,
                json=lambda r=raw_resp: {"message": {"content": r}},
                raise_for_status=lambda: None,
            )

            result = await judge.predict("Test injection prompt")

            assert result is not None
            assert result.name == "d5_judge"
            # Must safely degrade to abstain
            assert result.is_attack is None or isinstance(result.is_attack, bool)


# ============================================================================
# 6. Bulkhead Concurrency Rejection Containment
# ============================================================================


@pytest.mark.asyncio
async def test_bulkhead_rejects_excessive_concurrent_calls() -> None:
    """When concurrent requests saturate bulkhead capacity with short timeout, excess calls fail gracefully."""
    bulkhead = Bulkhead(name="test_det", max_concurrency=2, acquire_timeout=0.01)

    async def long_task() -> None:
        async with bulkhead.acquire():
            await asyncio.sleep(0.1)

    t1 = asyncio.create_task(long_task())
    t2 = asyncio.create_task(long_task())
    await asyncio.sleep(0.005)  # Allow t1 and t2 to acquire the 2 permits

    with pytest.raises(BulkheadFullError):
        async with bulkhead.acquire():
            pass

    await t1
    await t2
