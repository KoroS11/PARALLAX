"""Guard client abstractions for the PARALLAX sandbox.

Supports wrapping ParallelOrchestrator in-process or via the FastAPI HTTP gateway.
Provides mock implementations for offline unit tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import httpx

from parallax.contracts.api import (
    Decision,
    DetectorResultView,
    ScanMode,
    ScanResponse,
)
from parallax.orchestrator.parallel import ParallelOrchestrator


@runtime_checkable
class GuardClient(Protocol):
    """Protocol implemented by both in-process and remote HTTP guard wrappers."""

    async def scan(
        self,
        text: str,
        mode: ScanMode = ScanMode.FAST,
        trace_id: str | None = None,
    ) -> ScanResponse:
        """Scan input prompt text and return a ScanResponse verdict."""
        ...


class InProcessGuardClient:
    """Wraps a ParallelOrchestrator instance directly in-process (zero network I/O)."""

    def __init__(self, orchestrator: ParallelOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def scan(
        self,
        text: str,
        mode: ScanMode = ScanMode.FAST,
        trace_id: str | None = None,
    ) -> ScanResponse:
        """Forward scan request directly to the in-process orchestrator."""
        return await self.orchestrator.scan(text=text, mode=mode, trace_id=trace_id)


class HttpGuardClient:
    """Wraps the PARALLAX FastAPI gateway via HTTP REST API calls."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._external_client = client

    async def scan(
        self,
        text: str,
        mode: ScanMode = ScanMode.FAST,
        trace_id: str | None = None,
    ) -> ScanResponse:
        """Send POST /v1/scan to the PARALLAX API."""
        payload: dict[str, Any] = {
            "text": text,
            "mode": mode.value if hasattr(mode, "value") else str(mode),
        }
        if trace_id:
            payload["trace_id"] = trace_id

        if self._external_client is not None:
            resp = await self._external_client.post(
                f"{self.base_url}/v1/scan",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return ScanResponse.model_validate(resp.json())

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/scan",
                json=payload,
            )
            resp.raise_for_status()
            return ScanResponse.model_validate(resp.json())


class MockGuardClient:
    """Mock guard client for deterministic offline testing.

    Can be configured with a fixed decision or a custom inspection rule callback.
    """

    def __init__(
        self,
        default_decision: Decision = Decision.ALLOW,
        rule_fn: Callable[[str], Decision] | None = None,
        block_keywords: tuple[str, ...] | None = None,
    ) -> None:
        self.default_decision = default_decision
        self.rule_fn = rule_fn
        self.block_keywords = (
            block_keywords
            if block_keywords is not None
            else (
                "system override",
                "emergency drill",
                "disregard all",
                "disable_failsafe",
            )
        )
        self.call_history: list[str] = []

    async def scan(
        self,
        text: str,
        mode: ScanMode = ScanMode.FAST,
        trace_id: str | None = None,
    ) -> ScanResponse:
        """Simulate scanning text with configurable decision rules."""
        self.call_history.append(text)

        if self.rule_fn is not None:
            decision = self.rule_fn(text)
        elif any(kw in text.lower() for kw in self.block_keywords):
            decision = Decision.BLOCK
        else:
            decision = self.default_decision

        results = [
            DetectorResultView(
                name="mock_guard_d1",
                is_attack=decision == Decision.BLOCK,
                score=0.95 if decision == Decision.BLOCK else 0.05,
                attack_class="keyword_override" if decision == Decision.BLOCK else None,
                latency_ms=1.5,
                error=None,
            )
        ]

        return ScanResponse(
            decision=decision,
            final_technique="keyword_override" if decision == Decision.BLOCK else None,
            disagreement=False,
            confidence=0.95 if decision == Decision.BLOCK else 0.05,
            results=results,
            latency_ms=2.0,
            cached=False,
            trace_id=trace_id,
        )
