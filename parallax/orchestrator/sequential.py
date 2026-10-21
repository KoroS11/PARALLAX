"""Sequential execution orchestrator for latency baseline comparison (Claim C2)."""

import asyncio
import time
from collections.abc import Sequence

from parallax.contracts.api import ScanMode, ScanResponse
from parallax.contracts.detectors import Detector, DetectorResult
from parallax.orchestrator.fusion import AnyFlagFusion, FusionStrategy
from parallax.orchestrator.parallel import to_result_view


class SequentialOrchestrator:
    """Sequential pipeline executing detectors one by one to provide a baseline for Claim C2."""

    def __init__(
        self,
        detectors: Sequence[Detector],
        fusion: FusionStrategy | None = None,
        default_timeout: float = 2.0,
        per_detector_timeouts: dict[str, float] | None = None,
    ) -> None:
        self.detectors = list(detectors)
        self.fusion = fusion or AnyFlagFusion()
        self.default_timeout = default_timeout
        self.per_detector_timeouts = per_detector_timeouts or {}

    def filter_detectors(
        self,
        detectors: Sequence[Detector],
        mode: ScanMode,
    ) -> list[Detector]:
        """Filter detectors according to ScanMode."""
        if mode == ScanMode.FAST:
            return [
                d
                for d in detectors
                if "judge" not in d.name.lower() and "d5" not in d.name.lower()
            ]
        return list(detectors)

    async def _run_detector(self, detector: Detector, text: str) -> DetectorResult:
        """Run a single detector with timeout and error isolation."""
        t0 = time.perf_counter()
        timeout = self.per_detector_timeouts.get(detector.name, self.default_timeout)

        try:
            result = await asyncio.wait_for(detector.predict(text), timeout=timeout)
            if result.latency_ms <= 0.0:
                result.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return result
        except TimeoutError:
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            return DetectorResult(
                name=detector.name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=elapsed,
                error=f"Timeout after {timeout:.2f}s",
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            return DetectorResult(
                name=detector.name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=elapsed,
                error=f"Execution error: {exc}",
            )

    async def scan(
        self,
        text: str,
        mode: ScanMode = ScanMode.BENCH_SEQ,
        trace_id: str | None = None,
    ) -> ScanResponse:
        """Run sequential scan across all active detectors."""
        t_start = time.perf_counter()

        active_detectors = self.filter_detectors(self.detectors, mode)
        results: list[DetectorResult] = []

        # Execute strictly sequentially
        for detector in active_detectors:
            res = await self._run_detector(detector, text)
            results.append(res)

        fusion_res = self.fusion.fuse(results, prompt=text)
        total_latency = round((time.perf_counter() - t_start) * 1000.0, 2)

        return ScanResponse(
            decision=fusion_res.decision,
            final_technique=fusion_res.final_technique,
            disagreement=fusion_res.disagreement,
            confidence=fusion_res.confidence,
            results=[to_result_view(r) for r in results],
            latency_ms=total_latency,
            cached=False,
            trace_id=trace_id,
        )
