"""Parallel fan-out orchestrator with robust serving, cascade modes, and error isolation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from enum import Enum

from pydantic import ConfigDict, Field

from parallax.contracts.api import Decision, DetectorResultView, ScanMode, ScanResponse
from parallax.contracts.detectors import Detector, DetectorResult
from parallax.orchestrator.batching import MicroBatcher
from parallax.orchestrator.bulkhead import BulkheadManager
from parallax.orchestrator.cache import ScanCache
from parallax.orchestrator.cascade import CascadeConfig, CascadeRunner
from parallax.orchestrator.circuit_breaker import CircuitBreakerManager
from parallax.orchestrator.fusion import AnyFlagFusion, FusionStrategy
from parallax.preprocess.pipeline import preprocess


class OrchestrationMode(str, Enum):
    """Execution orchestration modes."""

    PARALLEL_FULL = "parallel_full"
    CASCADE = "cascade"
    SEQUENTIAL = "sequential"


class RobustScanResponse(ScanResponse):
    """Extended scan response model supporting abstain-aware degraded flag and triggered view."""

    model_config = ConfigDict(extra="ignore")

    degraded: bool = Field(
        default=False,
        description="True if one or more detectors abstained or encountered errors",
    )
    triggered_view: str | None = Field(
        default=None,
        description="Preprocessed text view (e.g. 'raw', 'decoded') yielding highest attack score",
    )
    view_details: dict[str, str] | None = Field(
        default=None,
        description="Mapping of detector names to the specific view triggering its verdict",
    )


def to_result_view(result: DetectorResult) -> DetectorResultView:
    """Convert a DetectorResult dataclass into a DetectorResultView Pydantic model."""
    return DetectorResultView(
        name=result.name,
        is_attack=result.is_attack,
        score=result.score,
        attack_class=result.attack_class,
        latency_ms=result.latency_ms,
        error=result.error,
    )


class ParallelOrchestrator:
    """Async multi-model orchestrator with circuit breakers, bulkheads, cascade, and view evaluation."""

    def __init__(
        self,
        detectors: Sequence[Detector],
        fusion: FusionStrategy | None = None,
        cache: ScanCache | None = None,
        default_timeout: float = 2.0,
        per_detector_timeouts: dict[str, float] | None = None,
        orchestration_mode: OrchestrationMode | str = OrchestrationMode.PARALLEL_FULL,
        cascade_config: CascadeConfig | None = None,
        circuit_breaker_manager: CircuitBreakerManager | None = None,
        bulkhead_manager: BulkheadManager | None = None,
        enable_circuit_breakers: bool = True,
        enable_bulkheads: bool = True,
        enable_batching: bool = False,
        batchers: dict[str, MicroBatcher] | None = None,
        evaluate_views: bool = False,
        fail_closed: bool = False,
        min_successful_detectors: int = 1,
        fail_closed_decision: Decision = Decision.FLAG,
    ) -> None:
        self.detectors = list(detectors)
        self.fusion = fusion or AnyFlagFusion(
            fail_closed=fail_closed,
            min_successful_detectors=min_successful_detectors,
            fail_closed_decision=fail_closed_decision,
        )
        self.cache = cache
        self.default_timeout = default_timeout
        self.per_detector_timeouts = per_detector_timeouts or {}

        # Orchestration Mode
        if isinstance(orchestration_mode, str):
            try:
                self.orchestration_mode = OrchestrationMode(orchestration_mode.lower())
            except ValueError:
                self.orchestration_mode = OrchestrationMode.PARALLEL_FULL
        else:
            self.orchestration_mode = orchestration_mode

        # Cascade runner
        self.cascade_runner = CascadeRunner(config=cascade_config)

        # Robustness controls
        self.enable_circuit_breakers = enable_circuit_breakers
        self.circuit_breakers = circuit_breaker_manager or CircuitBreakerManager()

        self.enable_bulkheads = enable_bulkheads
        self.bulkhead_manager = bulkhead_manager or BulkheadManager()

        self.enable_batching = enable_batching
        self.batchers: dict[str, MicroBatcher] = batchers or {}

        # Views & Fail policies
        self.evaluate_views = evaluate_views
        self.fail_closed = fail_closed
        self.min_successful_detectors = min_successful_detectors
        self.fail_closed_decision = fail_closed_decision

    def register_batcher(self, detector_name: str, batcher: MicroBatcher) -> None:
        """Register a micro-batcher queue for a specific detector."""
        self.batchers[detector_name] = batcher
        self.enable_batching = True

    def filter_detectors(
        self,
        detectors: Sequence[Detector],
        mode: ScanMode | OrchestrationMode | str,
    ) -> list[Detector]:
        """Filter detectors according to ScanMode or string mode."""
        mode_str = str(mode).lower()
        if "fast" in mode_str:
            return [
                d
                for d in detectors
                if "judge" not in d.name.lower() and "d5" not in d.name.lower()
            ]
        return list(detectors)

    async def _run_detector_raw(self, detector: Detector, text: str) -> DetectorResult:
        """Execute detector prediction with timeout and exception isolation."""
        t0 = time.perf_counter()
        timeout = self.per_detector_timeouts.get(detector.name, self.default_timeout)

        try:
            if self.enable_batching and detector.name in self.batchers:
                batcher = self.batchers[detector.name]
                result = await asyncio.wait_for(batcher.predict(text), timeout=timeout)
            else:
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

    async def _run_detector(self, detector: Detector, text: str) -> DetectorResult:
        """Execute detector protected by circuit breaker, bulkhead, and GPU semaphore."""
        # 1. Circuit breaker gate
        if self.enable_circuit_breakers:
            breaker = self.circuit_breakers.get_breaker(detector.name)
            if breaker.is_open:
                remaining = breaker.time_until_recovery()
                return DetectorResult(
                    name=detector.name,
                    is_attack=None,
                    score=None,
                    attack_class=None,
                    latency_ms=0.0,
                    error=(
                        f"Circuit breaker is OPEN for detector '{detector.name}'. "
                        f"Retry in {remaining:.2f}s."
                    ),
                )

        # 2. Concurrency Bulkhead & GPU lock
        async def _invoke() -> DetectorResult:
            if self.enable_bulkheads:
                async with self.bulkhead_manager.limit(detector.name):
                    return await self._run_detector_raw(detector, text)
            return await self._run_detector_raw(detector, text)

        result = await _invoke()

        # 3. Record outcome to circuit breaker
        if self.enable_circuit_breakers:
            breaker = self.circuit_breakers.get_breaker(detector.name)
            if result.error is not None or result.is_attack is None:
                breaker.record_failure()
            else:
                breaker.record_success()

        return result

    async def _evaluate_detector_views(
        self,
        detector: Detector,
        views: dict[str, str],
    ) -> DetectorResult:
        """Evaluate multiple preprocessed views concurrently and select highest scoring view."""
        if len(views) <= 1:
            only_name, only_text = next(iter(views.items()))
            res = await self._run_detector(detector, only_text)
            object.__setattr__(res, "highest_view", only_name)
            return res

        # Evaluate all views concurrently
        view_names = list(views.keys())
        tasks = [self._run_detector(detector, views[vn]) for vn in view_names]
        results = await asyncio.gather(*tasks)

        # Find view with highest attack score
        best_result = results[0]
        best_view = view_names[0]
        best_score = best_result.score if best_result.score is not None else -1.0

        for vn, res in zip(view_names, results):
            s = res.score if res.score is not None else -1.0
            if s > best_score:
                best_score = s
                best_result = res
                best_view = vn
            elif (
                s == best_score and res.is_attack is True and not best_result.is_attack
            ):
                best_result = res
                best_view = vn

        object.__setattr__(best_result, "highest_view", best_view)
        return best_result

    async def scan(
        self,
        text: str,
        mode: ScanMode | OrchestrationMode | str = ScanMode.FAST,
        trace_id: str | None = None,
        evaluate_views: bool | None = None,
    ) -> ScanResponse:
        """Execute scan across detectors using configured orchestration mode and safety guards."""
        t_start = time.perf_counter()

        # Resolve view evaluation setting
        should_eval_views = (
            evaluate_views if evaluate_views is not None else self.evaluate_views
        )

        # Preprocess views if requested
        if should_eval_views:
            prep_result = preprocess(text)
            views = prep_result.views
        else:
            views = {"raw": text}

        # Step 1: Cache check
        if self.cache is not None:
            cached_resp = self.cache.get(text)
            if cached_resp is not None:
                elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                return RobustScanResponse(
                    decision=cached_resp.decision,
                    final_technique=cached_resp.final_technique,
                    disagreement=cached_resp.disagreement,
                    confidence=cached_resp.confidence,
                    results=cached_resp.results,
                    latency_ms=elapsed_ms,
                    cached=True,
                    trace_id=trace_id or cached_resp.trace_id,
                    degraded=getattr(cached_resp, "degraded", False),
                    triggered_view=getattr(cached_resp, "triggered_view", None),
                )

        # Step 2: Filter active detectors
        active_detectors = self.filter_detectors(self.detectors, mode)

        # Step 3: Determine execution strategy
        mode_str = str(mode).lower()
        if (
            mode == ScanMode.BENCH_SEQ
            or mode_str == "sequential"
            or self.orchestration_mode == OrchestrationMode.SEQUENTIAL
        ):
            active_mode = OrchestrationMode.SEQUENTIAL
        elif (
            mode_str == "cascade"
            or self.orchestration_mode == OrchestrationMode.CASCADE
        ):
            active_mode = OrchestrationMode.CASCADE
        else:
            active_mode = OrchestrationMode.PARALLEL_FULL

        # Step 4: Execute according to mode
        results: list[DetectorResult] = []

        if active_mode == OrchestrationMode.SEQUENTIAL:
            for d in active_detectors:
                res = await self._evaluate_detector_views(d, views)
                results.append(res)

        elif active_mode == OrchestrationMode.CASCADE:
            # Multi-tier cascade execution
            async def _run_cascade_step(
                d: Detector, prompt_text: str
            ) -> DetectorResult:
                # If views enabled, evaluate views for this detector
                step_views = views if should_eval_views else {"raw": prompt_text}
                return await self._evaluate_detector_views(d, step_views)

            results, _ = await self.cascade_runner.execute(
                detectors=active_detectors,
                text=text,
                run_detector_fn=_run_cascade_step,
            )

        else:
            # PARALLEL_FULL fan-out
            tasks = [self._evaluate_detector_views(d, views) for d in active_detectors]
            if tasks:
                gathered = await asyncio.gather(*tasks)
                results = list(gathered)
            else:
                results = []

        # Step 5: Fusion aggregation
        fusion_res = self.fusion.fuse(results, prompt=text)

        # Step 6: Identify triggered views
        view_details: dict[str, str] = {}
        highest_score_view: str | None = None
        highest_score = -1.0

        for r in results:
            v_name = getattr(r, "highest_view", "raw")
            view_details[r.name] = v_name
            s = r.score if r.score is not None else -1.0
            if s > highest_score:
                highest_score = s
                highest_score_view = v_name

        # Determine degraded status
        is_degraded = getattr(fusion_res, "degraded", False) or any(
            r.is_attack is None or r.error is not None for r in results
        )

        # Step 7: Format response
        total_latency = round((time.perf_counter() - t_start) * 1000.0, 2)
        response = RobustScanResponse(
            decision=fusion_res.decision,
            final_technique=fusion_res.final_technique,
            disagreement=fusion_res.disagreement,
            confidence=fusion_res.confidence,
            results=[to_result_view(r) for r in results],
            latency_ms=total_latency,
            cached=False,
            trace_id=trace_id,
            degraded=is_degraded,
            triggered_view=highest_score_view,
            view_details=view_details,
        )

        # Also set attributes via object.__setattr__ to guarantee compatibility
        object.__setattr__(response, "degraded", is_degraded)
        object.__setattr__(response, "triggered_view", highest_score_view)
        object.__setattr__(response, "view_details", view_details)

        # Step 8: Populate cache
        if self.cache is not None:
            self.cache.set(text, response)

        return response
