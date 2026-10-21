"""Cascade orchestration mode: fast-tier early-exit and slow-tier task cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any

from parallax.contracts.detectors import Detector, DetectorResult

DEFAULT_FAST_DETECTOR_KEYWORDS = ("d1", "minilm", "d4", "rule")


@dataclass
class CascadeConfig:
    """Configuration knobs for cascade execution."""

    cascade_high_threshold: float = 0.85
    cascade_low_threshold: float = 0.15
    fast_detector_keywords: tuple[str, ...] = DEFAULT_FAST_DETECTOR_KEYWORDS
    speculative: bool = False


class CascadeRunner:
    """Executes multi-tier cascade detection with task cancellation and zero leaks.

    Tier 1 (Fast detectors, e.g. D1, D4) executes first.
    If Tier 1 result is confident:
      - Attack: any score >= cascade_high_threshold
      - Benign: all scores <= cascade_low_threshold
    The runner early-exits and cancels any speculative or slower detector tasks.
    If Tier 1 result is ambiguous, Tier 2 (slower detectors, e.g. D2, D3, D5) runs.
    """

    def __init__(self, config: CascadeConfig | None = None) -> None:
        self.config = config or CascadeConfig()

    def is_fast_detector(self, detector: Detector) -> bool:
        """Check if a detector belongs to the fast tier."""
        name = getattr(detector, "name", "").lower()
        return any(kw in name for kw in self.config.fast_detector_keywords)

    def partition_detectors(
        self,
        detectors: Sequence[Detector],
    ) -> tuple[list[Detector], list[Detector]]:
        """Split detectors into fast (Tier 1) and slow (Tier 2) tiers."""
        fast: list[Detector] = []
        slow: list[Detector] = []
        for d in detectors:
            if self.is_fast_detector(d):
                fast.append(d)
            else:
                slow.append(d)

        # Fallback: if no detectors classified as fast, treat first detector as fast
        if not fast and slow:
            fast.append(slow.pop(0))

        return fast, slow

    def evaluate_confidence(
        self,
        results: list[DetectorResult],
    ) -> tuple[bool, bool]:
        """Evaluate if Tier 1 results are confident.

        Returns (is_confident, is_attack).
        """
        valid_scores = [
            r.score for r in results if r.score is not None and r.is_attack is not None
        ]
        if not valid_scores:
            # Check discrete predictions if scores are absent
            valid_attacks = [r.is_attack for r in results if r.is_attack is not None]
            if not valid_attacks:
                return False, False
            if all(v is True for v in valid_attacks):
                return True, True
            if all(v is False for v in valid_attacks):
                return True, False
            return False, False

        # Confident attack
        if any(s >= self.config.cascade_high_threshold for s in valid_scores):
            return True, True

        # Confident benign
        if all(s <= self.config.cascade_low_threshold for s in valid_scores):
            return True, False

        # Ambiguous score range (e.g. 0.15 < s < 0.85)
        return False, False

    async def execute(
        self,
        detectors: Sequence[Detector],
        text: str,
        run_detector_fn: Callable[[Detector, str], Coroutine[Any, Any, DetectorResult]],
    ) -> tuple[list[DetectorResult], bool]:
        """Run cascade evaluation.

        Returns (results, early_exited).
        """
        fast_detectors, slow_detectors = self.partition_detectors(detectors)

        if not slow_detectors:
            # Only fast tier available
            results = await asyncio.gather(
                *[run_detector_fn(d, text) for d in fast_detectors]
            )
            return list(results), True

        slow_tasks: list[asyncio.Task[DetectorResult]] = []

        try:
            if self.config.speculative:
                # Launch slower detectors speculatively in background
                slow_tasks = [
                    asyncio.create_task(run_detector_fn(d, text))
                    for d in slow_detectors
                ]

            # Execute fast tier
            fast_tasks = [run_detector_fn(d, text) for d in fast_detectors]
            fast_results = list(await asyncio.gather(*fast_tasks))

            # Evaluate confidence
            is_confident, _ = self.evaluate_confidence(fast_results)

            if is_confident:
                # Early-exit: cancel any running or scheduled slow tasks
                for task in slow_tasks:
                    if not task.done():
                        task.cancel()
                if slow_tasks:
                    # Clean up cancelled tasks without leaking
                    await asyncio.gather(*slow_tasks, return_exceptions=True)
                return fast_results, True

            # Ambiguous: execute remaining detectors
            if self.config.speculative:
                slow_results = await asyncio.gather(*slow_tasks)
            else:
                slow_tasks = [
                    asyncio.create_task(run_detector_fn(d, text))
                    for d in slow_detectors
                ]
                slow_results = await asyncio.gather(*slow_tasks)

            return fast_results + list(slow_results), False

        except Exception:
            # Ensure cleanup on unexpected exceptions
            for task in slow_tasks:
                if not task.done():
                    task.cancel()
            if slow_tasks:
                await asyncio.gather(*slow_tasks, return_exceptions=True)
            raise
