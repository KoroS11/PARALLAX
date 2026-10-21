"""Concurrency bulkheads and GPU resource isolation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

from parallax.contracts.detectors import DetectorResult

# Detectors known to run on GPU or local Ollama judge
DEFAULT_GPU_DETECTOR_KEYWORDS = ("d2", "deberta", "d5", "judge")


class BulkheadFullError(Exception):
    """Raised when a detector's concurrency bulkhead limit is exceeded."""

    def __init__(self, detector_name: str, max_concurrency: int) -> None:
        self.detector_name = detector_name
        self.max_concurrency = max_concurrency
        super().__init__(
            f"Bulkhead capacity exceeded for detector '{detector_name}' "
            f"(max concurrency: {max_concurrency})."
        )


class Bulkhead:
    """Per-detector concurrency bulkhead limiter using asyncio.Semaphore."""

    def __init__(
        self,
        name: str,
        max_concurrency: int = 10,
        acquire_timeout: float | None = None,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")

        self.name = name
        self.max_concurrency = max_concurrency
        self.acquire_timeout = acquire_timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_count = 0
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        """Number of concurrently executing tasks within this bulkhead."""
        return self._active_count

    @property
    def available_slots(self) -> int:
        """Number of available concurrency slots."""
        return max(0, self.max_concurrency - self._active_count)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Async context manager acquiring a bulkhead execution slot."""
        if self.acquire_timeout is not None:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.acquire_timeout,
                )
            except TimeoutError as err:
                raise BulkheadFullError(self.name, self.max_concurrency) from err
        else:
            await self._semaphore.acquire()

        async with self._lock:
            self._active_count += 1

        try:
            yield
        finally:
            async with self._lock:
                self._active_count -= 1
            self._semaphore.release()


class GPULimiter:
    """Global GPU execution lock enforcing that only one GPU detector executes at once."""

    def __init__(
        self,
        gpu_semaphore: asyncio.Semaphore | None = None,
        gpu_detector_keywords: tuple[str, ...] = DEFAULT_GPU_DETECTOR_KEYWORDS,
    ) -> None:
        self._semaphore = gpu_semaphore or asyncio.Semaphore(1)
        self.gpu_detector_keywords = tuple(k.lower() for k in gpu_detector_keywords)
        self._active_gpu_detector: str | None = None
        self._lock = asyncio.Lock()

    @property
    def active_detector(self) -> str | None:
        """Name of the currently executing GPU detector, if any."""
        return self._active_gpu_detector

    def is_gpu_detector(self, name_or_obj: str | Any) -> bool:
        """Determine whether a detector targets GPU or Ollama local inference."""
        if (
            hasattr(name_or_obj, "device")
            and str(getattr(name_or_obj, "device", "")).lower() == "gpu"
        ):
            return True
        if hasattr(name_or_obj, "is_gpu") and bool(
            getattr(name_or_obj, "is_gpu", False)
        ):
            return True

        detector_name = (getattr(name_or_obj, "name", str(name_or_obj)) or "").lower()
        return any(kw in detector_name for kw in self.gpu_detector_keywords)

    @asynccontextmanager
    async def acquire(self, detector_name: str) -> AsyncIterator[None]:
        """Acquire the single global GPU execution slot."""
        await self._semaphore.acquire()
        async with self._lock:
            self._active_gpu_detector = detector_name
        try:
            yield
        finally:
            async with self._lock:
                self._active_gpu_detector = None
            self._semaphore.release()


class BulkheadManager:
    """Manages per-detector bulkheads and coordinated GPU semaphore access."""

    def __init__(
        self,
        default_max_concurrency: int = 10,
        per_detector_concurrency: dict[str, int] | None = None,
        acquire_timeout: float | None = None,
        gpu_limiter: GPULimiter | None = None,
    ) -> None:
        self.default_max_concurrency = default_max_concurrency
        self.per_detector_concurrency = per_detector_concurrency or {}
        self.acquire_timeout = acquire_timeout
        self.gpu_limiter = gpu_limiter or GPULimiter()
        self._bulkheads: dict[str, Bulkhead] = {}

    def get_bulkhead(self, name: str) -> Bulkhead:
        """Get or initialize bulkhead for a detector."""
        if name not in self._bulkheads:
            limit = self.per_detector_concurrency.get(
                name, self.default_max_concurrency
            )
            self._bulkheads[name] = Bulkhead(
                name=name,
                max_concurrency=limit,
                acquire_timeout=self.acquire_timeout,
            )
        return self._bulkheads[name]

    @asynccontextmanager
    async def limit(self, detector_name: str) -> AsyncIterator[None]:
        """Apply bulkhead and GPU serialization around an execution block."""
        bulkhead = self.get_bulkhead(detector_name)
        async with bulkhead.acquire():
            if self.gpu_limiter.is_gpu_detector(detector_name):
                async with self.gpu_limiter.acquire(detector_name):
                    yield
            else:
                yield

    async def execute(
        self,
        detector_name: str,
        func: Callable[..., Coroutine[Any, Any, DetectorResult]],
        *args: Any,
        **kwargs: Any,
    ) -> DetectorResult:
        """Execute a detector prediction bounded by its bulkhead and GPU semaphore."""
        try:
            async with self.limit(detector_name):
                return await func(*args, **kwargs)
        except BulkheadFullError as exc:
            return DetectorResult(
                name=detector_name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=0.0,
                error=str(exc),
            )
