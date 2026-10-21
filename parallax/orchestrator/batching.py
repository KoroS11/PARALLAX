"""Micro-batching queue for transformer-based detectors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from parallax.contracts.detectors import Detector, DetectorResult


class MicroBatcher:
    """Dynamic micro-batching queue for batch-capable Hugging Face / transformer detectors.

    Collects incoming asynchronous predict requests up to `max_batch_size` or
    until `max_wait_ms` has elapsed, then flushes them as a single batch.

    Parameters
    ----------
    detector : Detector
        Underlying detector instance.
    max_batch_size : int, default=16
        Maximum number of requests aggregated into one forward pass.
    max_wait_ms : float, default=5.0
        Maximum time in milliseconds to wait for additional inputs before flushing.
    """

    def __init__(
        self,
        detector: Detector,
        max_batch_size: int = 16,
        max_wait_ms: float = 5.0,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if max_wait_ms < 0:
            raise ValueError(f"max_wait_ms cannot be negative, got {max_wait_ms}")

        self.detector = detector
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.name = detector.name

        self._queue: list[tuple[str, asyncio.Future[DetectorResult]]] = []
        self._lock = asyncio.Lock()
        self._timer_task: asyncio.Task[None] | None = None
        self._closed = False

        # Metrics
        self.total_batches_processed = 0
        self.total_items_processed = 0

    @property
    def queue_size(self) -> int:
        """Current number of items queued waiting for batching."""
        return len(self._queue)

    @property
    def is_closed(self) -> bool:
        """Whether this batcher has been closed."""
        return self._closed

    async def _timer_worker(self) -> None:
        """Sleeps for max_wait_ms then flushes available items."""
        try:
            await asyncio.sleep(self.max_wait_ms / 1000.0)
            async with self._lock:
                if not self._queue:
                    return
                batch = list(self._queue)
                self._queue.clear()
                self._timer_task = None
            await self._dispatch_batch(batch)
        except asyncio.CancelledError:
            pass

    async def _dispatch_batch(
        self,
        batch: list[tuple[str, asyncio.Future[DetectorResult]]],
    ) -> None:
        """Execute inference on an extracted batch and resolve promises."""
        if not batch:
            return

        texts = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        self.total_batches_processed += 1
        self.total_items_processed += len(batch)

        t0 = time.perf_counter()
        try:
            # Check if underlying detector has native predict_batch support
            if hasattr(self.detector, "predict_batch") and callable(
                self.detector.predict_batch
            ):
                batch_results: Sequence[
                    DetectorResult
                ] = await self.detector.predict_batch(texts)
                if len(batch_results) != len(futures):
                    raise RuntimeError(
                        f"Detector '{self.name}' returned {len(batch_results)} results "
                        f"for a batch of {len(futures)} inputs."
                    )
                for fut, res in zip(futures, batch_results):
                    if not fut.done():
                        fut.set_result(res)
            else:
                # Concurrent individual execution fallback
                tasks = [self.detector.predict(t) for t in texts]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for fut, item_res in zip(futures, results):
                    if not fut.done():
                        if isinstance(item_res, BaseException):
                            fut.set_exception(item_res)
                        else:
                            fut.set_result(item_res)
        except Exception as exc:  # noqa: BLE001
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            for fut in futures:
                if not fut.done():
                    # Propagate fallback error result to individual requests
                    fallback = DetectorResult(
                        name=self.name,
                        is_attack=None,
                        score=None,
                        attack_class=None,
                        latency_ms=elapsed,
                        error=f"Micro-batch execution failed: {exc}",
                    )
                    fut.set_result(fallback)

    async def predict(self, text: str) -> DetectorResult:
        """Enqueue prompt text for micro-batch execution and await individual result."""
        if self._closed:
            raise RuntimeError(f"MicroBatcher for '{self.name}' is closed.")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[DetectorResult] = loop.create_future()
        batch_to_dispatch: list[tuple[str, asyncio.Future[DetectorResult]]] | None = (
            None
        )

        async with self._lock:
            self._queue.append((text, future))

            if len(self._queue) >= self.max_batch_size:
                # Immediate flush because capacity was reached
                if self._timer_task is not None and not self._timer_task.done():
                    self._timer_task.cancel()
                    self._timer_task = None
                batch_to_dispatch = list(self._queue)
                self._queue.clear()
            elif self._timer_task is None or self._timer_task.done():
                # Start window timer
                self._timer_task = asyncio.create_task(self._timer_worker())

        if batch_to_dispatch is not None:
            # Dispatch synchronously or as task
            asyncio.create_task(self._dispatch_batch(batch_to_dispatch))

        return await future

    async def flush(self) -> None:
        """Flush any currently queued items immediately."""
        batch_to_dispatch: list[tuple[str, asyncio.Future[DetectorResult]]] | None = (
            None
        )
        async with self._lock:
            if self._timer_task is not None and not self._timer_task.done():
                self._timer_task.cancel()
                self._timer_task = None
            if self._queue:
                batch_to_dispatch = list(self._queue)
                self._queue.clear()

        if batch_to_dispatch is not None:
            await self._dispatch_batch(batch_to_dispatch)

    async def close(self) -> None:
        """Close batcher, flushing remaining requests and cancelling timers."""
        self._closed = True
        await self.flush()
