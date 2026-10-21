"""Bounded asynchronous write-behind queue for database persistence.

Drains items into database sessions in background batches, logs and counts drops
when the queue is full, and insulates caller scan workflows from database outages.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

from sqlalchemy.exc import SQLAlchemyError

from parallax.storage.models import (
    DetectorResultRecord,
    EvalRunRecord,
    RunManifestRecord,
    SandboxEpisodeRecord,
    ScanRecord,
)
from parallax.storage.repositories import Database, process_text_privacy

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_CAPACITY = 1000
DEFAULT_BATCH_SIZE = 50
DEFAULT_FLUSH_INTERVAL = 0.05


class WriteBehindQueue:
    """Asynchronous write-behind queue with bounded capacity and error isolation."""

    def __init__(
        self,
        db: Database,
        maxsize: int = DEFAULT_QUEUE_CAPACITY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        self.db = db
        self._maxsize = maxsize
        self._batch_size = batch_size
        self._flush_interval_sec = flush_interval_sec

        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._worker_task: asyncio.Task[None] | None = None
        self._running: bool = False

        self._dropped_count: int = 0
        self._processed_count: int = 0
        self._error_count: int = 0

    @property
    def maxsize(self) -> int:
        """Capacity limit of the bounded queue."""
        return self._maxsize

    @property
    def dropped_count(self) -> int:
        """Total items dropped due to full queue capacity."""
        return self._dropped_count

    @property
    def processed_count(self) -> int:
        """Total items successfully drained and committed to database."""
        return self._processed_count

    @property
    def error_count(self) -> int:
        """Total items failed during database drain due to DB outages."""
        return self._error_count

    @property
    def qsize(self) -> int:
        """Number of items currently queued."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Whether the background worker loop is active."""
        return self._running

    async def start(self) -> None:
        """Start the background drain worker task."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop worker task and drain remaining queue items within timeout."""
        self._running = False
        if self._worker_task is None:
            return

        try:
            if not self._queue.empty():
                await asyncio.wait_for(self.flush(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                "WriteBehindQueue stop timed out waiting for queue to drain."
            )
        finally:
            if not self._worker_task.done():
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Worker task finished with exception during stop: %s", exc
                    )
            self._worker_task = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()

    def put(self, item: dict[str, Any]) -> bool:
        """Non-blocking put into bounded queue.

        If the queue is full, increments `dropped_count`, logs a warning,
        and returns `False` without raising an exception.
        """
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self._dropped_count += 1
            logger.warning(
                "Write-behind queue full (capacity %d); dropping item type '%s'",
                self._maxsize,
                item.get("type", "unknown"),
            )
            return False

    async def enqueue(self, item: dict[str, Any]) -> bool:
        """Enqueue an item into the queue."""
        return self.put(item)

    def enqueue_scan(
        self,
        text: str,
        decision: str,
        latency_ms: float = 0.0,
        results: list[Any] | None = None,
        disagreement: bool = False,
        trace_id: str | None = None,
        store_text: bool | None = None,
    ) -> bool:
        """Enqueue a prompt scan verdict and per-detector results."""
        item: dict[str, Any] = {
            "type": "scan",
            "text": text,
            "decision": decision,
            "latency_ms": latency_ms,
            "results": results or [],
            "disagreement": disagreement,
            "trace_id": trace_id,
            "store_text": store_text,
            "timestamp": datetime.now(UTC).replace(tzinfo=None),
        }
        return self.put(item)

    def enqueue_eval_run(
        self,
        run_id: str,
        experiment_id: str,
        status: str = "completed",
        metrics: dict[str, Any] | None = None,
    ) -> bool:
        """Enqueue an evaluation run record."""
        item: dict[str, Any] = {
            "type": "eval_run",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "status": status,
            "metrics": metrics or {},
            "timestamp": datetime.now(UTC).replace(tzinfo=None),
        }
        return self.put(item)

    def enqueue_episode(
        self,
        episode_id: str,
        victim_model: str,
        attack_technique: str,
        guard_enabled: bool,
        attack_succeeded: bool,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Enqueue a sandbox episode record."""
        item: dict[str, Any] = {
            "type": "episode",
            "episode_id": episode_id,
            "victim_model": victim_model,
            "attack_technique": attack_technique,
            "guard_enabled": guard_enabled,
            "attack_succeeded": attack_succeeded,
            "details": details or {},
            "timestamp": datetime.now(UTC).replace(tzinfo=None),
        }
        return self.put(item)

    async def flush(self) -> None:
        """Wait until all currently queued items have been processed by the worker."""
        await self._queue.join()

    async def _worker_loop(self) -> None:
        """Continuous background worker draining queued items into DB sessions."""
        while self._running or not self._queue.empty():
            try:
                first = await asyncio.wait_for(
                    self._queue.get(), timeout=self._flush_interval_sec
                )
                batch = [first]
            except TimeoutError:
                if not self._running:
                    break
                continue
            except asyncio.CancelledError:
                break

            # Collect additional items up to batch_size without blocking
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            try:
                await self._write_batch(batch)
                self._processed_count += len(batch)
            except (SQLAlchemyError, Exception) as e:  # noqa: BLE001
                self._error_count += len(batch)
                logger.error(
                    "Database outage during write-behind drain (%d items affected): %s",
                    len(batch),
                    e,
                )
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        """Write a batch of items to the database in a single session transaction."""
        scans: list[ScanRecord] = []
        eval_runs: list[EvalRunRecord] = []
        episodes: list[SandboxEpisodeRecord] = []
        manifests: list[RunManifestRecord] = []

        for item in batch:
            item_type = item.get("type", "scan")
            if item_type == "scan":
                text = item["text"]
                text_hash, text_truncated = process_text_privacy(
                    text, store_text=item.get("store_text")
                )
                scan = ScanRecord(
                    timestamp=item.get("timestamp"),
                    text_hash=text_hash,
                    text_truncated=text_truncated,
                    decision=item.get("decision", "allow"),
                    latency_ms=item.get("latency_ms", 0.0),
                    disagreement=item.get("disagreement", False),
                    trace_id=item.get("trace_id"),
                )
                for r in item.get("results", []):
                    r_dict = r if isinstance(r, dict) else getattr(r, "__dict__", {})
                    name = r_dict.get("name") or getattr(r, "name", "unknown")
                    is_attack = (
                        r_dict.get("is_attack")
                        if "is_attack" in r_dict
                        else getattr(r, "is_attack", None)
                    )
                    score = (
                        r_dict.get("score")
                        if "score" in r_dict
                        else getattr(r, "score", None)
                    )
                    attack_class = (
                        r_dict.get("attack_class")
                        if "attack_class" in r_dict
                        else getattr(r, "attack_class", None)
                    )
                    det_lat = (
                        r_dict.get("latency_ms")
                        if "latency_ms" in r_dict
                        else getattr(r, "latency_ms", 0.0)
                    )
                    error = (
                        r_dict.get("error")
                        if "error" in r_dict
                        else getattr(r, "error", None)
                    )

                    det_record = DetectorResultRecord(
                        detector_name=name,
                        is_attack=is_attack,
                        score=score,
                        attack_class=attack_class,
                        latency_ms=det_lat or 0.0,
                        error=error,
                    )
                    scan.detector_results.append(det_record)
                scans.append(scan)

            elif item_type == "eval_run":
                eval_runs.append(
                    EvalRunRecord(
                        run_id=item["run_id"],
                        experiment_id=item["experiment_id"],
                        status=item.get("status", "completed"),
                        metrics_json=item.get("metrics") or {},
                        timestamp=item.get("timestamp"),
                    )
                )

            elif item_type == "episode":
                episodes.append(
                    SandboxEpisodeRecord(
                        episode_id=item["episode_id"],
                        victim_model=item["victim_model"],
                        attack_technique=item["attack_technique"],
                        guard_enabled=item["guard_enabled"],
                        attack_succeeded=item["attack_succeeded"],
                        details_json=item.get("details") or {},
                        timestamp=item.get("timestamp"),
                    )
                )

            elif item_type == "manifest":
                manifests.append(
                    RunManifestRecord(
                        manifest_hash=item["manifest_hash"],
                        git_sha=item["git_sha"],
                        config_json=item.get("config") or {},
                        timestamp=item.get("timestamp"),
                    )
                )

        async with self.db.session() as session:
            if scans:
                session.add_all(scans)
            if eval_runs:
                session.add_all(eval_runs)
            if episodes:
                session.add_all(episodes)
            if manifests:
                session.add_all(manifests)
            await session.commit()
