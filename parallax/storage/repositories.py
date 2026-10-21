"""Database connection, async session management, and persistence repositories.

Handles SQLite connection pragmas (WAL mode, synchronous=NORMAL, busy_timeout=5000),
PostgreSQL URL compatibility, privacy-preserving text storage toggles,
and retention cleanup.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Self, cast

import numpy as np
from sqlalchemy import delete, desc, event, func, select
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from parallax.storage.models import (
    Base,
    DetectorResultRecord,
    EvalRunRecord,
    RunManifestRecord,
    SandboxEpisodeRecord,
    ScanRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_SQLITE_URL = "sqlite:///parallax.db"
DEFAULT_BUSY_TIMEOUT_MS = 5000
DEFAULT_TRUNCATION_LIMIT = 256


def _configure_sqlite_pragmas(
    engine: Engine, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS
) -> None:
    """Register connect hook to enforce SQLite WAL mode and busy timeout."""

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
        module_name = getattr(dbapi_connection.__class__, "__module__", "").lower()
        if "sqlite" in module_name:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            cursor.close()


def create_db_engine(
    url: str = DEFAULT_SQLITE_URL,
    echo: bool = False,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    **kwargs: Any,
) -> Engine:
    """Create SQLAlchemy engine with appropriate settings for SQLite or PostgreSQL.

    Automatically applies SQLite WAL mode and busy timeout when connecting to SQLite,
    while maintaining clean compatibility with PostgreSQL connection strings.
    """
    is_sqlite = url.startswith("sqlite")

    engine_kwargs: dict[str, Any] = {"echo": echo, **kwargs}
    if is_sqlite and ":memory:" not in url:
        # File-based SQLite: enable check_same_thread=False for multi-threading
        connect_args = engine_kwargs.get("connect_args", {})
        connect_args.setdefault("check_same_thread", False)
        engine_kwargs["connect_args"] = connect_args

    engine = create_engine(url, **engine_kwargs)

    if is_sqlite:
        _configure_sqlite_pragmas(engine, busy_timeout_ms=busy_timeout_ms)

    return engine


class AsyncSessionContext:
    """Async context wrapper around a synchronous SQLAlchemy Session."""

    def __init__(self, sync_session: Session) -> None:
        self._sync = sync_session

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            await self.rollback()
        await self.close()

    async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute a statement asynchronously in a thread worker."""
        return await asyncio.to_thread(self._sync.execute, statement, *args, **kwargs)

    async def scalar(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute and return scalar asynchronously."""
        return await asyncio.to_thread(self._sync.scalar, statement, *args, **kwargs)

    async def scalars(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute and return scalars asynchronously."""
        return await asyncio.to_thread(self._sync.scalars, statement, *args, **kwargs)

    async def commit(self) -> None:
        """Commit transaction asynchronously."""
        await asyncio.to_thread(self._sync.commit)

    async def rollback(self) -> None:
        """Rollback transaction asynchronously."""
        await asyncio.to_thread(self._sync.rollback)

    async def close(self) -> None:
        """Close session asynchronously."""
        await asyncio.to_thread(self._sync.close)

    async def get(self, entity: Any, ident: Any, *args: Any, **kwargs: Any) -> Any:
        """Get entity by primary key asynchronously."""
        return await asyncio.to_thread(self._sync.get, entity, ident, *args, **kwargs)

    async def delete(self, instance: Any) -> None:
        """Delete instance asynchronously."""
        await asyncio.to_thread(self._sync.delete, instance)

    async def refresh(self, instance: Any, *args: Any, **kwargs: Any) -> None:
        """Refresh instance asynchronously."""
        await asyncio.to_thread(self._sync.refresh, instance, *args, **kwargs)

    def add(self, instance: Any) -> None:
        """Add instance to sync session."""
        self._sync.add(instance)

    def add_all(self, instances: Iterable[Any]) -> None:
        """Add all instances to sync session."""
        self._sync.add_all(instances)

    @property
    def sync_session(self) -> Session:
        """Return underlying synchronous session."""
        return self._sync


class Database:
    """Database connection and session factory manager."""

    def __init__(
        self,
        url: str = DEFAULT_SQLITE_URL,
        echo: bool = False,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        **kwargs: Any,
    ) -> None:
        self.url = url
        self.engine = create_db_engine(
            url=url, echo=echo, busy_timeout_ms=busy_timeout_ms, **kwargs
        )
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def get_session(self) -> AsyncSessionContext:
        """Create a new AsyncSessionContext instance."""
        return AsyncSessionContext(self._session_factory())

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSessionContext]:
        """Async context manager yielding an AsyncSessionContext."""
        sess = self.get_session()
        async with sess:
            yield sess

    def create_all(self) -> None:
        """Synchronously create all registered schema tables."""
        Base.metadata.create_all(self.engine)

    async def acreate_all(self) -> None:
        """Asynchronously create all registered schema tables."""
        await asyncio.to_thread(self.create_all)

    def drop_all(self) -> None:
        """Synchronously drop all registered schema tables."""
        Base.metadata.drop_all(self.engine)

    async def adrop_all(self) -> None:
        """Asynchronously drop all registered schema tables."""
        await asyncio.to_thread(self.drop_all)

    def close(self) -> None:
        """Dispose of the database engine connections."""
        self.engine.dispose()

    async def aclose(self) -> None:
        """Asynchronously dispose of the database engine connections."""
        await asyncio.to_thread(self.close)


def should_store_text(override: bool | None = None) -> bool:
    """Evaluate whether raw prompt text storage is enabled via setting or environment."""
    if override is not None:
        return override
    env_val = os.getenv("PARALLAX_STORE_TEXT", "false").strip().lower()
    return env_val in ("1", "true", "yes", "on")


def process_text_privacy(
    text: str,
    store_text: bool | None = None,
    max_len: int = DEFAULT_TRUNCATION_LIMIT,
) -> tuple[str, str | None]:
    """Compute SHA-256 hash and truncated text according to privacy settings."""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if should_store_text(store_text):
        text_truncated = text[:max_len]
    else:
        text_truncated = None
    return text_hash, text_truncated


async def prune_old_records(
    db: Database,
    days: int = 30,
) -> dict[str, int]:
    """Prune scan, eval, episode, and manifest records older than N days.

    Returns dictionary with deleted record counts per table.
    """
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
    counts = {"scans": 0, "eval_runs": 0, "sandbox_episodes": 0, "run_manifests": 0}

    async with db.session() as session:
        # Scans (detector_results cascade-delete via foreign key ondelete=CASCADE)
        res_scans = await session.execute(
            delete(ScanRecord).where(ScanRecord.timestamp < cutoff)
        )
        counts["scans"] = res_scans.rowcount or 0

        # Eval runs
        res_eval = await session.execute(
            delete(EvalRunRecord).where(EvalRunRecord.timestamp < cutoff)
        )
        counts["eval_runs"] = res_eval.rowcount or 0

        # Sandbox episodes
        res_episodes = await session.execute(
            delete(SandboxEpisodeRecord).where(SandboxEpisodeRecord.timestamp < cutoff)
        )
        counts["sandbox_episodes"] = res_episodes.rowcount or 0

        # Run manifests
        res_manifests = await session.execute(
            delete(RunManifestRecord).where(RunManifestRecord.timestamp < cutoff)
        )
        counts["run_manifests"] = res_manifests.rowcount or 0

        await session.commit()

    return counts


class ScanRepository:
    """Repository for logging and querying scan records and detector results."""

    def __init__(self, db: Database, queue: Any | None = None) -> None:
        self.db = db
        self.queue = queue

    async def log_scan(
        self,
        text: str,
        decision: str,
        latency_ms: float = 0.0,
        results: list[Any] | None = None,
        disagreement: bool = False,
        trace_id: str | None = None,
        store_text: bool | None = None,
        safe: bool = True,
    ) -> ScanRecord | None:
        """Log a scan record and its detector results.

        If a write-behind queue is configured, defers to the queue asynchronously.
        If `safe=True`, database errors are logged and swallowed without raising.
        """
        if self.queue is not None:
            self.queue.enqueue_scan(
                text=text,
                decision=decision,
                latency_ms=latency_ms,
                results=results,
                disagreement=disagreement,
                trace_id=trace_id,
                store_text=store_text,
            )
            return None

        try:
            text_hash, text_truncated = process_text_privacy(
                text, store_text=store_text
            )
            scan = ScanRecord(
                text_hash=text_hash,
                text_truncated=text_truncated,
                decision=decision,
                latency_ms=latency_ms,
                disagreement=disagreement,
                trace_id=trace_id,
            )

            if results:
                for r in results:
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

            async with self.db.session() as session:
                session.add(scan)
                await session.commit()
                await session.refresh(scan)
            return scan

        except Exception as e:
            logger.error("Failed to persist scan record to database: %s", e)
            if not safe:
                raise
            return None

    async def get_scan_by_id(self, scan_id: int) -> ScanRecord | None:
        """Fetch scan record by ID with attached detector results."""
        async with self.db.session() as session:
            stmt = select(ScanRecord).where(ScanRecord.id == scan_id)
            res = await session.execute(stmt)
            return cast(ScanRecord | None, res.scalar_one_or_none())

    async def get_scan_by_trace_id(self, trace_id: str) -> ScanRecord | None:
        """Fetch scan record by trace ID."""
        async with self.db.session() as session:
            stmt = select(ScanRecord).where(ScanRecord.trace_id == trace_id)
            res = await session.execute(stmt)
            return cast(ScanRecord | None, res.scalars().first())

    async def list_recent_scans(
        self, limit: int = 100, offset: int = 0
    ) -> list[ScanRecord]:
        """Fetch recent scans ordered by timestamp descending."""
        async with self.db.session() as session:
            stmt = (
                select(ScanRecord)
                .order_by(desc(ScanRecord.timestamp))
                .limit(limit)
                .offset(offset)
            )
            res = await session.execute(stmt)
            return list(res.scalars().all())

    async def get_metrics_summary(self) -> dict[str, Any]:
        """Calculate aggregate scan metrics for dashboard and reporting."""
        async with self.db.session() as session:
            total_stmt = select(func.count(ScanRecord.id))
            total_scans = (await session.scalar(total_stmt)) or 0

            allow_stmt = select(func.count(ScanRecord.id)).where(
                ScanRecord.decision == "allow"
            )
            allow_count = (await session.scalar(allow_stmt)) or 0

            flag_stmt = select(func.count(ScanRecord.id)).where(
                ScanRecord.decision == "flag"
            )
            flag_count = (await session.scalar(flag_stmt)) or 0

            block_stmt = select(func.count(ScanRecord.id)).where(
                ScanRecord.decision == "block"
            )
            block_count = (await session.scalar(block_stmt)) or 0

            disagree_stmt = select(func.count(ScanRecord.id)).where(
                ScanRecord.disagreement.is_(True)
            )
            disagree_count = (await session.scalar(disagree_stmt)) or 0

            disagreement_rate = (
                float(disagree_count) / float(total_scans) if total_scans > 0 else 0.0
            )

            lat_stmt = select(ScanRecord.latency_ms)
            lat_res = await session.execute(lat_stmt)
            latencies = [lat for lat in lat_res.scalars().all() if lat is not None]

            if latencies:
                p50 = float(np.percentile(latencies, 50))
                p95 = float(np.percentile(latencies, 95))
            else:
                p50 = 0.0
                p95 = 0.0

            return {
                "total_scans": total_scans,
                "allow_count": allow_count,
                "flag_count": flag_count,
                "block_count": block_count,
                "disagreement_rate": round(disagreement_rate, 4),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
            }


class EvalRunRepository:
    """Repository for logging and querying evaluation runs."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def log_eval_run(
        self,
        run_id: str,
        experiment_id: str,
        status: str = "completed",
        metrics: dict[str, Any] | None = None,
        safe: bool = True,
    ) -> EvalRunRecord | None:
        """Create or update an evaluation run record."""
        try:
            record = EvalRunRecord(
                run_id=run_id,
                experiment_id=experiment_id,
                status=status,
                metrics_json=metrics or {},
            )
            async with self.db.session() as session:
                session.add(record)
                await session.commit()
                await session.refresh(record)
            return record
        except Exception as e:
            logger.error("Failed to log eval run %s: %s", run_id, e)
            if not safe:
                raise
            return None

    async def get_eval_run(self, run_id: str) -> EvalRunRecord | None:
        """Fetch an evaluation run by run_id."""
        async with self.db.session() as session:
            stmt = select(EvalRunRecord).where(EvalRunRecord.run_id == run_id)
            res = await session.execute(stmt)
            return cast(EvalRunRecord | None, res.scalar_one_or_none())

    async def list_eval_runs(self, limit: int = 100) -> list[EvalRunRecord]:
        """List evaluation runs ordered by timestamp descending."""
        async with self.db.session() as session:
            stmt = (
                select(EvalRunRecord)
                .order_by(desc(EvalRunRecord.timestamp))
                .limit(limit)
            )
            res = await session.execute(stmt)
            return list(res.scalars().all())


class SandboxEpisodeRepository:
    """Repository for logging and querying sandbox attack episodes."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def log_episode(
        self,
        episode_id: str,
        victim_model: str,
        attack_technique: str,
        guard_enabled: bool,
        attack_succeeded: bool,
        details: dict[str, Any] | None = None,
        safe: bool = True,
    ) -> SandboxEpisodeRecord | None:
        """Create and store a sandbox episode record."""
        try:
            record = SandboxEpisodeRecord(
                episode_id=episode_id,
                victim_model=victim_model,
                attack_technique=attack_technique,
                guard_enabled=guard_enabled,
                attack_succeeded=attack_succeeded,
                details_json=details or {},
            )
            async with self.db.session() as session:
                session.add(record)
                await session.commit()
                await session.refresh(record)
            return record
        except Exception as e:
            logger.error("Failed to log sandbox episode %s: %s", episode_id, e)
            if not safe:
                raise
            return None

    async def get_episode(self, episode_id: str) -> SandboxEpisodeRecord | None:
        """Fetch a sandbox episode by episode_id."""
        async with self.db.session() as session:
            stmt = select(SandboxEpisodeRecord).where(
                SandboxEpisodeRecord.episode_id == episode_id
            )
            res = await session.execute(stmt)
            return cast(SandboxEpisodeRecord | None, res.scalar_one_or_none())

    async def list_episodes(self, limit: int = 100) -> list[SandboxEpisodeRecord]:
        """List sandbox episodes ordered by timestamp descending."""
        async with self.db.session() as session:
            stmt = (
                select(SandboxEpisodeRecord)
                .order_by(desc(SandboxEpisodeRecord.timestamp))
                .limit(limit)
            )
            res = await session.execute(stmt)
            return list(res.scalars().all())


class RunManifestRepository:
    """Repository for recording configuration and reproducibility manifests."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def log_manifest(
        self,
        manifest_hash: str,
        git_sha: str,
        config: dict[str, Any] | None = None,
        safe: bool = True,
    ) -> RunManifestRecord | None:
        """Create and store a run manifest record."""
        try:
            record = RunManifestRecord(
                manifest_hash=manifest_hash,
                git_sha=git_sha,
                config_json=config or {},
            )
            async with self.db.session() as session:
                session.add(record)
                await session.commit()
                await session.refresh(record)
            return record
        except Exception as e:
            logger.error("Failed to log run manifest %s: %s", manifest_hash, e)
            if not safe:
                raise
            return None

    async def get_manifest(self, manifest_hash: str) -> RunManifestRecord | None:
        """Fetch a run manifest by manifest_hash."""
        async with self.db.session() as session:
            stmt = select(RunManifestRecord).where(
                RunManifestRecord.manifest_hash == manifest_hash
            )
            res = await session.execute(stmt)
            return cast(RunManifestRecord | None, res.scalar_one_or_none())
