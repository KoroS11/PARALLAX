"""Unit and integration tests for PARALLAX storage robustness.

Tests cover:
1. Concurrent writers without lock failures (WAL mode + busy_timeout)
2. Bounded queue overflow (drops are counted, never raised to caller)
3. Database unavailable isolation (caller scan logging never raises)
4. Alembic migration up/down on SQLite
5. Privacy toggle (PARALLAX_STORE_TEXT)
6. Retention pruning of expired records
7. Metrics summary calculation
8. Eval, sandbox episode, and run manifest repositories
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from alembic.config import Config
from sqlalchemy import inspect, text

from alembic import command
from parallax.contracts.detectors import DetectorResult
from parallax.storage.models import (
    DetectorResultRecord,
    EvalRunRecord,
    ScanRecord,
)
from parallax.storage.queue import WriteBehindQueue
from parallax.storage.repositories import (
    Database,
    EvalRunRepository,
    RunManifestRepository,
    SandboxEpisodeRepository,
    ScanRepository,
    create_db_engine,
    process_text_privacy,
    prune_old_records,
    should_store_text,
)


@pytest.fixture
def temp_db_path() -> Any:
    """Provide a temporary file path for a SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    p = Path(path)
    if p.exists():
        p.unlink(missing_ok=True)
    # Also clean up WAL and SHM files if present
    Path(f"{path}-wal").unlink(missing_ok=True)
    Path(f"{path}-shm").unlink(missing_ok=True)


@pytest.fixture
def test_db(temp_db_path: str) -> Any:
    """Initialize an active Database instance with created schema."""
    db = Database(url=f"sqlite:///{temp_db_path}")
    db.create_all()
    yield db
    db.close()


# ---------------------------------------------------------------------------
# 1. SQLite Pragmas and Connection Tests
# ---------------------------------------------------------------------------


def test_sqlite_wal_pragmas(temp_db_path: str) -> None:
    """Verify WAL mode, synchronous=NORMAL, and busy_timeout=5000 on SQLite."""
    engine = create_db_engine(f"sqlite:///{temp_db_path}")
    with engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        synchronous = conn.execute(text("PRAGMA synchronous")).scalar()
        busy_timeout = conn.execute(text("PRAGMA busy_timeout")).scalar()

    engine.dispose()
    assert str(journal_mode).lower() == "wal"
    assert synchronous in (1, "1", "NORMAL")
    assert busy_timeout == 5000


def test_postgres_url_compatibility() -> None:
    """Ensure Postgres URLs do not trigger SQLite-specific pragmas."""
    # Test engine configuration without actually connecting to non-existent Postgres
    engine = create_db_engine("postgresql+psycopg://user:pass@localhost:5432/testdb")
    assert engine.dialect.name == "postgresql"
    engine.dispose()


# ---------------------------------------------------------------------------
# 2. Concurrent Writers Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_writers(test_db: Database) -> None:
    """Test multiple asynchronous tasks concurrently writing scans into SQLite."""
    repo = ScanRepository(test_db)
    num_tasks = 12
    writes_per_task = 10

    async def writer_task(task_idx: int) -> None:
        for i in range(writes_per_task):
            prompt = f"Prompt from worker {task_idx} iter {i}"
            res = await repo.log_scan(
                text=prompt,
                decision="allow" if i % 2 == 0 else "flag",
                latency_ms=12.5 + i,
                results=[
                    {
                        "name": "minilm",
                        "is_attack": False,
                        "score": 0.05,
                        "latency_ms": 5.0,
                    },
                    {
                        "name": "deberta",
                        "is_attack": False,
                        "score": 0.12,
                        "latency_ms": 7.5,
                    },
                ],
                disagreement=False,
                trace_id=f"trace-{task_idx}-{i}",
                safe=False,
            )
            assert res is not None
            assert res.id > 0

    tasks = [asyncio.create_task(writer_task(t)) for t in range(num_tasks)]
    await asyncio.gather(*tasks)

    # Verify all records exist
    recent = await repo.list_recent_scans(limit=num_tasks * writes_per_task + 50)
    assert len(recent) == num_tasks * writes_per_task

    # Verify detector results attached
    first_scan = await repo.get_scan_by_id(recent[0].id)
    assert first_scan is not None
    assert len(first_scan.detector_results) == 2


# ---------------------------------------------------------------------------
# 3. Bounded Queue Overflow Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_queue_overflow(test_db: Database) -> None:
    """Verify bounded write-behind queue drops excess items and never raises."""
    capacity = 5
    total_items = 25
    queue = WriteBehindQueue(test_db, maxsize=capacity)

    # Enqueue items without starting background worker
    accepted = 0
    for i in range(total_items):
        ok = queue.enqueue_scan(
            text=f"Overflow test prompt {i}",
            decision="allow",
            latency_ms=10.0,
            trace_id=f"overflow-{i}",
        )
        if ok:
            accepted += 1

    assert accepted == capacity
    assert queue.dropped_count == total_items - capacity
    assert queue.qsize == capacity

    # Now start worker and flush; should process only the 5 retained items
    await queue.start()
    await queue.flush()
    await queue.stop()

    assert queue.processed_count == capacity

    repo = ScanRepository(test_db)
    scans = await repo.list_recent_scans(limit=100)
    assert len(scans) == capacity


# ---------------------------------------------------------------------------
# 4. Database Unavailable Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_database_unavailable_caller_isolation(temp_db_path: str) -> None:
    """Caller scan logging must succeed without raising when DB is unavailable."""
    db = Database(url=f"sqlite:///{temp_db_path}")
    await db.acreate_all()

    queue = WriteBehindQueue(db, maxsize=100)
    repo = ScanRepository(db, queue=queue)

    # Start queue worker
    await queue.start()

    # Simulate database outage by closing/disposing engine and corrupting session
    await db.aclose()
    db.session = MagicMock()
    db.session.side_effect = RuntimeError("Database connection lost (outage)")

    # Caller logging must never raise!
    try:
        res = await repo.log_scan(
            text="Simulated outage prompt",
            decision="block",
            latency_ms=25.0,
            trace_id="outage-trace-1",
        )
        assert res is None  # Queue-based returns None
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Caller logging raised during DB outage: {exc}")

    # Give worker a moment to process and encounter outage
    await asyncio.sleep(0.15)
    await queue.stop()

    assert queue.error_count > 0


@pytest.mark.asyncio
async def test_direct_scan_repository_safe_mode_on_db_outage(temp_db_path: str) -> None:
    """Direct repository log_scan in safe mode swallows DB errors."""
    db = Database(url=f"sqlite:///{temp_db_path}")
    repo = ScanRepository(db)

    # Force DB error on session
    db.session = MagicMock()
    db.session.side_effect = RuntimeError("Fatal DB error")

    # Safe mode should not raise
    result = await repo.log_scan(text="test safe", decision="allow", safe=True)
    assert result is None

    # Unsafe mode should raise
    with pytest.raises(RuntimeError):
        await repo.log_scan(text="test unsafe", decision="allow", safe=False)


# ---------------------------------------------------------------------------
# 5. Alembic Migration Up/Down Test
# ---------------------------------------------------------------------------


def test_alembic_migrations_up_and_down(temp_db_path: str) -> None:
    """Verify Alembic migrations apply upgrade to head and downgrade to base cleanly."""
    root_dir = Path(__file__).resolve().parent.parent
    ini_path = root_dir / "alembic.ini"
    assert ini_path.exists(), "alembic.ini not found"

    alembic_cfg = Config(str(ini_path))
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")
    alembic_cfg.set_main_option("script_location", str(root_dir / "alembic"))

    # Upgrade to head
    command.upgrade(alembic_cfg, "head")

    engine = create_db_engine(f"sqlite:///{temp_db_path}")
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    expected_tables = {
        "scans",
        "detector_results",
        "eval_runs",
        "sandbox_episodes",
        "run_manifests",
        "alembic_version",
    }
    assert expected_tables.issubset(tables)

    # Downgrade to base
    command.downgrade(alembic_cfg, "base")

    insp = inspect(engine)
    tables_after = set(insp.get_table_names())
    assert "scans" not in tables_after
    assert "detector_results" not in tables_after
    assert "eval_runs" not in tables_after
    assert "sandbox_episodes" not in tables_after
    assert "run_manifests" not in tables_after

    engine.dispose()


# ---------------------------------------------------------------------------
# 6. Privacy Toggle (PARALLAX_STORE_TEXT) Tests
# ---------------------------------------------------------------------------


def test_privacy_toggle_hash_and_truncation() -> None:
    """Test text hashing and truncation with privacy flag True/False."""
    sample_text = "Adversarial prompt text with sensitive contents " * 10
    expected_hash = hashlib.sha256(sample_text.encode("utf-8")).hexdigest()

    # Default: do not store raw text
    h, t = process_text_privacy(sample_text, store_text=False)
    assert h == expected_hash
    assert t is None

    # Store text enabled
    h, t = process_text_privacy(sample_text, store_text=True, max_len=30)
    assert h == expected_hash
    assert t == sample_text[:30]


def test_should_store_text_env_variable() -> None:
    """Test evaluation of PARALLAX_STORE_TEXT environment variable."""
    with patch.dict("os.environ", {"PARALLAX_STORE_TEXT": "1"}):
        assert should_store_text() is True

    with patch.dict("os.environ", {"PARALLAX_STORE_TEXT": "true"}):
        assert should_store_text() is True

    with patch.dict("os.environ", {"PARALLAX_STORE_TEXT": "0"}):
        assert should_store_text() is False

    with patch.dict("os.environ", {}, clear=True):
        assert should_store_text() is False


@pytest.mark.asyncio
async def test_scan_repository_privacy_integration(test_db: Database) -> None:
    """Verify ScanRepository respects store_text flag."""
    repo = ScanRepository(test_db)

    # Scan 1: store_text=False
    scan_private = await repo.log_scan(
        text="Private prompt",
        decision="allow",
        store_text=False,
    )
    assert scan_private is not None
    assert scan_private.text_truncated is None

    # Scan 2: store_text=True
    scan_stored = await repo.log_scan(
        text="Stored prompt text",
        decision="block",
        store_text=True,
    )
    assert scan_stored is not None
    assert scan_stored.text_truncated == "Stored prompt text"


# ---------------------------------------------------------------------------
# 7. Retention Pruning Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retention_pruning(test_db: Database) -> None:
    """Ensure prune_old_records removes records older than N days."""
    now = datetime.now(UTC).replace(tzinfo=None)
    old_time = now - timedelta(days=45)

    async with test_db.session() as session:
        # Old scan + detector result
        old_scan = ScanRecord(
            timestamp=old_time,
            text_hash="oldhash123",
            decision="allow",
            latency_ms=10.0,
        )
        old_scan.detector_results.append(
            DetectorResultRecord(detector_name="d1", latency_ms=5.0)
        )
        session.add(old_scan)

        # Recent scan
        new_scan = ScanRecord(
            timestamp=now,
            text_hash="newhash123",
            decision="block",
            latency_ms=15.0,
        )
        session.add(new_scan)

        # Old eval run
        old_eval = EvalRunRecord(
            run_id="old-run-1",
            experiment_id="E1",
            timestamp=old_time,
            status="completed",
        )
        session.add(old_eval)

        # Recent eval run
        new_eval = EvalRunRecord(
            run_id="new-run-1",
            experiment_id="E1",
            timestamp=now,
            status="completed",
        )
        session.add(new_eval)

        await session.commit()

    # Prune records older than 30 days
    counts = await prune_old_records(test_db, days=30)
    assert counts["scans"] == 1
    assert counts["eval_runs"] == 1

    repo = ScanRepository(test_db)
    scans = await repo.list_recent_scans()
    assert len(scans) == 1
    assert scans[0].text_hash == "newhash123"

    eval_repo = EvalRunRepository(test_db)
    runs = await eval_repo.list_eval_runs()
    assert len(runs) == 1
    assert runs[0].run_id == "new-run-1"


# ---------------------------------------------------------------------------
# 8. Metrics Summary Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_summary_calculation(test_db: Database) -> None:
    """Verify accurate aggregation of scans, decisions, disagreement, and latencies."""
    repo = ScanRepository(test_db)

    # Empty DB
    summary_empty = await repo.get_metrics_summary()
    assert summary_empty["total_scans"] == 0
    assert summary_empty["p50_latency_ms"] == 0.0

    # Add controlled scans: 2 allow, 1 flag, 1 block, 2 disagreements
    latencies = [10.0, 20.0, 30.0, 40.0]
    decisions = ["allow", "allow", "flag", "block"]
    disagreements = [False, True, True, False]

    for lat, dec, dis in zip(latencies, decisions, disagreements, strict=True):
        await repo.log_scan(
            text=f"Metrics test text {lat}",
            decision=dec,
            latency_ms=lat,
            disagreement=dis,
        )

    summary = await repo.get_metrics_summary()
    assert summary["total_scans"] == 4
    assert summary["allow_count"] == 2
    assert summary["flag_count"] == 1
    assert summary["block_count"] == 1
    assert summary["disagreement_rate"] == 0.5
    assert summary["p50_latency_ms"] == 25.0
    assert summary["p95_latency_ms"] == 38.5


# ---------------------------------------------------------------------------
# 9. Eval, Sandbox, and Manifest Repository CRUD Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eval_run_repository(test_db: Database) -> None:
    """Test EvalRunRepository create and read operations."""
    repo = EvalRunRepository(test_db)

    created = await repo.log_eval_run(
        run_id="run-101",
        experiment_id="E2",
        status="completed",
        metrics={"f1": 0.94, "asr": 0.05},
    )
    assert created is not None
    assert created.run_id == "run-101"

    fetched = await repo.get_eval_run("run-101")
    assert fetched is not None
    assert fetched.experiment_id == "E2"
    assert fetched.metrics_json == {"f1": 0.94, "asr": 0.05}

    runs = await repo.list_eval_runs()
    assert len(runs) == 1
    d = runs[0].to_dict()
    assert d["run_id"] == "run-101"


@pytest.mark.asyncio
async def test_sandbox_episode_repository(test_db: Database) -> None:
    """Test SandboxEpisodeRepository create and read operations."""
    repo = SandboxEpisodeRepository(test_db)

    ep = await repo.log_episode(
        episode_id="ep-55",
        victim_model="gemma3:4b",
        attack_technique="direct_injection",
        guard_enabled=True,
        attack_succeeded=False,
        details={"steps": 3},
    )
    assert ep is not None

    fetched = await repo.get_episode("ep-55")
    assert fetched is not None
    assert fetched.victim_model == "gemma3:4b"
    assert fetched.attack_succeeded is False

    episodes = await repo.list_episodes()
    assert len(episodes) == 1
    assert episodes[0].to_dict()["episode_id"] == "ep-55"


@pytest.mark.asyncio
async def test_run_manifest_repository(test_db: Database) -> None:
    """Test RunManifestRepository create and read operations."""
    repo = RunManifestRepository(test_db)

    manifest = await repo.log_manifest(
        manifest_hash="manifesthash999",
        git_sha="b878a6b1234567890abcdef1234567890abcdef1",
        config={"batch_size": 32},
    )
    assert manifest is not None

    fetched = await repo.get_manifest("manifesthash999")
    assert fetched is not None
    assert fetched.git_sha.startswith("b878a6b")
    assert fetched.to_dict()["manifest_hash"] == "manifesthash999"


# ---------------------------------------------------------------------------
# 10. WriteBehindQueue Multi-Type Batch Drain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_behind_queue_multi_type_drain(test_db: Database) -> None:
    """Test WriteBehindQueue draining scans, eval runs, episodes, and manifests."""
    async with WriteBehindQueue(test_db, maxsize=50) as queue:
        queue.enqueue_scan(
            text="Queue scan test",
            decision="allow",
            latency_ms=11.2,
            results=[
                DetectorResult(
                    name="minilm", is_attack=False, score=0.01, latency_ms=3.0
                )
            ],
            trace_id="qtrace-1",
        )
        queue.enqueue_eval_run(
            run_id="qrun-1",
            experiment_id="E6",
            status="completed",
            metrics={"asr": 0.12},
        )
        queue.enqueue_episode(
            episode_id="qep-1",
            victim_model="qwen3:8b",
            attack_technique="obfuscation",
            guard_enabled=False,
            attack_succeeded=True,
            details={"canary_leaked": True},
        )
        queue.put(
            {
                "type": "manifest",
                "manifest_hash": "qman-1",
                "git_sha": "1234567890abcdef1234567890abcdef12345678",
                "config": {"env": "test"},
                "timestamp": datetime.now(UTC).replace(tzinfo=None),
            }
        )

        await queue.flush()

    assert queue.processed_count == 4
    assert queue.error_count == 0

    scan_repo = ScanRepository(test_db)
    scan = await scan_repo.get_scan_by_trace_id("qtrace-1")
    assert scan is not None
    assert len(scan.detector_results) == 1
    assert scan.detector_results[0].detector_name == "minilm"

    eval_repo = EvalRunRepository(test_db)
    assert await eval_repo.get_eval_run("qrun-1") is not None

    sb_repo = SandboxEpisodeRepository(test_db)
    assert await sb_repo.get_episode("qep-1") is not None

    man_repo = RunManifestRepository(test_db)
    assert await man_repo.get_manifest("qman-1") is not None


# ---------------------------------------------------------------------------
# 11. AsyncSessionContext Additional Coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_session_context_operations(test_db: Database) -> None:
    """Test get, delete, refresh, scalars, and rollback on AsyncSessionContext."""
    async with test_db.session() as session:
        scan = ScanRecord(
            text_hash="hash_ops",
            decision="allow",
            latency_ms=5.0,
            trace_id="ops-trace",
        )
        session.add(scan)
        await session.commit()
        await session.refresh(scan)
        scan_id = scan.id

    async with test_db.session() as session:
        fetched = await session.get(ScanRecord, scan_id)
        assert fetched is not None
        assert fetched.trace_id == "ops-trace"

        # Delete
        await session.delete(fetched)
        await session.commit()

    async with test_db.session() as session:
        deleted = await session.get(ScanRecord, scan_id)
        assert deleted is None

    # Test rollback on exception
    try:
        async with test_db.session() as session:
            session.add(
                ScanRecord(text_hash="rollback_hash", decision="allow", latency_ms=1.0)
            )
            raise ValueError("Trigger rollback")
    except ValueError:
        pass

    async with test_db.session() as session:
        res = await session.execute(
            text("SELECT count(*) FROM scans WHERE text_hash='rollback_hash'")
        )
        count = res.scalar()
        assert count == 0


@pytest.mark.asyncio
async def test_models_to_dict_and_repr(test_db: Database) -> None:
    """Test to_dict serialization on ScanRecord and DetectorResultRecord."""
    repo = ScanRepository(test_db)
    scan = await repo.log_scan(
        text="Sample prompt for dict",
        decision="allow",
        latency_ms=10.0,
        results=[
            {"name": "minilm", "is_attack": False, "score": 0.05, "latency_ms": 3.0}
        ],
        store_text=True,
    )
    assert scan is not None
    d = scan.to_dict()
    assert d["decision"] == "allow"
    assert d["text_truncated"] == "Sample prompt for dict"
    assert len(d["detector_results"]) == 1
    assert d["detector_results"][0]["detector_name"] == "minilm"


@pytest.mark.asyncio
async def test_queue_lifecycle_edge_cases(test_db: Database) -> None:
    """Test queue double-start, stop without starting, is_running, and enqueue alias."""
    queue = WriteBehindQueue(test_db, maxsize=10)
    assert not queue.is_running

    # Stop without starting
    await queue.stop()

    # Start
    await queue.start()
    assert queue.is_running
    # Double-start should be a no-op
    await queue.start()

    # Enqueue alias
    ok = await queue.enqueue({"type": "unknown_type"})
    assert ok is True

    await queue.flush()
    await queue.stop()
    assert not queue.is_running


@pytest.mark.asyncio
async def test_repositories_error_branches(test_db: Database) -> None:
    """Test safe=True and safe=False on EvalRun, SandboxEpisode, and RunManifest."""
    eval_repo = EvalRunRepository(test_db)
    sb_repo = SandboxEpisodeRepository(test_db)
    man_repo = RunManifestRepository(test_db)

    with patch.object(test_db, "session", side_effect=RuntimeError("Simulated outage")):
        # Safe=True swallows error
        assert await eval_repo.log_eval_run("r1", "E1", safe=True) is None
        assert (
            await sb_repo.log_episode("ep1", "m", "t", True, False, safe=True) is None
        )
        assert await man_repo.log_manifest("m1", "g1", safe=True) is None

        # Safe=False raises
        with pytest.raises(RuntimeError):
            await eval_repo.log_eval_run("r1", "E1", safe=False)
        with pytest.raises(RuntimeError):
            await sb_repo.log_episode("ep1", "m", "t", True, False, safe=False)
        with pytest.raises(RuntimeError):
            await man_repo.log_manifest("m1", "g1", safe=False)


@pytest.mark.asyncio
async def test_database_drop_all_and_aclose(temp_db_path: str) -> None:
    """Test adrop_all and aclose on Database manager."""
    db = Database(url=f"sqlite:///{temp_db_path}")
    await db.acreate_all()
    await db.adrop_all()
    await db.aclose()
