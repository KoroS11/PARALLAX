"""SQLAlchemy 2.0 database models for PARALLAX storage layer.

Defines ORM entities for scans, detector results, eval runs, sandbox episodes,
and run manifests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utc_now() -> datetime:
    """Return current UTC datetime (timezone-naive for standard DB storage)."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Declarative base class for all PARALLAX storage models."""


class ScanRecord(Base):
    """Storage record for an individual prompt scan verdict."""

    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=_utc_now, nullable=False, index=True
    )
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    text_truncated: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    disagreement: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # 1-to-many relationship with per-detector results
    detector_results: Mapped[list[DetectorResultRecord]] = relationship(
        "DetectorResultRecord",
        back_populates="scan",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary view."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "text_hash": self.text_hash,
            "text_truncated": self.text_truncated,
            "decision": self.decision,
            "latency_ms": self.latency_ms,
            "disagreement": self.disagreement,
            "trace_id": self.trace_id,
            "detector_results": [r.to_dict() for r in self.detector_results],
        }


class DetectorResultRecord(Base):
    """Storage record for an individual detector's output in a scan."""

    __tablename__ = "detector_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("scans.id", ondelete="CASCADE"), nullable=False, index=True
    )
    detector_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    is_attack: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    attack_class: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    scan: Mapped[ScanRecord] = relationship(
        "ScanRecord", back_populates="detector_results"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary view."""
        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "detector_name": self.detector_name,
            "is_attack": self.is_attack,
            "score": self.score,
            "attack_class": self.attack_class,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class EvalRunRecord(Base):
    """Storage record for evaluation runs and summary metrics."""

    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=_utc_now, nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary view."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "status": self.status,
            "metrics_json": self.metrics_json or {},
        }


class SandboxEpisodeRecord(Base):
    """Storage record for a red-team / sandbox evaluation episode."""

    __tablename__ = "sandbox_episodes"

    episode_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    victim_model: Mapped[str] = mapped_column(String(128), nullable=False)
    attack_technique: Mapped[str] = mapped_column(String(128), nullable=False)
    guard_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    attack_succeeded: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=_utc_now, nullable=False, index=True
    )
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary view."""
        return {
            "episode_id": self.episode_id,
            "victim_model": self.victim_model,
            "attack_technique": self.attack_technique,
            "guard_enabled": self.guard_enabled,
            "attack_succeeded": self.attack_succeeded,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "details_json": self.details_json or {},
        }


class RunManifestRecord(Base):
    """Storage record for execution manifests (git commit, config, audit)."""

    __tablename__ = "run_manifests"

    manifest_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    git_sha: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=_utc_now, nullable=False, index=True
    )
    config_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary view."""
        return {
            "manifest_hash": self.manifest_hash,
            "git_sha": self.git_sha,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "config_json": self.config_json or {},
        }
