"""Initial schema for PARALLAX storage.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-10-03 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. scans table
    op.create_table(
        "scans",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("text_truncated", sa.Text(), nullable=True),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column(
            "disagreement", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("trace_id", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_scans_timestamp", "scans", ["timestamp"])
    op.create_index("ix_scans_text_hash", "scans", ["text_hash"])
    op.create_index("ix_scans_decision", "scans", ["decision"])
    op.create_index("ix_scans_disagreement", "scans", ["disagreement"])
    op.create_index("ix_scans_trace_id", "scans", ["trace_id"])

    # 2. detector_results table
    op.create_table(
        "detector_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("scan_id", sa.Integer(), nullable=False),
        sa.Column("detector_name", sa.String(length=64), nullable=False),
        sa.Column("is_attack", sa.Boolean(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("attack_class", sa.String(length=128), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["scan_id"], ["scans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_detector_results_scan_id", "detector_results", ["scan_id"])
    op.create_index(
        "ix_detector_results_detector_name", "detector_results", ["detector_name"]
    )

    # 3. eval_runs table
    op.create_table(
        "eval_runs",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("experiment_id", sa.String(length=64), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_eval_runs_experiment_id", "eval_runs", ["experiment_id"])
    op.create_index("ix_eval_runs_timestamp", "eval_runs", ["timestamp"])

    # 4. sandbox_episodes table
    op.create_table(
        "sandbox_episodes",
        sa.Column("episode_id", sa.String(length=64), nullable=False),
        sa.Column("victim_model", sa.String(length=128), nullable=False),
        sa.Column("attack_technique", sa.String(length=128), nullable=False),
        sa.Column("guard_enabled", sa.Boolean(), nullable=False),
        sa.Column("attack_succeeded", sa.Boolean(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("details_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("episode_id"),
    )
    op.create_index(
        "ix_sandbox_episodes_guard_enabled", "sandbox_episodes", ["guard_enabled"]
    )
    op.create_index(
        "ix_sandbox_episodes_attack_succeeded",
        "sandbox_episodes",
        ["attack_succeeded"],
    )
    op.create_index("ix_sandbox_episodes_timestamp", "sandbox_episodes", ["timestamp"])

    # 5. run_manifests table
    op.create_table(
        "run_manifests",
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("git_sha", sa.String(length=40), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("manifest_hash"),
    )
    op.create_index("ix_run_manifests_git_sha", "run_manifests", ["git_sha"])
    op.create_index("ix_run_manifests_timestamp", "run_manifests", ["timestamp"])


def downgrade() -> None:
    op.drop_table("run_manifests")
    op.drop_table("sandbox_episodes")
    op.drop_table("eval_runs")
    op.drop_table("detector_results")
    op.drop_table("scans")
