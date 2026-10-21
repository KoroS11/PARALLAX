"""Unit tests for PARALLAX Wave G Sample Size and Statistical Integrity Audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.stats_audit.run_audit import (
    compute_cluster_bootstrap,
    compute_cluster_stats,
)

OUTPUT_JSON = ROOT_DIR / "eval" / "stats_audit" / "sample_size_audit.json"
OUTPUT_MD = ROOT_DIR / "docs" / "AUDIT_SAMPLE_SIZES.md"


def test_cluster_bootstrap_ci_deterministic() -> None:
    """Verify cluster bootstrap produces deterministic, valid 95% intervals."""
    # 4 clusters, varying performance
    clusters = ["c1"] * 10 + ["c2"] * 10 + ["c3"] * 10 + ["c4"] * 10
    y_true = [True] * 40
    # c1: 100%, c2: 50%, c3: 0%, c4: 100% -> total 25/40 = 62.5%
    y_pred = [True] * 10 + [True] * 5 + [False] * 5 + [False] * 10 + [True] * 10

    pt, (low, high) = compute_cluster_bootstrap(
        y_true, y_pred, clusters, metric_type="recall", n_boot=500, seed=42
    )
    assert pt == 0.625
    assert 0.0 <= low <= pt <= high <= 1.0
    assert low < high


def test_cluster_stats_and_neff() -> None:
    """Verify Kish design effect and effective sample size calculation."""
    # Perfectly homogeneous across clusters: all 1.0 -> ICC should be 0.0, DEFF = 1.0, n_eff = N
    clusters = ["c1"] * 10 + ["c2"] * 10
    y = [1.0] * 20
    stats = compute_cluster_stats(y, clusters)
    assert stats["n"] == 20
    assert stats["clusters"] == 2
    assert stats["rho_anova"] == 0.0
    assert stats["deff_anova"] == 1.0
    assert stats["n_eff"] == 20.0

    # Strong cluster effect: c1 all 1.0, c2 all 0.0 -> ICC = 1.0
    y_clustered = [1.0] * 10 + [0.0] * 10
    stats_clustered = compute_cluster_stats(y_clustered, clusters)
    assert stats_clustered["rho_anova"] == 1.0
    assert stats_clustered["deff_anova"] > 1.0
    assert stats_clustered["n_eff"] < 20.0


def test_audit_artifacts_exist_and_valid() -> None:
    """Verify sample_size_audit.json and docs/AUDIT_SAMPLE_SIZES.md exist and conform."""
    assert OUTPUT_JSON.exists(), f"Missing {OUTPUT_JSON}"
    assert OUTPUT_MD.exists(), f"Missing {OUTPUT_MD}"

    data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert "table_4_1" in data
    assert "table_8_2" in data
    assert "table_8_3" in data
    assert "table_9_2" in data
    assert "table_9_3" in data
    assert "table_9_4" in data
    assert "table_9_5" in data

    # Check that under-powered disclaimer is present on n < 100
    for r in data["table_4_1"]["rows"]:
        assert r["disclaimer"] == "illustrative only, under-powered"
        assert r["n"] < 100

    for r in data["table_8_2"]["rows"]:
        assert r["disclaimer"] == "illustrative only, under-powered"
        assert r["n_attacks"] == 90
        assert r["n_benign"] == 64
        assert r["metrics"]["asr"]["ci_95_1000"] is not None
        assert len(r["metrics"]["asr"]["ci_95_1000"]) == 2

    for r in data["table_8_3"]["rows"]:
        assert r["disclaimer"] == "illustrative only, under-powered"
        assert r["n"] < 30

    md_text = OUTPUT_MD.read_text(encoding="utf-8")
    assert (
        "PARALLAX Wave G: Comprehensive Statistical Integrity & Sample Size Audit"
        in md_text
    )
    assert "illustrative only, under-powered" in md_text
    assert "n_eff" in md_text
    assert "1,000" in md_text
