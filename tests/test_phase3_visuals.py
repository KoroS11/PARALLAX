"""Test verification for Phase 3 Visuals, Tables, Sidecars, and Number Sheet."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "export" / "closeout" / "figures"
TABLES_DIR = REPO_ROOT / "export" / "closeout" / "tables"
NUMBERS_SHEET_PATH = REPO_ROOT / "export" / "closeout" / "NUMBERS_SHEET.md"
SOURCES_JSON_PATH = REPO_ROOT / "scripts" / "eval" / "figure_sources.json"
NUMBERS_PATH = REPO_ROOT / "numbers.json"

BANNED_WORDS = [
    "robust",
    "generalizes",
    "proves",
    "validated",
    "verified",
    "audited",
    "AUDIT_CRITERIA_SATISFIED",
    "integrity enforced",
    "signed off",
    "operational utility",
    "0 timeouts",
]


def test_f1_to_f15_figures_exist():
    """Verify that all figures F1 through F15 exist as PNG and SVG in dark and light themes."""
    fig_ids = [f"f{i}" for i in range(1, 16)]

    for fid in fig_ids:
        # Default png/svg
        matches_png = list(FIGURES_DIR.glob(f"{fid}_*.png"))
        matches_svg = list(FIGURES_DIR.glob(f"{fid}_*.svg"))
        assert len(matches_png) >= 2, f"Missing PNGs for {fid}: found {matches_png}"
        assert len(matches_svg) >= 2, f"Missing SVGs for {fid}: found {matches_svg}"


def test_f1_to_f15_tables_exist():
    """Verify that CSV tables exist for all figures."""
    expected_tables = [
        "f1_pipeline_stages.csv",
        "f2_data_composition.csv",
        "f3_detector_aucs.csv",
        "f4_held_out_arms.csv",
        "f5_pareto_frontier.csv",
        "f6_stacked_decisions.csv",
        "f7_router_performance.csv",
        "f8_judge_outcomes.csv",
        "f9_abstain_rows.csv",
        "f10_cost_vs_recall.csv",
        "f11_error_analysis.csv",
        "f12_confusion_matrices.csv",
        "f13_contamination_matrix.csv",
        "f14_provenance_timeline.csv",
        "f15_audit_defects.csv",
    ]
    for tbl in expected_tables:
        p = TABLES_DIR / tbl
        assert p.exists(), f"Table CSV missing: {tbl}"
        assert p.stat().st_size > 0, f"Table CSV is empty: {tbl}"


def test_figure_sources_sidecar_assertions():
    """Verify figure_sources.json contains entries and passing assertions for F1-F15."""
    assert SOURCES_JSON_PATH.exists(), f"Missing {SOURCES_JSON_PATH}"
    with open(SOURCES_JSON_PATH, "r", encoding="utf-8") as f:
        sidecars = json.load(f)

    for i in range(1, 16):
        key = f"F{i}"
        assert key in sidecars, f"Missing sidecar entry for {key}"
        sc = sidecars[key]
        assert "title" in sc
        assert "split" in sc
        assert "n" in sc
        assert "source_files" in sc
        assert "source_sha256" in sc
        assert "assertions" in sc
        assert len(sc["assertions"]) > 0


def test_numbers_sheet_traceability():
    """Verify NUMBERS_SHEET.md numbers match numbers.json."""
    assert NUMBERS_SHEET_PATH.exists()
    content = NUMBERS_SHEET_PATH.read_text(encoding="utf-8")

    with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
        num = json.load(f)

    r_loose = f"{num['arm_r.loose.recall']['value'] * 100:.2f}%"
    r_strict = f"{num['arm_r.strict.recall']['value'] * 100:.2f}%"
    r_pt = f"{num['arm_r.benign.pass_through']['value'] * 100:.2f}%"
    r_fbr = f"{num['arm_r.benign.block_rate']['value'] * 100:.2f}%"

    assert r_loose in content, (
        f"Arm R loose recall {r_loose} not found in NUMBERS_SHEET.md"
    )
    assert r_strict in content, (
        f"Arm R strict recall {r_strict} not found in NUMBERS_SHEET.md"
    )
    assert r_pt in content, f"Arm R pass through {r_pt} not found in NUMBERS_SHEET.md"
    assert r_fbr in content, f"Arm R FBR {r_fbr} not found in NUMBERS_SHEET.md"


def test_no_banned_words_in_numbers_sheet():
    """Verify zero banned words in NUMBERS_SHEET.md."""
    content = NUMBERS_SHEET_PATH.read_text(encoding="utf-8").lower()
    for word in BANNED_WORDS:
        assert word.lower() not in content, (
            f"Banned word '{word}' found in NUMBERS_SHEET.md"
        )
