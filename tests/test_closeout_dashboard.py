"""Test suite for Phase 5 Dashboard (export/closeout/dashboard.html)."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NUMBERS_PATH = REPO_ROOT / "numbers.json"
DASHBOARD_PATH = REPO_ROOT / "export" / "closeout" / "dashboard.html"

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


def test_dashboard_file_exists():
    """Verify that dashboard.html exists and is non-empty."""
    assert DASHBOARD_PATH.exists(), f"Dashboard missing at {DASHBOARD_PATH}"
    assert DASHBOARD_PATH.stat().st_size > 50000, "Dashboard file is unexpectedly small"


def test_no_external_network_requests():
    """Verify dashboard has zero external script or style requests."""
    content = DASHBOARD_PATH.read_text(encoding="utf-8")
    assert "<script src=" not in content, "External script link found in dashboard"
    assert '<link rel="stylesheet" href="http' not in content, (
        "External stylesheet found in dashboard"
    )


def test_top_banner_present():
    """Verify exact top banner wording."""
    content = DASHBOARD_PATH.read_text(encoding="utf-8")
    expected_banner = "held_out_v4 reused for design review; external sets scored once"
    assert expected_banner in content, (
        f"Missing required top banner: '{expected_banner}'"
    )


def test_dashboard_numbers_equal_source():
    """Verify that every number in the embedded JSON equals numbers.json."""
    content = DASHBOARD_PATH.read_text(encoding="utf-8")

    # Extract embedded JSON
    match = re.search(
        r'<script id="dashboard-data" type="application/json">\s*(\{.*?\})\s*</script>',
        content,
        re.DOTALL,
    )
    assert match, "Could not locate embedded dashboard data JSON"
    data = json.loads(match.group(1))

    with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
        numbers = json.load(f)

    # Check primary arm
    p = data["primary_arm"]
    assert p["loose_recall"] == round(numbers["arm_r.loose.recall"]["value"] * 100.0, 2)
    assert p["strict_recall"] == round(
        numbers["arm_r.strict.recall"]["value"] * 100.0, 2
    )
    assert p["pass_through"] == round(
        numbers["arm_r.benign.pass_through"]["value"] * 100.0, 2
    )
    assert p["block_rate"] == round(
        numbers["arm_r.benign.block_rate"]["value"] * 100.0, 2
    )
    assert p["extra_calls"] == round(numbers["arm_r.extra_calls_per_100"]["value"], 2)

    # Check router
    r = data["router"]
    assert r["auc_error"] == round(
        numbers["router.prob_error.auc_vs_fusion_error"]["value"], 4
    )
    assert r["auc_label"] == round(
        numbers["router.prob_error.auc_vs_label"]["value"], 4
    )


def test_no_hardcoded_metric_literals_outside_json():
    """Verify no hardcoded metric literals exist in HTML/JS outside embedded JSON."""
    content = DASHBOARD_PATH.read_text(encoding="utf-8")
    parts = content.split('<script id="dashboard-data" type="application/json">')
    html_before = parts[0]
    html_after = parts[1].split("</script>")[1]
    outer_content = html_before + html_after

    literals = ["84.17", "58.27", "68.75", "25.00"]
    for lit in literals:
        assert lit not in outer_content, (
            f"Hardcoded metric literal '{lit}' found outside embedded JSON"
        )


def test_zero_banned_words_in_dashboard(capsys):
    """Verify zero banned words in dashboard.html outside K1/K8 exclusions."""
    exclusion_list = [
        "Prompt payloads and injection texts in replay_items",
        "Raw corpus files under data/splits/",
        "Historical evaluation logs in results/final-v4/predictions.jsonl",
    ]
    print("\n[BANNED-WORD SCAN] K1/K8 Exclusion List:")
    for exc in exclusion_list:
        print(f"  - {exc}")

    content = DASHBOARD_PATH.read_text(encoding="utf-8")

    # Follow K1/K8 exclusion: remove prompt_preview strings from replay_items in embedded JSON
    sanitized_content = re.sub(
        r'"prompt_preview":\s*".*?"', '"prompt_preview": "[EXCLUDED_PAYLOAD]"', content
    ).lower()

    for word in BANNED_WORDS:
        assert word.lower() not in sanitized_content, (
            f"Banned word '{word}' found in dashboard.html outside K1/K8 exclusion"
        )
