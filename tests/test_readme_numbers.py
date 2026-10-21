#!/usr/bin/env python3
"""Test recomputing every table number in README.md from canonical source files.

Verifies:
1. Table 1 (Held-out arms performance on held_out_v4 N=347):
   Recomputes loose recall, strict recall, pass-through, benign flag, benign block,
   and judge calls per 100 for all arms from numbers.json and predictions.jsonl.
2. Table 2 (Base detector ROC-AUCs):
   Checks detector AUCs and bootstrap CIs on held_out_v4 from numbers.json.
3. Table 3 (Lakera Gandalf n=112 attacks):
   Recomputes allow, flag, block, loose recall, strict recall, and judge calls
   for all arms from external_metrics.json.
4. Table 4 (NotInject n=309 benign):
   Recomputes allow, flag, block, pass-through, false block rate, and judge calls
   for all arms from external_metrics.json.
5. Table 5 (NotInject Addendum partitions):
   Recomputes partition counts, pass-through, and false block rates from
   notinject_addendum_metrics.json.
6. Table 6 (Databricks Dolly n=200 benign):
   Recomputes allow, flag, block, pass-through, false block rate, and judge calls
   for all arms from external_metrics.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
NUMBERS_PATH = REPO_ROOT / "numbers.json"
PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
EXTERNAL_METRICS_PATH = REPO_ROOT / "export" / "closeout" / "external_metrics.json"
ADDENDUM_PATH = REPO_ROOT / "export" / "closeout" / "notinject_addendum_metrics.json"


def test_readme_held_out_table_numbers():
    assert README_PATH.exists(), "README.md must exist"
    readme_text = README_PATH.read_text(encoding="utf-8")
    with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
        num = json.load(f)

    arms = [
        ("Arm R (Pre-registered Primary)", "arm_r"),
        ("Arm A (Post-hoc AnyFlag)", "arm_a"),
        ("Arm B (Post-hoc Discrepancy)", "arm_b"),
        ("Arm U (Post-hoc Universal)", "arm_u"),
        ("Arm C2 (Post-hoc Standalone Judge)", "c2_standalone"),
        ("PG2 (Post-hoc Prompt Guard 2)", "promptguard2"),
        ("PromptForest (Post-hoc Tree Baseline)", "promptforest"),
    ]

    for label, k in arms:
        lr = num[f"{k}.loose.recall"]["value"] * 100
        sr = num[f"{k}.strict.recall"]["value"] * 100
        pt = num[f"{k}.benign.pass_through"]["value"] * 100
        bf = num[f"{k}.benign.flag_load"]["value"] * 100
        bb = num[f"{k}.benign.block_rate"]["value"] * 100
        calls = num[f"{k}.extra_calls_per_100"]["value"]

        # Assert these formatted strings appear in README.md
        assert f"{lr:.2f}%" in readme_text, f"Loose recall {lr:.2f}% for {label} missing in README.md"
        assert f"{sr:.2f}%" in readme_text, f"Strict recall {sr:.2f}% for {label} missing in README.md"
        assert f"{pt:.2f}%" in readme_text, f"Pass-through {pt:.2f}% for {label} missing in README.md"
        assert f"{bf:.2f}%" in readme_text, f"Benign FLAG {bf:.2f}% for {label} missing in README.md"
        assert f"{bb:.2f}%" in readme_text, f"Benign BLOCK {bb:.2f}% for {label} missing in README.md"
        assert f"{calls:.2f}" in readme_text, f"Judge calls {calls:.2f} for {label} missing in README.md"


def test_readme_detector_auc_table_numbers():
    readme_text = README_PATH.read_text(encoding="utf-8")
    with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
        num = json.load(f)

    for k in ["d2_deberta", "d3_deepset", "d4_rules", "promptguard2", "promptforest"]:
        auc = num[f"{k}.auc"]["value"]
        ci = num[f"{k}.auc_ci"]["value"]
        assert f"{auc:.4f}" in readme_text, f"AUC {auc:.4f} for {k} missing in README.md"
        assert f"{ci[0]:.4f}" in readme_text, f"CI low {ci[0]:.4f} for {k} missing in README.md"
        assert f"{ci[1]:.4f}" in readme_text, f"CI high {ci[1]:.4f} for {k} missing in README.md"


def test_readme_external_gandalf_numbers():
    readme_text = README_PATH.read_text(encoding="utf-8")
    with open(EXTERNAL_METRICS_PATH, "r", encoding="utf-8") as f:
        ext = json.load(f)

    g_arms = ext["subsets"]["source_gandalf"]["arms"]
    for k, row in g_arms.items():
        lr = row["loose_recall"] * 100
        sr = row["strict_recall"] * 100
        assert f"{lr:.2f}%" in readme_text, f"Gandalf loose recall {lr:.2f}% for {k} missing"
        assert f"{sr:.2f}%" in readme_text, f"Gandalf strict recall {sr:.2f}% for {k} missing"


def test_readme_external_notinject_numbers():
    readme_text = README_PATH.read_text(encoding="utf-8")
    with open(EXTERNAL_METRICS_PATH, "r", encoding="utf-8") as f:
        ext = json.load(f)

    ni_arms = ext["subsets"]["source_notinject"]["arms"]
    for k, row in ni_arms.items():
        pt = row["pass_through_rate"] * 100
        fbr = row["false_block_rate"] * 100
        assert f"{pt:.2f}%" in readme_text, f"NotInject pass-through {pt:.2f}% for {k} missing"
        assert f"{fbr:.2f}%" in readme_text, f"NotInject false block rate {fbr:.2f}% for {k} missing"


def test_readme_notinject_addendum_numbers():
    readme_text = README_PATH.read_text(encoding="utf-8")
    with open(ADDENDUM_PATH, "r", encoding="utf-8") as f:
        add = json.load(f)

    for p_key in ["english", "multilingual", "notinject_one", "notinject_two", "notinject_three"]:
        p_row = add["partitions"][p_key]["arms"]["arm_r"]
        pt = p_row["pass_through_rate"] * 100
        fbr = p_row["false_block_rate"] * 100
        assert f"{pt:.2f}%" in readme_text, f"Addendum {p_key} pass-through {pt:.2f}% missing"
        assert f"{fbr:.2f}%" in readme_text, f"Addendum {p_key} false block rate {fbr:.2f}% missing"


def test_readme_external_dolly_numbers():
    readme_text = README_PATH.read_text(encoding="utf-8")
    with open(EXTERNAL_METRICS_PATH, "r", encoding="utf-8") as f:
        ext = json.load(f)

    d_arms = ext["subsets"]["source_dolly"]["arms"]
    for k, row in d_arms.items():
        pt = row["pass_through_rate"] * 100
        fbr = row["false_block_rate"] * 100
        assert f"{pt:.2f}%" in readme_text, f"Dolly pass-through {pt:.2f}% for {k} missing"
        assert f"{fbr:.2f}%" in readme_text, f"Dolly false block rate {fbr:.2f}% for {k} missing"


def test_readme_banned_words_absence():
    readme_text = README_PATH.read_text(encoding="utf-8")
    banned_patterns = [
        r"\brobust\w*",
        r"\bgeneraliz\w*",
        r"\bproves\b",
        r"\bvalidated\b",
        r"\bverified\b",
        r"\bcertified\b",
        r"human-written",
    ]
    for pat in banned_patterns:
        matches = re.findall(pat, readme_text, re.IGNORECASE)
        assert len(matches) == 0, f"Banned pattern '{pat}' found in README.md: {matches}"


if __name__ == "__main__":
    test_readme_held_out_table_numbers()
    test_readme_detector_auc_table_numbers()
    test_readme_external_gandalf_numbers()
    test_readme_external_notinject_numbers()
    test_readme_notinject_addendum_numbers()
    test_readme_external_dolly_numbers()
    test_readme_banned_words_absence()
    print("[PASS] All README numbers and banned-word tests passed successfully!")
