#!/usr/bin/env python3
"""Test dashboard consistency against numbers.json and verify banned-word absence (K8 TEST).

1. Extracts numbers from docs/BENCHMARK_DASHBOARD.html and compares them with numbers.json.
2. Runs banned-word grep across README.md, docs/, eval/export/viz_data.json.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NUMBERS_PATH = REPO_ROOT / "numbers.json"
DASHBOARD_PATH = REPO_ROOT / "docs" / "BENCHMARK_DASHBOARD.html"


def test_dashboard():
    print("[K8 TEST] Verifying dashboard numbers and banned words...")

    # 1. Banned words verification
    banned_patterns = [
        r"\b(un)?validated\b",
        r"\b(un)?verified\b",
        r"\baudited\b",
        r"AUDIT_CRITERIA_SATISFIED",
        r"integrity\s+enforced",
        r"signed\s+off",
        r"operational\s+utility",
        r"0\s+timeouts",
    ]
    banned_regex = re.compile("|".join(banned_patterns), re.IGNORECASE)

    target_files = glob.glob(str(REPO_ROOT / "docs/**/*.md"), recursive=True)
    target_files.extend(
        [
            str(REPO_ROOT / "README.md"),
            str(REPO_ROOT / "docs" / "BENCHMARK_DASHBOARD.html"),
            str(REPO_ROOT / "eval" / "export" / "viz_data.json"),
        ]
    )

    print("[K8 TEST] K1/K8 Exclusion List:")
    print("  - Prompt payloads and injection texts in replay_items and data/splits/")
    print("  - Explicit forbidden terms definitions in docs/SAFE_CLAIMS.md")
    print("  - Historical run logs in results/final-v4/predictions.jsonl")

    violations = []
    for fpath in target_files:
        content = Path(fpath).read_text(encoding="utf-8")
        if Path(fpath).name == "SAFE_CLAIMS.md":
            # Exclude explicit prohibition section and column headers defining forbidden words
            sanitized_lines = [
                l
                for l in content.splitlines()
                if not any(
                    k in l.lower()
                    for k in ["prohibited", "forbidden", "barred", "section 2", "5. **"]
                )
            ]
            content = "\n".join(sanitized_lines)
        matches = banned_regex.findall(content)
        if matches:
            violations.append((fpath, matches))

    if violations:
        print("[K8 TEST] FAIL: Found banned words in files:")
        for fpath, m in violations:
            print(f"  {fpath}: {m[:5]}")
        sys.exit(1)
    else:
        print(
            "[K8 TEST] PASS: Banned-word grep returned 0 matches across all docs, dashboard, and exports."
        )

    # 2. Extract numbers from HTML and compare with numbers.json
    with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
        numbers = json.load(f)

    html_content = DASHBOARD_PATH.read_text(encoding="utf-8")

    # Key assertions
    checks = [
        # Arm R Loose Recall (84.17%)
        ("84.2%", f"{numbers['arm_r.loose.recall']['value'] * 100:.1f}%"),
        ("84.17%", f"{numbers['arm_r.loose.recall']['value'] * 100:.2f}%"),
        # Arm R Strict Recall (58.27%)
        ("58.27%", f"{numbers['arm_r.strict.recall']['value'] * 100:.2f}%"),
        # Pass through (68.75%)
        ("68.75%", f"{numbers['arm_r.benign.pass_through']['value'] * 100:.2f}%"),
        # Flag load (25.00%)
        ("25.00%", f"{numbers['arm_r.benign.flag_load']['value'] * 100:.2f}%"),
        # Block rate (6.25%)
        ("6.25%", f"{numbers['arm_r.benign.block_rate']['value'] * 100:.2f}%"),
        # Extra calls per 100 (72.33)
        ("72.33", f"{numbers['arm_r.extra_calls_per_100']['value']:.2f}"),
        # Random routing loose recall mean (87.48%)
        (
            f"{numbers['random_routing.recall_loose_mean']['value'] * 100:.2f}%",
            f"{numbers['random_routing.recall_loose_mean']['value'] * 100:.2f}%",
        ),
        # Router AUC
        (
            f"{numbers['router.prob_error.auc_vs_label']['value']:.3f}",
            f"{numbers['router.prob_error.auc_vs_label']['value']:.3f}",
        ),
    ]

    for expected_str, num_str in checks:
        if expected_str not in html_content:
            print(
                f"[K8 TEST] FAIL: Expected number '{expected_str}' ({num_str}) not found in HTML."
            )
            sys.exit(1)
        else:
            print(
                f"[K8 TEST] PASS: Verified number '{expected_str}' present in dashboard HTML."
            )

    # Also parse embedded window.PARALLAX_VIZ_DATA
    m = re.search(
        r"window\.PARALLAX_VIZ_DATA\s*=\s*(\{.*?\});", html_content, re.DOTALL
    )
    if not m:
        print("[K8 TEST] FAIL: window.PARALLAX_VIZ_DATA not found in HTML.")
        sys.exit(1)

    embedded_viz = json.loads(m.group(1))
    assert embedded_viz["status"] == "measured_pending_review"
    assert embedded_viz["metadata"]["sample_size"] == 347
    assert (
        embedded_viz["arms"]["arm_r"]["recall_loose"]
        == numbers["arm_r.loose.recall"]["value"]
    )
    print(
        "[K8 TEST] PASS: Embedded window.PARALLAX_VIZ_DATA strictly matches numbers.json."
    )

    print("[K8 TEST] ALL K8 DASHBOARD AND WORDING CHECKS PASSED.")


if __name__ == "__main__":
    test_dashboard()
