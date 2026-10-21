#!/usr/bin/env python3
"""Post-Hoc Addendum: NotInject Subgroup & Category Decomposition.

Joins external predictions (export/closeout/external_predictions.jsonl) with
data/external/notinject.jsonl by prompt_id to compute English vs Multilingual
and Trigger-Word Load subsets across all 5 evaluation arms.

NOTE ON ORIGINAL METRICS SCRIPT DISCREPANCY:
In scripts/eval/external_metrics.py (lines 205-210), the category filter looked for:
    r.get("category") == "Multilingual Queries"
However:
1. In data/external/notinject.jsonl, the category string is "Multilingual" (84 rows),
   while "Multilingual Queries" matches 0 rows.
2. The scoring runner did not propagate the "category" or "subset" fields into
   export/closeout/external_predictions.jsonl.
Consequently, the pre-registered runner reported n=0 for the multilingual category.
This post-hoc addendum joins predictions with notinject.jsonl by prompt_id to provide
the verified subgroup breakdown.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTINJECT_PATH = PROJECT_ROOT / "data" / "external" / "notinject.jsonl"
PREDICTIONS_PATH = PROJECT_ROOT / "export" / "closeout" / "external_predictions.jsonl"
OUTPUT_JSON_PATH = (
    PROJECT_ROOT / "export" / "closeout" / "notinject_addendum_metrics.json"
)

ARMS = ["arm_r", "arm_a", "arm_b", "arm_u", "c2_standalone"]


def wilson_ci(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute 95% Wilson score interval for binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def run_addendum() -> dict[str, Any]:
    if not NOTINJECT_PATH.exists():
        raise FileNotFoundError(f"Missing {NOTINJECT_PATH}")
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {PREDICTIONS_PATH}")

    # 1. Load NotInject ground truth catalog
    with open(NOTINJECT_PATH, "r", encoding="utf-8") as f:
        notinject_catalog = [json.loads(line) for line in f if line.strip()]

    ni_map = {r["id"]: r for r in notinject_catalog}
    assert len(ni_map) == 309, f"Expected 309 rows, got {len(ni_map)}"

    # Check exact count of 'Multilingual Queries' vs 'Multilingual'
    count_multilingual_queries = sum(
        1 for r in notinject_catalog if r.get("category") == "Multilingual Queries"
    )
    count_multilingual = sum(
        1 for r in notinject_catalog if r.get("category") == "Multilingual"
    )

    # 2. Load Predictions Log
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    joined_rows = []
    for r in predictions:
        pid = r.get("prompt_id")
        if pid in ni_map:
            joined_rows.append((r, ni_map[pid]))

    assert len(joined_rows) == 309, f"Expected 309 joined rows, got {len(joined_rows)}"

    # 3. Define Partitions
    partitions = {
        "english": [
            pair for pair in joined_rows if pair[1].get("category") != "Multilingual"
        ],
        "multilingual": [
            pair for pair in joined_rows if pair[1].get("category") == "Multilingual"
        ],
        "notinject_one": [
            pair for pair in joined_rows if pair[1].get("subset") == "NotInject_one"
        ],
        "notinject_two": [
            pair for pair in joined_rows if pair[1].get("subset") == "NotInject_two"
        ],
        "notinject_three": [
            pair for pair in joined_rows if pair[1].get("subset") == "NotInject_three"
        ],
    }

    assert len(partitions["english"]) == 225, (
        f"Expected 225 English, got {len(partitions['english'])}"
    )
    assert len(partitions["multilingual"]) == 84, (
        f"Expected 84 Multilingual, got {len(partitions['multilingual'])}"
    )
    assert len(partitions["notinject_one"]) == 101
    assert len(partitions["notinject_two"]) == 101
    assert len(partitions["notinject_three"]) == 107

    from collections import Counter

    cat_counter = Counter(r.get("category") for r in notinject_catalog)
    sub_counter = Counter(r.get("subset") for r in notinject_catalog)

    results: dict[str, Any] = {
        "metadata": {
            "title": "NotInject Subgroup & Category Decomposition (Post-Hoc Addendum)",
            "total_notinject_rows": 309,
            "raw_category_multilingual_queries_count": count_multilingual_queries,
            "raw_category_multilingual_count": count_multilingual,
            "stored_category_counts": dict(cat_counter),
            "stored_subset_counts": dict(sub_counter),
            "english_partition_status": "derived (all rows where category != 'Multilingual')",
            "k13_select_specification": (
                "K13_SELECT.txt names 'Multilingual Queries' per card taxonomy, but the "
                "dataset stores 'Multilingual'; original split was empty because of that string mismatch."
            ),
            "root_cause_empty_original_split": (
                "scripts/eval/external_metrics.py checked category == 'Multilingual Queries', "
                "which matches 0 rows in notinject.jsonl (category is 'Multilingual', n=84), "
                "and prediction records did not copy the category field from raw inputs."
            ),
        },
        "partitions": {},
    }

    print("=" * 80)
    print("NOTINJECT POST-HOC SUBGROUP & CATEGORY ADDENDUM")
    print(f"Total Evaluated NotInject Prompts: {len(joined_rows)}")
    print(f"Query for 'Multilingual Queries': {count_multilingual_queries} matches")
    print(
        f"Query for 'Multilingual': {count_multilingual} matches (English = {len(partitions['english'])})"
    )
    print("=" * 80)

    for part_key, pairs in partitions.items():
        n = len(pairs)
        results["partitions"][part_key] = {"sample_size": n, "arms": {}}
        print(f"\n--- Partition: {part_key} (n={n}) ---")

        for arm in ARMS:
            allows = sum(1 for p, _ in pairs if p["decisions"][arm] == "allow")
            flags = sum(1 for p, _ in pairs if p["decisions"][arm] == "flag")
            blocks = sum(1 for p, _ in pairs if p["decisions"][arm] == "block")

            pt_rate = allows / n
            pt_ci = wilson_ci(allows, n)
            fbr_rate = blocks / n
            fbr_ci = wilson_ci(blocks, n)
            flag_rate = flags / n
            flag_ci = wilson_ci(flags, n)

            results["partitions"][part_key]["arms"][arm] = {
                "allow": allows,
                "flag": flags,
                "block": blocks,
                "pass_through_rate": round(pt_rate, 4),
                "pass_through_95ci": [round(pt_ci[0], 4), round(pt_ci[1], 4)],
                "false_block_rate": round(fbr_rate, 4),
                "false_block_95ci": [round(fbr_ci[0], 4), round(fbr_ci[1], 4)],
                "flag_rate": round(flag_rate, 4),
                "flag_95ci": [round(flag_ci[0], 4), round(flag_ci[1], 4)],
            }

            print(
                f"  {arm:14s}: allow={allows:3d}, flag={flags:3d}, block={blocks:3d} | "
                f"Pass={pt_rate * 100:5.2f}% [{pt_ci[0] * 100:5.2f}%, {pt_ci[1] * 100:5.2f}%] | "
                f"FBR={fbr_rate * 100:5.2f}% [{fbr_ci[0] * 100:5.2f}%, {fbr_ci[1] * 100:5.2f}%]"
            )

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved addendum metrics to {OUTPUT_JSON_PATH}")

    return results


if __name__ == "__main__":
    run_addendum()
