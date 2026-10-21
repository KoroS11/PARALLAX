#!/usr/bin/env python3
"""Judge Abstain Analysis & Forensic Audit (K4).

Analyzes the 10 attack rows with judge verdict abstain.
Calculates:
- Fisher's exact test p-value for abstains (10/139 vs 0/208)
- Latency comparison: abstain vs non-abstain
- Complete decision trace across arms: fusion, arm_a, arm_b, arm_r, arm_u, c2_standalone
- Verification of unified abstain protocol claims
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
OUTPUT_REPORT_PATH = REPO_ROOT / "docs" / "JUDGE_OUTCOMES_AUDIT.md"


def analyze_abstains() -> dict[str, Any]:
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    atks = [r for r in rows if r["label_binary"]]
    bens = [r for r in rows if not r["label_binary"]]

    abstain_atks = [r for r in atks if r["judge"]["verdict"] == "abstain"]
    non_abstain_atks = [r for r in atks if r["judge"]["verdict"] != "abstain"]

    abstain_bens = [r for r in bens if r["judge"]["verdict"] == "abstain"]
    non_abstain_bens = [r for r in bens if r["judge"]["verdict"] != "abstain"]

    # 1. Contingency table and Fisher exact test
    #              Abstain   Non-Abstain
    # Attack         10          129
    # Benign          0          208
    table = [
        [len(abstain_atks), len(non_abstain_atks)],
        [len(abstain_bens), len(non_abstain_bens)],
    ]
    odds_ratio, p_value = stats.fisher_exact(table)

    # 2. Latency comparison
    abstain_lats = [float(r["judge"]["latency_ms"]) for r in abstain_atks]
    non_abstain_lats = [
        float(r["judge"]["latency_ms"])
        for r in rows
        if r["judge"]["verdict"] != "abstain"
    ]

    p50_abstain = float(np.percentile(abstain_lats, 50))
    p95_abstain = float(np.percentile(abstain_lats, 95))
    mean_abstain = float(np.mean(abstain_lats))

    p50_non_abstain = float(np.percentile(non_abstain_lats, 50))
    p95_non_abstain = float(np.percentile(non_abstain_lats, 95))
    mean_non_abstain = float(np.mean(non_abstain_lats))

    # 3. Analyze the 10 rows
    row_details = []
    for r in abstain_atks:
        row_details.append(
            {
                "prompt_id": r["prompt_id"],
                "source": r["source"],
                "author": r.get("author", "n/a"),
                "family": r["family"],
                "scenario": r["scenario"],
                "fusion_decision": r["fusion"]["decision"],
                "arm_a": r["decisions"]["arm_a"],
                "arm_b": r["decisions"]["arm_b"],
                "arm_r": r["decisions"]["arm_r"],
                "arm_u": r["decisions"]["arm_u"],
                "c2_standalone": r["decisions"]["c2_standalone"],
                "router_should_route": r["router"]["should_route"],
                "trigger_b_triggered": r["trigger_arm_b"]["triggered"],
                "judge_confidence": r["judge"]["confidence"],
                "judge_latency_ms": r["judge"]["latency_ms"],
            }
        )

    return {
        "n_attacks_abstain": len(abstain_atks),
        "n_attacks_total": len(atks),
        "n_benign_abstain": len(abstain_bens),
        "n_benign_total": len(bens),
        "fisher_exact_p": float(p_value),
        "odds_ratio": float(odds_ratio),
        "latency_abstain": {
            "p50_ms": p50_abstain,
            "p95_ms": p95_abstain,
            "mean_ms": mean_abstain,
        },
        "latency_non_abstain": {
            "p50_ms": p50_non_abstain,
            "p95_ms": p95_non_abstain,
            "mean_ms": mean_non_abstain,
        },
        "rows": row_details,
    }


def main() -> int:
    data = analyze_abstains()
    print(
        f"Abstains: attacks {data['n_attacks_abstain']}/{data['n_attacks_total']}, benign {data['n_benign_abstain']}/{data['n_benign_total']}"
    )
    print(f"Fisher exact test p-value: {data['fisher_exact_p']:.6e}")
    print(
        f"Latency: Abstain mean={data['latency_abstain']['mean_ms']:.2f} ms (p50={data['latency_abstain']['p50_ms']:.2f} ms)"
    )
    print(
        f"Latency: Non-abstain mean={data['latency_non_abstain']['mean_ms']:.2f} ms (p50={data['latency_non_abstain']['p50_ms']:.2f} ms)"
    )

    # Print markdown table of 10 rows
    print(
        "\n| prompt_id | source | fusion | arm_a | arm_b | arm_r | arm_u | c2_standalone | router.should_route | trigger_arm_b |"
    )
    print("|---|---|---|---|---|---|---|---|:---:|:---:|")
    for r in data["rows"]:
        print(
            f"| {r['prompt_id']} | {r['source']} | {r['fusion_decision']} | {r['arm_a']} | {r['arm_b']} | {r['arm_r']} | {r['arm_u']} | {r['c2_standalone']} | {r['router_should_route']} | {r['trigger_b_triggered']} |"
        )

    return 0


if __name__ == "__main__":
    main()
