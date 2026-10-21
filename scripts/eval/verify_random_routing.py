#!/usr/bin/env python3
"""Random Routing Simulation & Primary Arm Verification (K3 TEST).

Simulates random routing at equal budget (k=251 calls / 347, seed=0, 2000 draws).
Reproduces:
- Recall loose mean: 0.874 +- 0.005
- Benign BLOCK mean: 0.090 +- 0.005
Updates numbers.json with simulation metrics and intervals.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
NUMBERS_PATH = REPO_ROOT / "numbers.json"
PRIMARY_ARM_CHECK_PATH = REPO_ROOT / "docs" / "PRIMARY_ARM_CHECK.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_random_routing_simulation() -> tuple[dict[str, float], dict[str, list[float]]]:
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    n = len(rows)
    k_routed = sum(1 for r in rows if r["router"]["should_route"])
    atks_idx = [i for i, r in enumerate(rows) if r["label_binary"]]
    bens_idx = [i for i, r in enumerate(rows) if not r["label_binary"]]

    rng = np.random.default_rng(0)
    recalls: list[float] = []
    ben_blocks: list[float] = []
    ben_flags: list[float] = []

    for _ in range(2000):
        routed_set = set(rng.choice(n, size=k_routed, replace=False))
        # Loose recall (flag or block on attacks)
        tp = sum(
            1
            for i in atks_idx
            if (
                rows[i]["decisions"]["arm_u"]
                if i in routed_set
                else rows[i]["decisions"]["arm_a"]
            )
            in ("flag", "block")
        )
        recalls.append(tp / len(atks_idx))

        # Benign block rate
        b_blk = sum(
            1
            for i in bens_idx
            if (
                rows[i]["decisions"]["arm_u"]
                if i in routed_set
                else rows[i]["decisions"]["arm_a"]
            )
            == "block"
        )
        ben_blocks.append(b_blk / len(bens_idx))

        # Benign flag rate
        b_flg = sum(
            1
            for i in bens_idx
            if (
                rows[i]["decisions"]["arm_u"]
                if i in routed_set
                else rows[i]["decisions"]["arm_a"]
            )
            == "flag"
        )
        ben_flags.append(b_flg / len(bens_idx))

    means = {
        "recall_loose_mean": float(np.mean(recalls)),
        "benign_block_mean": float(np.mean(ben_blocks)),
        "benign_flag_mean": float(np.mean(ben_flags)),
    }
    intervals = {
        "recall_loose_ci": [
            float(np.percentile(recalls, 2.5)),
            float(np.percentile(recalls, 97.5)),
        ],
        "benign_block_ci": [
            float(np.percentile(ben_blocks, 2.5)),
            float(np.percentile(ben_blocks, 97.5)),
        ],
        "benign_flag_ci": [
            float(np.percentile(ben_flags, 2.5)),
            float(np.percentile(ben_flags, 97.5)),
        ],
    }
    return means, intervals


def main() -> int:
    print("[verify_random_routing] === Running K3 Simulation & Rule Check ===")
    errors: list[str] = []

    # 1. Run simulation
    means, intervals = run_random_routing_simulation()
    rec_mean = means["recall_loose_mean"]
    blk_mean = means["benign_block_mean"]

    print(
        f"[verify_random_routing] Random routing loose recall mean: {rec_mean:.4f} (95% CI: {intervals['recall_loose_ci']})"
    )
    print(
        f"[verify_random_routing] Random routing benign block mean: {blk_mean:.4f} (95% CI: {intervals['benign_block_ci']})"
    )

    if abs(rec_mean - 0.874) > 0.005:
        errors.append(
            f"Recall loose mean {rec_mean:.4f} deviates from 0.874 by > 0.005"
        )
    else:
        print("[verify_random_routing] PASS: Recall loose mean within 0.874 +- 0.005.")

    if abs(blk_mean - 0.090) > 0.005:
        errors.append(
            f"Benign block mean {blk_mean:.4f} deviates from 0.090 by > 0.005"
        )
    else:
        print("[verify_random_routing] PASS: Benign block mean within 0.090 +- 0.005.")

    # 2. Check PRIMARY_ARM_CHECK.md exists and contains verbatim rule text
    if not PRIMARY_ARM_CHECK_PATH.exists():
        errors.append(f"Missing {PRIMARY_ARM_CHECK_PATH}")
    else:
        content = PRIMARY_ARM_CHECK_PATH.read_text(encoding="utf-8")
        verbatim_snippets = [
            "Primary System: Arm R (Learned Review Router with Policy B)",
            "Evaluation on `held_out_v4.jsonl` must be executed EXACTLY ONCE under `scripts/gpu_lock.py`",
            "D1 (MiniLM): PERMANENTLY DROPPED",
        ]
        for snip in verbatim_snippets:
            if snip not in content:
                errors.append(
                    f"PRIMARY_ARM_CHECK.md missing verbatim snippet: '{snip}'"
                )
            else:
                print(
                    f"[verify_random_routing] PASS: Verbatim protocol snippet verified: '{snip}'"
                )

    # 3. Update numbers.json with simulation numbers
    if NUMBERS_PATH.exists():
        with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
            numbers = json.load(f)

        pred_sha = sha256_file(PREDICTIONS_PATH)
        for k, v in means.items():
            numbers[f"random_routing.{k}"] = {
                "value": v,
                "n": 2000,
                "source_file": "results/final-v4/predictions.jsonl",
                "sha256": pred_sha,
            }
        for k, v in intervals.items():
            numbers[f"random_routing.{k}"] = {
                "value": v,
                "n": 2000,
                "source_file": "results/final-v4/predictions.jsonl",
                "sha256": pred_sha,
            }

        with open(NUMBERS_PATH, "w", encoding="utf-8") as f:
            json.dump(numbers, f, indent=2)
        print(
            "[verify_random_routing] PASS: numbers.json updated with random routing simulation values."
        )

    if errors:
        print("\n[verify_random_routing] FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("[verify_random_routing] ALL K3 VERIFICATIONS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
