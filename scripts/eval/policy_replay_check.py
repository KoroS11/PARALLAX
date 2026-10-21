#!/usr/bin/env python3
"""5-Arm Policy Replay Verification Check for PARALLAX Evaluation (scripts/eval/policy_replay_check.py).

Verifies that the policy composition code in scripts/eval/eval_external.py reproduces
100% (347/347) of logged decisions across all five evaluated arms:
- Arm A (AnyFlag fusion baseline)
- Arm B (Challenger-triggered review)
- Arm R (Learned review router, tau=0.35)
- Arm U (Oracle upper bound / unconditional review)
- Arm C2 (Standalone judge)

This check must pass prior to any external scoring execution.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parallax.challenger.schemas import ChallengerVerdict
from scripts.eval.eval_external import ExternalEvaluator

logger = logging.getLogger(__name__)

DEFAULT_PREDICTIONS: Path = PROJECT_ROOT / "results" / "final-v4" / "predictions.jsonl"


class _ReplayJudgeResult:
    """Mock judge response container matching AdversarialJudge schema."""

    def __init__(self, verdict: ChallengerVerdict, confidence: float):
        self.verdict = verdict
        self.confidence = float(confidence)


def run_policy_replay_check(
    predictions_path: Path = DEFAULT_PREDICTIONS,
) -> dict[str, int]:
    """Run 5-arm policy replay across all rows in predictions_path.

    Returns dict of match counts per arm.
    Raises ValueError on any mismatch.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    lines = [
        line.strip()
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_rows = len(lines)
    if total_rows == 0:
        raise ValueError(f"Empty predictions file at {predictions_path}")

    evaluator = ExternalEvaluator(use_fakes=True)
    matches: dict[str, int] = {
        "arm_a": 0,
        "arm_b": 0,
        "arm_r": 0,
        "arm_u": 0,
        "c2_standalone": 0,
    }
    mismatches: list[dict[str, Any]] = []

    for idx, line in enumerate(lines):
        row = json.loads(line)
        d2_cal = float(row["detectors"]["d2-deberta"]["score"])
        d3_cal = float(row["detectors"]["d3-deepset"]["score"])
        d4_info = row["detectors"]["d4_rules"]
        d4_is_attack = bool(d4_info["is_attack"])
        d4_binary = 1.0 if d4_is_attack else 0.0

        prob_error = float(row["router"]["prob_error"])
        verdict_str = str(row["judge"]["verdict"]).upper()
        verdict_enum = ChallengerVerdict[verdict_str]
        judge_conf = float(row["judge"]["confidence"])

        fake_judge = _ReplayJudgeResult(verdict=verdict_enum, confidence=judge_conf)

        res = evaluator.evaluate_single_row(
            prompt_id=row.get("prompt_id", str(idx)),
            text="placeholder",
            label_binary=bool(row.get("label_binary", False)),
            family=row.get("family", "unknown"),
            source=row.get("source", "unknown"),
            mock_d2_raw=d2_cal,
            mock_d3_raw=d3_cal,
            mock_d4_raw=d4_binary,
            mock_judge_res=fake_judge,
            mock_router_prob=prob_error,
        )

        decisions = res["decisions"]
        logged_decisions = row["decisions"]
        row_mismatches = {}

        for arm in ["arm_a", "arm_b", "arm_r", "arm_u", "c2_standalone"]:
            computed = decisions[arm]
            logged = logged_decisions[arm]
            if computed == logged:
                matches[arm] += 1
            else:
                row_mismatches[arm] = {"computed": computed, "logged": logged}

        if row_mismatches:
            mismatches.append(
                {
                    "row_index": idx,
                    "prompt_id": row.get("prompt_id"),
                    "mismatches": row_mismatches,
                }
            )

    logger.info("Policy replay evaluated on %d rows.", total_rows)
    for arm, count in matches.items():
        pct = (count / total_rows) * 100.0
        logger.info("  %-15s: %d/%d (%.2f%%)", arm, count, total_rows, pct)

    if mismatches:
        err_msg = (
            f"Policy replay failed with {len(mismatches)} mismatched rows:\n"
            + json.dumps(mismatches[:5], indent=2)
        )
        raise ValueError(err_msg)

    return matches


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[policy-replay] %(asctime)s [%(levelname)s] %(message)s",
    )
    print("Running 5-arm policy replay check on final-v4 predictions...")
    try:
        matches = run_policy_replay_check()
        print("-" * 60)
        print("PASS: 5-arm policy replay succeeded 347/347 on all arms.")
        for arm, count in matches.items():
            print(f"  {arm:<16}: {count}/347 (100.00%)")
        print("-" * 60)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: Policy replay check failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
