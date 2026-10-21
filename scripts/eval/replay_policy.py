#!/usr/bin/env python3
"""Hermetic Policy Replay (K12c Item 2).

Replays the 959a99a policy decision code without any model inference:
Takes the logged detector scores, fusion decisions, router error probabilities,
and judge verdicts from results/final-v4/predictions.jsonl and reproduces:
- arm_a
- arm_b
- arm_r
- arm_u
- c2_standalone

Compares reproduced decisions against logged decisions for all 347 rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallax.challenger.challenger_v2 import (
    AuthorityPolicy,
    ChallengerV2Reviewer,
    ChallengerVerdict,
)
from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult

PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"


def replay_all_rows() -> dict[str, Any]:
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    reviewer = ChallengerV2Reviewer(
        policy=AuthorityPolicy.POLICY_B,
        downgrade_confidence_threshold=0.65,
    )

    mismatches = {
        "arm_a": 0,
        "arm_b": 0,
        "arm_r": 0,
        "arm_u": 0,
        "c2_standalone": 0,
    }
    matches = {
        "arm_a": 0,
        "arm_b": 0,
        "arm_r": 0,
        "arm_u": 0,
        "c2_standalone": 0,
    }

    total_rows = len(records)

    for r in records:
        # 1. Base arm_a from logged fusion
        fusion_dec_str = r["fusion"]["decision"].lower()
        arm_a = Decision.BLOCK if fusion_dec_str == "block" else Decision.ALLOW

        # 2. Judge verdict reconstruction
        v_str = r["judge"]["verdict"].lower()
        if v_str == "block":
            verdict = ChallengerVerdict.BLOCK
        elif v_str == "allow":
            verdict = ChallengerVerdict.ALLOW
        elif v_str == "escalate":
            verdict = ChallengerVerdict.ESCALATE
        else:
            verdict = ChallengerVerdict.ABSTAIN

        j_conf = r["judge"]["confidence"]
        ens_conf = r["fusion"]["confidence"]

        # Detector results reconstruction
        dets = r["detectors"]
        det_results = [
            DetectorResult(
                name="d2_deberta",
                score=dets["d2-deberta"]["score"],
                is_attack=dets["d2-deberta"]["is_attack"],
            ),
            DetectorResult(
                name="d3_deepset",
                score=dets["d3-deepset"]["score"],
                is_attack=dets["d3-deepset"]["is_attack"],
            ),
            DetectorResult(
                name="d4_rules",
                score=dets["d4_rules"]["score"],
                is_attack=dets["d4_rules"]["is_attack"],
            ),
        ]

        # Replay Policy B on Arm U (review on all rows)
        arm_u, _ = reviewer.apply_authority_policy(
            verdict=verdict,
            original_decision=arm_a,
            unanimous=True,
            mean_confidence=j_conf,
            detector_results=det_results,
            ensemble_confidence=ens_conf,
        )

        # Replay Policy B on Arm B (when discrepancy trigger fired)
        b_trig = r["trigger_arm_b"]["triggered"]
        arm_b = arm_u if b_trig else arm_a

        # Replay Policy B on Arm R (when router routed)
        should_route = r["router"]["should_route"]
        arm_r = arm_u if should_route else arm_a

        # Standalone C2
        if verdict == ChallengerVerdict.BLOCK:
            c2 = Decision.BLOCK
        elif verdict == ChallengerVerdict.ALLOW:
            c2 = Decision.ALLOW
        else:
            c2 = Decision.FLAG

        logged_dec = r["decisions"]

        for arm_name, replayed_dec in [
            ("arm_a", arm_a),
            ("arm_b", arm_b),
            ("arm_r", arm_r),
            ("arm_u", arm_u),
            ("c2_standalone", c2),
        ]:
            if replayed_dec.value == logged_dec[arm_name]:
                matches[arm_name] += 1
            else:
                mismatches[arm_name] += 1

    return {
        "total_rows": total_rows,
        "matches": matches,
        "mismatches": mismatches,
    }


def main():
    print("[replay_policy] Starting offline Policy B reproduction check...")
    res = replay_all_rows()
    print(f"[replay_policy] Evaluated total rows: {res['total_rows']}")
    print(f"[replay_policy] Matches:    {res['matches']}")
    print(f"[replay_policy] Mismatches: {res['mismatches']}")

    total_mismatches = sum(res["mismatches"].values())
    if total_mismatches == 0:
        print(
            f"[replay_policy] PASS: 100% agreement ({res['total_rows']} rows) across all 5 arms."
        )
        return 0
    else:
        print(f"[replay_policy] FAIL: Encountered {total_mismatches} mismatches.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
