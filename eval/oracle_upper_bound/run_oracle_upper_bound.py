"""Theoretical Oracle Upper Bound Evaluation for PARALLAX.

STRICT ISOLATION NOTICE:
This module computes theoretical upper bounds (assuming a hypothetical perfect
reviewer that resolves all routed prompts with 100% fidelity).
Outputs are stored exclusively in eval/oracle_upper_bound/.
Nothing in this module or directory may be read by REPORT.md, STATUS.md,
or scripts/eval/eval_final.py.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig
from parallax.contracts.api import Decision
from parallax.eval.metrics import compute_binary_metrics
from parallax.orchestrator.fusion import AnyFlagFusion
from parallax.router.learned_router import LearnedReviewRouter

logging.basicConfig(level=logging.INFO, format="[oracle-eval] %(message)s")
logger = logging.getLogger("oracle-eval")

OUTPUT_DIR = REPO_ROOT / "eval" / "oracle_upper_bound"
RESULTS_FILE = OUTPUT_DIR / "oracle_results.json"
README_FILE = OUTPUT_DIR / "README.md"


def evaluate_oracle_on_records(
    samples: list[dict[str, Any]],
    scored_results: list[list[Any]],
    router_artifact_path: Path,
) -> dict[str, Any]:
    """Compute theoretical upper bounds assuming perfect resolution on routed prompts."""
    arm_a_fusion = AnyFlagFusion()
    rule_trigger = ChallengerTrigger(TriggerConfig())
    router = (
        LearnedReviewRouter.load(router_artifact_path)
        if router_artifact_path.exists()
        else None
    )

    total_n = len(samples)
    y_true = [1 if s.get("label_binary", s.get("is_attack")) else 0 for s in samples]

    base_preds: list[int] = []
    rule_routed: set[int] = set()
    learned_routed: set[int] = set()

    for idx, (s, det_results) in enumerate(zip(samples, scored_results, strict=True)):
        fused = arm_a_fusion.fuse(det_results)
        base_preds.append(1 if fused.decision in (Decision.FLAG, Decision.BLOCK) else 0)

        # Rule trigger routing
        t_res = rule_trigger.evaluate(
            detector_results=det_results,
            fused_score=fused.confidence,
            confidence=fused.confidence,
            original_decision=fused.decision,
            encoding_present=False,
            encoding_types=[],
        )
        if t_res.should_challenge:
            rule_routed.add(idx)

        # Learned router routing
        if router:
            r_dec = router.route(s.get("text", ""), det_results, fused)
            if r_dec.should_route:
                learned_routed.add(idx)

    # 1. Arm B Oracle (rule-routed prompts resolved with ground truth)
    rule_oracle_preds = [
        y_true[idx] if idx in rule_routed else base_preds[idx] for idx in range(total_n)
    ]
    rule_m = compute_binary_metrics(y_true, rule_oracle_preds)
    rule_fpr = rule_m.confusion_matrix.matrix[0][1] / max(1, rule_m.support_negative)

    # 2. Arm R Oracle (learned-routed prompts resolved with ground truth)
    learned_oracle_preds = [
        y_true[idx] if idx in learned_routed else base_preds[idx]
        for idx in range(total_n)
    ]
    learned_m = compute_binary_metrics(y_true, learned_oracle_preds)
    learned_fpr = learned_m.confusion_matrix.matrix[0][1] / max(
        1, learned_m.support_negative
    )

    # 3. Arm U Oracle (all prompts resolved with ground truth)
    universal_m = compute_binary_metrics(y_true, y_true)

    return {
        "total_samples": total_n,
        "arm_b_rule_oracle": {
            "name": "Arm B Oracle Upper Bound (Rule Trigger + Perfect Challenger)",
            "recall": round(rule_m.recall, 4),
            "fpr": round(rule_fpr, 4),
            "precision": round(rule_m.precision, 4),
            "f1": round(rule_m.f1, 4),
            "routed_count": len(rule_routed),
            "extra_calls_per_100": round(
                (len(rule_routed) / max(1, total_n)) * 100.0, 2
            ),
        },
        "arm_r_learned_oracle": {
            "name": "Arm R Oracle Upper Bound (Learned Router + Perfect Challenger)",
            "recall": round(learned_m.recall, 4),
            "fpr": round(learned_fpr, 4),
            "precision": round(learned_m.precision, 4),
            "f1": round(learned_m.f1, 4),
            "routed_count": len(learned_routed),
            "extra_calls_per_100": round(
                (len(learned_routed) / max(1, total_n)) * 100.0, 2
            ),
        },
        "arm_u_universal_oracle": {
            "name": "Arm U Oracle Upper Bound (Universal Perfect Challenger)",
            "recall": round(universal_m.recall, 4),
            "fpr": 0.0,
            "precision": round(universal_m.precision, 4),
            "f1": round(universal_m.f1, 4),
            "routed_count": total_n,
            "extra_calls_per_100": 100.0,
        },
    }


def main() -> None:
    logger.info("Computing theoretical oracle upper bounds...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    held_out_path = REPO_ROOT / "data" / "splits" / "held_out.jsonl"
    router_artifact = REPO_ROOT / "artifacts" / "router" / "learned_router.joblib"
    cache_path = REPO_ROOT / "data" / "cache" / "detector_scores.json"

    samples: list[dict[str, Any]] = []
    with open(held_out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    cached_scores: dict[str, Any] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_scores = json.load(f)

    from parallax.contracts.detectors import DetectorResult

    scored_results: list[list[DetectorResult]] = []
    for s in samples:
        s_id = s["id"]
        c = cached_scores.get(s_id, {})
        d1 = DetectorResult(
            name="d1-minilm",
            is_attack=bool(c.get("d1_score", 0.0) >= 0.5),
            score=c.get("d1_score", 0.0),
            latency_ms=c.get("d1_latency", 15.0),
        )
        d2 = DetectorResult(
            name="d2-deberta",
            is_attack=bool(c.get("d2_score", 0.0) >= 0.5),
            score=c.get("d2_score", 0.0),
            latency_ms=c.get("d2_latency", 25.0),
        )
        d3 = DetectorResult(
            name="d3-deepset",
            is_attack=bool(c.get("d3_score", 0.0) >= 0.5),
            score=c.get("d3_score", 0.0),
            latency_ms=c.get("d3_latency", 25.0),
        )
        scored_results.append([d1, d2, d3])

    results = evaluate_oracle_on_records(samples, scored_results, router_artifact)
    results["metadata"] = {
        "run_id": f"oracle-eval-{uuid4().hex[:8]}",
        "timestamp": datetime.now(UTC).isoformat(),
        "disclaimer": "THEORETICAL UPPER BOUND ONLY. Excluded from headline tables and reports.",
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved oracle upper bounds to %s", RESULTS_FILE)

    readme_content = f"""# Theoretical Oracle Upper Bound Benchmark

> [!WARNING]
> This folder contains theoretical upper bounds calculated under the assumption
> that a reviewer perfectly resolves all routed queries.
> **These results are strictly excluded from headline reports, STATUS.md, and REPORT.md.**

## Held-Out Results (Theoretical Upper Bound)

- **Arm B (Rule Trigger Oracle)**: Recall {results["arm_b_rule_oracle"]["recall"] * 100:.1f}%, FPR {results["arm_b_rule_oracle"]["fpr"] * 100:.1f}% ({results["arm_b_rule_oracle"]["extra_calls_per_100"]:.1f} extra calls/100)
- **Arm R (Learned Router Oracle)**: Recall {results["arm_r_learned_oracle"]["recall"] * 100:.1f}%, FPR {results["arm_r_learned_oracle"]["fpr"] * 100:.1f}% ({results["arm_r_learned_oracle"]["extra_calls_per_100"]:.1f} extra calls/100)
- **Arm U (Universal Oracle)**: Recall {results["arm_u_universal_oracle"]["recall"] * 100:.1f}%, FPR {results["arm_u_universal_oracle"]["fpr"] * 100:.1f}% (100.0 extra calls/100)

*Generated: {results["metadata"]["timestamp"]}*
"""
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info("Wrote %s", README_FILE)


if __name__ == "__main__":
    main()
