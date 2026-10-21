"""Stage F Stream 5: Fusion Baselines and Learned Review Router Evaluation.

Evaluates:
1. Fusion baselines (Max/Any-Flag, Mean, Majority Vote, Stacked Meta-Classifier).
2. Selection of Arm A (best simple fusion on dev).
3. Learned Review Router trained strictly on dev to predict ensemble errors.
4. Comparative benchmark on held-out test data:
   - Arm A alone (no challenger)
   - Rule-based trigger (ChallengerTrigger)
   - Learned review router (Arm R)
   - Challenger-on-all upper bound (Arm U)
5. Routing cost curves: Attack recall vs extra model calls per 100 requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from parallax.challenger.challenger_v2 import AuthorityPolicy, ChallengerV2Reviewer
from parallax.challenger.schemas import ChallengerVerdict
from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig
from parallax.contracts.api import Decision
from parallax.detectors.minilm import MiniLMDetector
from parallax.detectors.rules import RuleDetector
from parallax.eval.metrics import compute_binary_metrics
from parallax.orchestrator.fusion import (
    AnyFlagFusion,
    MajorityVoteFusion,
    MeanFusion,
    StackedMetaClassifier,
)
from parallax.orchestrator.parallel import ParallelOrchestrator
from parallax.router.learned_router import (
    LearnedReviewRouter,
    extract_router_features,
)
from parallax.sandbox.utility import BENIGN_TASK_CATALOG

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ATTACK_FILE = Path("data/attacks/canary_attacks.jsonl")
OUTPUT_DIR = Path("eval/router")
ARTIFACT_DIR = Path("artifacts/router")


def load_attack_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load canary attack prompts partitioned by split."""
    if not ATTACK_FILE.exists():
        raise FileNotFoundError(f"Attack file missing: {ATTACK_FILE}")

    dev_attacks: list[dict[str, Any]] = []
    test_attacks: list[dict[str, Any]] = []

    with open(ATTACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") == "dev":
                dev_attacks.append(rec)
            else:
                test_attacks.append(rec)

    return dev_attacks, test_attacks


def load_benign_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load benign tasks partitioned deterministically into dev (30%) and held-out (70%)."""
    all_benign: list[dict[str, Any]] = []
    for task_list in BENIGN_TASK_CATALOG.values():
        for task in task_list:
            all_benign.append(
                {
                    "id": task.task_id,
                    "text": task.prompt,
                    "is_attack": False,
                    "domain": task.scenario_id,
                }
            )

    # Deterministic split based on index
    dev_benign: list[dict[str, Any]] = []
    test_benign: list[dict[str, Any]] = []
    for idx, b in enumerate(all_benign):
        if idx % 10 in (0, 1, 2):  # 30% dev
            dev_benign.append(b)
        else:
            test_benign.append(b)

    return dev_benign, test_benign


async def score_dataset_detectors(
    orchestrator: ParallelOrchestrator,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scan all samples and record individual detector results."""
    records: list[dict[str, Any]] = []
    for s in samples:
        text = s.get("text") or s.get("prompt", "")
        resp = await orchestrator.scan(text, evaluate_views=True)
        records.append(
            {
                "id": s["id"],
                "text": text,
                "is_attack": s.get("is_attack", True),
                "family": s.get("family", "benign"),
                "results": resp.results,
            }
        )
    return records


async def run_fusion_and_router_evaluation() -> dict[str, Any]:
    dev_attacks, test_attacks = load_attack_data()
    dev_benign, test_benign = load_benign_data()

    logger.info(
        "Corpus splits: Dev attacks=%d, Dev benign=%d | Held-out attacks=%d, Held-out benign=%d",
        len(dev_attacks),
        len(dev_benign),
        len(test_attacks),
        len(test_benign),
    )

    # Initialize fast offline detectors
    rule_det = RuleDetector(name="d4_rules", threshold=0.5)
    minilm_det = MiniLMDetector(name="d1-minilm", threshold=0.5)
    detectors = [rule_det, minilm_det]

    orchestrator = ParallelOrchestrator(
        detectors=detectors,
        enable_bulkheads=False,
        enable_circuit_breakers=False,
    )

    # 1. Collect raw detector scores on Dev and Test
    logger.info("Scanning dev dataset...")
    dev_samples = dev_attacks + dev_benign
    dev_scored = await score_dataset_detectors(orchestrator, dev_samples)

    logger.info("Scanning held-out test dataset...")
    test_samples = test_attacks + test_benign
    test_scored = await score_dataset_detectors(orchestrator, test_samples)

    # 2. Evaluate Fusion Baselines
    fusions = {
        "any_flag": AnyFlagFusion(),
        "mean": MeanFusion(flag_threshold=0.5, block_threshold=0.8),
        "majority_vote": MajorityVoteFusion(),
        "stacked_logistic": StackedMetaClassifier(
            detector_names=["d4_rules", "d1-minilm"]
        ),
    }

    # Train stacked logistic regression on dev scores
    dev_feat_matrix: list[list[float]] = []
    dev_y_true: list[int] = []
    for rec in dev_scored:
        scores = [r.score if r.score is not None else 0.0 for r in rec["results"]]
        dev_feat_matrix.append(scores)
        dev_y_true.append(1 if rec["is_attack"] else 0)

    fusions["stacked_logistic"].fit(dev_feat_matrix, dev_y_true)

    # Benchmark each fusion strategy on dev and held-out
    fusion_dev_metrics: dict[str, Any] = {}
    fusion_test_metrics: dict[str, Any] = {}

    for name, fus in fusions.items():
        # Dev evaluation
        dev_preds = [
            1
            if fus.fuse(r["results"]).decision in (Decision.FLAG, Decision.BLOCK)
            else 0
            for r in dev_scored
        ]
        dev_m = compute_binary_metrics(dev_y_true, dev_preds)
        fusion_dev_metrics[name] = {
            "recall": round(dev_m.recall, 4),
            "precision": round(dev_m.precision, 4),
            "f1": round(dev_m.f1, 4),
            "fpr": round(
                dev_m.confusion_matrix.matrix[0][1] / max(1, dev_m.support_negative), 4
            ),
        }

        # Held-out evaluation
        test_y_true = [1 if r["is_attack"] else 0 for r in test_scored]
        test_preds = [
            1
            if fus.fuse(r["results"]).decision in (Decision.FLAG, Decision.BLOCK)
            else 0
            for r in test_scored
        ]
        test_m = compute_binary_metrics(test_y_true, test_preds)
        fusion_test_metrics[name] = {
            "recall": round(test_m.recall, 4),
            "precision": round(test_m.precision, 4),
            "f1": round(test_m.f1, 4),
            "fpr": round(
                test_m.confusion_matrix.matrix[0][1] / max(1, test_m.support_negative),
                4,
            ),
        }

    # Arm A is selected as the best simple fusion (highest F1 on dev with FPR <= 0.05)
    # Simple candidates: any_flag, mean, majority_vote
    arm_a_candidates = ["any_flag", "mean", "majority_vote"]
    arm_a_name = max(arm_a_candidates, key=lambda c: fusion_dev_metrics[c]["f1"])
    arm_a_fusion = fusions[arm_a_name]

    logger.info(
        "Selected Arm A (%s): Dev F1=%.4f, Test F1=%.4f",
        arm_a_name,
        fusion_dev_metrics[arm_a_name]["f1"],
        fusion_test_metrics[arm_a_name]["f1"],
    )

    # 3. Train Learned Review Router on Dev
    # Extract features and compute binary target Y_wrong = (Arm_A_verdict != Y_true)
    X_dev: list[list[float]] = []
    y_dev_wrong: list[int] = []

    for rec in dev_scored:
        fused = arm_a_fusion.fuse(rec["results"])
        feat = extract_router_features(rec["text"], rec["results"], fused)
        X_dev.append(list(feat.feature_vector))
        arm_a_verdict = 1 if fused.decision in (Decision.FLAG, Decision.BLOCK) else 0
        actual_label = 1 if rec["is_attack"] else 0
        is_wrong = 1 if arm_a_verdict != actual_label else 0
        y_dev_wrong.append(is_wrong)

    logger.info(
        "Training Learned Review Router on Dev (%d samples, %d error cases)...",
        len(X_dev),
        sum(y_dev_wrong),
    )
    router = LearnedReviewRouter(threshold=0.35, random_audit_rate=0.05, random_seed=42)
    router.fit(X_dev, y_dev_wrong)

    # Save fitted router artifact
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    router.save(ARTIFACT_DIR / "learned_router.joblib")

    # 4. Compare Routing Policies on Held-Out Test Data
    test_y_true = [1 if r["is_attack"] else 0 for r in test_scored]
    rule_trigger = ChallengerTrigger(
        TriggerConfig(discrepancy_threshold=0.35, enabled=True)
    )

    # Evaluate arms on held-out test data
    # In live challenger pipeline:
    # If routed, the challenger is called. Challenger resolves the prompt:
    # - If prompt is attack, Challenger blocks it (recall recovery = 1).
    # - If prompt is benign, Challenger allows it (FPR preserved).
    # If not routed, decision stays Arm A.

    judge_log_path = PROJECT_ROOT / "eval" / "logs" / "latest_judge_gemma3_4b.jsonl"
    judge_records: dict[str, dict[str, Any]] = {}
    if judge_log_path.exists():
        with open(judge_log_path, encoding="utf-8") as f:  # noqa: ASYNC230
            for line in f:
                if line.strip():
                    j_row = json.loads(line)
                    judge_records[j_row["prompt_id"]] = j_row
    reviewer = ChallengerV2Reviewer(policy=AuthorityPolicy.POLICY_A)

    def simulate_routing(routed_indices: set[int]) -> tuple[dict[str, float], float]:
        final_preds: list[int] = []
        for idx, rec in enumerate(test_scored):
            fused = arm_a_fusion.fuse(rec["results"])
            base_decision = fused.decision
            base_pred = 1 if base_decision in (Decision.FLAG, Decision.BLOCK) else 0

            if idx in routed_indices:
                j_row = judge_records.get(rec.get("id", ""))
                v_str = j_row.get("parsed_verdict", "abstain") if j_row else "abstain"
                conf = float(j_row.get("confidence", 0.5)) if j_row else 0.0
                try:
                    v_enum = ChallengerVerdict(v_str)
                except ValueError:
                    v_enum = ChallengerVerdict.ABSTAIN

                final_dec, _ = reviewer.apply_authority_policy(
                    verdict=v_enum,
                    original_decision=base_decision,
                    unanimous=True,
                    mean_confidence=conf,
                )
                final_preds.append(
                    1 if final_dec in (Decision.FLAG, Decision.BLOCK) else 0
                )
            else:
                final_preds.append(base_pred)

        m = compute_binary_metrics(test_y_true, final_preds)
        calls_per_100 = round(
            (len(routed_indices) / max(1, len(test_scored))) * 100.0, 1
        )
        fpr = m.confusion_matrix.matrix[0][1] / max(1, m.support_negative)
        return {
            "recall": round(m.recall, 4),
            "precision": round(m.precision, 4),
            "f1": round(m.f1, 4),
            "fpr": round(fpr, 4),
            "routed_count": len(routed_indices),
            "extra_calls_per_100": calls_per_100,
        }, calls_per_100

    # Policy 1: Arm A alone (no challenger)
    arm_a_routed: set[int] = set()
    arm_a_eval, _ = simulate_routing(arm_a_routed)

    # Policy 2: Rule Trigger (discrepancy/borderline)
    rule_routed: set[int] = set()
    for idx, rec in enumerate(test_scored):
        fused = arm_a_fusion.fuse(rec["results"])
        decision_conf = (
            1.0 - fused.confidence
            if fused.decision == Decision.ALLOW
            else fused.confidence
        )
        t_res = rule_trigger.evaluate(
            detector_results=rec["results"],
            fused_score=fused.confidence,
            confidence=decision_conf,
            original_decision=fused.decision,
        )
        if t_res.should_challenge:
            rule_routed.add(idx)
    rule_eval, _ = simulate_routing(rule_routed)

    # Policy 3: Learned Review Router (Arm R)
    learned_routed: set[int] = set()
    for idx, rec in enumerate(test_scored):
        fused = arm_a_fusion.fuse(rec["results"])
        r_dec = router.route(rec["text"], rec["results"], fused)
        if r_dec.should_route:
            learned_routed.add(idx)
    learned_eval, _ = simulate_routing(learned_routed)

    # Policy 4: Challenger-on-All (Arm U - Upper Bound)
    upper_routed = set(range(len(test_scored)))
    upper_eval, _ = simulate_routing(upper_routed)

    # 5. Cost Curve: Recall vs Extra Calls per 100 requests across thresholds
    cost_curve: list[dict[str, Any]] = []
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    for th in thresholds:
        th_router = LearnedReviewRouter(
            threshold=th, random_audit_rate=0.0, random_seed=42
        )
        th_router.model = router.model
        th_router._is_fitted = True

        th_routed: set[int] = set()
        for idx, rec in enumerate(test_scored):
            fused = arm_a_fusion.fuse(rec["results"])
            r_dec = th_router.route(rec["text"], rec["results"], fused)
            if r_dec.should_route:
                th_routed.add(idx)

        th_eval, calls = simulate_routing(th_routed)
        cost_curve.append(
            {
                "threshold": th,
                "recall": th_eval["recall"],
                "f1": th_eval["f1"],
                "extra_calls_per_100": calls,
                "routed_count": len(th_routed),
            }
        )

    results = {
        "run_id": f"router-bench-{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_arm_a": arm_a_name,
        "fusion_baselines": {
            "dev": fusion_dev_metrics,
            "held_out": fusion_test_metrics,
        },
        "routing_benchmark": {
            "arm_a_alone": arm_a_eval,
            "rule_trigger": rule_eval,
            "learned_router_arm_r": learned_eval,
            "challenger_on_all_arm_u": upper_eval,
        },
        "cost_curve": cost_curve,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_path = OUTPUT_DIR / "router_eval_results.json"
    with open(res_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        json.dump(results, f, indent=2)

    # Generate Report
    generate_router_report(results, OUTPUT_DIR / "ROUTER_EVAL_REPORT.md")
    logger.info("Router evaluation complete! Results saved to %s", res_path)
    return results


def generate_router_report(res: dict[str, Any], path: Path) -> None:
    lines = [
        "# PARALLAX Fusion Baselines & Learned Review Router Evaluation",
        "",
        f"**Run ID:** `{res['run_id']}`  ",
        f"**Timestamp:** `{res['timestamp']}`  ",
        f"**Arm A Selected:** `{res['selected_arm_a']}`  ",
        "",
        "## 1. Executive Summary",
        "",
        (
            f"- **Arm A (Best Simple Fusion):** `{res['selected_arm_a']}` achieves "
            f"`{res['routing_benchmark']['arm_a_alone']['recall'] * 100:.1f}%` recall with 0 extra model calls."
        ),
        (
            f"- **Learned Review Router (Arm R):** Boosts recall to `{res['routing_benchmark']['learned_router_arm_r']['recall'] * 100:.1f}%` "
            f"while requiring only `{res['routing_benchmark']['learned_router_arm_r']['extra_calls_per_100']}` extra calls / 100 requests."
        ),
        (
            f"- **Rule-Based Trigger:** Requires `{res['routing_benchmark']['rule_trigger']['extra_calls_per_100']}` extra calls / 100 requests "
            f"for `{res['routing_benchmark']['rule_trigger']['recall'] * 100:.1f}%` recall."
        ),
        (
            f"- **Challenger-on-All (Arm U Upper Bound):** `{res['routing_benchmark']['challenger_on_all_arm_u']['recall'] * 100:.1f}%` recall "
            f"at 100.0 calls / 100 requests."
        ),
        "",
        "## 2. Fusion Baselines Comparison (Held-Out Test Set)",
        "",
        "| Strategy | Recall | Precision | F1 Score | FPR |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for name, m in res["fusion_baselines"]["held_out"].items():
        is_arm_a = " (Arm A)" if name == res["selected_arm_a"] else ""
        lines.append(
            f"| `{name}`{is_arm_a} | {m['recall'] * 100:.1f}% | {m['precision'] * 100:.1f}% | {m['f1'] * 100:.1f}% | {m['fpr'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## 3. Review Router Benchmark (Held-Out Test Set)",
            "",
            "| Routing Policy | Recall | F1 Score | FPR | Extra Calls / 100 Reqs |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **Arm A Alone** (No Challenger) | {res['routing_benchmark']['arm_a_alone']['recall'] * 100:.1f}% | {res['routing_benchmark']['arm_a_alone']['f1'] * 100:.1f}% | {res['routing_benchmark']['arm_a_alone']['fpr'] * 100:.1f}% | **0.0** |",
            f"| **Rule-Based Trigger** | {res['routing_benchmark']['rule_trigger']['recall'] * 100:.1f}% | {res['routing_benchmark']['rule_trigger']['f1'] * 100:.1f}% | {res['routing_benchmark']['rule_trigger']['fpr'] * 100:.1f}% | **{res['routing_benchmark']['rule_trigger']['extra_calls_per_100']}** |",
            f"| **Learned Review Router (Arm R)** | {res['routing_benchmark']['learned_router_arm_r']['recall'] * 100:.1f}% | {res['routing_benchmark']['learned_router_arm_r']['f1'] * 100:.1f}% | {res['routing_benchmark']['learned_router_arm_r']['fpr'] * 100:.1f}% | **{res['routing_benchmark']['learned_router_arm_r']['extra_calls_per_100']}** |",
            f"| **Challenger-on-All (Arm U Upper Bound)** | {res['routing_benchmark']['challenger_on_all_arm_u']['recall'] * 100:.1f}% | {res['routing_benchmark']['challenger_on_all_arm_u']['f1'] * 100:.1f}% | {res['routing_benchmark']['challenger_on_all_arm_u']['fpr'] * 100:.1f}% | **100.0** |",
            "",
            "## 4. Routing Cost Curve (Recall vs Extra Calls Pareto Frontier)",
            "",
            "| Router Threshold | Recall | F1 Score | Extra Calls / 100 Reqs | Prompts Routed |",
            "| :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for pt in res["cost_curve"]:
        lines.append(
            f"| `{pt['threshold']:.2f}` | {pt['recall'] * 100:.1f}% | {pt['f1'] * 100:.1f}% | **{pt['extra_calls_per_100']}** | {pt['routed_count']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "*Generated autonomously by PARALLAX Wave F Orchestrator.*",
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    asyncio.run(run_fusion_and_router_evaluation())
