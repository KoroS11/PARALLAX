"""Run frozen PARALLAX arms and PromptForest baseline on the external benchmark.

Zero-tuning evaluation across:
- PromptForest 2/3-Detector Baseline
- Individual Detectors (d1-minilm, d2-deberta, d3-deepset)
- Arm A (AnyFlag Fusion)
- Mean Fusion
- Arm B (Rule-based Challenger Trigger Policy with live Ollama Judge)
- Arm R (Learned Review Router with live Ollama Judge)
- Oracle Upper Bound (Theoretical - labeled explicitly and excluded from headline tables)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from parallax.challenger.challenger_v2 import (
    AuthorityPolicy,
    ChallengerV2Reviewer,
)
from parallax.challenger.judge import AdversarialJudge
from parallax.challenger.schemas import ChallengerVerdict
from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig
from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult
from parallax.detectors.deberta import DeBERTaDetector
from parallax.detectors.minilm import MiniLMDetector
from parallax.detectors.promptforest import PromptForestAdapter
from parallax.eval.metrics import compute_binary_metrics, compute_wilson_ci
from parallax.orchestrator.fusion import AnyFlagFusion, MeanFusion
from parallax.preprocess.pipeline import preprocess
from parallax.router.learned_router import LearnedReviewRouter
from scripts.gpu_lock import gpu_lock

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("external_eval")

BENCHMARK_PATH = PROJECT_ROOT / "data" / "external_eval" / "external_benchmark.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "external_eval" / "external_eval_results.json"
ROUTER_ARTIFACT = PROJECT_ROOT / "artifacts" / "router" / "learned_router.joblib"
RAW_SCORES_PATH = PROJECT_ROOT / "data" / "external_eval" / "external_raw_scores.json"
JUDGE_CACHE_PATH = (
    PROJECT_ROOT / "data" / "external_eval" / "external_judge_scores.json"
)


def compute_cluster_and_wilson_ci(
    y_true: list[int],
    y_pred: list[int],
    clusters: list[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute 95% cluster-resampling bootstrap and Wilson score intervals for recall and FPR."""
    cluster_map: dict[str, list[int]] = {}
    for idx, c in enumerate(clusters):
        cluster_map.setdefault(c, []).append(idx)
    cluster_keys = sorted(cluster_map.keys())
    num_clusters = len(cluster_keys)

    rng = np.random.default_rng(seed)
    recalls: list[float] = []
    fprs: list[float] = []

    for _ in range(n_boot):
        sampled_c = rng.choice(cluster_keys, size=num_clusters, replace=True)
        sampled_idx: list[int] = []
        for sc in sampled_c:
            sampled_idx.extend(cluster_map[sc])

        sub_true = [y_true[i] for i in sampled_idx]
        sub_pred = [y_pred[i] for i in sampled_idx]

        pos_count = sum(1 for yt in sub_true if yt == 1)
        neg_count = sum(1 for yt in sub_true if yt == 0)

        if pos_count > 0:
            tp = sum(
                1
                for yt, yp in zip(sub_true, sub_pred, strict=False)
                if yt == 1 and yp == 1
            )
            recalls.append(tp / pos_count)
        if neg_count > 0:
            fp = sum(
                1
                for yt, yp in zip(sub_true, sub_pred, strict=False)
                if yt == 0 and yp == 1
            )
            fprs.append(fp / neg_count)

    boot_rec_ci = (
        round(float(np.percentile(recalls, 2.5)), 4) if recalls else 0.0,
        round(float(np.percentile(recalls, 97.5)), 4) if recalls else 0.0,
    )
    boot_fpr_ci = (
        round(float(np.percentile(fprs, 2.5)), 4) if fprs else 0.0,
        round(float(np.percentile(fprs, 97.5)), 4) if fprs else 0.0,
    )

    # Wilson score intervals
    total_pos = sum(1 for yt in y_true if yt == 1)
    total_neg = sum(1 for yt in y_true if yt == 0)
    actual_tp = sum(
        1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 1 and yp == 1
    )
    actual_fp = sum(
        1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 0 and yp == 1
    )

    wilson_rec = compute_wilson_ci(actual_tp, total_pos)
    wilson_fpr = compute_wilson_ci(actual_fp, total_neg)

    return {
        "recall_ci_95": list(boot_rec_ci),
        "recall_wilson_ci_95": list(wilson_rec),
        "fpr_ci_95": list(boot_fpr_ci),
        "fpr_wilson_ci_95": list(wilson_fpr),
    }


def score_missing_samples(
    missing_samples: list[dict[str, Any]], cached_data: dict[str, Any]
) -> None:
    """Score missing samples synchronously calling async scoring routine."""

    async def _async_score() -> None:
        pf_adapter = PromptForestAdapter(base_url="http://127.0.0.1:8765")
        with gpu_lock(timeout_seconds=600):
            d1 = MiniLMDetector(name="d1-minilm")
            d2 = DeBERTaDetector(name="d2-deberta")
            d3 = DeBERTaDetector(
                name="d3-deepset", model_name="deepset/deberta-v3-base-injection"
            )

            logger.info(
                "Scoring %d missing samples across detectors...", len(missing_samples)
            )
            for idx, sample in enumerate(missing_samples, 1):
                text = sample["text"]
                r1 = d1.predict_sync(text)
                r2 = await d2.predict(text)
                r3 = await d3.predict(text)
                pf_res = await pf_adapter.predict(text)

                cached_data[sample["id"]] = {
                    "sample": sample,
                    "d1": {
                        "name": r1.name,
                        "score": r1.score,
                        "is_attack": r1.is_attack,
                        "latency_ms": r1.latency_ms,
                        "error": r1.error,
                    },
                    "d2": {
                        "name": r2.name,
                        "score": r2.score,
                        "is_attack": r2.is_attack,
                        "latency_ms": r2.latency_ms,
                        "error": r2.error,
                    },
                    "d3": {
                        "name": r3.name,
                        "score": r3.score,
                        "is_attack": r3.is_attack,
                        "latency_ms": r3.latency_ms,
                        "error": r3.error,
                    },
                    "promptforest": {
                        "name": pf_res.name,
                        "score": pf_res.score,
                        "is_attack": pf_res.is_attack,
                        "latency_ms": pf_res.latency_ms,
                        "error": pf_res.error,
                        "uncertainty": pf_res.uncertainty,
                    },
                }
                if idx % 20 == 0 or idx == len(missing_samples):
                    logger.info(
                        "Scored %d/%d missing samples...", idx, len(missing_samples)
                    )

    asyncio.run(_async_score())


async def evaluate_routed_prompts_live(
    scored_records: list[dict[str, Any]],
    routed_indices: set[int],
    base_preds: list[int],
    cached_judge_results: dict[str, Any],
) -> tuple[list[int], list[float], int]:
    """Execute live Ollama judge on routed prompts with error handling and abstention."""
    judge_log_path = PROJECT_ROOT / "eval" / "logs" / "latest_judge_gemma3_4b.jsonl"
    judge_log_map: dict[str, dict[str, Any]] = {}
    if judge_log_path.exists():
        with open(judge_log_path, "r", encoding="utf-8") as f:  # noqa: ASYNC230
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    judge_log_map[row["prompt_id"]] = row

    reviewer = ChallengerV2Reviewer(policy=AuthorityPolicy.POLICY_A)
    judge = AdversarialJudge(model_name="gemma3:4b", base_url="http://localhost:11434")
    final_preds = list(base_preds)
    latencies: list[float] = []
    error_count = 0

    for idx in sorted(routed_indices):
        rec = scored_records[idx]
        sample_id = rec["sample"]["id"]
        text = rec["sample"]["text"]
        base_decision = Decision.FLAG if base_preds[idx] == 1 else Decision.ALLOW

        # 1. Check judge log from run_judge_all.py
        if sample_id in judge_log_map:
            j_row = judge_log_map[sample_id]
            verdict_str = j_row.get("parsed_verdict", "abstain")
            conf = float(j_row.get("confidence", 0.5))
            lat = float(j_row.get("latency_ms", 0.0))
            latencies.append(lat)
            try:
                v_enum = ChallengerVerdict(verdict_str)
            except ValueError:
                v_enum = ChallengerVerdict.ABSTAIN

            final_decision, _ = reviewer.apply_authority_policy(
                verdict=v_enum,
                original_decision=base_decision,
                unanimous=True,
                mean_confidence=conf,
            )
            final_preds[idx] = (
                1 if final_decision in (Decision.FLAG, Decision.BLOCK) else 0
            )
            continue

        # 2. Check cached judge results
        if sample_id in cached_judge_results:
            j_data = cached_judge_results[sample_id]
            verdict_str = j_data.get("verdict", "abstain")
            conf = float(j_data.get("confidence", 0.5))
            lat = float(j_data.get("latency_ms", 0.0))
            latencies.append(lat)
            try:
                v_enum = ChallengerVerdict(verdict_str)
            except ValueError:
                v_enum = ChallengerVerdict.ABSTAIN

            final_decision, _ = reviewer.apply_authority_policy(
                verdict=v_enum,
                original_decision=base_decision,
                unanimous=True,
                mean_confidence=conf,
            )
            final_preds[idx] = (
                1 if final_decision in (Decision.FLAG, Decision.BLOCK) else 0
            )
            continue

        # 3. Live call fallback
        try:
            with gpu_lock(timeout_seconds=300):
                j_res = await judge.evaluate(
                    prompt=text,
                    counter_evidence_hypothesis="External eval adversarial review",
                    detector_summary={"results": [r.name for r in rec["results"]]},
                )
            lat = j_res.latency_ms
            latencies.append(lat)
            cached_judge_results[sample_id] = {
                "sample_id": sample_id,
                "verdict": j_res.verdict.value,
                "confidence": j_res.confidence,
                "reasoning": j_res.reasoning,
                "latency_ms": lat,
                "raw_response": j_res.raw_response,
                "error": j_res.error,
            }
            final_decision, _ = reviewer.apply_authority_policy(
                verdict=j_res.verdict,
                original_decision=base_decision,
                unanimous=True,
                mean_confidence=j_res.confidence,
            )
            final_preds[idx] = (
                1 if final_decision in (Decision.FLAG, Decision.BLOCK) else 0
            )
        except Exception as exc:  # noqa: BLE001
            error_count += 1
            logger.warning(
                "Judge call failed for sample %s: %s (keeping base pred)",
                sample_id,
                exc,
            )
            final_preds[idx] = base_preds[idx]
            latencies.append(0.0)
            cached_judge_results[sample_id] = {
                "sample_id": sample_id,
                "verdict": ChallengerVerdict.ABSTAIN.value,
                "confidence": 0.5,
                "reasoning": f"Exception: {exc}",
                "latency_ms": 0.0,
                "raw_response": "",
                "error": str(exc),
            }

    return final_preds, latencies, error_count


def main() -> None:
    logger.info("Loading external benchmark from %s...", BENCHMARK_PATH)
    samples: list[dict[str, Any]] = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    total_n = len(samples)
    y_true = [1 if s["is_attack"] else 0 for s in samples]
    clusters = [s.get("family", s.get("source", "default")) for s in samples]
    logger.info(
        "Loaded %d samples (%d attacks, %d benign).",
        total_n,
        sum(y_true),
        total_n - sum(y_true),
    )

    arm_a_fusion = AnyFlagFusion()
    mean_fusion = MeanFusion()
    rule_trigger = ChallengerTrigger(config=TriggerConfig())
    router = LearnedReviewRouter.load(ROUTER_ARTIFACT)

    cached_data: dict[str, Any] = {}
    if RAW_SCORES_PATH.exists():
        try:
            with open(RAW_SCORES_PATH, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            logger.info(
                "Loaded %d cached scores from %s", len(cached_data), RAW_SCORES_PATH
            )
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load cache: %s", e)
            cached_data = {}

    cached_judge_results: dict[str, Any] = {}
    if JUDGE_CACHE_PATH.exists():
        try:
            with open(JUDGE_CACHE_PATH, "r", encoding="utf-8") as f:
                cached_judge_results = json.load(f)
            logger.info(
                "Loaded %d cached judge results from %s",
                len(cached_judge_results),
                JUDGE_CACHE_PATH,
            )
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load judge cache: %s", e)
            cached_judge_results = {}

    missing_samples = [s for s in samples if s["id"] not in cached_data]
    if missing_samples:
        score_missing_samples(missing_samples, cached_data)
        with open(RAW_SCORES_PATH, "w", encoding="utf-8") as f:
            json.dump(cached_data, f, indent=2)
        logger.info("Saved all %d raw scores to %s", len(cached_data), RAW_SCORES_PATH)

    scored_records: list[dict[str, Any]] = []
    pf_errors_count = 0

    for sample in samples:
        cd = cached_data[sample["id"]]
        r1 = DetectorResult(
            name=cd["d1"]["name"],
            is_attack=cd["d1"]["is_attack"],
            score=cd["d1"]["score"],
            latency_ms=cd["d1"]["latency_ms"],
            error=cd["d1"]["error"],
        )
        r2 = DetectorResult(
            name=cd["d2"]["name"],
            is_attack=cd["d2"]["is_attack"],
            score=cd["d2"]["score"],
            latency_ms=cd["d2"]["latency_ms"],
            error=cd["d2"]["error"],
        )
        r3 = DetectorResult(
            name=cd["d3"]["name"],
            is_attack=cd["d3"]["is_attack"],
            score=cd["d3"]["score"],
            latency_ms=cd["d3"]["latency_ms"],
            error=cd["d3"]["error"],
        )

        pf_entry = cd.get("promptforest", {})
        pf_score = pf_entry.get("score")
        pf_is_atk = pf_entry.get("is_attack")
        pf_latency = pf_entry.get("latency_ms")
        pf_error = pf_entry.get("error")

        if pf_score is None or pf_error is not None:
            pf_errors_count += 1
            pf_is_atk = False
        elif pf_is_atk is None:
            pf_is_atk = bool(pf_score >= 0.5)

        if pf_latency is None:
            pf_latency = 0.0

        scored_records.append(
            {
                "sample": sample,
                "d1": r1,
                "d2": r2,
                "d3": r3,
                "results": [r1, r2, r3],
                "promptforest_score": pf_score,
                "promptforest_is_attack": pf_is_atk,
                "promptforest_latency": pf_latency,
                "promptforest_error": pf_error,
            }
        )

    # 1. Evaluate Individual Detectors
    evaluations: dict[str, Any] = {}

    for det_key, name in [
        ("d1", "MiniLM"),
        ("d2", "DeBERTa (ProtectAI)"),
        ("d3", "DeBERTa (Deepset)"),
    ]:
        preds = [
            1 if r[det_key].score is not None and r[det_key].score >= 0.5 else 0
            for r in scored_records
        ]
        latencies = [r[det_key].latency_ms for r in scored_records]
        m = compute_binary_metrics(y_true, preds)
        fpr = m.confusion_matrix.matrix[0][1] / max(1, m.support_negative)
        ci = compute_cluster_and_wilson_ci(y_true, preds, clusters)
        evaluations[det_key] = {
            "name": name,
            "recall": round(m.recall, 4),
            "recall_ci_95": ci["recall_ci_95"],
            "recall_wilson_ci_95": ci["recall_wilson_ci_95"],
            "fpr": round(fpr, 4),
            "fpr_ci_95": ci["fpr_ci_95"],
            "fpr_wilson_ci_95": ci["fpr_wilson_ci_95"],
            "precision": round(m.precision, 4),
            "f1": round(m.f1, 4),
            "accuracy": round(m.accuracy, 4),
            "p50_latency_ms": round(float(np.median(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        }

    # 2. Evaluate PromptForest Baseline
    pf_preds = [1 if r["promptforest_is_attack"] else 0 for r in scored_records]
    pf_latencies = [r["promptforest_latency"] for r in scored_records]
    pf_m = compute_binary_metrics(y_true, pf_preds)
    pf_fpr = pf_m.confusion_matrix.matrix[0][1] / max(1, pf_m.support_negative)
    pf_ci = compute_cluster_and_wilson_ci(y_true, pf_preds, clusters)
    evaluations["promptforest"] = {
        "name": "PromptForest Live (2-Detector Ensemble)",
        "recall": round(pf_m.recall, 4),
        "recall_ci_95": pf_ci["recall_ci_95"],
        "recall_wilson_ci_95": pf_ci["recall_wilson_ci_95"],
        "fpr": round(pf_fpr, 4),
        "fpr_ci_95": pf_ci["fpr_ci_95"],
        "fpr_wilson_ci_95": pf_ci["fpr_wilson_ci_95"],
        "precision": round(pf_m.precision, 4),
        "f1": round(pf_m.f1, 4),
        "accuracy": round(pf_m.accuracy, 4),
        "p50_latency_ms": round(float(np.median(pf_latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(pf_latencies, 95)), 2),
        "detector_errors_count": pf_errors_count,
    }

    # 3. Evaluate Arm A (AnyFlag Fusion)
    arm_a_preds: list[int] = []
    arm_a_latencies: list[float] = []
    for r in scored_records:
        fused = arm_a_fusion.fuse(r["results"])
        arm_a_preds.append(
            1 if fused.decision in (Decision.FLAG, Decision.BLOCK) else 0
        )
        arm_a_latencies.append(max(res.latency_ms for res in r["results"]))
    arm_a_m = compute_binary_metrics(y_true, arm_a_preds)
    arm_a_fpr = arm_a_m.confusion_matrix.matrix[0][1] / max(1, arm_a_m.support_negative)
    arm_a_ci = compute_cluster_and_wilson_ci(y_true, arm_a_preds, clusters)
    evaluations["arm_a_anyflag"] = {
        "name": "Arm A (AnyFlag Fusion)",
        "recall": round(arm_a_m.recall, 4),
        "recall_ci_95": arm_a_ci["recall_ci_95"],
        "recall_wilson_ci_95": arm_a_ci["recall_wilson_ci_95"],
        "fpr": round(arm_a_fpr, 4),
        "fpr_ci_95": arm_a_ci["fpr_ci_95"],
        "fpr_wilson_ci_95": arm_a_ci["fpr_wilson_ci_95"],
        "precision": round(arm_a_m.precision, 4),
        "f1": round(arm_a_m.f1, 4),
        "accuracy": round(arm_a_m.accuracy, 4),
        "p50_latency_ms": round(float(np.median(arm_a_latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(arm_a_latencies, 95)), 2),
        "extra_calls_per_100": 0.0,
    }

    # 4. Evaluate Mean Fusion
    mean_preds: list[int] = []
    for r in scored_records:
        fused = mean_fusion.fuse(r["results"])
        mean_preds.append(1 if fused.decision in (Decision.FLAG, Decision.BLOCK) else 0)
    mean_m = compute_binary_metrics(y_true, mean_preds)
    mean_fpr = mean_m.confusion_matrix.matrix[0][1] / max(1, mean_m.support_negative)
    mean_ci = compute_cluster_and_wilson_ci(y_true, mean_preds, clusters)
    evaluations["mean_fusion"] = {
        "name": "Mean Fusion",
        "recall": round(mean_m.recall, 4),
        "recall_ci_95": mean_ci["recall_ci_95"],
        "recall_wilson_ci_95": mean_ci["recall_wilson_ci_95"],
        "fpr": round(mean_fpr, 4),
        "fpr_ci_95": mean_ci["fpr_ci_95"],
        "fpr_wilson_ci_95": mean_ci["fpr_wilson_ci_95"],
        "precision": round(mean_m.precision, 4),
        "f1": round(mean_m.f1, 4),
        "accuracy": round(mean_m.accuracy, 4),
        "extra_calls_per_100": 0.0,
    }

    # 5. Evaluate Arm B (Rule-based Trigger Policy with Real Live Judge)
    rule_routed: set[int] = set()
    for idx, r in enumerate(scored_records):
        fused = arm_a_fusion.fuse(r["results"])
        prep = preprocess(r["sample"]["text"])
        decision_conf = (
            1.0 - fused.confidence
            if fused.decision == Decision.ALLOW
            else fused.confidence
        )
        t_res = rule_trigger.evaluate(
            detector_results=r["results"],
            fused_score=fused.confidence,
            confidence=decision_conf,
            original_decision=fused.decision,
            encoding_present=prep.encoding_present,
            encoding_types=prep.encoding_types,
        )
        if t_res.should_challenge:
            rule_routed.add(idx)

    rule_preds, _rule_judge_lats, rule_judge_errs = asyncio.run(
        evaluate_routed_prompts_live(
            scored_records, rule_routed, arm_a_preds, cached_judge_results
        )
    )
    rule_m = compute_binary_metrics(y_true, rule_preds)
    rule_fpr = rule_m.confusion_matrix.matrix[0][1] / max(1, rule_m.support_negative)
    rule_ci = compute_cluster_and_wilson_ci(y_true, rule_preds, clusters)
    rule_calls = round((len(rule_routed) / total_n) * 100.0, 2)
    evaluations["arm_b_rule_trigger"] = {
        "name": "Arm B (Rule Trigger + Real Review)",
        "recall": round(rule_m.recall, 4),
        "recall_ci_95": rule_ci["recall_ci_95"],
        "recall_wilson_ci_95": rule_ci["recall_wilson_ci_95"],
        "fpr": round(rule_fpr, 4),
        "fpr_ci_95": rule_ci["fpr_ci_95"],
        "fpr_wilson_ci_95": rule_ci["fpr_wilson_ci_95"],
        "precision": round(rule_m.precision, 4),
        "f1": round(rule_m.f1, 4),
        "accuracy": round(rule_m.accuracy, 4),
        "extra_calls_per_100": rule_calls,
        "routed_count": len(rule_routed),
        "judge_errors": rule_judge_errs,
    }

    # 6. Evaluate Arm R (Learned Review Router with Real Live Judge)
    learned_routed: set[int] = set()
    for idx, r in enumerate(scored_records):
        fused = arm_a_fusion.fuse(r["results"])
        r_dec = router.route(r["sample"]["text"], r["results"], fused)
        if r_dec.should_route:
            learned_routed.add(idx)

    learned_preds, _learned_judge_lats, learned_judge_errs = asyncio.run(
        evaluate_routed_prompts_live(
            scored_records, learned_routed, arm_a_preds, cached_judge_results
        )
    )
    learned_m = compute_binary_metrics(y_true, learned_preds)
    learned_fpr = learned_m.confusion_matrix.matrix[0][1] / max(
        1, learned_m.support_negative
    )
    learned_ci = compute_cluster_and_wilson_ci(y_true, learned_preds, clusters)
    learned_calls = round((len(learned_routed) / total_n) * 100.0, 2)
    evaluations["arm_r_learned_router"] = {
        "name": "Arm R (Learned Review Router + Real Review)",
        "recall": round(learned_m.recall, 4),
        "recall_ci_95": learned_ci["recall_ci_95"],
        "recall_wilson_ci_95": learned_ci["recall_wilson_ci_95"],
        "fpr": round(learned_fpr, 4),
        "fpr_ci_95": learned_ci["fpr_ci_95"],
        "fpr_wilson_ci_95": learned_ci["fpr_wilson_ci_95"],
        "precision": round(learned_m.precision, 4),
        "f1": round(learned_m.f1, 4),
        "accuracy": round(learned_m.accuracy, 4),
        "extra_calls_per_100": learned_calls,
        "routed_count": len(learned_routed),
        "judge_errors": learned_judge_errs,
    }

    # Save judge cache
    with open(JUDGE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cached_judge_results, f, indent=2)

    # 8. Breakdown by Technique Family
    family_breakdown: dict[str, Any] = {}
    families = sorted({s["family"] for s in samples})
    for fam in families:
        fam_indices = [i for i, s in enumerate(samples) if s["family"] == fam]
        fam_true = [y_true[i] for i in fam_indices]
        fam_arm_a = [arm_a_preds[i] for i in fam_indices]
        fam_pf = [pf_preds[i] for i in fam_indices]
        fam_arm_r = [learned_preds[i] for i in fam_indices]

        n_fam = len(fam_indices)
        is_attack_fam = fam_true[0] == 1 if fam_true else False
        if is_attack_fam:
            fam_metric = {
                "count": n_fam,
                "arm_a_recall": round(float(np.mean(fam_arm_a)), 4),
                "promptforest_recall": round(float(np.mean(fam_pf)), 4),
                "arm_r_recall": round(float(np.mean(fam_arm_r)), 4),
            }
        else:
            fam_metric = {
                "count": n_fam,
                "arm_a_fpr": round(float(np.mean(fam_arm_a)), 4),
                "promptforest_fpr": round(float(np.mean(fam_pf)), 4),
                "arm_r_fpr": round(float(np.mean(fam_arm_r)), 4),
            }
        family_breakdown[fam] = fam_metric

    run_id = f"ext-eval-{uuid.uuid4().hex[:8]}"
    output_payload = {
        "benchmark": "data/external_eval/external_benchmark.jsonl",
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "total_samples": total_n,
        "attacks_count": sum(y_true),
        "benign_count": total_n - sum(y_true),
        "evaluations": evaluations,
        "family_breakdown": family_breakdown,
    }

    logger.info("Writing external evaluation results to %s...", OUT_PATH)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("External evaluation completed successfully with run_id %s!", run_id)


if __name__ == "__main__":
    main()
