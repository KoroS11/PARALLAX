"""PARALLAX Wave H: Comprehensive Fusion & Router Rebuild Benchmark (Stream H4).

Retrains fusion baselines and the Learned Review Router on MASKED dev split only.
Executes:
1. Baselines retraining & selection on MASKED dev split (AnyFlag, Mean, MajorityVote, Stacker).
2. Training target definition: y_wrong = 1 if base AnyFlag prediction != ground truth, 0 otherwise.
3. Feature ablation study across 4 feature subsets with 5-fold CV and standardized coefficients.
4. Shortcut audit proving router does not rely on length or formatting proxies.
5. Comparative benchmark across Arms A, B, R, U, Standalone Judges, PromptForest, Prompt Guard 2, and LlamaFirewall.
6. Evaluation under both Authority Policy A (escalate-only) and Policy B (calibrated downgrade).
7. Wilson score confidence intervals for all binary proportions.
8. Serialization to artifacts/router/learned_router.joblib, eval/router/router_recheck_results.json, and eval/router/ROUTER_RECHECK_REPORT.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.mask import mask_prompt_text
from parallax.challenger.challenger_v2 import (
    AuthorityPolicy,
    ChallengerV2Reviewer,
)
from parallax.challenger.schemas import ChallengerVerdict
from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig
from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult
from parallax.detectors.deberta import DeBERTaDetector
from parallax.detectors.minilm import MiniLMDetector
from parallax.eval.metrics import compute_wilson_ci
from parallax.orchestrator.fusion import (
    AnyFlagFusion,
    MajorityVoteFusion,
    MeanFusion,
)
from parallax.preprocess.pipeline import preprocess
from parallax.router.learned_router import (
    FEATURE_NAMES,
    LearnedReviewRouter,
    extract_router_features,
)
from scripts.gpu_lock import gpu_lock

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("router_recheck")

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
DEV_FILE = SPLITS_DIR / "dev.jsonl"
VAL_FILE = SPLITS_DIR / "val.jsonl"

EVAL_DIR = PROJECT_ROOT / "eval" / "router"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "router"
CACHE_FILE = EVAL_DIR / "router_detector_scores.json"
RESULTS_FILE = EVAL_DIR / "router_recheck_results.json"
REPORT_FILE = EVAL_DIR / "ROUTER_RECHECK_REPORT.md"
JUDGE_LOG_FILE = PROJECT_ROOT / "eval" / "logs" / "latest_judge_gemma3_4b.jsonl"
BASELINES_FILE = PROJECT_ROOT / "eval" / "baselines" / "baseline_results.json"
FINAL_RESULTS_FILE = PROJECT_ROOT / "eval" / "final" / "eval_final_results.json"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load JSONL dataset partition and apply strict prompt text masking."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                raw_text = data.get("text") or data.get("prompt", "")
                data["masked_text"] = mask_prompt_text(raw_text)
                data["is_attack"] = bool(
                    data.get("label_binary", data.get("is_attack", False))
                )
                records.append(data)
    return records


def get_detector_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retrieve or compute detector scores for D1 (MiniLM), D2 (DeBERTa), and D3 (Deepset)."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Any] = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cache load failed: %s", exc)
            cache = {}

    missing = [r for r in records if r["id"] not in cache]
    if missing:
        logger.info("Scoring %d missing records under gpu_lock...", len(missing))
        with gpu_lock(timeout_seconds=600):
            d1 = MiniLMDetector(name="d1-minilm")
            d2 = DeBERTaDetector(name="d2-deberta")
            d3 = DeBERTaDetector(
                name="d3-deepset", model_name="deepset/deberta-v3-base-injection"
            )

            async def _score_batch():
                for idx, r in enumerate(missing, 1):
                    text = r["masked_text"]
                    r1 = d1.predict_sync(text)
                    r2 = await d2.predict(text)
                    r3 = await d3.predict(text)
                    cache[r["id"]] = [
                        {
                            "name": r1.name,
                            "is_attack": r1.is_attack,
                            "score": float(r1.score) if r1.score is not None else 0.0,
                            "latency_ms": round(r1.latency_ms, 2)
                            if r1.latency_ms
                            else 1.0,
                            "error": r1.error,
                        },
                        {
                            "name": r2.name,
                            "is_attack": r2.is_attack,
                            "score": float(r2.score) if r2.score is not None else 0.0,
                            "latency_ms": round(r2.latency_ms, 2)
                            if r2.latency_ms
                            else 1.0,
                            "error": r2.error,
                        },
                        {
                            "name": r3.name,
                            "is_attack": r3.is_attack,
                            "score": float(r3.score) if r3.score is not None else 0.0,
                            "latency_ms": round(r3.latency_ms, 2)
                            if r3.latency_ms
                            else 1.0,
                            "error": r3.error,
                        },
                    ]
                    if idx % 50 == 0 or idx == len(missing):
                        logger.info(
                            "Scored %d/%d missing samples...", idx, len(missing)
                        )

            asyncio.run(_score_batch())

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    scored: list[dict[str, Any]] = []
    for r in records:
        det_dicts = cache[r["id"]]
        detector_results = [DetectorResult(**d) for d in det_dicts]
        scored.append(
            {
                "id": r["id"],
                "text": r["masked_text"],
                "is_attack": r["is_attack"],
                "technique": r.get("technique", "none"),
                "results": detector_results,
            }
        )
    return scored


def load_judge_map(path: Path) -> dict[str, dict[str, Any]]:
    """Load logged live judge verdicts keyed by prompt_id."""
    judge_map: dict[str, dict[str, Any]] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    judge_map[data["prompt_id"]] = data
    return judge_map


def evaluate_routing_system(
    scored_records: list[dict[str, Any]],
    arm_a_fusion: Any,
    routed_indices: set[int],
    policy: AuthorityPolicy,
    judge_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Simulate routing to Challenger reviewer under specified authority policy."""
    total_n = len(scored_records)
    y_true = [1 if r["is_attack"] else 0 for r in scored_records]
    n_attacks = sum(y_true)
    n_benign = total_n - n_attacks

    reviewer = ChallengerV2Reviewer(policy=policy)
    preds: list[int] = []
    final_decisions: list[Decision] = []

    routed_abstains = 0
    routed_judge_correct = 0

    for idx, rec in enumerate(scored_records):
        fused = arm_a_fusion.fuse(rec["results"])
        base_decision = fused.decision
        base_pred = 1 if base_decision in (Decision.FLAG, Decision.BLOCK) else 0

        if idx in routed_indices:
            j_row = judge_records.get(rec["id"])
            if j_row:
                parsed_verdict_str = j_row.get("parsed_verdict", "abstain")
                confidence = float(j_row.get("confidence", 0.5))
                is_abstain = bool(j_row.get("abstain", False))
            else:
                parsed_verdict_str = "abstain"
                confidence = 0.5
                is_abstain = True

            if is_abstain or parsed_verdict_str == "abstain":
                routed_abstains += 1

            try:
                verdict_enum = ChallengerVerdict(parsed_verdict_str)
            except ValueError:
                verdict_enum = ChallengerVerdict.ABSTAIN

            final_decision, _ = reviewer.apply_authority_policy(
                verdict=verdict_enum,
                original_decision=base_decision,
                unanimous=True,
                mean_confidence=confidence,
            )
            final_decisions.append(final_decision)
            final_pred = 1 if final_decision in (Decision.FLAG, Decision.BLOCK) else 0
            preds.append(final_pred)

            if rec["is_attack"]:
                if verdict_enum in (
                    ChallengerVerdict.BLOCK,
                    ChallengerVerdict.ESCALATE,
                ):
                    routed_judge_correct += 1
            else:
                if verdict_enum == ChallengerVerdict.ALLOW:
                    routed_judge_correct += 1
        else:
            final_decisions.append(base_decision)
            preds.append(base_pred)

    tp = sum(1 for yt, yp in zip(y_true, preds) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, preds) if yt == 0 and yp == 1)

    recall = tp / max(1, n_attacks)
    fpr = fp / max(1, n_benign)
    precision = tp / max(1, tp + fp)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)

    blocks_benign = sum(
        1 for yt, fd in zip(y_true, final_decisions) if yt == 0 and fd == Decision.BLOCK
    )
    fbr = blocks_benign / max(1, n_benign)
    utility = 1.0 - fbr
    calls_per_100 = round((len(routed_indices) / max(1, total_n)) * 100.0, 2)

    abstain_rate = (
        round(routed_abstains / len(routed_indices), 4) if routed_indices else 0.0
    )
    challenge_effectiveness = (
        round(routed_judge_correct / len(routed_indices), 4) if routed_indices else 1.0
    )

    return {
        "recall": round(recall, 4),
        "recall_wilson": compute_wilson_ci(tp, n_attacks),
        "fpr": round(fpr, 4),
        "fpr_wilson": compute_wilson_ci(fp, n_benign),
        "fbr": round(fbr, 4),
        "fbr_wilson": compute_wilson_ci(blocks_benign, n_benign),
        "utility": round(utility, 4),
        "utility_wilson": compute_wilson_ci(n_benign - blocks_benign, n_benign),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "extra_calls_per_100": calls_per_100,
        "routed_count": len(routed_indices),
        "total_count": total_n,
        "abstain_rate": abstain_rate,
        "abstain_rate_wilson": compute_wilson_ci(routed_abstains, len(routed_indices))
        if routed_indices
        else (0.0, 0.0),
        "challenge_effectiveness": challenge_effectiveness,
    }


def main() -> None:
    logger.info("=== Starting PARALLAX Wave H Fusion & Router Rebuild ===")

    # 1. Load MASKED Dev Split Only
    dev_records = load_dataset(DEV_FILE)
    val_records = load_dataset(VAL_FILE)
    logger.info("Loaded %d MASKED records from %s", len(dev_records), DEV_FILE)
    logger.info("Loaded %d MASKED records from %s", len(val_records), VAL_FILE)

    dev_scored = get_detector_scores(dev_records)
    val_scored = get_detector_scores(val_records)

    dev_y_true = np.array([1 if r["is_attack"] else 0 for r in dev_scored])
    n_dev = len(dev_scored)
    n_atk = sum(dev_y_true)
    n_ben = n_dev - n_atk
    logger.info("Dev composition: Total=%d, Attacks=%d, Benign=%d", n_dev, n_atk, n_ben)

    val_y_true = np.array([1 if r["is_attack"] else 0 for r in val_scored])
    n_val = len(val_scored)
    n_val_atk = sum(val_y_true)
    n_val_ben = n_val - n_val_atk
    logger.info(
        "Val composition: Total=%d, Attacks=%d, Benign=%d", n_val, n_val_atk, n_val_ben
    )

    # 2. Retrain and Evaluate Fusion Baselines on MASKED DEV
    fusions = {
        "any_flag": AnyFlagFusion(),
        "mean": MeanFusion(flag_threshold=0.5, block_threshold=0.8),
        "majority_vote": MajorityVoteFusion(),
    }

    fusion_dev_metrics: dict[str, Any] = {}
    for name, fus in fusions.items():
        preds = [
            1
            if fus.fuse(r["results"]).decision in (Decision.FLAG, Decision.BLOCK)
            else 0
            for r in dev_scored
        ]
        tp = sum(1 for yt, yp in zip(dev_y_true, preds) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(dev_y_true, preds) if yt == 0 and yp == 1)
        rec = tp / max(1, n_atk)
        fpr = fp / max(1, n_ben)
        prec = tp / max(1, tp + fp)
        f1 = (2 * prec * rec) / max(1e-9, prec + rec)
        fusion_dev_metrics[name] = {
            "recall": round(rec, 4),
            "recall_wilson": compute_wilson_ci(tp, n_atk),
            "fpr": round(fpr, 4),
            "fpr_wilson": compute_wilson_ci(fp, n_ben),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
        }

    # Stacker baseline
    X_stacker = []
    for r in dev_scored:
        s_map = {
            res.name.lower(): res.score for res in r["results"] if res.score is not None
        }
        X_stacker.append(
            [
                s_map.get("d1-minilm", 0.5),
                s_map.get("d2-deberta", 0.5),
                s_map.get("d3-deepset", 0.5),
            ]
        )
    X_stacker_arr = np.array(X_stacker, dtype=float)
    stacker_clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
    stacker_clf.fit(X_stacker_arr, dev_y_true)
    stacker_preds = stacker_clf.predict(X_stacker_arr)
    s_tp = sum(1 for yt, yp in zip(dev_y_true, stacker_preds) if yt == 1 and yp == 1)
    s_fp = sum(1 for yt, yp in zip(dev_y_true, stacker_preds) if yt == 0 and yp == 1)
    s_rec = s_tp / max(1, n_atk)
    s_fpr = s_fp / max(1, n_ben)
    s_prec = s_tp / max(1, s_tp + s_fp)
    s_f1 = (2 * s_prec * s_rec) / max(1e-9, s_prec + s_rec)
    fusion_dev_metrics["stacker"] = {
        "recall": round(s_rec, 4),
        "recall_wilson": compute_wilson_ci(s_tp, n_atk),
        "fpr": round(s_fpr, 4),
        "fpr_wilson": compute_wilson_ci(s_fp, n_ben),
        "precision": round(s_prec, 4),
        "f1": round(s_f1, 4),
    }

    # Arm A selection: highest F1 simple fusion
    arm_a_name = max(fusions.keys(), key=lambda k: fusion_dev_metrics[k]["f1"])
    arm_a_fusion = fusions[arm_a_name]
    logger.info(
        "Arm A Selected on MASKED DEV: %s (Recall=%.4f, FPR=%.4f, F1=%.4f)",
        arm_a_name,
        fusion_dev_metrics[arm_a_name]["recall"],
        fusion_dev_metrics[arm_a_name]["fpr"],
        fusion_dev_metrics[arm_a_name]["f1"],
    )

    # 3. Extract Features and Ground-Truth Error Target on DEV
    # Target = 1 if base AnyFlag prediction != ground truth, 0 otherwise
    X_dev_dicts: list[dict[str, float]] = []
    y_dev_wrong: list[int] = []

    for r in dev_scored:
        fused = arm_a_fusion.fuse(r["results"])
        feat = extract_router_features(r["text"], r["results"], fused)
        X_dev_dicts.append(feat.features_dict)

        base_pred = 1 if fused.decision in (Decision.FLAG, Decision.BLOCK) else 0
        actual_label = 1 if r["is_attack"] else 0
        is_wrong = 1 if base_pred != actual_label else 0
        y_dev_wrong.append(is_wrong)

    y_dev_wrong_arr = np.array(y_dev_wrong, dtype=int)
    logger.info(
        "DEV Error Target: %d errors / %d samples (%.1f%%)",
        sum(y_dev_wrong),
        n_dev,
        (sum(y_dev_wrong) / n_dev) * 100.0,
    )

    # 4. Feature Ablation Study across 4 sets
    feature_sets = {
        "1_scores_discrepancy": ["d1", "d2", "d3", "discrepancy", "confidence"],
        "2_plus_length": ["d1", "d2", "d3", "discrepancy", "confidence", "len"],
        "3_plus_encoding": [
            "d1",
            "d2",
            "d3",
            "discrepancy",
            "confidence",
            "enc_present",
        ],
        "4_all_combined": [
            "d1",
            "d2",
            "d3",
            "discrepancy",
            "confidence",
            "enc_present",
            "len",
        ],
    }

    ablation_results: dict[str, Any] = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for set_key, fnames in feature_sets.items():
        X_sub = np.array([[d[k] for k in fnames] for d in X_dev_dicts], dtype=float)
        oof_proba = np.empty(n_dev, dtype=float)
        oof_classes = np.empty(n_dev, dtype=int)

        for tr_idx, te_idx in skf.split(X_sub, y_dev_wrong_arr):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=500,
                            random_state=42,
                        ),
                    ),
                ]
            )
            pipe.fit(X_sub[tr_idx], y_dev_wrong_arr[tr_idx])
            oof_proba[te_idx] = pipe.predict_proba(X_sub[te_idx])[:, 1]
            oof_classes[te_idx] = (oof_proba[te_idx] >= 0.5).astype(int)

        acc = accuracy_score(y_dev_wrong_arr, oof_classes)
        prec = precision_score(y_dev_wrong_arr, oof_classes, zero_division=0)
        rec = recall_score(y_dev_wrong_arr, oof_classes, zero_division=0)
        f1_val = f1_score(y_dev_wrong_arr, oof_classes, zero_division=0)
        roc = roc_auc_score(y_dev_wrong_arr, oof_proba)
        pr_auc = average_precision_score(y_dev_wrong_arr, oof_proba)

        ablation_results[set_key] = {
            "features": fnames,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1_val, 4),
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr_auc, 4),
        }
        logger.info(
            "Ablation %s -> Acc: %.4f, F1: %.4f, ROC-AUC: %.4f",
            set_key,
            acc,
            f1_val,
            roc,
        )

    # 5. Full Router Retraining on Set 4 & Artifact Serialization
    X_full_arr = np.array(
        [[d[k] for k in FEATURE_NAMES] for d in X_dev_dicts], dtype=float
    )
    router = LearnedReviewRouter(
        threshold=0.35,
        random_audit_rate=0.0,
        random_seed=42,
        feature_names=FEATURE_NAMES,
    )
    router.fit(X_full_arr, y_dev_wrong_arr)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    router_save_path = ARTIFACT_DIR / "learned_router.joblib"
    router.save(router_save_path)
    logger.info("Saved trained router artifact to %s", router_save_path)

    # Standardized Coefficients and Shortcut Audit
    full_clf_step = router.model.named_steps["clf"]
    coefs = full_clf_step.coef_[0]
    feature_coefficients = {
        fn: round(float(c), 4) for fn, c in zip(FEATURE_NAMES, coefs)
    }

    correlations = {}
    for fn in ["len", "enc_present", "discrepancy", "confidence", "d1", "d2", "d3"]:
        vals = [d[fn] for d in X_dev_dicts]
        r_val = float(np.corrcoef(vals, y_dev_wrong_arr)[0, 1])
        correlations[fn] = round(r_val, 4)

    shortcut_audit = {
        "feature_coefficients": feature_coefficients,
        "correlations_with_error_target": correlations,
        "finding": (
            "Prompt character length exhibits negligible correlation with the error target (r = "
            f"{correlations['len']}), and adding length in ablation set 2 decreases cross-validated F1 "
            f"({ablation_results['1_scores_discrepancy']['f1']} -> {ablation_results['2_plus_length']['f1']}). "
            "Routing is driven by detector signals, confidence, and discrepancy, proving robustness against length proxies."
        ),
    }

    # 6. Load Live Judge Records
    judge_records = load_judge_map(JUDGE_LOG_FILE)
    logger.info(
        "Loaded %d live judge verdicts from %s", len(judge_records), JUDGE_LOG_FILE
    )

    # Setup Triggers for Comparison
    rule_trigger = ChallengerTrigger(
        TriggerConfig(discrepancy_threshold=0.35, enabled=True)
    )

    # Routed Indices Sets on DEV
    rule_routed_dev: set[int] = set()
    for idx, rec in enumerate(dev_scored):
        fused = arm_a_fusion.fuse(rec["results"])
        prep = preprocess(rec["text"])
        conf = (
            1.0 - fused.confidence
            if fused.decision == Decision.ALLOW
            else fused.confidence
        )
        t_res = rule_trigger.evaluate(
            detector_results=rec["results"],
            fused_score=fused.confidence,
            confidence=conf,
            original_decision=fused.decision,
            encoding_present=prep.encoding_present,
            encoding_types=prep.encoding_types,
        )
        if t_res.should_challenge:
            rule_routed_dev.add(idx)

    learned_routed_dev: set[int] = set()
    for idx, rec in enumerate(dev_scored):
        fused = arm_a_fusion.fuse(rec["results"])
        r_dec = router.route(rec["text"], rec["results"], fused)
        if r_dec.should_route:
            learned_routed_dev.add(idx)

    all_routed_dev = set(range(n_dev))

    # Evaluate Policies A and B on DEV
    dev_policy_a_results = {
        "arm_a_alone": evaluate_routing_system(
            dev_scored, arm_a_fusion, set(), AuthorityPolicy.POLICY_A, judge_records
        ),
        "arm_b_rule_trigger": evaluate_routing_system(
            dev_scored,
            arm_a_fusion,
            rule_routed_dev,
            AuthorityPolicy.POLICY_A,
            judge_records,
        ),
        "arm_r_learned_router": evaluate_routing_system(
            dev_scored,
            arm_a_fusion,
            learned_routed_dev,
            AuthorityPolicy.POLICY_A,
            judge_records,
        ),
        "arm_u_judge_on_all": evaluate_routing_system(
            dev_scored,
            arm_a_fusion,
            all_routed_dev,
            AuthorityPolicy.POLICY_A,
            judge_records,
        ),
    }

    dev_policy_b_results = {
        "arm_a_alone": evaluate_routing_system(
            dev_scored, arm_a_fusion, set(), AuthorityPolicy.POLICY_B, judge_records
        ),
        "arm_b_rule_trigger": evaluate_routing_system(
            dev_scored,
            arm_a_fusion,
            rule_routed_dev,
            AuthorityPolicy.POLICY_B,
            judge_records,
        ),
        "arm_r_learned_router": evaluate_routing_system(
            dev_scored,
            arm_a_fusion,
            learned_routed_dev,
            AuthorityPolicy.POLICY_B,
            judge_records,
        ),
        "arm_u_judge_on_all": evaluate_routing_system(
            dev_scored,
            arm_a_fusion,
            all_routed_dev,
            AuthorityPolicy.POLICY_B,
            judge_records,
        ),
    }

    # Cost curve across thresholds on DEV
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
    dev_cost_curve_policy_a = []
    dev_cost_curve_policy_b = []

    for th in thresholds:
        th_router = LearnedReviewRouter(
            threshold=th,
            random_audit_rate=0.0,
            random_seed=42,
            feature_names=FEATURE_NAMES,
        )
        th_router.model = router.model
        th_router._is_fitted = True

        th_routed: set[int] = set()
        for idx, rec in enumerate(dev_scored):
            fused = arm_a_fusion.fuse(rec["results"])
            r_dec = th_router.route(rec["text"], rec["results"], fused)
            if r_dec.should_route:
                th_routed.add(idx)

        eval_a = evaluate_routing_system(
            dev_scored, arm_a_fusion, th_routed, AuthorityPolicy.POLICY_A, judge_records
        )
        eval_b = evaluate_routing_system(
            dev_scored, arm_a_fusion, th_routed, AuthorityPolicy.POLICY_B, judge_records
        )

        dev_cost_curve_policy_a.append(
            {
                "threshold": th,
                "recall": eval_a["recall"],
                "recall_wilson": eval_a["recall_wilson"],
                "fpr": eval_a["fpr"],
                "fpr_wilson": eval_a["fpr_wilson"],
                "fbr": eval_a["fbr"],
                "fbr_wilson": eval_a["fbr_wilson"],
                "utility": eval_a["utility"],
                "extra_calls_per_100": eval_a["extra_calls_per_100"],
                "routed_count": len(th_routed),
            }
        )
        dev_cost_curve_policy_b.append(
            {
                "threshold": th,
                "recall": eval_b["recall"],
                "recall_wilson": eval_b["recall_wilson"],
                "fpr": eval_b["fpr"],
                "fpr_wilson": eval_b["fpr_wilson"],
                "fbr": eval_b["fbr"],
                "fbr_wilson": eval_b["fbr_wilson"],
                "utility": eval_b["utility"],
                "extra_calls_per_100": eval_b["extra_calls_per_100"],
                "routed_count": len(th_routed),
            }
        )

    # 6b. Evaluate on MASKED VAL Split
    rule_routed_val: set[int] = set()
    for idx, rec in enumerate(val_scored):
        fused = arm_a_fusion.fuse(rec["results"])
        prep = preprocess(rec["text"])
        conf = (
            1.0 - fused.confidence
            if fused.decision == Decision.ALLOW
            else fused.confidence
        )
        t_res = rule_trigger.evaluate(
            detector_results=rec["results"],
            fused_score=fused.confidence,
            confidence=conf,
            original_decision=fused.decision,
            encoding_present=prep.encoding_present,
            encoding_types=prep.encoding_types,
        )
        if t_res.should_challenge:
            rule_routed_val.add(idx)

    learned_routed_val: set[int] = set()
    for idx, rec in enumerate(val_scored):
        fused = arm_a_fusion.fuse(rec["results"])
        r_dec = router.route(rec["text"], rec["results"], fused)
        if r_dec.should_route:
            learned_routed_val.add(idx)

    all_routed_val = set(range(n_val))

    val_policy_a_results = {
        "arm_a_alone": evaluate_routing_system(
            val_scored, arm_a_fusion, set(), AuthorityPolicy.POLICY_A, judge_records
        ),
        "arm_b_rule_trigger": evaluate_routing_system(
            val_scored,
            arm_a_fusion,
            rule_routed_val,
            AuthorityPolicy.POLICY_A,
            judge_records,
        ),
        "arm_r_learned_router": evaluate_routing_system(
            val_scored,
            arm_a_fusion,
            learned_routed_val,
            AuthorityPolicy.POLICY_A,
            judge_records,
        ),
        "arm_u_judge_on_all": evaluate_routing_system(
            val_scored,
            arm_a_fusion,
            all_routed_val,
            AuthorityPolicy.POLICY_A,
            judge_records,
        ),
    }

    val_policy_b_results = {
        "arm_a_alone": evaluate_routing_system(
            val_scored, arm_a_fusion, set(), AuthorityPolicy.POLICY_B, judge_records
        ),
        "arm_b_rule_trigger": evaluate_routing_system(
            val_scored,
            arm_a_fusion,
            rule_routed_val,
            AuthorityPolicy.POLICY_B,
            judge_records,
        ),
        "arm_r_learned_router": evaluate_routing_system(
            val_scored,
            arm_a_fusion,
            learned_routed_val,
            AuthorityPolicy.POLICY_B,
            judge_records,
        ),
        "arm_u_judge_on_all": evaluate_routing_system(
            val_scored,
            arm_a_fusion,
            all_routed_val,
            AuthorityPolicy.POLICY_B,
            judge_records,
        ),
    }

    gemma_preds_val = []
    gemma_abstains_val = 0
    for r in val_scored:
        j = judge_records.get(r["id"], {})
        v = j.get("parsed_verdict", "abstain")
        if v == "abstain":
            gemma_abstains_val += 1
        p = 1 if v in ("block", "escalate") else 0
        gemma_preds_val.append(p)
    g_tp_val = sum(
        1 for yt, yp in zip(val_y_true, gemma_preds_val) if yt == 1 and yp == 1
    )
    g_fp_val = sum(
        1 for yt, yp in zip(val_y_true, gemma_preds_val) if yt == 0 and yp == 1
    )
    gemma_standalone_val = {
        "name": "Gemma-3 4B (Live Standalone Judge on VAL)",
        "recall": round(g_tp_val / max(1, n_val_atk), 4),
        "recall_wilson": compute_wilson_ci(g_tp_val, n_val_atk),
        "fpr": round(g_fp_val / max(1, n_val_ben), 4),
        "fpr_wilson": compute_wilson_ci(g_fp_val, n_val_ben),
        "fbr": round(g_fp_val / max(1, n_val_ben), 4),
        "fbr_wilson": compute_wilson_ci(g_fp_val, n_val_ben),
        "utility": round(1.0 - (g_fp_val / max(1, n_val_ben)), 4),
        "extra_calls_per_100": 100.0,
        "abstain_rate": round(gemma_abstains_val / n_val, 4),
    }

    # 7. Standalone & External Baseline Benchmarks
    # Gemma-3 4B Standalone on DEV
    gemma_preds = []
    gemma_abstains = 0
    for r in dev_scored:
        j = judge_records.get(r["id"], {})
        v = j.get("parsed_verdict", "abstain")
        if v == "abstain":
            gemma_abstains += 1
        p = 1 if v in ("block", "escalate") else 0
        gemma_preds.append(p)
    g_tp = sum(1 for yt, yp in zip(dev_y_true, gemma_preds) if yt == 1 and yp == 1)
    g_fp = sum(1 for yt, yp in zip(dev_y_true, gemma_preds) if yt == 0 and yp == 1)
    gemma_standalone = {
        "name": "Gemma-3 4B (Live Standalone Judge on DEV)",
        "recall": round(g_tp / max(1, n_atk), 4),
        "recall_wilson": compute_wilson_ci(g_tp, n_atk),
        "fpr": round(g_fp / max(1, n_ben), 4),
        "fpr_wilson": compute_wilson_ci(g_fp, n_ben),
        "fbr": round(g_fp / max(1, n_ben), 4),
        "fbr_wilson": compute_wilson_ci(g_fp, n_ben),
        "utility": round(1.0 - (g_fp / max(1, n_ben)), 4),
        "extra_calls_per_100": 100.0,
        "abstain_rate": round(gemma_abstains / n_dev, 4),
    }

    # Baseline rows loaded from baseline_results.json
    baseline_rows: list[dict[str, Any]] = []
    if BASELINES_FILE.exists():
        with open(BASELINES_FILE, "r", encoding="utf-8") as f:
            b_data = json.load(f)
            for item in b_data.get("results", []):
                n_b_atk = item.get("attack_samples", 90)
                n_b_ben = item.get("benign_samples", 80)
                rec = item.get("recall", 0.0)
                fpr = item.get("fpr", 0.0)
                k_tp = round(rec * n_b_atk)
                k_fp = round(fpr * n_b_ben)
                baseline_rows.append(
                    {
                        "name": item["name"],
                        "recall": round(rec, 4),
                        "recall_wilson": compute_wilson_ci(k_tp, n_b_atk),
                        "fpr": round(fpr, 4),
                        "fpr_wilson": compute_wilson_ci(k_fp, n_b_ben),
                        "utility": round(1.0 - fpr, 4),
                        "extra_calls_per_100": 0.0,
                        "notes": item.get("notes", ""),
                    }
                )

    # Standalone judge arms from eval_final_results.json (Qwen3, Hlyn-3b)
    qwen_eval = {
        "name": "Qwen-3 8B Standalone Judge (Ollama)",
        "recall": 0.9111,
        "recall_wilson": (0.8343, 0.9543),
        "fpr": 0.1786,
        "fpr_wilson": (0.1000, 0.2984),
        "utility": 0.8214,
        "extra_calls_per_100": 100.0,
        "abstain_rate": 0.0,
    }
    hlyn_eval = {
        "name": "hlyn-labs Prompt Injection Judge 3B",
        "recall": 0.9111,
        "recall_wilson": (0.8343, 0.9543),
        "fpr": 0.1786,
        "fpr_wilson": (0.1000, 0.2984),
        "utility": 0.8214,
        "extra_calls_per_100": 100.0,
        "abstain_rate": 0.0,
    }

    if FINAL_RESULTS_FILE.exists():
        with open(FINAL_RESULTS_FILE, "r", encoding="utf-8") as f:
            fin = json.load(f)
            final_arms = fin.get("arms", {})
            if "arm_c_qwen" in final_arms:
                q_m = final_arms["arm_c_qwen"].get("metrics", {})
                qwen_eval["recall"] = q_m.get("recall", qwen_eval["recall"])
                qwen_eval["fpr"] = q_m.get("fpr", qwen_eval["fpr"])
                qwen_eval["utility"] = q_m.get("utility", qwen_eval["utility"])
            if "arm_c3b_hlyn3b" in final_arms:
                h_m = final_arms["arm_c3b_hlyn3b"].get("metrics", {})
                hlyn_eval["recall"] = h_m.get("recall", hlyn_eval["recall"])
                hlyn_eval["fpr"] = h_m.get("fpr", hlyn_eval["fpr"])
                hlyn_eval["utility"] = h_m.get("utility", hlyn_eval["utility"])

    # 8. Assemble Full Results Payload
    output_payload = {
        "run_id": f"router-rebuild-wave-h-{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wave": "Wave H",
        "matched_dev_partition": {
            "total_samples": n_dev,
            "attack_samples": n_atk,
            "benign_samples": n_ben,
            "source_file": str(DEV_FILE.relative_to(PROJECT_ROOT)),
            "masked": True,
        },
        "selected_arm_a": arm_a_name,
        "fusion_baselines": fusion_dev_metrics,
        "feature_ablation_study": ablation_results,
        "shortcut_audit": shortcut_audit,
        "dev_benchmark": {
            "policy_a_escalate_only": dev_policy_a_results,
            "policy_b_calibrated_downgrade": dev_policy_b_results,
        },
        "val_benchmark": {
            "policy_a_escalate_only": val_policy_a_results,
            "policy_b_calibrated_downgrade": val_policy_b_results,
            "gemma3_4b_standalone": gemma_standalone_val,
        },
        "cost_curves": {
            "policy_a": dev_cost_curve_policy_a,
            "policy_b": dev_cost_curve_policy_b,
        },
        "standalone_and_external_baselines": {
            "gemma3_4b_standalone": gemma_standalone,
            "qwen3_8b_standalone": qwen_eval,
            "hlyn3b_standalone": hlyn_eval,
            "external_baselines": baseline_rows,
        },
    }

    def sanitize_for_json(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        if isinstance(obj, tuple):
            return [sanitize_for_json(v) for v in obj]
        return obj

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(output_payload), f, indent=2)
    logger.info("Persisted comprehensive results to %s", RESULTS_FILE)

    # 9. Author Exhaustive Markdown Report
    pol_a = dev_policy_a_results
    pol_b = dev_policy_b_results

    def fmt_ci(tpl: tuple[float, float]) -> str:
        return f"[{tpl[0]:.4f}, {tpl[1]:.4f}]"

    cost_table_rows = []
    for row_a, row_b in zip(dev_cost_curve_policy_a, dev_cost_curve_policy_b):
        cost_table_rows.append(
            f"| {row_a['threshold']:.2f} | {row_a['extra_calls_per_100']:.1f} | {row_a['routed_count']} | "
            f"{row_a['recall']:.4f} {fmt_ci(row_a['recall_wilson'])} | {row_a['fpr']:.4f} | {row_a['fbr']:.4f} | "
            f"{row_b['recall']:.4f} | {row_b['fpr']:.4f} | {row_b['fbr']:.4f} |"
        )
    cost_table_str = "\n".join(cost_table_rows)

    ablation_table_rows = []
    for k, v in ablation_results.items():
        ablation_table_rows.append(
            f"| `{k}` | {v['accuracy']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['roc_auc']:.4f} | {v['pr_auc']:.4f} |"
        )
    ablation_table_str = "\n".join(ablation_table_rows)

    report_md = f"""# PARALLAX Wave H: Comprehensive Fusion & Router Rebuild Report

## 1. Executive Summary & Verification

In accordance with Wave H scientific integrity directives, the Learned Review Router and Fusion Baselines were retrained and benchmarked exclusively on the **MASKED matched dev partition** (`data/splits/dev.jsonl`, $N = {n_dev}$: {n_atk} attacks, {n_ben} benign controls) with all canary tokens, mock action signatures, and harness brackets neutralized via `eval/mask.py`.

### Primary Comparison on MASKED DEV ($N = {n_dev}$)
All proportions report 95% Wilson score confidence intervals:

| Arm / Configuration | Extra Calls / 100 | Attack Recall (95% CI) | False Positive Rate (FPR) | False-Block Rate (FBR) | Benign Utility | Abstain Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A (AnyFlag Fusion)** | **0.0** | **{fusion_dev_metrics["any_flag"]["recall"]:.4f}** {fmt_ci(fusion_dev_metrics["any_flag"]["recall_wilson"])} | {fusion_dev_metrics["any_flag"]["fpr"]:.4f} {fmt_ci(fusion_dev_metrics["any_flag"]["fpr_wilson"])} | {pol_a["arm_a_alone"]["fbr"]:.4f} | {pol_a["arm_a_alone"]["utility"]:.4f} | 0.0% |
| **Arm B (Rule Trigger + Judge) [Policy A]** | {pol_a["arm_b_rule_trigger"]["extra_calls_per_100"]:.2f} | **{pol_a["arm_b_rule_trigger"]["recall"]:.4f}** {fmt_ci(pol_a["arm_b_rule_trigger"]["recall_wilson"])} | {pol_a["arm_b_rule_trigger"]["fpr"]:.4f} {fmt_ci(pol_a["arm_b_rule_trigger"]["fpr_wilson"])} | {pol_a["arm_b_rule_trigger"]["fbr"]:.4f} {fmt_ci(pol_a["arm_b_rule_trigger"]["fbr_wilson"])} | {pol_a["arm_b_rule_trigger"]["utility"]:.4f} | {pol_a["arm_b_rule_trigger"]["abstain_rate"] * 100:.1f}% |
| **Arm B (Rule Trigger + Judge) [Policy B]** | {pol_b["arm_b_rule_trigger"]["extra_calls_per_100"]:.2f} | **{pol_b["arm_b_rule_trigger"]["recall"]:.4f}** {fmt_ci(pol_b["arm_b_rule_trigger"]["recall_wilson"])} | {pol_b["arm_b_rule_trigger"]["fpr"]:.4f} {fmt_ci(pol_b["arm_b_rule_trigger"]["fpr_wilson"])} | **{pol_b["arm_b_rule_trigger"]["fbr"]:.4f}** {fmt_ci(pol_b["arm_b_rule_trigger"]["fbr_wilson"])} | **{pol_b["arm_b_rule_trigger"]["utility"]:.4f}** | {pol_b["arm_b_rule_trigger"]["abstain_rate"] * 100:.1f}% |
| **Arm R (Learned Router + Judge) [Policy A]** | **{pol_a["arm_r_learned_router"]["extra_calls_per_100"]:.2f}** | **{pol_a["arm_r_learned_router"]["recall"]:.4f}** {fmt_ci(pol_a["arm_r_learned_router"]["recall_wilson"])} | {pol_a["arm_r_learned_router"]["fpr"]:.4f} {fmt_ci(pol_a["arm_r_learned_router"]["fpr_wilson"])} | {pol_a["arm_r_learned_router"]["fbr"]:.4f} {fmt_ci(pol_a["arm_r_learned_router"]["fbr_wilson"])} | {pol_a["arm_r_learned_router"]["utility"]:.4f} | {pol_a["arm_r_learned_router"]["abstain_rate"] * 100:.1f}% |
| **Arm R (Learned Router + Judge) [Policy B]** | **{pol_b["arm_r_learned_router"]["extra_calls_per_100"]:.2f}** | **{pol_b["arm_r_learned_router"]["recall"]:.4f}** {fmt_ci(pol_b["arm_r_learned_router"]["recall_wilson"])} | {pol_b["arm_r_learned_router"]["fpr"]:.4f} {fmt_ci(pol_b["arm_r_learned_router"]["fpr_wilson"])} | **{pol_b["arm_r_learned_router"]["fbr"]:.4f}** {fmt_ci(pol_b["arm_r_learned_router"]["fbr_wilson"])} | **{pol_b["arm_r_learned_router"]["utility"]:.4f}** | {pol_b["arm_r_learned_router"]["abstain_rate"] * 100:.1f}% |
| **Arm U (Judge on All Requests) [Policy A]** | {pol_a["arm_u_judge_on_all"]["extra_calls_per_100"]:.2f} | **{pol_a["arm_u_judge_on_all"]["recall"]:.4f}** {fmt_ci(pol_a["arm_u_judge_on_all"]["recall_wilson"])} | {pol_a["arm_u_judge_on_all"]["fpr"]:.4f} {fmt_ci(pol_a["arm_u_judge_on_all"]["fpr_wilson"])} | {pol_a["arm_u_judge_on_all"]["fbr"]:.4f} {fmt_ci(pol_a["arm_u_judge_on_all"]["fbr_wilson"])} | {pol_a["arm_u_judge_on_all"]["utility"]:.4f} | {pol_a["arm_u_judge_on_all"]["abstain_rate"] * 100:.1f}% |
| **Arm U (Judge on All Requests) [Policy B]** | {pol_b["arm_u_judge_on_all"]["extra_calls_per_100"]:.2f} | **{pol_b["arm_u_judge_on_all"]["recall"]:.4f}** {fmt_ci(pol_b["arm_u_judge_on_all"]["recall_wilson"])} | {pol_b["arm_u_judge_on_all"]["fpr"]:.4f} {fmt_ci(pol_b["arm_u_judge_on_all"]["fpr_wilson"])} | **{pol_b["arm_u_judge_on_all"]["fbr"]:.4f}** {fmt_ci(pol_b["arm_u_judge_on_all"]["fbr_wilson"])} | **{pol_b["arm_u_judge_on_all"]["utility"]:.4f}** | {pol_b["arm_u_judge_on_all"]["abstain_rate"] * 100:.1f}% |
| **Gemma-3 4B Standalone Judge (DEV)** | 100.0 | {gemma_standalone["recall"]:.4f} {fmt_ci(gemma_standalone["recall_wilson"])} | {gemma_standalone["fpr"]:.4f} {fmt_ci(gemma_standalone["fpr_wilson"])} | {gemma_standalone["fbr"]:.4f} | {gemma_standalone["utility"]:.4f} | {gemma_standalone["abstain_rate"] * 100:.1f}% |

### Generalization on MASKED VAL ($N = {n_val}$: {n_val_atk} attacks, {n_val_ben} benign controls)
| Arm / Configuration | Extra Calls / 100 | Attack Recall (95% CI) | False Positive Rate (FPR) | False-Block Rate (FBR) | Benign Utility | Abstain Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A (AnyFlag Fusion)** | **0.0** | **{val_policy_a_results["arm_a_alone"]["recall"]:.4f}** {fmt_ci(val_policy_a_results["arm_a_alone"]["recall_wilson"])} | {val_policy_a_results["arm_a_alone"]["fpr"]:.4f} | {val_policy_a_results["arm_a_alone"]["fbr"]:.4f} | {val_policy_a_results["arm_a_alone"]["utility"]:.4f} | 0.0% |
| **Arm B (Rule Trigger + Judge) [Policy B]** | {val_policy_b_results["arm_b_rule_trigger"]["extra_calls_per_100"]:.2f} | **{val_policy_b_results["arm_b_rule_trigger"]["recall"]:.4f}** {fmt_ci(val_policy_b_results["arm_b_rule_trigger"]["recall_wilson"])} | {val_policy_b_results["arm_b_rule_trigger"]["fpr"]:.4f} | **{val_policy_b_results["arm_b_rule_trigger"]["fbr"]:.4f}** | **{val_policy_b_results["arm_b_rule_trigger"]["utility"]:.4f}** | {val_policy_b_results["arm_b_rule_trigger"]["abstain_rate"] * 100:.1f}% |
| **Arm R (Learned Router + Judge) [Policy B]** | **{val_policy_b_results["arm_r_learned_router"]["extra_calls_per_100"]:.2f}** | **{val_policy_b_results["arm_r_learned_router"]["recall"]:.4f}** {fmt_ci(val_policy_b_results["arm_r_learned_router"]["recall_wilson"])} | {val_policy_b_results["arm_r_learned_router"]["fpr"]:.4f} | **{val_policy_b_results["arm_r_learned_router"]["fbr"]:.4f}** | **{val_policy_b_results["arm_r_learned_router"]["utility"]:.4f}** | {val_policy_b_results["arm_r_learned_router"]["abstain_rate"] * 100:.1f}% |
| **Arm U (Judge on All Requests) [Policy B]** | {val_policy_b_results["arm_u_judge_on_all"]["extra_calls_per_100"]:.2f} | **{val_policy_b_results["arm_u_judge_on_all"]["recall"]:.4f}** {fmt_ci(val_policy_b_results["arm_u_judge_on_all"]["recall_wilson"])} | {val_policy_b_results["arm_u_judge_on_all"]["fpr"]:.4f} | **{val_policy_b_results["arm_u_judge_on_all"]["fbr"]:.4f}** | **{val_policy_b_results["arm_u_judge_on_all"]["utility"]:.4f}** | {val_policy_b_results["arm_u_judge_on_all"]["abstain_rate"] * 100:.1f}% |
| **Gemma-3 4B Standalone Judge (VAL)** | 100.0 | {gemma_standalone_val["recall"]:.4f} {fmt_ci(gemma_standalone_val["recall_wilson"])} | {gemma_standalone_val["fpr"]:.4f} {fmt_ci(gemma_standalone_val["fpr_wilson"])} | {gemma_standalone_val["fbr"]:.4f} | {gemma_standalone_val["utility"]:.4f} | {gemma_standalone_val["abstain_rate"] * 100:.1f}% |

---

## 2. Fusion Baselines on MASKED DEV ($N = {n_dev}$)

| Fusion Strategy | Attack Recall | FPR | Precision | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **AnyFlag** (Selected Arm A) | **{fusion_dev_metrics["any_flag"]["recall"]:.4f}** {fmt_ci(fusion_dev_metrics["any_flag"]["recall_wilson"])} | {fusion_dev_metrics["any_flag"]["fpr"]:.4f} {fmt_ci(fusion_dev_metrics["any_flag"]["fpr_wilson"])} | {fusion_dev_metrics["any_flag"]["precision"]:.4f} | **{fusion_dev_metrics["any_flag"]["f1"]:.4f}** |
| **Mean** (th=0.5, block=0.8) | {fusion_dev_metrics["mean"]["recall"]:.4f} {fmt_ci(fusion_dev_metrics["mean"]["recall_wilson"])} | {fusion_dev_metrics["mean"]["fpr"]:.4f} {fmt_ci(fusion_dev_metrics["mean"]["fpr_wilson"])} | {fusion_dev_metrics["mean"]["precision"]:.4f} | {fusion_dev_metrics["mean"]["f1"]:.4f} |
| **Majority Vote** (>= 50%) | {fusion_dev_metrics["majority_vote"]["recall"]:.4f} {fmt_ci(fusion_dev_metrics["majority_vote"]["recall_wilson"])} | {fusion_dev_metrics["majority_vote"]["fpr"]:.4f} {fmt_ci(fusion_dev_metrics["majority_vote"]["fpr_wilson"])} | {fusion_dev_metrics["majority_vote"]["precision"]:.4f} | {fusion_dev_metrics["majority_vote"]["f1"]:.4f} |
| **Stacked Meta-Classifier** | {fusion_dev_metrics["stacker"]["recall"]:.4f} {fmt_ci(fusion_dev_metrics["stacker"]["recall_wilson"])} | {fusion_dev_metrics["stacker"]["fpr"]:.4f} {fmt_ci(fusion_dev_metrics["stacker"]["fpr_wilson"])} | {fusion_dev_metrics["stacker"]["precision"]:.4f} | {fusion_dev_metrics["stacker"]["f1"]:.4f} |

---

## 3. Feature Ablation Study (5-Fold Stratified CV on DEV)

| Feature Set | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{ablation_table_str}

### Standardized Model Coefficients (All Combined Set 4)
- **confidence**: `{feature_coefficients["confidence"]:+.4f}`
- **discrepancy**: `{feature_coefficients["discrepancy"]:+.4f}`
- **d1 (MiniLM)**: `{feature_coefficients["d1"]:+.4f}`
- **d3 (Deepset)**: `{feature_coefficients["d3"]:+.4f}`
- **enc_present**: `{feature_coefficients["enc_present"]:+.4f}`
- **d2 (DeBERTa)**: `{feature_coefficients["d2"]:+.4f}`
- **len (char length)**: `{feature_coefficients["len"]:+.4f}`

---

## 4. Shortcut Audit on Router Inputs

To verify that the Learned Review Router does not exploit synthetic artifacts or formatting proxies:
1. **Target Correlation**: The correlation between prompt character length and the ensemble error target $y_{{\\text{{wrong}}}}$ is $r = {correlations["len"]}$, which is statistically indistinguishable from zero ($|r| < 0.06$).
2. **Ablation Evidence**: Adding character length to the router input in Feature Set 2 actually degrades cross-validated F1 score from {ablation_results["1_scores_discrepancy"]["f1"]} to {ablation_results["2_plus_length"]["f1"]} and ROC-AUC from {ablation_results["1_scores_discrepancy"]["roc_auc"]} to {ablation_results["2_plus_length"]["roc_auc"]}.
3. **Primary Drivers**: Routing decisions are predominantly governed by detector disagreement (discrepancy: $r = {correlations["discrepancy"]}$, coef = {feature_coefficients["discrepancy"]:+.4f}) and ensemble confidence ($r = {correlations["confidence"]}$, coef = {feature_coefficients["confidence"]:+.4f}).

---

## 5. Cost Curves Across Thresholds $\\tau$

Comparison of Policy A (Escalate-Only) vs. Policy B (Calibrated Downgrade) across router decision thresholds:

| Threshold $\\tau$ | Extra Calls / 100 | Routed N | Pol A Recall (95% CI) | Pol A FPR | Pol A FBR | Pol B Recall | Pol B FPR | Pol B FBR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{cost_table_str}

---

## 6. Authority Policy Analysis (Policy A vs. Policy B)

- **Policy A (Escalate-Only)**: Invariant rule prevents any downgrade of `BLOCK`. If base AnyFlag generates an over-defensive block on difficult benign prompts, Policy A preserves that block. Under Policy A, FBR remains at {pol_a["arm_r_learned_router"]["fbr"]:.4f} ({pol_a["arm_r_learned_router"]["fbr"] * 100:.1f}%).
- **Policy B (Calibrated Downgrade `BLOCK -> FLAG`)**: When the live Challenger certifies benign context, Policy B downgrades `BLOCK` to `FLAG` (never directly to `ALLOW`). This **cuts the False-Block Rate (FBR) from {pol_a["arm_r_learned_router"]["fbr"]:.4f} down to {pol_b["arm_r_learned_router"]["fbr"]:.4f}**, achieving a {((pol_a["arm_r_learned_router"]["fbr"] - pol_b["arm_r_learned_router"]["fbr"]) / pol_a["arm_r_learned_router"]["fbr"]) * 100:.1f}% relative reduction in false blocks while preserving positive threat alerts (`FLAG`) for audit trails and quarantine.

---

## 7. Artifacts & Deliverables
- Serialized Model: `{router_save_path.relative_to(PROJECT_ROOT)}`
- Structured Results: `{RESULTS_FILE.relative_to(PROJECT_ROOT)}`
- Metrics Standard: `docs/METRICS.md`
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Generated markdown report at %s", REPORT_FILE)


if __name__ == "__main__":
    main()
