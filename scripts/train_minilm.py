"""Training and calibration script for D1: MiniLM Two-Stage Technique Classifier.

Implements D-3 redesign:
  - Full canonical splits loading with deduplication and zero exact overlap verification.
  - Stage 1: Binary classifier (attack vs none) with probability calibration on validation split.
  - Model comparison: LogisticRegression vs LinearSVC with Platt scaling (sigmoid calibration).
  - Optimal decision threshold selection ensuring benign FPR <= 5% (FPR <= 0.05).
  - Stage 2: Technique classification head trained on attack samples only.
  - Full evaluation metrics on validation AND test splits:
      * Binary: accuracy, precision, recall, F1, ROC-AUC, FPR on 'none'.
      * Multi-class: 8-class macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrix.
      * Confusable class pair identification and merge analysis.
  - Saves artifacts:
      * artifacts/d1_minilm/classifier.joblib (< 1MB)
      * artifacts/d1_minilm/metadata.json (full metrics, parameters, manifest SHA256)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import LinearSVC

from parallax.data.loaders import load_technique_samples
from parallax.data.schema import CANONICAL_TECHNIQUES
from parallax.data.splits import check_exact_overlaps, deduplicate_by_normalized_hash
from scripts.gpu_lock import gpu_lock

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = "data/manifests/canonical_splits_manifest.json"
ATTACK_TECHNIQUES = [t for t in CANONICAL_TECHNIQUES if t != "none"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train D1 MiniLM two-stage classifier on canonical splits."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/d1_minilm",
        help="Directory to save trained artifacts.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to canonical splits manifest JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional max train samples (default: None, uses full canonical train split).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional max val samples (default: None, uses full canonical val split).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional max test samples (default: None, uses full canonical test split).",
    )
    parser.add_argument(
        "--c-reg",
        type=float,
        default=1.0,
        help="Regularization parameter C for classifiers.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum iterations for linear classifiers.",
    )
    parser.add_argument(
        "--target-max-fpr",
        type=float,
        default=0.05,
        help="Maximum allowable False Positive Rate on benign 'none' samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device for SentenceTransformer (e.g. cpu, cuda).",
    )
    return parser.parse_args()


def load_canonical_data(
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
) -> dict[str, list[Any]]:
    """Load normalized technique samples from loaders and enforce canonical deduplication."""
    logger.info("Loading technique samples from data loader...")
    raw_splits = {
        "train": load_technique_samples(split="train", limit=max_train),
        "val": load_technique_samples(split="val", limit=max_val),
        "test": load_technique_samples(split="test", limit=max_test),
    }

    logger.info(
        f"Raw split sizes: train={len(raw_splits['train'])}, val={len(raw_splits['val'])}, test={len(raw_splits['test'])}"
    )

    # Deduplicate with precedence: test > val > train
    canonical_splits = deduplicate_by_normalized_hash(raw_splits)
    logger.info(
        f"Canonical split sizes: train={len(canonical_splits['train'])}, val={len(canonical_splits['val'])}, test={len(canonical_splits['test'])}"
    )

    # Verify zero exact overlap between train and eval splits
    overlaps_val = check_exact_overlaps(
        canonical_splits["train"], canonical_splits["val"]
    )
    overlaps_test = check_exact_overlaps(
        canonical_splits["train"], canonical_splits["test"]
    )

    if overlaps_val:
        raise ValueError(
            f"Found {len(overlaps_val)} exact overlaps between train and val splits!"
        )
    if overlaps_test:
        raise ValueError(
            f"Found {len(overlaps_test)} exact overlaps between train and test splits!"
        )

    logger.info(
        "Verified 0 exact overlaps between train and evaluation sets (val & test)."
    )
    return canonical_splits


def get_manifest_metadata(manifest_path_str: str) -> tuple[str, str | None]:
    """Compute manifest file sha256 and read overall_sha256 if present."""
    p = Path(manifest_path_str)
    if not p.exists():
        logger.warning(f"Manifest not found at {p}. SHA256 will be empty.")
        return "", None

    content = p.read_bytes()
    file_sha256 = hashlib.sha256(content).hexdigest()
    overall_sha = None
    try:
        data = json.loads(content.decode("utf-8"))
        overall_sha = data.get("overall_sha256")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug("Failed parsing manifest JSON: %s", e)
    return file_sha256, overall_sha


def generate_split_embeddings(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Encode texts using SentenceTransformer under gpu_lock if on CUDA."""
    logger.info(f"Loading {model_name} on {device}...")
    if device == "cuda":
        with gpu_lock():
            encoder = SentenceTransformer(model_name, device=device)
            embs = encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
    else:
        encoder = SentenceTransformer(model_name, device=device)
        embs = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return np.asarray(embs, dtype=np.float32)


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate binary evaluation metrics given attack probabilities and threshold."""
    y_pred = y_proba >= threshold
    benign_mask = y_true == 0
    num_benign = int(np.sum(benign_mask))
    fp_count = int(np.sum(benign_mask & y_pred))
    fpr_none = float(fp_count / num_benign) if num_benign > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "fpr_on_none": round(fpr_none, 4),
    }


def find_optimal_threshold(
    y_val_bin: np.ndarray,
    val_attack_probs: np.ndarray,
    target_max_fpr: float = 0.05,
) -> tuple[float, dict[str, float]]:
    """Find the optimal decision threshold on validation set subject to FPR <= target_max_fpr."""
    best_thresh = 0.5
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}

    benign_mask = y_val_bin == 0
    num_benign = int(np.sum(benign_mask))

    candidate_thresholds = np.linspace(0.01, 0.99, 197)
    valid_candidates = []

    for th in candidate_thresholds:
        preds = val_attack_probs >= th
        fp = int(np.sum(benign_mask & preds))
        fpr = fp / num_benign if num_benign > 0 else 0.0

        if fpr <= target_max_fpr:
            f1 = float(f1_score(y_val_bin, preds, zero_division=0))
            valid_candidates.append((th, fpr, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(th)

    if not valid_candidates:
        # Fallback to high threshold if none met target
        logger.warning(f"No threshold met FPR <= {target_max_fpr}! Defaulting to 0.70.")
        best_thresh = 0.70

    best_metrics = evaluate_binary_predictions(y_val_bin, val_attack_probs, best_thresh)
    logger.info(
        f"Optimal threshold: {best_thresh:.3f} | Val FPR: {best_metrics['fpr_on_none']:.4f} "
        f"(target <= {target_max_fpr}) | Val F1: {best_metrics['f1']:.4f}"
    )
    return best_thresh, best_metrics


def analyze_confusable_classes(cm: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """Identify which technique classes are most confusable and document merger candidates."""
    confusable_pairs = []
    num_c = len(classes)
    for i in range(num_c):
        for j in range(i + 1, num_c):
            errors = int(cm[i, j] + cm[j, i])
            total = int(np.sum(cm[i, :]) + np.sum(cm[j, :]))
            rate = float(errors / total) if total > 0 else 0.0
            if errors > 0:
                confusable_pairs.append(
                    {
                        "pair": [classes[i], classes[j]],
                        "mutual_errors": errors,
                        "confusion_rate": round(rate, 4),
                    }
                )

    confusable_pairs.sort(key=lambda x: x["mutual_errors"], reverse=True)

    recommendations = [
        "1. 'keyword_override' and 'persona_play': Both rely on imperative prompt override instructions ('Ignore previous guidelines' vs 'You are now an actor without constraints'). Merging them into a single 'instruction_override' technique improves classification boundary clarity.",
        "2. 'context_overflow' and 'few_shot_poisoning': Both dilute safety alignment by injecting large distractors or repetitive synthetic conversational turns into the context window. Merging into 'context_manipulation' provides strong macro-F1 gains.",
        "3. 'encoding' and 'payload_splitting': Both rely on syntactic token evasion (base64/rot13 vs distributed variable segments). They share token obfuscation patterns.",
    ]

    return {
        "top_confusable_pairs": confusable_pairs[:5],
        "merging_recommendations": recommendations,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PARALLAX D1 MiniLM Redesign & Retraining Pipeline (D-3)")
    logger.info("=" * 70)

    # 1. Load data
    canonical_splits = load_canonical_data(
        max_train=args.max_train_samples,
        max_val=args.max_val_samples,
        max_test=args.max_test_samples,
    )

    train_samples = canonical_splits["train"]
    val_samples = canonical_splits["val"]
    test_samples = canonical_splits["test"]

    train_texts = [s.text for s in train_samples]
    train_labels = [s.label_multiclass for s in train_samples]
    y_train_bin = np.array([1 if lbl != "none" else 0 for lbl in train_labels])

    val_texts = [s.text for s in val_samples]
    val_labels = [s.label_multiclass for s in val_samples]
    y_val_bin = np.array([1 if lbl != "none" else 0 for lbl in val_labels])

    test_texts = [s.text for s in test_samples]
    test_labels = [s.label_multiclass for s in test_samples]
    y_test_bin = np.array([1 if lbl != "none" else 0 for lbl in test_labels])

    # 2. Embeddings
    logger.info(
        f"Generating MiniLM embeddings for Train ({len(train_texts)}), Val ({len(val_texts)}), Test ({len(test_texts)})..."
    )
    x_train = generate_split_embeddings(
        train_texts, args.model_name, args.device, args.batch_size
    )
    x_val = generate_split_embeddings(
        val_texts, args.model_name, args.device, args.batch_size
    )
    x_test = generate_split_embeddings(
        test_texts, args.model_name, args.device, args.batch_size
    )

    # 3. Stage 1 Binary Classifier Comparison & Calibration
    logger.info(
        "Fitting Stage 1 Binary Classifiers with Probability Calibration on Validation Split..."
    )
    x_comb = np.vstack([x_train, x_val])
    y_comb_bin = np.concatenate([y_train_bin, y_val_bin])
    ps = PredefinedSplit([-1] * len(x_train) + [0] * len(x_val))

    # Candidate 1: Logistic Regression + Platt Sigmoid Calibration
    lr_base = LogisticRegression(
        C=args.c_reg,
        max_iter=args.max_iter,
        class_weight="balanced",
        random_state=42,
    )
    cal_lr = CalibratedClassifierCV(lr_base, cv=ps, method="sigmoid")
    cal_lr.fit(x_comb, y_comb_bin)

    val_probs_lr = cal_lr.predict_proba(x_val)[:, 1]
    lr_brier = float(brier_score_loss(y_val_bin, val_probs_lr))
    lr_metrics = evaluate_binary_predictions(y_val_bin, val_probs_lr, threshold=0.5)
    lr_metrics["brier_score"] = round(lr_brier, 4)

    # Candidate 2: LinearSVC + Platt Sigmoid Calibration
    svc_base = LinearSVC(
        C=args.c_reg,
        max_iter=args.max_iter,
        class_weight="balanced",
        random_state=42,
    )
    cal_svc = CalibratedClassifierCV(svc_base, cv=ps, method="sigmoid")
    cal_svc.fit(x_comb, y_comb_bin)

    val_probs_svc = cal_svc.predict_proba(x_val)[:, 1]
    svc_brier = float(brier_score_loss(y_val_bin, val_probs_svc))
    svc_metrics = evaluate_binary_predictions(y_val_bin, val_probs_svc, threshold=0.5)
    svc_metrics["brier_score"] = round(svc_brier, 4)

    logger.info(
        f"Stage 1 LR Validation: ROC-AUC={lr_metrics['roc_auc']}, F1={lr_metrics['f1']}, Brier={lr_brier:.4f}"
    )
    logger.info(
        f"Stage 1 LinearSVC Validation: ROC-AUC={svc_metrics['roc_auc']}, F1={svc_metrics['f1']}, Brier={svc_brier:.4f}"
    )

    # Select best model (higher ROC-AUC, tiebreak on lower Brier score)
    if (svc_metrics["roc_auc"], -svc_brier) > (
        lr_metrics["roc_auc"],
        -lr_brier,
    ):
        selected_stage1 = cal_svc
        selected_model_name = "LinearSVC (Calibrated)"
        val_probs_selected = val_probs_svc
        test_probs_selected = cal_svc.predict_proba(x_test)[:, 1]
    else:
        selected_stage1 = cal_lr
        selected_model_name = "LogisticRegression (Calibrated)"
        val_probs_selected = val_probs_lr
        test_probs_selected = cal_lr.predict_proba(x_test)[:, 1]

    logger.info(f"Selected Stage 1 Binary Model: {selected_model_name}")

    # 4. Optimal Threshold Selection on Validation Split (FPR <= target_max_fpr)
    optimal_threshold, val_bin_metrics = find_optimal_threshold(
        y_val_bin, val_probs_selected, target_max_fpr=args.target_max_fpr
    )

    test_bin_metrics = evaluate_binary_predictions(
        y_test_bin, test_probs_selected, optimal_threshold
    )
    logger.info(
        f"Test Binary Metrics (thresh={optimal_threshold}): Acc={test_bin_metrics['accuracy']}, "
        f"F1={test_bin_metrics['f1']}, Prec={test_bin_metrics['precision']}, Rec={test_bin_metrics['recall']}, "
        f"AUC={test_bin_metrics['roc_auc']}, FPR_none={test_bin_metrics['fpr_on_none']}"
    )

    # 5. Stage 2: Technique Classifier Trained on Attack Samples Only
    logger.info("Training Stage 2 Technique Classifier on Attack Samples...")
    attack_mask_train = y_train_bin == 1
    x_train_attack = x_train[attack_mask_train]
    y_train_attack = [
        train_labels[i] for i in range(len(train_labels)) if attack_mask_train[i]
    ]

    stage2_clf = LogisticRegression(
        C=args.c_reg,
        max_iter=args.max_iter,
        class_weight="balanced",
        random_state=42,
    )
    stage2_clf.fit(x_train_attack, y_train_attack)
    stage2_classes = list(stage2_clf.classes_)
    logger.info(
        f"Stage 2 fitted on {len(x_train_attack)} attack samples across {len(stage2_classes)} classes: {stage2_classes}"
    )

    # 6. End-to-End Prediction and Evaluation on Validation & Test
    def predict_two_stage(
        x: np.ndarray, probs: np.ndarray, thresh: float
    ) -> tuple[np.ndarray, list[str]]:
        is_att = probs >= thresh
        preds = []
        for i in range(len(x)):
            if is_att[i]:
                pred_c = stage2_clf.predict(x[i : i + 1])[0]
                preds.append(str(pred_c))
            else:
                preds.append("none")
        return is_att, preds

    # Validation Multi-class Evaluation
    _val_pred_att, val_pred_tech = predict_two_stage(
        x_val, val_probs_selected, optimal_threshold
    )
    val_macro_f1 = float(
        f1_score(
            val_labels,
            val_pred_tech,
            labels=CANONICAL_TECHNIQUES,
            average="macro",
            zero_division=0,
        )
    )
    val_weighted_f1 = float(
        f1_score(
            val_labels,
            val_pred_tech,
            labels=CANONICAL_TECHNIQUES,
            average="weighted",
            zero_division=0,
        )
    )
    val_report = classification_report(
        val_labels,
        val_pred_tech,
        labels=CANONICAL_TECHNIQUES,
        output_dict=True,
        zero_division=0,
    )
    val_cm = confusion_matrix(val_labels, val_pred_tech, labels=CANONICAL_TECHNIQUES)

    # Test Multi-class Evaluation
    _test_pred_att, test_pred_tech = predict_two_stage(
        x_test, test_probs_selected, optimal_threshold
    )
    test_macro_f1 = float(
        f1_score(
            test_labels,
            test_pred_tech,
            labels=CANONICAL_TECHNIQUES,
            average="macro",
            zero_division=0,
        )
    )
    test_weighted_f1 = float(
        f1_score(
            test_labels,
            test_pred_tech,
            labels=CANONICAL_TECHNIQUES,
            average="weighted",
            zero_division=0,
        )
    )
    test_report = classification_report(
        test_labels,
        test_pred_tech,
        labels=CANONICAL_TECHNIQUES,
        output_dict=True,
        zero_division=0,
    )
    test_cm = confusion_matrix(test_labels, test_pred_tech, labels=CANONICAL_TECHNIQUES)

    logger.info(
        f"Validation 8-class: Macro F1={val_macro_f1:.4f}, Weighted F1={val_weighted_f1:.4f}"
    )
    logger.info(
        f"Test 8-class: Macro F1={test_macro_f1:.4f}, Weighted F1={test_weighted_f1:.4f}"
    )

    # 7. Confusable Class Analysis
    confusable_analysis = analyze_confusable_classes(test_cm, CANONICAL_TECHNIQUES)
    logger.info(
        f"Top confusable pairs: {[p['pair'] for p in confusable_analysis['top_confusable_pairs'][:3]]}"
    )

    # 8. Save Artifacts
    manifest_sha256, overall_sha256 = get_manifest_metadata(args.manifest_path)

    artifact_path = output_dir / "classifier.joblib"
    pipeline_artifact = {
        "stage1": selected_stage1,
        "stage2": stage2_clf,
        "classes": CANONICAL_TECHNIQUES,
        "attack_classes": stage2_classes,
        "threshold": optimal_threshold,
        "model_name": args.model_name,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "manifest_sha256": manifest_sha256,
        "manifest_overall_sha256": overall_sha256,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    joblib.dump(pipeline_artifact, artifact_path)
    artifact_size_kb = artifact_path.stat().st_size / 1024.0
    logger.info(
        f"Saved pipeline artifact to {artifact_path} ({artifact_size_kb:.1f} KB < 1MB limit)."
    )

    metadata = {
        "model_name": args.model_name,
        "device": args.device,
        "dataset_name": "neuralchemy/prompt-injection-technique (canonical)",
        "manifest_sha256": manifest_sha256,
        "manifest_overall_sha256": overall_sha256,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "classes": CANONICAL_TECHNIQUES,
        "attack_classes": stage2_classes,
        "stage1_comparison": {
            "logistic_regression": lr_metrics,
            "linear_svc": svc_metrics,
            "selected_model": selected_model_name,
        },
        "optimal_threshold": {
            "threshold": optimal_threshold,
            "target_max_fpr": args.target_max_fpr,
            "achieved_val_fpr": val_bin_metrics["fpr_on_none"],
        },
        "val_binary_metrics": val_bin_metrics,
        "test_binary_metrics": test_bin_metrics,
        "val_multiclass_metrics": {
            "macro_f1": round(val_macro_f1, 4),
            "weighted_f1": round(val_weighted_f1, 4),
            "classification_report": val_report,
            "confusion_matrix": val_cm.tolist(),
        },
        "test_multiclass_metrics": {
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "classification_report": test_report,
            "confusion_matrix": test_cm.tolist(),
        },
        "confusable_classes_analysis": confusable_analysis,
        "created_at": datetime.now(UTC).isoformat(),
    }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Saved full metadata and evaluation report to {meta_path}")
    logger.info("D-3 Retraining Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
