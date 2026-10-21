#!/usr/bin/env python3
"""Detector ROC-AUC Table and Bootstrap CI Computation (K12e Item 2).

Loads detector scores across splits:
- held_out_v4 (N=347): Loaded from results/final-v4/predictions.jsonl.
- dev (N=297) and val (N=277):
  * d4_rules: Scored deterministically using RuleDetector (regex heuristics).
  * d2-deberta, d3-deepset, promptguard2: Summary scalar AUCs recorded during
    Wave H.3 calibration in docs/DETECTOR_CALIBRATION.md. Per-sample prediction
    arrays were not dumped to disk; under K12e zero-model-run rules, bootstrap CIs
    cannot be re-sampled without cached per-prompt score logs.
  * promptforest: Not evaluated on dev/val (external service failed with connection
    errors during pilot; only evaluated on held_out_v4).

Computes exact ROC-AUC and 2,000-resample bootstrap 95% CI (seed 0).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallax.detectors.rules import RuleDetector

PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
DEV_PATH = REPO_ROOT / "data" / "splits" / "dev.jsonl"
VAL_PATH = REPO_ROOT / "data" / "splits" / "val.jsonl"


def bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, seed: int = 0, n_resamples: int = 2000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    low, high = np.percentile(boot_aucs, [2.5, 97.5])
    return float(low), float(high)


def compute_held_out_detector_aucs():
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    y_true = np.array([1 if r["label_binary"] else 0 for r in rows])
    detectors = ["d2-deberta", "d3-deepset", "d4_rules", "promptguard2", "promptforest"]

    results = {}
    for d in detectors:
        scores = np.array([r["detectors"][d]["score"] for r in rows])
        auc = float(roc_auc_score(y_true, scores))
        ci_low, ci_high = bootstrap_auc_ci(y_true, scores, seed=0, n_resamples=2000)
        results[d] = {
            "auc": auc,
            "ci": (ci_low, ci_high),
            "n": len(y_true),
        }
    return results


def compute_d4_rules_splits():
    detector = RuleDetector()
    results = {}
    for split_name, path in [("dev", DEV_PATH), ("val", VAL_PATH)]:
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        y_true = np.array([1 if r.get("label_binary", False) else 0 for r in rows])
        texts = [r["text"] for r in rows]
        scores = np.array([detector.predict_sync(t).score for t in texts])
        auc = float(roc_auc_score(y_true, scores))
        ci_low, ci_high = bootstrap_auc_ci(y_true, scores, seed=0, n_resamples=2000)
        results[split_name] = {
            "auc": auc,
            "ci": (ci_low, ci_high),
            "n": len(y_true),
        }
    return results


def main():
    print("=" * 80)
    print("COMPUTING DETECTOR ROC-AUC & 95% BOOTSTRAP CI (SEED=0, RESAMPLES=2000)")
    print("=" * 80)

    held_out = compute_held_out_detector_aucs()
    d4_splits = compute_d4_rules_splits()

    print("\n--- HELD_OUT_V4 (N=347: 139 attacks, 208 benign) ---")
    for d, res in held_out.items():
        print(
            f"{d:15s}: AUC = {res['auc']:.4f}, 95% CI = [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]"
        )

    print("\n--- D4_RULES SPLITS ---")
    for s, res in d4_splits.items():
        print(
            f"d4_rules on {s:4s} (N={res['n']}): AUC = {res['auc']:.4f}, 95% CI = [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]"
        )

    print("\n" + "=" * 80)
    print("DETECTOR ROC-AUC TABLE (MARKDOWN):")
    print("=" * 80)

    # Format Markdown Table
    # Note on dev/val neural detectors:
    # d3 dev: 0.8350 (from scripts/detectors/calibrate_detectors.py / docs/DETECTOR_CALIBRATION.md)
    # d3 val: 0.8452 (from scripts/detectors/calibrate_detectors.py / docs/DETECTOR_CALIBRATION.md)
    # d2 val: 0.7500 (from docs/DETECTOR_CALIBRATION.md)
    # pg2 val: 0.8294 (from docs/DETECTOR_CALIBRATION.md)
    # promptforest: not evaluated on dev/val
    table = [
        "| Detector | Model ID | dev ($N=297$) | val ($N=277$) | held_out_v4 ($N=347$) | Source of Dev/Val Score |",
        "|---|---|:---:|:---:|:---:|---|",
        f"| **d2-deberta** | `protectai/deberta-v3-base-prompt-injection-v2` | 0.7480 [0.694, 0.798] | 0.7500 *(CI not cached)* | **{held_out['d2-deberta']['auc']:.4f}** [{held_out['d2-deberta']['ci'][0]:.4f}, {held_out['d2-deberta']['ci'][1]:.4f}] | `generate_k10_handoff.py:397` (dev), `DETECTOR_CALIBRATION.md:14` (val) |",
        f"| **d3-deepset** | `deepset/deberta-v3-base-injection` | 0.8350 [0.789, 0.876] | 0.8452 *(CI not cached)* | **{held_out['d3-deepset']['auc']:.4f}** [{held_out['d3-deepset']['ci'][0]:.4f}, {held_out['d3-deepset']['ci'][1]:.4f}] | `generate_k10_handoff.py:398` (dev), `DETECTOR_CALIBRATION.md:15` (val) |",
        f"| **d4_rules** | `RuleDetector` (regex heuristics) | {d4_splits['dev']['auc']:.4f} [{d4_splits['dev']['ci'][0]:.4f}, {d4_splits['dev']['ci'][1]:.4f}] | {d4_splits['val']['auc']:.4f} [{d4_splits['val']['ci'][0]:.4f}, {d4_splits['val']['ci'][1]:.4f}] | **{held_out['d4_rules']['auc']:.4f}** [{held_out['d4_rules']['ci'][0]:.4f}, {held_out['d4_rules']['ci'][1]:.4f}] | Evaluated by `scripts/eval/detector_auc_table.py` |",
        f"| **Prompt Guard 2** | `meta-llama/Llama-Prompt-Guard-2-86M` | 0.8290 *(CI not cached)* | 0.8294 *(CI not cached)* | **{held_out['promptguard2']['auc']:.4f}** [{held_out['promptguard2']['ci'][0]:.4f}, {held_out['promptguard2']['ci'][1]:.4f}] | `calibrate_detectors.py:381` (dev), `DETECTOR_CALIBRATION.md:17` (val) |",
        f"| **PromptForest** | External Tree Baseline | *not evaluated* | *not evaluated* | **{held_out['promptforest']['auc']:.4f}** [{held_out['promptforest']['ci'][0]:.4f}, {held_out['promptforest']['ci'][1]:.4f}] | External baseline; connection failed in dev/val pilot |",
    ]
    md_str = "\n".join(table)
    print(md_str)
    return md_str


if __name__ == "__main__":
    main()
