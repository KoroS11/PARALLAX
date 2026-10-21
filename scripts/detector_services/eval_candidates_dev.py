"""Evaluate 4 candidate prompt injection detectors on the PARALLAX DEV split.

Candidate models evaluated:
1. hlyn-labs/prompt-injection-judge-deberta-70m (CC-BY-NC-4.0, gated: auto, use_fast=False)
2. protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0, ungated)
3. bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1 (AGPL-3.0, ungated)
4. deepset/deberta-v3-base-injection (MIT, ungated)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi, get_token
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parallax.eval.metrics import (
    compute_binary_metrics,
    compute_pr_auc,
    compute_roc_auc,
)
from parallax.sandbox.utility import BENIGN_TASK_CATALOG
from scripts.gpu_lock import gpu_lock

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("eval_candidates")

CANDIDATES = [
    {
        "id": "hlyn-labs/prompt-injection-judge-deberta-70m",
        "name": "hlyn_70m",
        "use_fast": False,
        "pos_label_idx": 1,
        "expected_pos_label": "LABEL_1",
    },
    {
        "id": "protectai/deberta-v3-base-prompt-injection-v2",
        "name": "protectai_base_v2",
        "use_fast": True,
        "pos_label_idx": 1,
        "expected_pos_label": "INJECTION",
    },
    {
        "id": "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1",
        "name": "bastion_xsmall",
        "use_fast": True,
        "pos_label_idx": 1,
        "expected_pos_label": "attack",
    },
    {
        "id": "deepset/deberta-v3-base-injection",
        "name": "deepset_base",
        "use_fast": True,
        "pos_label_idx": 1,
        "expected_pos_label": "INJECTION",
    },
]


def load_dev_data() -> list[dict[str, Any]]:
    """Load the 54 DEV split samples (30 attacks, 24 benign)."""
    samples: list[dict[str, Any]] = []

    # Attacks from canary_attacks.jsonl
    attack_file = PROJECT_ROOT / "data" / "attacks" / "canary_attacks.jsonl"
    with open(attack_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") == "dev":
                samples.append(
                    {
                        "id": rec.get("id"),
                        "text": rec.get("text") or rec.get("prompt"),
                        "is_attack": True,
                        "type": "attack",
                        "technique": rec.get("technique"),
                    }
                )

    # Benign from BENIGN_TASK_CATALOG
    all_benign: list[dict[str, Any]] = []
    for task_list in BENIGN_TASK_CATALOG.values():
        for task in task_list:
            all_benign.append(
                {
                    "id": task.task_id,
                    "text": task.prompt,
                    "is_attack": False,
                    "type": "benign",
                    "scenario": task.scenario_id,
                }
            )

    dev_benign = [b for idx, b in enumerate(all_benign) if idx % 10 in (0, 1, 2)]
    samples.extend(dev_benign)
    return samples


def get_hub_info(model_id: str, token: str | None) -> dict[str, Any]:
    """Retrieve Hub metadata, license and gating status."""
    api = HfApi(token=token)
    try:
        info = api.model_info(model_id, timeout=10)
        card_data = info.card_data or {}
        lic = getattr(card_data, "license", None) or (
            card_data.get("license") if isinstance(card_data, dict) else None
        )
        return {
            "gated": getattr(info, "gated", False),
            "license": lic or "unknown",
            "private": getattr(info, "private", False),
            "downloads": getattr(info, "downloads", 0),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "gated": "unknown", "license": "unknown"}


def evaluate_candidate(
    cand: dict[str, Any],
    dev_samples: list[dict[str, Any]],
    token: str | None,
    device: str,
) -> dict[str, Any]:
    """Evaluate one candidate model on all dev samples."""
    model_id = cand["id"]
    logger.info("Evaluating candidate: %s on device: %s", model_id, device)
    hub_info = get_hub_info(model_id, token)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=cand["use_fast"],
        token=token,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        token=token,
    )
    model.eval()
    model.to(device)

    y_true: list[int] = []
    y_score: list[float] = []
    latencies: list[float] = []
    sample_results: list[dict[str, Any]] = []

    pos_idx = cand["pos_label_idx"]

    # Warmup
    warmup_text = "Hello, please summarize this document."
    inputs = tokenizer(
        warmup_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        _ = model(**inputs)

    for sample in dev_samples:
        text = sample["text"]
        true_label = 1 if sample["is_attack"] else 0
        y_true.append(true_label)

        t0 = time.perf_counter()
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        attack_prob = float(probs[pos_idx])
        y_score.append(round(attack_prob, 6))

        sample_results.append(
            {
                "id": sample["id"],
                "true_label": true_label,
                "score": round(attack_prob, 6),
                "latency_ms": round(latency_ms, 2),
            }
        )

    # Compute metrics at standard threshold 0.5
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    binary_metrics = compute_binary_metrics(y_true, y_pred)
    pr_auc = compute_pr_auc(y_true, y_score)
    roc_auc = compute_roc_auc(y_true, y_score)

    mean_latency = float(np.mean(latencies))
    p50_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "candidate": cand["name"],
        "model_id": model_id,
        "hub_info": hub_info,
        "samples_evaluated": len(dev_samples),
        "f1": round(binary_metrics.f1, 4),
        "precision": round(binary_metrics.precision, 4),
        "recall": round(binary_metrics.recall, 4),
        "accuracy": round(binary_metrics.accuracy, 4),
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "latency_ms_mean": round(mean_latency, 2),
        "latency_ms_p50": round(p50_latency, 2),
        "latency_ms_p95": round(p95_latency, 2),
        "samples": sample_results,
    }


def main() -> None:
    token = os.environ.get("HF_TOKEN") or get_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Using device: %s (CUDA available: %s)", device, torch.cuda.is_available()
    )

    dev_samples = load_dev_data()
    logger.info("Loaded %d DEV split samples (30 attacks, 24 benign)", len(dev_samples))

    results: list[dict[str, Any]] = []

    with gpu_lock(timeout_seconds=600):
        logger.info("Acquired GPU lock. Starting candidate evaluations...")
        for cand in CANDIDATES:
            cand_res = evaluate_candidate(cand, dev_samples, token, device)
            results.append(cand_res)
            logger.info(
                "[%s] PR-AUC: %s | F1: %.4f | Recall: %.4f | Prec: %.4f | Latency p50: %.1fms",
                cand["name"],
                cand_res["pr_auc"],
                cand_res["f1"],
                cand_res["recall"],
                cand_res["precision"],
                cand_res["latency_ms_p50"],
            )

    output_dir = PROJECT_ROOT / "eval" / "detector_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "dev_candidate_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved candidate evaluation results to %s", out_file)

    # Print summary table
    print("\n" + "=" * 90)
    print("DEV SPLIT DETECTOR CANDIDATE EVALUATION SUMMARY")
    print("=" * 90)
    header = f"{'Candidate':<20} | {'Gated':<8} | {'License':<14} | {'PR-AUC':<8} | {'F1':<6} | {'Recall':<6} | {'Prec':<6} | {'p50 (ms)':<8}"
    print(header)
    print("-" * 90)
    for r in results:
        hub = r["hub_info"]
        gated_str = str(hub.get("gated", "unknown"))
        lic_str = str(hub.get("license", "unknown"))
        row = (
            f"{r['candidate']:<20} | {gated_str:<8} | {lic_str:<14} | "
            f"{r['pr_auc'] if r['pr_auc'] is not None else 0.0:<8.4f} | {r['f1']:<6.4f} | "
            f"{r['recall']:<6.4f} | {r['precision']:<6.4f} | {r['latency_ms_p50']:<8.1f}"
        )
        print(row)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
