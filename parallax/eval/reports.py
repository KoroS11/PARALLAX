"""Evaluation reporting, run manifest generation, and visualization for PARALLAX.

Includes:
- Run manifest writer: records Git commit SHA, branch, dataset hashes, seeds,
  installed library versions, and hardware specifications (CPU, RAM, GPU/VRAM).
- Markdown report table generator: produces clean GitHub-flavored markdown summaries.
- JSON results exporter: exports structured metric trees and evaluation runs.
- Visualization generators: confusion matrix heatmap, ROC curve, PR curve,
  and latency comparisons using Matplotlib and Seaborn (saved to reports/).
"""

import ctypes
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from parallax.contracts.data import DatasetSample
from parallax.eval.metrics import ConfusionMatrixResult
from parallax.eval.runners import EvalRunResult

matplotlib.use("Agg")  # Headless backend for CLI / testing environments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run Manifest Writer
# ---------------------------------------------------------------------------


def _get_git_info() -> dict[str, Any]:
    """Retrieve Git commit SHA, current branch, and dirty status."""
    sha = "unknown"
    branch = "unknown"
    is_dirty = False

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as err:
        logger.debug("Could not resolve git SHA: %s", err)

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as err:
        logger.debug("Could not resolve git branch: %s", err)

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        is_dirty = len(status_out) > 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as err:
        logger.debug("Could not resolve git status: %s", err)

    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": is_dirty,
    }


def _get_total_ram_gb() -> float | None:
    """Retrieve total system physical RAM in gigabytes."""
    try:
        import psutil

        return float(round(psutil.virtual_memory().total / (1024**3), 2))
    except (ImportError, OSError) as err:
        logger.debug("psutil RAM check failed: %s", err)

    if platform.system() == "Windows":
        try:

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(round(stat.ullTotalPhys / (1024**3), 2))
        except (OSError, AttributeError) as err:
            logger.debug("Windows ctypes RAM check failed: %s", err)

    return None


def _get_gpu_info() -> dict[str, Any]:
    """Retrieve CUDA GPU availability, device name, and total VRAM in GB."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            return {
                "cuda_available": False,
                "device_count": 0,
                "device_name": None,
                "total_vram_gb": None,
            }

        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count > 0 else None
        vram_gb = None
        if count > 0:
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 2)

        return {
            "cuda_available": True,
            "device_count": count,
            "device_name": name,
            "total_vram_gb": vram_gb,
        }
    except (ImportError, RuntimeError) as err:
        logger.debug("PyTorch GPU check failed: %s", err)
        return {
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "total_vram_gb": None,
        }


def _get_library_versions() -> dict[str, str]:
    """Retrieve versions of installed critical libraries."""
    key_libs = [
        "scikit-learn",
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "transformers",
        "sentence-transformers",
        "pydantic",
        "fastapi",
        "pytest",
        "matplotlib",
        "seaborn",
    ]
    versions = {"python": sys.version.split()[0]}
    for lib in key_libs:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "not installed"
    return versions


def compute_dataset_hash(
    file_or_data: Path | str | Sequence[DatasetSample],
) -> str:
    """Compute SHA-256 fingerprint for a dataset file or list of samples."""
    hasher = hashlib.sha256()

    if isinstance(file_or_data, (str, Path)) and os.path.exists(file_or_data):
        with open(file_or_data, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    elif isinstance(file_or_data, Sequence):
        for s in file_or_data:
            if isinstance(s, DatasetSample):
                entry = f"{s.text}|{s.label_binary}|{s.label_multiclass}|{s.source}"
                hasher.update(entry.encode("utf-8"))
            else:
                hasher.update(str(s).encode("utf-8"))
    else:
        hasher.update(str(file_or_data).encode("utf-8"))

    return hasher.hexdigest()


def generate_run_manifest(
    run_id: str,
    experiment_id: str,
    seed: int = 42,
    dataset_hashes: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate comprehensive run manifest for reproducibility and auditability."""
    git_info = _get_git_info()
    ram_gb = _get_total_ram_gb()
    gpu_info = _get_gpu_info()
    lib_versions = _get_library_versions()

    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "random_seed": seed,
        "git": git_info,
        "dataset_hashes": dataset_hashes or {},
        "hardware": {
            "platform": platform.platform(),
            "os_system": platform.system(),
            "cpu_architecture": platform.machine(),
            "cpu_count_logical": os.cpu_count(),
            "total_ram_gb": ram_gb,
            "gpu": gpu_info,
        },
        "environment": {
            "python_executable": sys.executable,
            "libraries": lib_versions,
        },
        "metadata": extra_metadata or {},
    }
    return manifest


def save_run_manifest(
    manifest: dict[str, Any],
    filepath: Path | str,
) -> Path:
    """Save run manifest to JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return p


# ---------------------------------------------------------------------------
# JSON Results Exporter
# ---------------------------------------------------------------------------


def export_results_json(
    eval_results: EvalRunResult | list[EvalRunResult] | dict[str, Any],
    filepath: Path | str,
    manifest: dict[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    """Export evaluation results and manifest into a single structured JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(eval_results, EvalRunResult):
        results_data: Any = eval_results.to_dict()
    elif isinstance(eval_results, list):
        results_data = [
            r.to_dict() if isinstance(r, EvalRunResult) else r for r in eval_results
        ]
    else:
        results_data = eval_results

    payload = {
        "manifest": manifest or {},
        "results": results_data,
        "exported_at": datetime.now(UTC).isoformat(),
    }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, default=str, ensure_ascii=False)
    return p


# ---------------------------------------------------------------------------
# Markdown Report Generator
# ---------------------------------------------------------------------------


def generate_markdown_report(
    eval_results: EvalRunResult | list[EvalRunResult] | dict[str, Any],
    title: str = "PARALLAX Evaluation Report",
    manifest: dict[str, Any] | None = None,
) -> str:
    """Generate a clean GitHub-flavored markdown report from evaluation runs."""
    results_list: list[EvalRunResult] = []
    if isinstance(eval_results, EvalRunResult):
        results_list = [eval_results]
    elif isinstance(eval_results, list):
        results_list = [r for r in eval_results if isinstance(r, EvalRunResult)]

    lines: list[str] = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]

    # Manifest section
    if manifest:
        git_sha = manifest.get("git", {}).get("git_sha", "unknown")
        branch = manifest.get("git", {}).get("git_branch", "unknown")
        ram = manifest.get("hardware", {}).get("total_ram_gb", "unknown")
        gpu_name = manifest.get("hardware", {}).get("gpu", {}).get("device_name", "N/A")
        seed = manifest.get("random_seed", "N/A")

        lines.extend(
            [
                "## Environment & Provenance",
                "",
                f"- **Git Commit**: `{git_sha}` (branch: `{branch}`)",
                f"- **Random Seed**: `{seed}`",
                f"- **Hardware**: RAM: `{ram} GB`, GPU: `{gpu_name}`",
                "",
            ]
        )

    # Overall Summary Table
    if results_list:
        lines.extend(
            [
                "## Evaluation Results Summary",
                "",
                "| Experiment | Detector | Samples | Precision | Recall | F1 | Accuracy | ROC-AUC | ECE | Recall@1%FPR | p50 (ms) | p95 (ms) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )

        for r in results_list:
            bm = r.binary_metrics
            prec = f"{bm.precision:.4f}" if bm else "N/A"
            rec = f"{bm.recall:.4f}" if bm else "N/A"
            f1 = f"{bm.f1:.4f}" if bm else "N/A"
            acc = f"{bm.accuracy:.4f}" if bm else "N/A"
            roc = f"{bm.roc_auc:.4f}" if (bm and bm.roc_auc is not None) else "N/A"
            ece = f"{bm.ece:.4f}" if (bm and bm.ece is not None) else "N/A"
            rec_1pct = (
                f"{bm.recall_at_1pct_fpr.recall:.4f}"
                if (bm and bm.recall_at_1pct_fpr)
                else "N/A"
            )
            p50 = f"{r.latency_stats.get('p50_ms', 0.0):.2f}"
            p95 = f"{r.latency_stats.get('p95_ms', 0.0):.2f}"

            lines.append(
                f"| {r.experiment_id} | {r.detector_name} | {r.evaluated_samples}/{r.total_samples} "
                f"| {prec} | {rec} | {f1} | {acc} | {roc} | {ece} | {rec_1pct} | {p50} | {p95} |"
            )
        lines.append("")

        # Per-class multiclass breakdown
        for r in results_list:
            if r.multiclass_report and r.multiclass_report.per_class:
                lines.extend(
                    [
                        f"### Multi-Class Technique Breakdown ({r.experiment_id} - {r.detector_name})",
                        "",
                        f"- **Macro-F1**: `{r.multiclass_report.macro_f1:.4f}` | **Weighted-F1**: `{r.multiclass_report.weighted_f1:.4f}`",
                        "",
                        "| Technique / Class | Precision | Recall | F1 | Support |",
                        "|---|---|---|---|---|",
                    ]
                )
                for cls_name, cm in sorted(r.multiclass_report.per_class.items()):
                    lines.append(
                        f"| {cls_name} | {cm.precision:.4f} | {cm.recall:.4f} | {cm.f1:.4f} | {cm.support} |"
                    )
                lines.append("")

        # Per-technique attack recall
        for r in results_list:
            if r.per_technique_recall:
                lines.extend(
                    [
                        f"### Per-Technique Attack Recall ({r.experiment_id} - {r.detector_name})",
                        "",
                        "| Attack Technique | Recall Rate |",
                        "|---|---|",
                    ]
                )
                for tech, rec_val in sorted(r.per_technique_recall.items()):
                    lines.append(f"| {tech} | {rec_val * 100.0:.2f}% |")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualizations: Plots (saved to reports/)
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: ConfusionMatrixResult | list[list[int]],
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    output_path: Path | str = "reports/confusion_matrix.png",
    normalize: bool = False,
) -> Path:
    """Generate and save a confusion matrix heatmap using Seaborn and Matplotlib."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(cm, ConfusionMatrixResult):
        matrix = cm.to_numpy()
        class_labels = cm.labels
    else:
        matrix = np.array(cm, dtype=int)
        class_labels = labels or [str(i) for i in range(len(matrix))]

    if normalize:
        display_matrix = matrix.astype(float)
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        display_matrix = display_matrix / row_sums
        fmt = ".2f"
    else:
        display_matrix = matrix.astype(int)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        display_matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        cbar=True,
    )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()

    fig.savefig(out_p)
    plt.close(fig)
    return out_p


def plot_roc_curve(
    y_true: Sequence[bool | int],
    y_scores_dict: dict[str, Sequence[float]],
    title: str = "Receiver Operating Characteristic (ROC)",
    output_path: Path | str = "reports/roc_curves.png",
) -> Path:
    """Plot and save ROC curves for one or more detectors."""
    from sklearn.metrics import auc as sk_auc
    from sklearn.metrics import roc_curve as sk_roc_curve

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    y_arr = np.asarray(y_true, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    for name, scores in y_scores_dict.items():
        score_arr = np.asarray(scores, dtype=float)
        try:
            fpr, tpr, _ = sk_roc_curve(y_arr, score_arr)
            auc_val = float(sk_auc(fpr, tpr)) if len(fpr) > 1 else 0.0
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)
        except (ValueError, TypeError) as err:
            logger.debug("Failed ROC curve plotting for %s: %s", name, err)

    # 45-degree diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.500)")

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig.savefig(out_p)
    plt.close(fig)
    return out_p


def plot_pr_curve(
    y_true: Sequence[bool | int],
    y_scores_dict: dict[str, Sequence[float]],
    title: str = "Precision-Recall Curve",
    output_path: Path | str = "reports/pr_curves.png",
) -> Path:
    """Plot and save Precision-Recall curves for one or more detectors."""
    from sklearn.metrics import auc as sk_auc
    from sklearn.metrics import precision_recall_curve as sk_pr_curve

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    y_arr = np.asarray(y_true, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    for name, scores in y_scores_dict.items():
        score_arr = np.asarray(scores, dtype=float)
        try:
            prec, rec, _ = sk_pr_curve(y_arr, score_arr)
            auc_val = float(sk_auc(rec, prec))
            ax.plot(rec, prec, label=f"{name} (PR-AUC = {auc_val:.3f})", lw=2)
        except (ValueError, TypeError) as err:
            logger.debug("Failed PR curve plotting for %s: %s", name, err)

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("Recall (TPR)", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig.savefig(out_p)
    plt.close(fig)
    return out_p


def plot_latency_comparison(
    latencies_dict: dict[str, Sequence[float]],
    title: str = "Latency Comparison (p50 / p95)",
    output_path: Path | str = "reports/latency_comparison.png",
) -> Path:
    """Plot and save latency benchmark comparison bar chart."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    names = list(latencies_dict.keys())
    p50_vals = [float(np.percentile(latencies_dict[k], 50)) for k in names]
    p95_vals = [float(np.percentile(latencies_dict[k], 95)) for k in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(x - width / 2, p50_vals, width, label="p50 Latency (ms)", color="#3498db")
    ax.bar(x + width / 2, p95_vals, width, label="p95 Latency (ms)", color="#e74c3c")

    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig.savefig(out_p)
    plt.close(fig)
    return out_p
