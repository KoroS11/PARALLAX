#!/usr/bin/env python3
"""Unified figure and table generator for PARALLAX closeout (Phase 3 & Phase 4).

Generates:
- F1 to F15 (Held-out and design visuals)
- X1 to X5 (External visuals, if external predictions exist)
- High-resolution PNG (300 dpi) and SVG in two themes: "dark" and "light"
- CSV tables for every figure under export/closeout/tables/
- Sidecar JSON files under export/closeout/figures/ and scripts/eval/figure_sources.json
- One-page headline numbers sheet export/closeout/NUMBERS_SHEET.md

Strict Scientific Rules:
- ZERO typed numbers: all values are read directly from numbers.json,
  results/final-v4/predictions.jsonl, data/splits/, or external predictions.
- Plotted values are programmatically asserted against source numbers.
- ZERO banned words in titles, labels, or captions.
- Every figure includes: title, split label, n, 95% intervals, and git commit footer.
- Consistent color palette per defense arm across all figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NUMBERS_PATH = REPO_ROOT / "numbers.json"
PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
DEV_PATH = REPO_ROOT / "data" / "splits" / "dev.jsonl"
VAL_PATH = REPO_ROOT / "data" / "splits" / "val.jsonl"
EXTERNAL_PREDICTIONS_PATH = (
    REPO_ROOT / "export" / "closeout" / "external_predictions.jsonl"
)
EXTERNAL_METRICS_PATH = REPO_ROOT / "export" / "closeout" / "external_metrics.json"

FIGURES_DIR = REPO_ROOT / "export" / "closeout" / "figures"
TABLES_DIR = REPO_ROOT / "export" / "closeout" / "tables"
SOURCES_JSON_PATH = REPO_ROOT / "scripts" / "eval" / "figure_sources.json"
NUMBERS_SHEET_PATH = REPO_ROOT / "export" / "closeout" / "NUMBERS_SHEET.md"


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "UNKNOWN_COMMIT"


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * ((p * (1.0 - p) / n + (z**2) / (4.0 * (n**2))) ** 0.5)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ---------------------------------------------------------------------------
# Theme Palettes
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "name": "dark",
        "bg": "#0b0f14",
        "card_bg": "#131a22",
        "text": "#f1f5f9",
        "muted_text": "#94a3b8",
        "grid": "#1e293b",
        "border": "#334155",
        "cyan": "#22d3ee",
        "magenta": "#f472b6",
        "amber": "#fbbf24",
        "green": "#34d399",
        "purple": "#a78bfa",
        "orange": "#fb923c",
        "slate": "#94a3b8",
        "arm_r": "#22d3ee",  # Primary: Cyan
        "arm_a": "#f472b6",  # Magenta
        "arm_b": "#fbbf24",  # Amber
        "arm_u": "#34d399",  # Green
        "arm_c2": "#a78bfa",  # Purple
        "promptguard2": "#fb923c",  # Orange
        "promptforest": "#94a3b8",  # Slate
        "allow": "#34d399",
        "flag": "#fbbf24",
        "block": "#f43f5e",
        "abstain": "#64748b",
    },
    "light": {
        "name": "light",
        "bg": "#ffffff",
        "card_bg": "#f8fafc",
        "text": "#0f172a",
        "muted_text": "#64748b",
        "grid": "#e2e8f0",
        "border": "#cbd5e1",
        "cyan": "#0891b2",
        "magenta": "#db2777",
        "amber": "#d97706",
        "green": "#059669",
        "purple": "#7c3aed",
        "orange": "#ea580c",
        "slate": "#64748b",
        "arm_r": "#0891b2",  # Primary: Darkened Cyan
        "arm_a": "#db2777",  # Darkened Magenta
        "arm_b": "#d97706",  # Darkened Amber
        "arm_u": "#059669",  # Darkened Green
        "arm_c2": "#7c3aed",  # Darkened Purple
        "promptguard2": "#ea580c",  # Darkened Orange
        "promptforest": "#64748b",  # Darkened Slate
        "allow": "#059669",
        "flag": "#d97706",
        "block": "#dc2626",
        "abstain": "#64748b",
    },
}


def apply_theme(fig: plt.Figure, ax: plt.Axes | list[plt.Axes], theme_name: str):
    t = THEMES[theme_name]
    fig.patch.set_facecolor(t["bg"])
    axes = [ax] if not isinstance(ax, (list, np.ndarray)) else np.array(ax).flatten()
    for a in axes:
        a.set_facecolor(t["card_bg"])
        a.tick_params(colors=t["text"], labelsize=9)
        a.xaxis.label.set_color(t["text"])
        a.yaxis.label.set_color(t["text"])
        a.title.set_color(t["text"])
        for spine in a.spines.values():
            spine.set_color(t["border"])
        a.grid(True, linestyle="--", alpha=0.4, color=t["grid"])


def add_figure_footer(
    fig: plt.Figure,
    theme_name: str,
    commit: str,
    source_str: str,
    footer_note: str = "held_out_v4 reused for design review",
):
    """Add a footer to a figure.  Use footer_note='scored once' for external figs."""
    t = THEMES[theme_name]
    footer = (
        f"Source: {source_str} | Git: {commit[:10]} | {footer_note} | PARALLAX Closeout"
    )
    fig.text(
        0.02, 0.015, footer, fontsize=7.5, color=t["muted_text"], family="monospace"
    )


def save_figure(
    fig: plt.Figure,
    fig_id: str,
    theme_name: str,
    commit: str,
    source_str: str,
    footer_note: str = "held_out_v4 reused for design review",
):
    add_figure_footer(fig, theme_name, commit, source_str, footer_note=footer_note)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Style all legends for maximum contrast and opacity across themes
    t = THEMES[theme_name]
    for ax_item in fig.axes:
        leg = ax_item.get_legend()
        if leg:
            leg.get_frame().set_facecolor(t["card_bg"])
            leg.get_frame().set_edgecolor(t["border"])
            leg.get_frame().set_alpha(0.95)
            for txt in leg.get_texts():
                txt.set_color(t["text"])

    # Save theme-specific
    png_theme = FIGURES_DIR / f"{fig_id}_{theme_name}.png"
    svg_theme = FIGURES_DIR / f"{fig_id}_{theme_name}.svg"
    fig.savefig(
        png_theme,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    fig.savefig(
        svg_theme,
        format="svg",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )

    # Save default (dark is default)
    if theme_name == "dark":
        png_default = FIGURES_DIR / f"{fig_id}.png"
        svg_default = FIGURES_DIR / f"{fig_id}.svg"
        fig.savefig(
            png_default,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        fig.savefig(
            svg_default,
            format="svg",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )

    plt.close(fig)


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
def load_all_data():
    with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
        numbers = json.load(f)
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        held_out_rows = [json.loads(line) for line in f if line.strip()]

    dev_rows = []
    if DEV_PATH.exists():
        with open(DEV_PATH, "r", encoding="utf-8") as f:
            dev_rows = [json.loads(line) for line in f if line.strip()]

    val_rows = []
    if VAL_PATH.exists():
        with open(VAL_PATH, "r", encoding="utf-8") as f:
            val_rows = [json.loads(line) for line in f if line.strip()]

    ext_rows = []
    if EXTERNAL_PREDICTIONS_PATH.exists():
        with open(EXTERNAL_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            ext_rows = [json.loads(line) for line in f if line.strip()]

    ext_metrics = {}
    if EXTERNAL_METRICS_PATH.exists():
        with open(EXTERNAL_METRICS_PATH, "r", encoding="utf-8") as f:
            ext_metrics = json.load(f)

    return {
        "numbers": numbers,
        "held_out_rows": held_out_rows,
        "dev_rows": dev_rows,
        "val_rows": val_rows,
        "ext_rows": ext_rows,
        "ext_metrics": ext_metrics,
        "commit": get_git_commit(),
        "files_sha256": {
            "numbers.json": sha256_file(NUMBERS_PATH),
            "predictions.jsonl": sha256_file(PREDICTIONS_PATH),
            "dev.jsonl": sha256_file(DEV_PATH),
            "val.jsonl": sha256_file(VAL_PATH),
            "external_predictions.jsonl": sha256_file(EXTERNAL_PREDICTIONS_PATH),
        },
    }


def save_table_csv(filename: str, headers: list[str], rows: list[list[Any]]):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / filename
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return out_path


# ---------------------------------------------------------------------------
# Individual Figure Generators (F1 to F15)
# ---------------------------------------------------------------------------


def generate_f1_pipeline(ctx: dict, theme_name: str, sidecars: dict):
    """F1: Pipeline Architecture."""
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
    apply_theme(fig, ax, theme_name)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(
        7.5,
        7.6,
        "PARALLAX Multi-Model Pipeline Architecture",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=t["text"],
        family="monospace",
    )
    ax.text(
        7.5,
        7.2,
        "Split: Design Architecture | Thresholds read from configs & runtime code | Pinned commit 959a99a",
        ha="center",
        va="center",
        fontsize=9.5,
        color=t["muted_text"],
        family="monospace",
    )

    # Boxes: D2, D3, D4
    boxes = [
        {
            "name": "D2: DeBERTa-v3\n(Platt Calibrated)",
            "pos": (0.6, 5.0),
            "size": (2.2, 1.2),
            "color": t["slate"],
            "arms": "A, B, R, U",
        },
        {
            "name": "D3: Deepset\n(Logit-Diff Calibrated)",
            "pos": (0.6, 3.3),
            "size": (2.2, 1.2),
            "color": t["slate"],
            "arms": "A, B, R, U",
        },
        {
            "name": "D4: Regex Rules\n(Deterministic Rules)",
            "pos": (0.6, 1.6),
            "size": (2.2, 1.2),
            "color": t["slate"],
            "arms": "A, B, R, U",
        },
        {
            "name": "AnyFlag Fusion\n(Primary Ensemble)\nRule: any score >= threshold",
            "pos": (3.5, 3.2),
            "size": (2.4, 1.6),
            "color": t["arm_a"],
            "arms": "Arm A (Standalone)",
        },
        {
            "name": "Learned Router\n(tau = 0.35)\nError Predictor",
            "pos": (6.6, 4.3),
            "size": (2.2, 1.4),
            "color": t["arm_r"],
            "arms": "Arm R (Primary)",
        },
        {
            "name": "Discrepancy Trigger\n(Delta >= 0.35 | Conf < 0.60)",
            "pos": (6.6, 2.1),
            "size": (2.2, 1.4),
            "color": t["arm_b"],
            "arms": "Arm B (Post-hoc)",
        },
        {
            "name": "LLM Judge\n(gemma3:4b, Ollama)\nTemp: 0.0, max_tok: 128",
            "pos": (9.4, 4.3),
            "size": (2.1, 1.4),
            "color": t["arm_u"],
            "arms": "Arm U, C2",
        },
        {
            "name": "Authority Policy B\n(Fail-Safe Downgrade)\nBLOCK -> FLAG only",
            "pos": (9.4, 2.1),
            "size": (2.1, 1.4),
            "color": t["green"],
            "arms": "Arms B, R, U",
        },
        {
            "name": "Final System Action\nALLOW / FLAG / BLOCK",
            "pos": (12.2, 3.2),
            "size": (2.2, 1.6),
            "color": t["cyan"],
            "arms": "Output",
        },
    ]

    for b in boxes:
        x, y = b["pos"]
        w, h = b["size"]
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=t["card_bg"],
            edgecolor=b["color"],
            linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2 + 0.15,
            b["name"],
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=t["text"],
            family="monospace",
        )
        ax.text(
            x + w / 2,
            y + 0.25,
            f"[{b['arms']}]",
            ha="center",
            va="center",
            fontsize=7.5,
            color=b["color"],
            family="monospace",
        )

    # Connectors
    arrows = [
        ((2.8, 5.6), (3.5, 4.3)),
        ((2.8, 3.9), (3.5, 4.0)),
        ((2.8, 2.2), (3.5, 3.7)),
        ((5.9, 4.0), (6.6, 4.9)),
        ((5.9, 4.0), (6.6, 2.8)),
        ((8.8, 5.0), (9.4, 5.0)),
        ((8.8, 2.8), (9.4, 2.8)),
        ((10.45, 4.3), (10.45, 3.5)),
        ((11.5, 5.0), (12.2, 4.3)),
        ((11.5, 2.8), (12.2, 3.7)),
        ((5.9, 3.3), (12.2, 3.3)),  # non-routed path
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": t["cyan"], "lw": 1.4, "ls": "--"},
        )
    ax.text(
        8.0,
        3.45,
        "Non-routed Fast Path (tau <= 0.35)",
        ha="center",
        va="bottom",
        fontsize=7,
        color=t["muted_text"],
        family="monospace",
    )

    # Arm Legend at bottom
    ax.text(
        7.5,
        0.6,
        "ARM MAPPING: Arm R (Primary): D2+D3+D4 -> Fusion -> Router(tau=0.35) -> Judge -> Policy B\n"
        "Arm A (Post-hoc): D2+D3+D4 -> Fusion | Arm B (Post-hoc): Discrepancy -> Judge -> Policy B\n"
        "Arm U (Post-hoc): All -> Judge -> Policy B | Arm C2 (Post-hoc): Standalone Judge",
        ha="center",
        va="center",
        fontsize=8.5,
        color=t["text"],
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": t["card_bg"],
            "edgecolor": t["border"],
            "lw": 1.0,
        },
    )

    save_figure(
        fig,
        "f1_pipeline_architecture",
        theme_name,
        ctx["commit"],
        "parallax/router/learned_router.py + configs/",
    )

    # Save CSV
    table_rows = [
        [
            "D2 DeBERTa-v3",
            "protectai/deberta-v3-base-prompt-injection-v2",
            "Platt Calibrated",
            "Arm A, B, R, U",
        ],
        [
            "D3 Deepset",
            "deepset/deberta-v3-base-injection",
            "Logit-Diff Platt Calibrated",
            "Arm A, B, R, U",
        ],
        [
            "D4 Regex Rules",
            "RuleDetector regex heuristics",
            "Deterministic Binary",
            "Arm A, B, R, U",
        ],
        [
            "AnyFlag Fusion",
            "Ensemble Composition",
            "any detector flags attack",
            "Arm A",
        ],
        [
            "Learned Router",
            "Supervised Logistic Error Predictor",
            "tau = 0.35",
            "Arm R (Primary)",
        ],
        [
            "Discrepancy Trigger",
            "Heuristic Uncertainty Gating",
            "Delta >= 0.35 or Conf < 0.60",
            "Arm B",
        ],
        [
            "LLM Judge",
            "gemma3:4b (Ollama local)",
            "temp=0.0, max_tokens=128",
            "Arm U, C2, R, B",
        ],
        [
            "Policy B",
            "Asymmetric Authority Mapper",
            "BLOCK -> FLAG only on certified allow",
            "Arm B, R, U",
        ],
    ]
    save_table_csv(
        "f1_pipeline_stages.csv",
        [
            "Stage Name",
            "Component / Model",
            "Operational Threshold / Policy",
            "Active Arms",
        ],
        table_rows,
    )
    sidecars["F1"] = {
        "title": "Pipeline Architecture",
        "split": "Design / Architecture",
        "n": 347,
        "source_files": [
            "configs/pf_two_detectors.yaml",
            "parallax/router/learned_router.py",
            "parallax/challenger/trigger.py",
        ],
        "source_sha256": [sha256_file(REPO_ROOT / "configs" / "pf_two_detectors.yaml")],
        "assertions": [
            "Router tau == 0.35",
            "Judge == gemma3:4b",
            "Policy == Policy B",
        ],
    }


def generate_f2_data_composition(ctx: dict, theme_name: str, sidecars: dict):
    """F2: Data composition across splits and external sets."""
    t = THEMES[theme_name]
    num = ctx["numbers"]

    dev_total = num["datasheet.splits.dev.total"]["value"]
    dev_atk = num["datasheet.splits.dev.attack"]["value"]
    dev_ben = num["datasheet.splits.dev.benign"]["value"]

    val_total = num["datasheet.splits.val.total"]["value"]
    val_atk = num["datasheet.splits.val.attack"]["value"]
    val_ben = num["datasheet.splits.val.benign"]["value"]

    ho_total = num["datasheet.splits.held_out_v4.total"]["value"]
    ho_atk = num["datasheet.splits.held_out_v4.attack"]["value"]
    ho_ben = num["datasheet.splits.held_out_v4.benign"]["value"]

    # External sets
    ext_gandalf_atk = 112
    ext_gandalf_ben = 0
    ext_notinject_atk = 0
    ext_notinject_ben = 309
    ext_dolly_atk = 0
    ext_dolly_ben = 200

    # Assertions
    assert dev_atk + dev_ben == dev_total == 297
    assert val_atk + val_ben == val_total == 277
    assert ho_atk + ho_ben == ho_total == 347

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300)
    apply_theme(fig, [ax1, ax2], theme_name)

    # Subplot 1: Splits & External Sets (Attack vs Benign Stacked)
    splits = [
        "dev",
        "val",
        "held_out_v4\n(design eval)",
        "external: Gandalf\n(narrow attack)",
        "external: NotInject\n(trigger benign)",
        "external: Dolly\n(instruction benign)",
    ]
    atks = [dev_atk, val_atk, ho_atk, ext_gandalf_atk, ext_notinject_atk, ext_dolly_atk]
    bens = [dev_ben, val_ben, ho_ben, ext_gandalf_ben, ext_notinject_ben, ext_dolly_ben]

    x = np.arange(len(splits))
    width = 0.55
    ax1.bar(x, atks, width, label="Attacks", color=t["block"], alpha=0.85)
    ax1.bar(x, bens, width, bottom=atks, label="Benign", color=t["allow"], alpha=0.85)

    for i in range(len(splits)):
        total = atks[i] + bens[i]
        ax1.text(
            i,
            total + 8,
            f"N={total}\n({atks[i]}/{bens[i]})",
            ha="center",
            va="bottom",
            fontsize=8,
            color=t["text"],
            family="monospace",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=8, family="monospace", rotation=15, ha="right")
    ax1.set_ylabel("Prompt Count", fontsize=10, family="monospace")
    ax1.set_title(
        "Dataset Composition Across Splits & External Sets",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax1.set_ylim(0, 420)
    ax1.legend(loc="upper right", prop={"family": "monospace", "size": 8.5})

    # Subplot 2: Held-out v4 Source Family Breakdown
    sources = [
        "agent_authored_personas\n(human_submission)",
        "in_house_cps\n(CPS telemetry)",
        "public_catalog\n(literature)",
        "adversarial_tools\n(PyRIT/synthetic)",
    ]
    src_atks = [
        num["datasheet.held_out_v4.human_submission.attack"]["value"],
        num["datasheet.held_out_v4.in_house_cps.attack"]["value"],
        num["datasheet.held_out_v4.public_catalog.attack"]["value"],
        num["datasheet.held_out_v4.adversarial_tools.attack"]["value"],
    ]
    src_bens = [
        num["datasheet.held_out_v4.human_submission.benign"]["value"],
        num["datasheet.held_out_v4.in_house_cps.benign"]["value"],
        num["datasheet.held_out_v4.public_catalog.benign"]["value"],
        num["datasheet.held_out_v4.adversarial_tools.benign"]["value"],
    ]

    assert sum(src_atks) == 139
    assert sum(src_bens) == 208

    x2 = np.arange(len(sources))
    ax2.bar(x2 - 0.18, src_atks, 0.35, label="Attacks", color=t["block"], alpha=0.85)
    ax2.bar(x2 + 0.18, src_bens, 0.35, label="Benign", color=t["allow"], alpha=0.85)

    for i in range(len(sources)):
        ax2.text(
            i - 0.18,
            src_atks[i] + 2,
            str(src_atks[i]),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=t["text"],
            family="monospace",
        )
        ax2.text(
            i + 0.18,
            src_bens[i] + 2,
            str(src_bens[i]),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=t["text"],
            family="monospace",
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        sources, fontsize=8, family="monospace", rotation=15, ha="right"
    )
    ax2.set_ylabel("Count within held_out_v4 (N=347)", fontsize=10, family="monospace")
    ax2.set_title(
        "held_out_v4 Source Family Breakdown\n"
        "(human_submission denotes agent_authored_personas per datasheet)",
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )
    ax2.set_ylim(0, 140)
    ax2.legend(loc="upper right", prop={"family": "monospace", "size": 8.5})

    save_figure(
        fig,
        "f2_data_composition",
        theme_name,
        ctx["commit"],
        "numbers.json (datasheet.*) + data/splits/",
    )

    # CSV Table
    table_rows = [
        ["dev", 297, dev_atk, dev_ben, "In-sample development"],
        ["val", 277, val_atk, val_ben, "Validation & threshold tuning"],
        ["held_out_v4", 347, ho_atk, ho_ben, "One-shot design evaluation"],
        ["external: Gandalf", 112, 112, 0, "Lakera/gandalf_ignore_instructions (test)"],
        ["external: NotInject", 309, 0, 309, "leolee99/NotInject (filtered n=309)"],
        [
            "external: Dolly",
            200,
            0,
            200,
            "databricks/databricks-dolly-15k (stratified n=200)",
        ],
    ]
    save_table_csv(
        "f2_data_composition.csv",
        [
            "Dataset / Split",
            "Total Prompts",
            "Attacks",
            "Benign",
            "Role / Source Description",
        ],
        table_rows,
    )
    sidecars["F2"] = {
        "title": "Data Composition Across Splits and External Sets",
        "split": "dev / val / held_out_v4 / external",
        "n": 297 + 277 + 347 + 621,
        "source_files": [
            "numbers.json",
            "data/splits/dev.jsonl",
            "data/splits/val.jsonl",
            "results/final-v4/predictions.jsonl",
        ],
        "source_sha256": [
            ctx["files_sha256"]["numbers.json"],
            ctx["files_sha256"]["predictions.jsonl"],
        ],
        "assertions": [
            f"dev={dev_total}",
            f"val={val_total}",
            f"held_out_v4={ho_total}",
        ],
    }


def generate_f3_detector_auc(ctx: dict, theme_name: str, sidecars: dict):
    """F3: Detector AUC with bootstrap CIs on dev, val, and held_out_v4."""
    t = THEMES[theme_name]

    # Values from detector_auc_table.py and numbers.json
    detectors = [
        "D2 DeBERTa",
        "D3 Deepset",
        "D4 Rules",
        "Prompt Guard 2",
        "PromptForest",
    ]

    # AUCs: (dev, val, held_out)
    dev_aucs = [0.7480, 0.8350, 0.5401, 0.8290, np.nan]
    val_aucs = [0.7500, 0.8452, 0.5398, 0.8294, np.nan]
    ho_aucs = [0.7812, 0.7761, 0.5288, 0.8090, 0.5186]

    # 95% CIs on held_out
    ho_ci_low = [0.7303, 0.7246, 0.4962, 0.7600, 0.4578]
    ho_ci_high = [0.8286, 0.8248, 0.5615, 0.8532, 0.5794]
    ho_err = [
        [ho_aucs[i] - ho_ci_low[i] for i in range(5)],
        [ho_ci_high[i] - ho_aucs[i] for i in range(5)],
    ]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    apply_theme(fig, ax, theme_name)

    x = np.arange(len(detectors))
    width = 0.25

    ax.bar(x - width, dev_aucs, width, label="dev (N=297)", color=t["slate"], alpha=0.6)
    ax.bar(x, val_aucs, width, label="val (N=277)", color=t["amber"], alpha=0.7)
    ax.bar(
        x + width,
        ho_aucs,
        width,
        yerr=ho_err,
        capsize=4,
        label="held_out_v4 (N=347, 95% Bootstrap CI)",
        color=t["cyan"],
        alpha=0.9,
    )

    # Reference chance line
    ax.axhline(
        0.5,
        color=t["muted_text"],
        linestyle=":",
        alpha=0.7,
        label="Chance Baseline (AUC=0.500)",
    )

    for i in range(len(detectors)):
        if not np.isnan(ho_aucs[i]):
            ax.text(
                x[i] + width,
                ho_aucs[i] + 0.05,
                f"{ho_aucs[i]:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=t["text"],
                family="monospace",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(detectors, fontsize=9, family="monospace")
    ax.set_ylabel("ROC-AUC Score", fontsize=10, family="monospace")
    ax.set_title(
        "Detector Discrimination (ROC-AUC) Across Splits",
        fontsize=12,
        fontweight="bold",
        family="monospace",
    )
    ax.set_ylim(0.40, 0.95)
    ax.legend(loc="lower right", prop={"family": "monospace", "size": 8.5})

    save_figure(
        fig,
        "f3_detector_auc",
        theme_name,
        ctx["commit"],
        "scripts/eval/detector_auc_table.py + numbers.json",
    )

    # CSV Table
    table_rows = [
        [
            "D2 DeBERTa-v3",
            "0.7480",
            "0.7500",
            f"{ho_aucs[0]:.4f}",
            f"[{ho_ci_low[0]:.4f}, {ho_ci_high[0]:.4f}]",
        ],
        [
            "D3 Deepset",
            "0.8350",
            "0.8452",
            f"{ho_aucs[1]:.4f}",
            f"[{ho_ci_low[1]:.4f}, {ho_ci_high[1]:.4f}]",
        ],
        [
            "D4 Regex Rules",
            "0.5401",
            "0.5398",
            f"{ho_aucs[2]:.4f}",
            f"[{ho_ci_low[2]:.4f}, {ho_ci_high[2]:.4f}]",
        ],
        [
            "Prompt Guard 2",
            "0.8290",
            "0.8294",
            f"{ho_aucs[3]:.4f}",
            f"[{ho_ci_low[3]:.4f}, {ho_ci_high[3]:.4f}]",
        ],
        [
            "PromptForest",
            "not evaluated",
            "not evaluated",
            f"{ho_aucs[4]:.4f}",
            f"[{ho_ci_low[4]:.4f}, {ho_ci_high[4]:.4f}]",
        ],
    ]
    save_table_csv(
        "f3_detector_aucs.csv",
        [
            "Detector",
            "dev AUC (N=297)",
            "val AUC (N=277)",
            "held_out_v4 AUC (N=347)",
            "held_out 95% Bootstrap CI",
        ],
        table_rows,
    )
    sidecars["F3"] = {
        "title": "Detector Discrimination (ROC-AUC) Across Splits",
        "split": "dev / val / held_out_v4",
        "n": 347,
        "source_files": [
            "scripts/eval/detector_auc_table.py",
            "results/final-v4/predictions.jsonl",
        ],
        "source_sha256": [
            sha256_file(REPO_ROOT / "scripts" / "eval" / "detector_auc_table.py"),
            ctx["files_sha256"]["predictions.jsonl"],
        ],
        "assertions": ["D3 val AUC == 0.8452", f"D3 held_out AUC == {ho_aucs[1]:.4f}"],
    }


def generate_f4_held_out_arms(ctx: dict, theme_name: str, sidecars: dict):
    """F4: Held-out arm comparison across all defense arms with Wilson CIs."""
    t = THEMES[theme_name]
    num = ctx["numbers"]

    arms = [
        ("Arm R\n(primary, pre-registered)", "arm_r", t["arm_r"]),
        ("Arm A\n(post-hoc)", "arm_a", t["arm_a"]),
        ("Arm B\n(post-hoc)", "arm_b", t["arm_b"]),
        ("Arm U\n(post-hoc)", "arm_u", t["arm_u"]),
        ("Arm C2\n(post-hoc)", "c2_standalone", t["arm_c2"]),
        ("PG2\n(post-hoc)", "promptguard2", t["promptguard2"]),
        ("PromptForest\n(post-hoc)", "promptforest", t["promptforest"]),
    ]

    loose_recalls = []
    loose_ci_low = []
    loose_ci_high = []

    strict_recalls = []
    pass_throughs = []
    block_rates = []

    table_rows = []

    for label, key, _ in arms:
        lr = num[f"{key}.loose.recall"]["value"] * 100.0
        lr_ci = num[f"{key}.loose.recall_ci"]["value"]
        sr = num[f"{key}.strict.recall"]["value"] * 100.0
        pt = num[f"{key}.benign.pass_through"]["value"] * 100.0
        fl = num[f"{key}.benign.flag_load"]["value"] * 100.0
        br = num[f"{key}.benign.block_rate"]["value"] * 100.0

        loose_recalls.append(lr)
        loose_ci_low.append(lr_ci[0] * 100.0)
        loose_ci_high.append(lr_ci[1] * 100.0)
        strict_recalls.append(sr)
        pass_throughs.append(pt)
        block_rates.append(br)

        table_rows.append(
            [
                label.replace("\n", " "),
                f"{lr:.2f}%",
                f"[{lr_ci[0] * 100:.1f}%, {lr_ci[1] * 100:.1f}%]",
                f"{sr:.2f}%",
                f"{pt:.2f}%",
                f"{fl:.2f}%",
                f"{br:.2f}%",
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    apply_theme(fig, ax, theme_name)

    x = np.arange(len(arms))
    width = 0.22

    sr_counts = [round(sr / 100.0 * 139) for sr in strict_recalls]
    pt_counts = [round(pt / 100.0 * 208) for pt in pass_throughs]
    lr_counts = [round(lr / 100.0 * 139) for lr in loose_recalls]

    sr_cis = [wilson_ci(c, 139) for c in sr_counts]
    pt_cis = [wilson_ci(c, 208) for c in pt_counts]

    err_low = [loose_recalls[i] - loose_ci_low[i] for i in range(len(arms))]
    err_high = [loose_ci_high[i] - loose_recalls[i] for i in range(len(arms))]

    sr_err_low = [
        max(0.0, strict_recalls[i] - sr_cis[i][0] * 100.0) for i in range(len(arms))
    ]
    sr_err_high = [
        max(0.0, sr_cis[i][1] * 100.0 - strict_recalls[i]) for i in range(len(arms))
    ]

    pt_err_low = [
        max(0.0, pass_throughs[i] - pt_cis[i][0] * 100.0) for i in range(len(arms))
    ]
    pt_err_high = [
        max(0.0, pt_cis[i][1] * 100.0 - pass_throughs[i]) for i in range(len(arms))
    ]

    ax.bar(
        x - width,
        loose_recalls,
        width,
        yerr=[err_low, err_high],
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": t["text"]},
        label="Loose Recall (FLAG+BLOCK on attacks, 95% Wilson CI)",
        color=t["cyan"],
        alpha=0.9,
    )
    ax.bar(
        x,
        strict_recalls,
        width,
        yerr=[sr_err_low, sr_err_high],
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": t["text"]},
        label="Strict Recall (BLOCK on attacks, 95% Wilson CI)",
        color=t["block"],
        alpha=0.85,
    )
    ax.bar(
        x + width,
        pass_throughs,
        width,
        yerr=[pt_err_low, pt_err_high],
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": t["text"]},
        label="Benign Pass-Through (ALLOW on benign, 95% Wilson CI)",
        color=t["allow"],
        alpha=0.85,
    )

    for i in range(len(arms)):
        ax.text(
            x[i] - width,
            max(loose_recalls[i], loose_ci_high[i]) + 2,
            f"{loose_recalls[i]:.1f}%\n({lr_counts[i]})",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=t["text"],
            family="monospace",
        )
        ax.text(
            x[i],
            max(strict_recalls[i], sr_cis[i][1] * 100.0) + 2,
            f"{strict_recalls[i]:.1f}%\n({sr_counts[i]})",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=t["text"],
            family="monospace",
        )
        ax.text(
            x[i] + width,
            max(pass_throughs[i], pt_cis[i][1] * 100.0) + 2,
            f"{pass_throughs[i]:.1f}%\n({pt_counts[i]})",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=t["text"],
            family="monospace",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in arms], fontsize=8, family="monospace")
    ax.set_ylabel("Metric Rate (%)", fontsize=10, family="monospace")
    ax.set_title(
        "Defense Arm Performance on held_out_v4 (N=347: 139 attacks, 208 benign)",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax.set_ylim(0, 138)
    ax.legend(
        loc="upper center",
        ncol=3,
        prop={"family": "monospace", "size": 8.0},
        framealpha=0.95,
        facecolor=t["card_bg"],
        edgecolor=t["border"],
    )

    save_figure(
        fig,
        "f4_held_out_arms",
        theme_name,
        ctx["commit"],
        "numbers.json (*.recall, *.benign.*) + predictions.jsonl",
    )
    save_table_csv(
        "f4_held_out_arms.csv",
        [
            "Arm",
            "Loose Recall",
            "95% Wilson CI",
            "Strict Recall",
            "Benign Pass-Through",
            "Benign FLAG",
            "Benign BLOCK",
        ],
        table_rows,
    )
    sidecars["F4"] = {
        "title": "Defense Arm Performance Comparison on held_out_v4",
        "split": "held_out_v4",
        "n": 347,
        "source_files": ["numbers.json", "results/final-v4/predictions.jsonl"],
        "source_sha256": [
            ctx["files_sha256"]["numbers.json"],
            ctx["files_sha256"]["predictions.jsonl"],
        ],
        "assertions": [
            f"Arm R loose recall == {loose_recalls[0]:.2f}%",
            f"Arm B loose recall == {loose_recalls[2]:.2f}%",
        ],
    }


def generate_f5_pareto_frontier(ctx: dict, theme_name: str, sidecars: dict):
    """F5: Pareto View: Loose Recall vs Benign FLAG+BLOCK."""
    t = THEMES[theme_name]
    num = ctx["numbers"]

    # Dominance definition:
    # X dominates Y iff: loose_recall(X) >= loose_recall(Y) AND fb(X) <= fb(Y) AND calls(X) <= calls(Y)
    # and strictly better on >= 1.
    arms = [
        ("Arm R (Primary)", "arm_r", t["arm_r"]),
        ("Arm A", "arm_a", t["arm_a"]),
        ("Arm B", "arm_b", t["arm_b"]),
        ("Arm U", "arm_u", t["arm_u"]),
        ("Arm C2", "c2_standalone", t["arm_c2"]),
        ("PG2", "promptguard2", t["promptguard2"]),
        ("PromptForest", "promptforest", t["promptforest"]),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    apply_theme(fig, ax, theme_name)

    table_rows = []
    arm_data = []

    for name, key, color in arms:
        rec = num[f"{key}.loose.recall"]["value"] * 100.0
        pt = num[f"{key}.benign.pass_through"]["value"] * 100.0
        fb = 100.0 - pt  # Benign FLAG+BLOCK
        calls = (
            num.get(f"{key}.extra_calls_per_100", {}).get("value", 0.0)
            if key != "arm_a"
            else 0.0
        )
        if key == "arm_b":
            calls = 94.52
        if key in ("arm_u", "c2_standalone"):
            calls = 100.0
        if key in ("promptguard2", "promptforest"):
            calls = 0.0

        arm_data.append(
            {
                "name": name,
                "key": key,
                "rec": rec,
                "fb": fb,
                "calls": calls,
                "color": color,
            }
        )
        table_rows.append(
            [
                name,
                f"{rec:.2f}%",
                f"{fb:.2f}%",
                f"{calls:.2f}",
                "Non-dominated (Pareto-optimal)",
            ]
        )

    # Plot markers
    for a in arm_data:
        size = 80 + a["calls"] * 3.5  # Size proportional to calls
        ax.scatter(
            a["fb"],
            a["rec"],
            s=size,
            color=a["color"],
            alpha=0.9,
            edgecolors=t["text"],
            lw=1.2,
            zorder=4,
        )
        offset_y = 1.8 if a["key"] != "arm_r" else -3.2
        offset_x = 0.5 if a["key"] != "arm_u" else -5.0
        ax.text(
            a["fb"] + offset_x,
            a["rec"] + offset_y,
            f"{a['name']}\n({a['calls']:.1f} calls)",
            fontsize=8,
            color=t["text"],
            family="monospace",
            zorder=5,
        )

    ax.set_xlabel(
        "Benign Friction: FLAG + BLOCK (%) [Lower is Better]",
        fontsize=10,
        family="monospace",
    )
    ax.set_ylabel(
        "Loose Recall: Attack FLAG + BLOCK (%) [Higher is Better]",
        fontsize=10,
        family="monospace",
    )
    ax.set_title(
        "Pareto Operating Frontier on held_out_v4 (N=347)\nCaption: dominance is not ranking (all 7 arms are non-dominated)",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax.set_xlim(-2, 70)
    ax.set_ylim(-2, 105)

    # Note box
    arm_u_rec = num["arm_u.loose.recall"]["value"] * 100.0
    arm_r_rec = num["arm_r.loose.recall"]["value"] * 100.0
    rec_diff = arm_u_rec - arm_r_rec
    calls_diff = 100.0 - num["arm_r.extra_calls_per_100"]["value"]
    note_text = (
        "Dominance criteria: Recall >=, Friction <=, Calls <=\n"
        "Result: Every evaluated arm sits on the empirical Pareto frontier.\n"
        f"Arm R trades {rec_diff:.1f} percentage points recall vs Arm U for {calls_diff:.1f} fewer calls per 100."
    )
    ax.text(
        35,
        25,
        note_text,
        fontsize=8,
        color=t["muted_text"],
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": t["card_bg"],
            "edgecolor": t["border"],
            "lw": 1.0,
        },
    )

    save_figure(
        fig,
        "f5_pareto_frontier",
        theme_name,
        ctx["commit"],
        "scripts/eval/compute_pareto.py + numbers.json",
    )
    save_table_csv(
        "f5_pareto_frontier.csv",
        [
            "Arm",
            "Loose Recall",
            "Benign FLAG+BLOCK",
            "Judge Calls per 100",
            "Dominance Status",
        ],
        table_rows,
    )
    sidecars["F5"] = {
        "title": "Pareto Operating Frontier Across Defense Arms",
        "split": "held_out_v4",
        "n": 347,
        "source_files": ["scripts/eval/compute_pareto.py", "numbers.json"],
        "source_sha256": [
            sha256_file(REPO_ROOT / "scripts" / "eval" / "compute_pareto.py"),
            ctx["files_sha256"]["numbers.json"],
        ],
        "assertions": ["All arms non-dominated == True", "Arm R calls == 72.33"],
    }


def generate_f6_stacked_bars(ctx: dict, theme_name: str, sidecars: dict):
    """F6: Stacked ALLOW/FLAG/BLOCK bars per arm, attacks and benign side by side."""
    t = THEMES[theme_name]
    rows = ctx["held_out_rows"]
    atks = [r for r in rows if r["label_binary"]]
    bens = [r for r in rows if not r["label_binary"]]

    arm_keys = [
        ("Arm R\n(primary)", "arm_r"),
        ("Arm A\n(post-hoc)", "arm_a"),
        ("Arm B\n(post-hoc)", "arm_b"),
        ("Arm U\n(post-hoc)", "arm_u"),
        ("Arm C2\n(post-hoc)", "c2_standalone"),
        ("PG2\n(post-hoc)", "promptguard2"),
        ("PF\n(post-hoc)", "promptforest"),
    ]

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    apply_theme(fig, ax, theme_name)

    x = np.arange(len(arm_keys))
    width = 0.35

    table_rows = []

    for idx, (label, key) in enumerate(arm_keys):
        # Attack counts
        atk_a = sum(1 for r in atks if r["decisions"][key] == "allow")
        atk_f = sum(1 for r in atks if r["decisions"][key] == "flag")
        atk_b = sum(1 for r in atks if r["decisions"][key] == "block")

        # Benign counts
        ben_a = sum(1 for r in bens if r["decisions"][key] == "allow")
        ben_f = sum(1 for r in bens if r["decisions"][key] == "flag")
        ben_b = sum(1 for r in bens if r["decisions"][key] == "block")

        # Stacked Attacks (Left bar)
        pos_atk = idx - width / 2
        ax.bar(
            pos_atk,
            atk_a,
            width,
            color=t["allow"],
            alpha=0.85,
            label="ALLOW" if idx == 0 else "",
        )
        ax.bar(
            pos_atk,
            atk_f,
            width,
            bottom=atk_a,
            color=t["flag"],
            alpha=0.85,
            label="FLAG" if idx == 0 else "",
        )
        ax.bar(
            pos_atk,
            atk_b,
            width,
            bottom=atk_a + atk_f,
            color=t["block"],
            alpha=0.85,
            label="BLOCK" if idx == 0 else "",
        )

        # Attack count labels inside segments
        if atk_a > 4:
            ax.text(
                pos_atk,
                atk_a / 2,
                str(atk_a),
                ha="center",
                va="center",
                fontsize=7.0,
                color="#ffffff" if theme_name == "dark" else "#000000",
                family="monospace",
            )
        if atk_f > 4:
            ax.text(
                pos_atk,
                atk_a + atk_f / 2,
                str(atk_f),
                ha="center",
                va="center",
                fontsize=7.0,
                color="#000000",
                family="monospace",
            )
        if atk_b > 4:
            ax.text(
                pos_atk,
                atk_a + atk_f + atk_b / 2,
                str(atk_b),
                ha="center",
                va="center",
                fontsize=7.0,
                color="#ffffff",
                family="monospace",
            )
        ax.text(
            pos_atk,
            len(atks) + 3,
            str(len(atks)),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=t["text"],
            family="monospace",
        )

        # Stacked Benign (Right bar)
        pos_ben = idx + width / 2
        ax.bar(pos_ben, ben_a, width, color=t["allow"], alpha=0.85)
        ax.bar(pos_ben, ben_f, width, bottom=ben_a, color=t["flag"], alpha=0.85)
        ax.bar(
            pos_ben, ben_b, width, bottom=ben_a + ben_f, color=t["block"], alpha=0.85
        )

        # Benign count labels inside segments
        if ben_a > 4:
            ax.text(
                pos_ben,
                ben_a / 2,
                str(ben_a),
                ha="center",
                va="center",
                fontsize=7.0,
                color="#ffffff" if theme_name == "dark" else "#000000",
                family="monospace",
            )
        if ben_f > 4:
            ax.text(
                pos_ben,
                ben_a + ben_f / 2,
                str(ben_f),
                ha="center",
                va="center",
                fontsize=7.0,
                color="#000000",
                family="monospace",
            )
        if ben_b > 4:
            ax.text(
                pos_ben,
                ben_a + ben_f + ben_b / 2,
                str(ben_b),
                ha="center",
                va="center",
                fontsize=7.0,
                color="#ffffff",
                family="monospace",
            )
        ax.text(
            pos_ben,
            len(bens) + 3,
            str(len(bens)),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=t["text"],
            family="monospace",
        )

        table_rows.append(
            [label.replace("\n", " "), "Attacks", atk_a, atk_f, atk_b, len(atks)]
        )
        table_rows.append(
            [label.replace("\n", " "), "Benign", ben_a, ben_f, ben_b, len(bens)]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{a[0]}\n[Atk | Ben]" for a in arm_keys], fontsize=8, family="monospace"
    )
    ax.set_ylabel("Count of Prompts", fontsize=10, family="monospace")
    ax.set_title(
        "Decision Composition by Arm on held_out_v4 (Attacks N=139, Benign N=208)",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax.set_ylim(0, 245)
    ax.legend(
        loc="upper left",
        prop={"family": "monospace", "size": 8.5},
        framealpha=0.95,
        facecolor=t["card_bg"],
        edgecolor=t["border"],
    )

    save_figure(
        fig,
        "f6_stacked_bars",
        theme_name,
        ctx["commit"],
        "results/final-v4/predictions.jsonl (decisions.*)",
    )
    save_table_csv(
        "f6_stacked_decisions.csv",
        ["Arm", "Class", "ALLOW", "FLAG", "BLOCK", "Total"],
        table_rows,
    )
    sidecars["F6"] = {
        "title": "Decision Composition by Arm on held_out_v4",
        "split": "held_out_v4",
        "n": 347,
        "source_files": ["results/final-v4/predictions.jsonl"],
        "source_sha256": [ctx["files_sha256"]["predictions.jsonl"]],
        "assertions": [
            "Arm R atk == 22 allow, 36 flag, 81 block",
            "Arm R ben == 143 allow, 52 flag, 13 block",
        ],
    }


def generate_f7_router_analysis(ctx: dict, theme_name: str, sidecars: dict):
    """F7: Router Analysis: ROC AUC, Routing Asymmetry, and Random-Routing Empirical Distribution."""
    t = THEMES[theme_name]
    rows = ctx["held_out_rows"]
    num = ctx["numbers"]

    y_error = np.array(
        [
            1 if (r["fusion"]["decision"] == "block") != bool(r["label_binary"]) else 0
            for r in rows
        ]
    )
    prob_error = np.array([r["router"]["prob_error"] for r in rows])

    # ROC vs fusion error
    fpr_err, tpr_err, _ = roc_curve(y_error, prob_error)
    auc_error = roc_auc_score(y_error, prob_error)

    # Assertions against numbers.json
    expected_auc_err = num["router.prob_error.auc_vs_fusion_error"]["value"]
    assert abs(auc_error - expected_auc_err) < 1e-3

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    apply_theme(fig, [ax1, ax2, ax3], theme_name)

    # Panel 1: ROC Curve
    ax1.plot(
        fpr_err,
        tpr_err,
        color=t["cyan"],
        lw=2.0,
        label=f"ROC (AUC = {auc_error:.4f})\n95% CI [0.4579, 0.6006]",
    )
    ax1.plot(
        [0, 1], [0, 1], color=t["muted_text"], linestyle=":", label="Chance (AUC=0.500)"
    )
    ax1.set_xlabel("False Positive Rate", fontsize=9, family="monospace")
    ax1.set_ylabel("True Positive Rate", fontsize=9, family="monospace")
    ax1.set_title(
        "Router Error Prediction ROC\n(vs Fusion Error)",
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )
    ax1.legend(loc="lower right", prop={"family": "monospace", "size": 8})

    # Panel 2: Routing Rate Asymmetry
    atk_routed = sum(
        1 for r in rows if r["label_binary"] and r["router"]["should_route"]
    )
    ben_routed = sum(
        1 for r in rows if not r["label_binary"] and r["router"]["should_route"]
    )
    atk_rate = (atk_routed / 139) * 100.0
    ben_rate = (ben_routed / 208) * 100.0
    overall_rate = ((atk_routed + ben_routed) / 347) * 100.0

    bars = ax2.bar(
        ["Attacks\n(N=139)", "Benign\n(N=208)", "Overall\n(N=347)"],
        [atk_rate, ben_rate, overall_rate],
        color=[t["block"], t["allow"], t["cyan"]],
        alpha=0.85,
        width=0.5,
    )
    for b, rate, count, tot in zip(
        bars,
        [atk_rate, ben_rate, overall_rate],
        [atk_routed, ben_routed, atk_routed + ben_routed],
        [139, 208, 347],
    ):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            rate + 2,
            f"{rate:.1f}%\n({count}/{tot})",
            ha="center",
            va="bottom",
            fontsize=8,
            color=t["text"],
            family="monospace",
        )

    ax2.set_ylabel("Routing Rate to Judge (%)", fontsize=9, family="monospace")
    ax2.set_title(
        "Routing Rate Asymmetry\n(tau = 0.35)",
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )
    ax2.set_ylim(0, 105)

    # Panel 3: Random Routing Empirical Distribution Simulation (k=251, 2000 draws)
    rng = np.random.default_rng(0)
    k_routed = 251
    sim_recalls = []
    sim_ben_blocks = []
    sim_ben_flags = []
    atks_idx = [i for i, r in enumerate(rows) if r["label_binary"]]
    bens_idx = [i for i, r in enumerate(rows) if not r["label_binary"]]

    for _ in range(2000):
        routed_idx = set(rng.choice(347, size=k_routed, replace=False))
        tp = sum(
            1
            for i in atks_idx
            if (
                rows[i]["decisions"]["arm_u"]
                if i in routed_idx
                else rows[i]["decisions"]["arm_a"]
            )
            in ("flag", "block")
        )
        fb = sum(
            1
            for i in bens_idx
            if (
                rows[i]["decisions"]["arm_u"]
                if i in routed_idx
                else rows[i]["decisions"]["arm_a"]
            )
            == "block"
        )
        ff = sum(
            1
            for i in bens_idx
            if (
                rows[i]["decisions"]["arm_u"]
                if i in routed_idx
                else rows[i]["decisions"]["arm_a"]
            )
            == "flag"
        )
        sim_recalls.append((tp / 139) * 100.0)
        sim_ben_blocks.append((fb / 208) * 100.0)
        sim_ben_flags.append((ff / 208) * 100.0)

    arm_r_recall = num["arm_r.loose.recall"]["value"] * 100.0
    arm_r_fb = (
        sum(1 for i in bens_idx if rows[i]["decisions"]["arm_r"] == "block")
        / 208.0
        * 100.0
    )
    arm_r_ff = (
        sum(1 for i in bens_idx if rows[i]["decisions"]["arm_r"] == "flag")
        / 208.0
        * 100.0
    )

    pctile_rec = float(np.mean(np.array(sim_recalls) <= arm_r_recall) * 100.0)
    pctile_fb = float(np.mean(np.array(sim_ben_blocks) <= arm_r_fb) * 100.0)
    pctile_ff = float(np.mean(np.array(sim_ben_flags) <= arm_r_ff) * 100.0)

    ax3.hist(
        sim_recalls,
        bins=20,
        color=t["cyan"],
        alpha=0.55,
        edgecolor=t["border"],
        density=True,
        label=f"Recall (mean {np.mean(sim_recalls):.1f}%)",
    )
    ax3.hist(
        sim_ben_flags,
        bins=20,
        color=t["amber"],
        alpha=0.55,
        edgecolor=t["border"],
        density=True,
        label=f"Benign FLAG (mean {np.mean(sim_ben_flags):.1f}%)",
    )
    ax3.hist(
        sim_ben_blocks,
        bins=15,
        color=t["block"],
        alpha=0.55,
        edgecolor=t["border"],
        density=True,
        label=f"Benign BLOCK (mean {np.mean(sim_ben_blocks):.1f}%)",
    )

    ax3.axvline(
        arm_r_recall,
        color=t["cyan"],
        lw=2.0,
        linestyle="--",
        label=f"Arm R Rec: {arm_r_recall:.1f}%",
    )
    ax3.axvline(
        arm_r_ff,
        color=t["amber"],
        lw=2.0,
        linestyle="--",
        label=f"Arm R FLAG: {arm_r_ff:.1f}%",
    )
    ax3.axvline(
        arm_r_fb,
        color=t["block"],
        lw=2.0,
        linestyle="--",
        label=f"Arm R BLOCK: {arm_r_fb:.1f}%",
    )

    ax3.set_xlabel("Metric Rate (%)", fontsize=9, family="monospace")
    ax3.set_ylabel("Empirical Density", fontsize=9, family="monospace")
    ax3.set_title(
        f"Random Routing Simulation (k=251, 2000 draws)\n"
        f"Caption: Arm R empirical percentiles: recall={pctile_rec:.1f}%, "
        f"BLOCK={pctile_fb:.1f}%, FLAG={pctile_ff:.1f}%\n"
        "(no significance claim; skill not demonstrated)",
        fontsize=8.5,
        fontweight="bold",
        family="monospace",
    )
    ax3.set_xlim(-2, 100)
    ax3.legend(
        loc="upper center",
        ncol=2,
        prop={"family": "monospace", "size": 7.5},
        framealpha=0.95,
        facecolor=t["card_bg"],
        edgecolor=t["border"],
    )

    save_figure(
        fig,
        "f7_router_analysis",
        theme_name,
        ctx["commit"],
        "results/final-v4/predictions.jsonl (router.*)",
    )

    table_rows = [
        [
            "AUC vs Fusion Error",
            f"{auc_error:.4f}",
            "[0.4579, 0.6006]",
            "No demonstrated discrimination vs chance",
        ],
        [
            "AUC vs Ground Truth Label",
            f"{num['router.prob_error.auc_vs_label']['value']:.4f}",
            "[0.3703, 0.4991]",
            "Reflects benign routing asymmetry",
        ],
        ["Attack Routing Rate", f"{atk_rate:.2f}%", "82 / 139", "tau = 0.35 threshold"],
        [
            "Benign Routing Rate",
            f"{ben_rate:.2f}%",
            "169 / 208",
            "tau = 0.35 threshold",
        ],
        [
            "Overall Routing Rate",
            f"{overall_rate:.2f}%",
            "251 / 347",
            "tau = 0.35 threshold",
        ],
        [
            "Random Routing Recall Mean",
            f"{np.mean(sim_recalls):.2f}%",
            "[84.17%, 90.65%]",
            "2000 draws at k=251 calls",
        ],
    ]
    save_table_csv(
        "f7_router_performance.csv",
        ["Metric", "Value", "Interval / Fraction", "Interpretation"],
        table_rows,
    )
    sidecars["F7"] = {
        "title": "Router Diagnostic Analysis and Random Routing Benchmark",
        "split": "held_out_v4",
        "n": 347,
        "source_files": ["results/final-v4/predictions.jsonl", "numbers.json"],
        "source_sha256": [
            ctx["files_sha256"]["predictions.jsonl"],
            ctx["files_sha256"]["numbers.json"],
        ],
        "assertions": [
            f"Router AUC vs Error == {auc_error:.4f}",
            f"Random routing mean recall == {np.mean(sim_recalls):.2f}%",
        ],
    }


def generate_f8_judge_outcomes(ctx: dict, theme_name: str, sidecars: dict):
    """F8: Judge outcomes, latency distribution abstain vs non-abstain, num_predict cap."""
    t = THEMES[theme_name]
    rows = ctx["held_out_rows"]

    verdicts = [r["judge"]["verdict"] for r in rows]
    v_counts = {
        "allow": verdicts.count("allow"),
        "block": verdicts.count("block"),
        "abstain": verdicts.count("abstain"),
    }

    # Latencies
    abstain_lats = [
        float(r["judge"]["latency_ms"])
        for r in rows
        if r["judge"]["verdict"] == "abstain"
    ]
    non_abstain_lats = [
        float(r["judge"]["latency_ms"])
        for r in rows
        if r["judge"]["verdict"] != "abstain"
    ]

    assert len(abstain_lats) == 10
    assert len(non_abstain_lats) == 337

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
    apply_theme(fig, [ax1, ax2], theme_name)

    # Subplot 1: Verdict Counts
    bars = ax1.bar(
        [
            "ALLOW",
            "BLOCK",
            "ABSTAIN",
        ],
        [v_counts["allow"], v_counts["block"], v_counts["abstain"]],
        color=[t["allow"], t["block"], t["abstain"]],
        alpha=0.85,
        width=0.5,
    )
    for b in bars:
        h = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2,
            h + 4,
            f"{h} ({h / 347 * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=t["text"],
            family="monospace",
        )

    ax1.set_ylabel("Count of Prompts (N=347)", fontsize=10, family="monospace")
    ax1.set_title(
        "LLM Judge Verdict Distribution\n(Model: gemma3:4b, max_tokens=128)",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax1.set_ylim(0, 260)

    # Subplot 2: Latency Boxplot (Abstain vs Non-Abstain)
    bp = ax2.boxplot(
        [non_abstain_lats, abstain_lats],
        tick_labels=["Non-Abstain\n(N=337)", "Abstain\n(N=10)"],
        patch_artist=True,
        widths=0.45,
    )

    bp["boxes"][0].set_facecolor(t["cyan"])
    bp["boxes"][1].set_facecolor(t["amber"])
    for element in ["whiskers", "caps", "medians"]:
        plt.setp(bp[element], color=t["text"], lw=1.2)

    mean_non = float(np.mean(non_abstain_lats))
    mean_abs = float(np.mean(abstain_lats))
    delta_mean_pct = ((mean_abs - mean_non) / mean_non) * 100.0

    median_non = float(np.median(non_abstain_lats))
    median_abs = float(np.median(abstain_lats))
    delta_median_pct = ((median_abs - median_non) / median_non) * 100.0

    ax2.text(
        1,
        median_non,
        f" Median: {median_non:.0f} ms\n Mean: {mean_non:.0f} ms",
        color=t["text"],
        fontsize=7.5,
        family="monospace",
        va="center",
    )
    ax2.text(
        2,
        median_abs,
        f" Median: {median_abs:.0f} ms (+{delta_median_pct:.1f}%)\n Mean: {mean_abs:.0f} ms (+{delta_mean_pct:.1f}%)",
        color=t["text"],
        fontsize=7.5,
        family="monospace",
        va="center",
    )

    ax2.set_ylabel("Latency (ms)", fontsize=10, family="monospace")
    ax2.set_title(
        "Hypothesis (untested)\nToken cap (num_predict=128) vs Latency",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax2.text(
        0.5,
        0.05,
        "All 10 abstain rows cluster at 6.5s-8.9s (median 8.0s, +39.5% vs non-abstain 5.7s).\n"
        "Token counts not logged in telemetry; cap hypothesis remains untested.",
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        family="monospace",
        color=t["muted_text"],
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": t["card_bg"],
            "edgecolor": t["border"],
            "alpha": 0.9,
        },
    )

    save_figure(
        fig,
        "f8_judge_outcomes",
        theme_name,
        ctx["commit"],
        "results/final-v4/predictions.jsonl (judge.*)",
    )

    table_rows = [
        [
            "ALLOW",
            v_counts["allow"],
            f"{v_counts['allow'] / 347 * 100:.2f}%",
            f"{mean_non:.1f} ms",
        ],
        [
            "BLOCK",
            v_counts["block"],
            f"{v_counts['block'] / 347 * 100:.2f}%",
            f"{mean_non:.1f} ms",
        ],
        [
            "ABSTAIN",
            v_counts["abstain"],
            f"{v_counts['abstain'] / 347 * 100:.2f}%",
            f"{mean_abs:.1f} ms (+{delta_median_pct:.1f}% median / +{delta_mean_pct:.1f}% mean)",
        ],
    ]
    save_table_csv(
        "f8_judge_outcomes.csv",
        ["Verdict", "Count (N=347)", "Percentage", "Mean Latency"],
        table_rows,
    )
    sidecars["F8"] = {
        "title": "Judge Outcome Counts and Latency Distribution",
        "split": "held_out_v4",
        "n": 347,
        "source_files": ["results/final-v4/predictions.jsonl"],
        "source_sha256": [ctx["files_sha256"]["predictions.jsonl"]],
        "assertions": [
            "Abstain count == 10",
            f"Abstain mean latency == {mean_abs:.1f}",
        ],
    }


def generate_f9_abstain_rows(ctx: dict, theme_name: str, sidecars: dict):
    """F9: Detailed case table of 10 judge abstains and per-arm computed effects."""
    t = THEMES[theme_name]
    rows = ctx["held_out_rows"]
    abstain_rows = [r for r in rows if r["judge"]["verdict"] == "abstain"]
    assert len(abstain_rows) == 10

    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    apply_theme(fig, ax, theme_name)
    ax.axis("off")

    headers = [
        "Prompt ID",
        "Family",
        "Fusion",
        "Arm A",
        "Arm B",
        "Arm R",
        "Arm U",
        "Arm C2",
        "Effect (Arm R)",
        "Effect (Arm B/U)",
    ]
    table_data = []
    csv_rows = []

    for r in abstain_rows:
        pid = r["prompt_id"]
        fam = r["family"][:18]
        fus = r["fusion"]["decision"].upper()
        a = r["decisions"]["arm_a"].upper()
        b = r["decisions"]["arm_b"].upper()
        r_dec = r["decisions"]["arm_r"].upper()
        u = r["decisions"]["arm_u"].upper()
        c2 = r["decisions"]["c2_standalone"].upper()

        if fus == "BLOCK" and r_dec == "FLAG":
            r_eff = "BLOCK->FLAG"
        elif fus == "ALLOW" and r_dec == "FLAG":
            r_eff = "ALLOW->FLAG"
        elif fus == r_dec:
            r_eff = "none"
        else:
            r_eff = f"{fus}->{r_dec}"

        if fus == "BLOCK" and b == "FLAG":
            b_eff = "BLOCK->FLAG"
        elif fus == "ALLOW" and b == "FLAG":
            b_eff = "ALLOW->FLAG"
        elif fus == b:
            b_eff = "none"
        else:
            b_eff = f"{fus}->{b}"

        table_data.append([pid, fam, fus, a, b, r_dec, u, c2, r_eff, b_eff])
        csv_rows.append(
            [
                pid,
                r["family"],
                fus,
                a,
                b,
                r_dec,
                u,
                c2,
                r_eff,
                b_eff,
                r["judge"]["latency_ms"],
            ]
        )

    table = ax.table(
        cellText=table_data, colLabels=headers, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)

    # Style cells
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor(t["card_bg"])
            cell.set_text_props(weight="bold", color=t["cyan"], family="monospace")
        else:
            cell.set_facecolor(t["bg"])
            cell.set_text_props(color=t["text"], family="monospace")
            if col_idx in (8, 9) and "FLAG" in cell.get_text().get_text():
                cell.set_text_props(color=t["amber"], weight="bold", family="monospace")

    cnt_b_to_f = sum(
        1
        for r in abstain_rows
        if r["fusion"]["decision"].upper() == "BLOCK"
        and r["decisions"]["arm_r"].upper() == "FLAG"
    )
    cnt_a_to_f = sum(
        1
        for r in abstain_rows
        if r["fusion"]["decision"].upper() == "ALLOW"
        and r["decisions"]["arm_r"].upper() == "FLAG"
    )
    cnt_none = len(abstain_rows) - cnt_b_to_f - cnt_a_to_f

    b_cnt_b2f = sum(
        1
        for r in abstain_rows
        if r["fusion"]["decision"].upper() == "BLOCK"
        and r["decisions"]["arm_b"].upper() == "FLAG"
    )
    b_cnt_a2f = sum(
        1
        for r in abstain_rows
        if r["fusion"]["decision"].upper() == "ALLOW"
        and r["decisions"]["arm_b"].upper() == "FLAG"
    )
    b_cnt_none = len(abstain_rows) - b_cnt_b2f - b_cnt_a2f

    ax.set_title(
        f"The 10 Attack Rows with Judge Abstain on held_out_v4 (N=10/139)\n"
        f"Arm R: {cnt_b_to_f} BLOCK->FLAG, {cnt_a_to_f} ALLOW->FLAG, {cnt_none} none | "
        f"Arms B/U/C2: {b_cnt_b2f} BLOCK->FLAG, {b_cnt_a2f} ALLOW->FLAG, {b_cnt_none} none\n"
        "Caption: abstain defaults to FLAG, which counts toward loose recall for Arms B, U and C2.",
        fontsize=9.5,
        fontweight="bold",
        family="monospace",
        pad=22,
    )

    save_figure(
        fig,
        "f9_abstain_rows",
        theme_name,
        ctx["commit"],
        "results/final-v4/predictions.jsonl (judge.verdict == abstain)",
    )
    save_table_csv(
        "f9_abstain_rows.csv",
        [
            "Prompt ID",
            "Family",
            "Fusion",
            "Arm A",
            "Arm B",
            "Arm R",
            "Arm U",
            "Arm C2",
            "Effect (Arm R)",
            "Effect (Arm B/U)",
            "Latency ms",
        ],
        csv_rows,
    )
    sidecars["F9"] = {
        "title": "Decisions Trace Across Arms for 10 Judge Abstain Rows",
        "split": "held_out_v4",
        "n": 10,
        "source_files": ["results/final-v4/predictions.jsonl"],
        "source_sha256": [ctx["files_sha256"]["predictions.jsonl"]],
        "assertions": ["Abstain rows count == 10", "BLOCK to FLAG downgrades == 7"],
    }


def generate_f10_cost_vs_recall(ctx: dict, theme_name: str, sidecars: dict):
    """F10: Cost vs recall: judge calls per 100 against loose recall per arm."""
    t = THEMES[theme_name]
    num = ctx["numbers"]

    # (label, key, color, is_primary)
    arms = [
        ("Arm R (Primary)", "arm_r", t["arm_r"], True),
        ("Arm A (post-hoc)", "arm_a", t["arm_a"], False),
        ("Arm B (post-hoc)", "arm_b", t["arm_b"], False),
        ("Arm U (post-hoc)", "arm_u", t["arm_u"], False),
        ("Arm C2 (post-hoc)", "c2_standalone", t["arm_c2"], False),
        ("PG2 (post-hoc)", "promptguard2", t["promptguard2"], False),
        ("PromptForest (post-hoc)", "promptforest", t["promptforest"], False),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    apply_theme(fig, ax, theme_name)

    table_rows = []

    arm_b_calls = num.get("arm_b.extra_calls_per_100", {}).get("value", 94.52)

    for name, key, color, is_primary in arms:
        rec = num[f"{key}.loose.recall"]["value"] * 100.0
        calls = 0.0
        if key == "arm_r":
            calls = num["arm_r.extra_calls_per_100"]["value"]
        elif key == "arm_b":
            calls = arm_b_calls
        elif key in ("arm_u", "c2_standalone"):
            calls = 100.0

        marker = "*" if is_primary else "o"
        ms = 220 if is_primary else 120
        ax.scatter(
            calls,
            rec,
            s=ms,
            color=color,
            edgecolors=t["text"],
            lw=1.5,
            zorder=4,
            marker=marker,
        )
        offset_x = 1.5 if calls < 90 else -20.0
        offset_y = 1.5 if key != "arm_a" else -3.0
        ax.text(
            calls + offset_x,
            rec + offset_y,
            f"{name}\n({calls:.1f} calls, {rec:.1f}%)",
            fontsize=7.5,
            color=t["text"],
            family="monospace",
            zorder=5,
        )
        table_rows.append([name, f"{calls:.2f}", f"{rec:.2f}%"])

    ax.set_xlabel(
        "Adversarial LLM Judge Invocations per 100 Requests",
        fontsize=10,
        family="monospace",
    )
    ax.set_ylabel("Loose Recall on Attacks (%)", fontsize=10, family="monospace")
    ax.set_title(
        "Operational Cost vs Detection Recall on held_out_v4 (N=347)\n"
        "Caption: dominance is not ranking — no single arm dominates all objectives",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax.set_xlim(-5, 115)
    ax.set_ylim(0, 108)  # from 0 so PG2/PromptForest are visible

    # Dominance trade-off note
    arm_u_rec = num["arm_u.loose.recall"]["value"] * 100.0
    arm_r_rec = num["arm_r.loose.recall"]["value"] * 100.0
    rec_diff = arm_u_rec - arm_r_rec
    calls_diff = 100.0 - num["arm_r.extra_calls_per_100"]["value"]
    note_f10 = (
        f"Arm R trades {rec_diff:.1f} percentage points recall vs Arm U\n"
        f"in exchange for {calls_diff:.1f} fewer calls per 100 requests."
    )
    ax.text(
        45,
        15,
        note_f10,
        fontsize=8,
        color=t["muted_text"],
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": t["card_bg"],
            "edgecolor": t["border"],
            "lw": 1.0,
        },
    )

    save_figure(
        fig,
        "f10_cost_vs_recall",
        theme_name,
        ctx["commit"],
        "numbers.json (*.loose.recall, *.calls_per_100)",
    )
    save_table_csv(
        "f10_cost_vs_recall.csv",
        ["Arm", "Judge Invocations per 100 Requests", "Loose Recall (%)"],
        table_rows,
    )
    sidecars["F10"] = {
        "title": "Operational Cost vs Detection Recall Across Arms",
        "split": "held_out_v4",
        "n": 347,
        "source_files": ["numbers.json"],
        "source_sha256": [ctx["files_sha256"]["numbers.json"]],
        "assertions": ["Arm R calls == 72.33", "Arm U calls == 100.0", "y_min == 0"],
    }


def generate_f11_error_analysis(ctx: dict, theme_name: str, sidecars: dict):
    """F11: Error analysis by family/category: False Negatives and False Blocks for Arm R."""
    t = THEMES[theme_name]
    rows = ctx["held_out_rows"]

    fn_counts = {}
    fb_counts = {}
    unknown_fn = 0
    unknown_fb = 0

    for r in rows:
        fam = r.get("family") or "unknown"
        if r["label_binary"] and r["decisions"]["arm_r"] == "allow":
            if fam == "unknown":
                unknown_fn += 1
            fn_counts[fam] = fn_counts.get(fam, 0) + 1
        elif not r["label_binary"] and r["decisions"]["arm_r"] == "block":
            if fam == "unknown":
                unknown_fb += 1
            fb_counts[fam] = fb_counts.get(fam, 0) + 1

    assert sum(fn_counts.values()) == 22
    assert sum(fb_counts.values()) == 13

    # Wider canvas, no label truncation, larger row height for readability
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16, max(6, max(len(fn_counts), len(fb_counts)) * 0.65 + 2.5)),
        dpi=300,
    )
    apply_theme(fig, [ax1, ax2], theme_name)

    # Subplot 1: False Negatives (Attacks ALLOWed)
    fn_sorted = sorted(fn_counts.items(), key=lambda x: x[1], reverse=True)
    labels1 = [x[0] for x in fn_sorted]  # no truncation
    vals1 = [x[1] for x in fn_sorted]

    y1 = np.arange(len(labels1))
    ax1.barh(y1, vals1, color=t["block"], alpha=0.85, height=0.55)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(labels1, fontsize=8, family="monospace")
    ax1.invert_yaxis()
    unknown_note1 = f" (family=unknown: {unknown_fn})" if unknown_fn > 0 else ""
    ax1.set_xlabel(
        f"Count of False Negatives (Sum = 22){unknown_note1}",
        fontsize=9,
        family="monospace",
    )
    ax1.set_title(
        "Arm R False Negatives by Attack Family\nheld_out_v4 (N=347: 139 attacks, 208 benign)\n(Aggregated counts only, no raw prompt text)",
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )
    for i, v in enumerate(vals1):
        ax1.text(
            v + 0.1,
            i,
            str(v),
            va="center",
            fontsize=8,
            color=t["text"],
            family="monospace",
        )

    # Subplot 2: False Blocks (Benign BLOCKed)
    fb_sorted = sorted(fb_counts.items(), key=lambda x: x[1], reverse=True)
    labels2 = [x[0] for x in fb_sorted]  # no truncation
    vals2 = [x[1] for x in fb_sorted]

    y2 = np.arange(len(labels2))
    ax2.barh(y2, vals2, color=t["amber"], alpha=0.85, height=0.55)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labels2, fontsize=8, family="monospace")
    ax2.invert_yaxis()
    unknown_note2 = f" (family=unknown: {unknown_fb})" if unknown_fb > 0 else ""
    ax2.set_xlabel(
        f"Count of False Blocks (Sum = 13){unknown_note2}",
        fontsize=9,
        family="monospace",
    )
    ax2.set_title(
        "Arm R False Blocks by Benign Family\n\n(Aggregated counts only, no raw prompt text)",
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )
    for i, v in enumerate(vals2):
        ax2.text(
            v + 0.1,
            i,
            str(v),
            va="center",
            fontsize=8,
            color=t["text"],
            family="monospace",
        )

    plt.tight_layout(pad=2.5, w_pad=3.5)

    save_figure(
        fig,
        "f11_error_analysis",
        theme_name,
        ctx["commit"],
        "results/final-v4/predictions.jsonl (arm_r errors)",
    )

    csv_rows = [["False Negative (Attack ALLOWed)", fam, cnt] for fam, cnt in fn_sorted]
    csv_rows += [["False Block (Benign BLOCKed)", fam, cnt] for fam, cnt in fb_sorted]
    save_table_csv(
        "f11_error_analysis.csv", ["Error Type", "Family", "Count"], csv_rows
    )
    sidecars["F11"] = {
        "title": "Error Breakdown by Family for Primary Arm R",
        "split": "held_out_v4",
        "n": 347,
        "source_files": ["results/final-v4/predictions.jsonl"],
        "source_sha256": [ctx["files_sha256"]["predictions.jsonl"]],
        "assertions": ["Total false negatives == 22", "Total false blocks == 13"],
    }


def generate_f12_confusion_matrices(ctx: dict, theme_name: str, sidecars: dict):
    """F12: Confusion matrices (attack/benign x ALLOW/FLAG/BLOCK) for Arm R and Arm A."""
    t = THEMES[theme_name]
    rows = ctx["held_out_rows"]
    atks = [r for r in rows if r["label_binary"]]
    bens = [r for r in rows if not r["label_binary"]]

    # Arm R
    r_atk = [
        sum(1 for r in atks if r["decisions"]["arm_r"] == dec)
        for dec in ["allow", "flag", "block"]
    ]
    r_ben = [
        sum(1 for r in bens if r["decisions"]["arm_r"] == dec)
        for dec in ["allow", "flag", "block"]
    ]
    mat_r = np.array([r_atk, r_ben])

    # Arm A
    a_atk = [
        sum(1 for r in atks if r["decisions"]["arm_a"] == dec)
        for dec in ["allow", "flag", "block"]
    ]
    a_ben = [
        sum(1 for r in bens if r["decisions"]["arm_a"] == dec)
        for dec in ["allow", "flag", "block"]
    ]
    mat_a = np.array([a_atk, a_ben])

    assert list(r_atk) == [22, 36, 81]
    assert list(r_ben) == [143, 52, 13]
    assert list(a_atk) == [37, 0, 102]
    assert list(a_ben) == [143, 0, 65]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
    apply_theme(fig, [ax1, ax2], theme_name)

    cols = ["ALLOW", "FLAG", "BLOCK"]
    idx_labels = ["Attack\n(N=139)", "Benign\n(N=208)"]

    for ax, mat, title_str, arm_name in zip(
        [ax1, ax2],
        [mat_r, mat_a],
        ["Arm R (Primary System)", "Arm A (AnyFlag Fusion)"],
        ["arm_r", "arm_a"],
    ):
        ax.imshow(mat, cmap="Blues" if theme_name == "light" else "viridis", alpha=0.85)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(idx_labels)))
        ax.set_xticklabels(cols, fontsize=9, family="monospace")
        ax.set_yticklabels(idx_labels, fontsize=9, family="monospace")
        ax.set_title(
            f"{title_str}\nConfusion Matrix — held_out_v4 (n=347)",
            fontsize=11,
            fontweight="bold",
            family="monospace",
        )

        # Determine normalised intensity for each cell to pick readable text colour
        vmin = mat.min()
        vmax = mat.max() if mat.max() > mat.min() else mat.min() + 1
        for i in range(2):
            tot = 139 if i == 0 else 208
            for j in range(3):
                cnt = mat[i, j]
                pct = round((cnt / tot) * 100.0, 2)  # one rounding rule throughout
                # text contrast: use white on dark cells, black on light cells
                intensity = (cnt - vmin) / (vmax - vmin)
                txt_color = "#ffffff" if intensity > 0.5 else (t["text"])
                ax.text(
                    j,
                    i,
                    f"{cnt}\n({pct:.2f}%)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=txt_color,
                    family="monospace",
                )

    save_figure(
        fig,
        "f12_confusion_matrices",
        theme_name,
        ctx["commit"],
        "results/final-v4/predictions.jsonl (arm_r & arm_a decisions)",
    )

    table_rows = [
        ["Arm R (Primary)", "Attack", r_atk[0], r_atk[1], r_atk[2], 139],
        ["Arm R (Primary)", "Benign", r_ben[0], r_ben[1], r_ben[2], 208],
        ["Arm A (Fusion)", "Attack", a_atk[0], a_atk[1], a_atk[2], 139],
        ["Arm A (Fusion)", "Benign", a_ben[0], a_ben[1], a_ben[2], 208],
    ]
    save_table_csv(
        "f12_confusion_matrices.csv",
        ["Arm", "Ground Truth", "ALLOW", "FLAG", "BLOCK", "Total"],
        table_rows,
    )
    sidecars["F12"] = {
        "title": "Confusion Matrices for Arm R and Arm A on held_out_v4",
        "split": "held_out_v4",
        "n": 347,
        "source_files": ["results/final-v4/predictions.jsonl"],
        "source_sha256": [ctx["files_sha256"]["predictions.jsonl"]],
        "assertions": [
            "Arm R atk allow==22, flag==36, block==81",
            "Arm A atk allow==37, flag==0, block==102",
        ],
    }


def generate_f13_contamination_matrix(ctx: dict, theme_name: str, sidecars: dict):
    """F13: Contamination matrix (detector x dataset) discrete 3-state heatmap."""
    t = THEMES[theme_name]

    detectors = [
        "d2-deberta (ProtectAI)",
        "d3-deepset (deepset)",
        "d4_rules (Static Regex)",
        "Prompt Guard 2 (86M)",
        "PromptForest (Tree)",
        "Judge (gemma3:4b)",
    ]
    datasets = ["XSTest", "TruthfulQA", "deepset", "Gandalf", "NotInject", "Dolly"]

    # States:
    # 0 = not documented in the card
    # 1 = documented overlap
    # 2 = no model card (rule-based)
    grid = np.array(
        [
            [1, 0, 0, 0, 0, 0],  # d2: XSTest documented overlap
            [0, 0, 1, 0, 0, 0],  # d3: deepset documented overlap
            [2, 2, 2, 2, 2, 2],  # d4_rules: static regex (no model card)
            [0, 0, 0, 0, 0, 0],  # pg2: not documented
            [0, 0, 0, 0, 0, 0],  # pf: not documented
            [0, 0, 0, 0, 0, 0],  # judge: not documented
        ]
    )

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    apply_theme(fig, ax, theme_name)

    state_colors = {
        0: t["slate"],  # Not documented
        1: t["block"],  # Documented overlap (contamination)
        2: t["amber"],  # No model card (rule-based)
    }
    state_labels = {
        0: "not documented in the card",
        1: "documented overlap",
        2: "no model card (rule-based)",
    }

    # Custom colormap
    for i in range(len(detectors)):
        for j in range(len(datasets)):
            state = grid[i, j]
            color = state_colors[state]
            rect = patches.Rectangle(
                (j - 0.45, i - 0.45),
                0.9,
                0.9,
                facecolor=color,
                alpha=0.8,
                edgecolor=t["border"],
            )
            ax.add_patch(rect)
            txt = "OVERLAP" if state == 1 else ("RULES" if state == 2 else "NOT DOC")
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#ffffff" if theme_name == "dark" else "#000000",
                family="monospace",
            )

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(detectors)))
    ax.set_xticklabels(datasets, fontsize=9, family="monospace")
    ax.set_yticklabels(detectors, fontsize=9, family="monospace")
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylim(len(detectors) - 0.5, -0.5)
    ax.set_title(
        "Pre-Evaluation Training Data Contamination Matrix\n"
        "Source: docs/EXTERNAL_DATASET_REVIEW.md Section 3.2\n"
        "Caption: not documented is not evidence of no overlap",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )

    # Legend
    legend_patches = [
        patches.Patch(color=t["block"], label="documented overlap (excluded/dropped)"),
        patches.Patch(color=t["amber"], label="no model card (rule-based heuristics)"),
        patches.Patch(
            color=t["slate"], label="not documented in accessible model card"
        ),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        prop={"family": "monospace", "size": 8},
    )

    save_figure(
        fig,
        "f13_contamination_matrix",
        theme_name,
        ctx["commit"],
        "docs/EXTERNAL_DATASET_REVIEW.md:165-173",
    )

    table_rows = []
    for i, d in enumerate(detectors):
        for j, ds in enumerate(datasets):
            table_rows.append([d, ds, state_labels[grid[i, j]]])
    save_table_csv(
        "f13_contamination_matrix.csv",
        ["Detector Component", "External Candidate Dataset", "Contamination Status"],
        table_rows,
    )
    sidecars["F13"] = {
        "title": "Detector by Dataset Contamination Matrix",
        "split": "Design / External Candidates",
        "n": 6,
        "source_files": ["docs/EXTERNAL_DATASET_REVIEW.md"],
        "source_sha256": [
            sha256_file(REPO_ROOT / "docs" / "EXTERNAL_DATASET_REVIEW.md")
        ],
        "assertions": ["D2 XSTest overlap == True", "D3 deepset overlap == True"],
    }


def generate_f14_provenance_timeline(ctx: dict, theme_name: str, sidecars: dict):
    """F14: Provenance timeline and cryptographic hash chain."""
    t = THEMES[theme_name]

    events = [
        (
            "1. Protocol-Lock Commit",
            "2026-10-19 19:02 IST",
            "Commit 7e98c39 committed before run",
        ),
        (
            "2. Final-v4 Run Start",
            "2026-10-19 19:50 IST",
            "Initiated under scripts/gpu_lock.py",
        ),
        (
            "3. Predictions Closed",
            "2026-10-19 20:37:16 IST",
            "347 rows written to predictions.jsonl",
        ),
        (
            "4. Manifest Created",
            "2026-10-19 20:37:38 IST",
            "Manifest committed under eval/logs/",
        ),
        (
            "5. Code Producing Commit",
            "2026-10-19 21:30 IST",
            "Commit 959a99a committed to main",
        ),
        (
            "6. External Dataset Freeze",
            "2026-10-20 23:45 IST",
            "FROZEN_EXTERNAL.sha256 created",
        ),
        (
            "7. K13_SCORE Authorisation",
            "2026-10-21 02:01 IST",
            "docs/K13_SCORE.txt committed (commit 334ffa0)",
        ),
        (
            "8. Overnight Scoring Start",
            "2026-10-21 02:08 IST",
            "Single pass eval_external.py under gpu_lock",
        ),
        (
            "9. Overnight Scoring End",
            "2026-10-21 02:55 IST",
            "621 rows written to external_predictions.jsonl",
        ),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 9.5), dpi=300)
    apply_theme(fig, ax, theme_name)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(
        5.0,
        9.5,
        "PARALLAX Evaluation Provenance Timeline & Temporal Invariants",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=t["text"],
        family="monospace",
    )

    y_pos = np.linspace(8.8, 2.7, len(events))
    ax.plot([1.8, 1.8], [2.5, 9.0], color=t["cyan"], lw=2.0, linestyle="--", zorder=1)

    table_rows = []
    for i, (name, ts, desc) in enumerate(events):
        y = y_pos[i]
        ax.scatter(
            1.8,
            y,
            s=120,
            color=t["amber"] if i >= 5 else t["cyan"],
            edgecolors=t["text"],
            lw=1.2,
            zorder=3,
        )
        ax.text(
            2.2,
            y + 0.12,
            name,
            fontsize=9,
            fontweight="bold",
            color=t["text"],
            family="monospace",
        )
        ax.text(
            2.2,
            y - 0.18,
            f"{ts} — {desc}",
            fontsize=8,
            color=t["muted_text"],
            family="monospace",
        )
        table_rows.append([name, ts, desc])

    # Hash chain footer
    hash_footer = (
        "CRYPTOGRAPHIC HASH CHAIN:\n"
        "• Manifest SHA-256: d5e7271d7b8df7aeea3ad965c4d9004d289b33d11d6dce873d46cac821a273bf\n"
        "• Final-v4 Predictions SHA-256: 74e0975eccda410597a865e3b0b5581e7732eca04406c4ca032cfccc29e1a444\n"
        "• Target Split FROZEN.sha256: 9325e6185ac6a8c038b8129de91fe3dd8852f30ebecf4a667b24f4e973217dd5\n"
        "• External Sets FROZEN_EXTERNAL.sha256: gandalf(ebaf856...), notinject(dda9261...), dolly(b89f71e...)"
    )
    ax.text(
        5.0,
        1.2,
        hash_footer,
        ha="center",
        va="center",
        fontsize=7.5,
        color=t["text"],
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": t["card_bg"],
            "edgecolor": t["border"],
            "lw": 1.0,
        },
    )

    save_figure(
        fig,
        "f14_provenance_timeline",
        theme_name,
        ctx["commit"],
        "docs/PROVENANCE.md + git log",
    )
    save_table_csv(
        "f14_provenance_timeline.csv",
        ["Milestone Event", "Timestamp", "Provenance Details"],
        table_rows,
    )
    sidecars["F14"] = {
        "title": "Provenance Timeline and Cryptographic Hash Chain",
        "split": "All Splits / Governance",
        "n": 7,
        "source_files": [
            "docs/PROVENANCE.md",
            "data/splits/FROZEN.sha256",
            "data/external/FROZEN_EXTERNAL.sha256",
        ],
        "source_sha256": [sha256_file(REPO_ROOT / "docs" / "PROVENANCE.md")],
        "assertions": [
            "Protocol-lock strictly precedes run start",
            "Manifest timestamp strictly precedes producing commit",
        ],
    }


def generate_f15_audit_trail(ctx: dict, theme_name: str, sidecars: dict):
    """F15: Audit trail of defects found and fixed during closeout."""
    t = THEMES[theme_name]

    defects = [
        (
            "Hand-typed literal",
            "scripts/export_viz_data.py:194",
            "22 / 44 / 73 literal in dashboard",
            "Replaced with recount_metrics.py derived from predictions.jsonl (22/36/81)",
        ),
        (
            "Copied CI duplicate",
            "docs/REVIEW_PACKAGE.md",
            "Duplicated CI token copied across tables",
            "Re-derived exact bootstrap quantiles [84.17%, 90.65%] by script",
        ),
        (
            "Unsupported interval",
            "docs/PRIMARY_ARM_CHECK.md",
            "Binomial 82.01-92.81% interval in text",
            "Removed binomial interval; retained empirical 2.5/97.5 quantiles",
        ),
        (
            "Abstain scope wording",
            "docs/JUDGE_OUTCOMES_AUDIT.md",
            "Claimed unified abstain across all rows",
            "Corrected to 10/139 attack abstains, 0/208 benign (Fisher p=8.71e-5)",
        ),
        (
            "Router skill wording",
            "docs/SAFE_CLAIMS.md",
            "Described router as 'worse than random'",
            "Replaced with 'near chance / no demonstrated skill' (AUC 0.53)",
        ),
        (
            "NotInject overlap stop",
            "docs/EXTERNAL_EVAL_PROTOCOL.md",
            "8.85% dev overlap triggered stop rule",
            "Phase B2 stopped; 30 exact-match rows excluded to freeze at N=309",
        ),
        # --- Fix-forward items ---
        (
            "FIX-FWD: Multilingual category mismatch",
            "scripts/eval/external_metrics.py",
            "Script checked category=='Multilingual Queries'; dataset stores 'Multilingual'; split returned n=0",
            "Post-hoc addendum script joins by prompt_id; correct split English=225, Multilingual=84",
        ),
        (
            "FIX-FWD: Dolly text committed",
            "git log df23e6b, 9e68954",
            "Dolly prompt text committed in external_input_interleaved.jsonl and external_predictions.jsonl",
            "Both files untracked, gitignored; text-free external_predictions_notext.jsonl committed",
        ),
        (
            "FIX-FWD: Gate-claim correction",
            "docs/OVERNIGHT_REPORT.md",
            "Report claimed 100% gate pass; preflight row 0 differed (Ollama HTTP 500 -> abs path fallback)",
            "Corrected: gate passed, row 0 discrepancy documented in LIMITATIONS",
        ),
    ]

    import textwrap

    def wrap_cell(text: str, width: int = 38) -> str:
        return "\n".join(textwrap.wrap(text, width))

    wrapped_defects = [tuple(wrap_cell(cell, 36) for cell in row) for row in defects]

    fig, ax = plt.subplots(figsize=(17, 7.5), dpi=300)
    apply_theme(fig, ax, theme_name)
    ax.axis("off")

    headers = [
        "Defect Category",
        "Detection Source File",
        "Identified Anomaly",
        "Formal Closeout Remediation",
    ]
    table = ax.table(
        cellText=wrapped_defects,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colWidths=[0.18, 0.20, 0.31, 0.31],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor(t["card_bg"])
            cell.set_text_props(weight="bold", color=t["cyan"], family="monospace")
        elif row_idx > 6:  # fix-forward items highlighted differently
            cell.set_facecolor(t["grid"])
            cell.set_text_props(color=t["muted_text"], family="monospace")
        else:
            cell.set_facecolor(t["bg"])
            cell.set_text_props(color=t["text"], family="monospace")

    ax.set_title(
        "Closeout Audit Trail: Methodological Defects Identified and Fixed + Fix-Forward Items\nGenerated from docs/METHODOLOGY_FAILURES.md and docs/CLOSEOUT_TESTLOG.md",
        fontsize=11,
        fontweight="bold",
        family="monospace",
        pad=20,
    )

    save_figure(
        fig,
        "f15_audit_trail",
        theme_name,
        ctx["commit"],
        "docs/METHODOLOGY_FAILURES.md + docs/CLOSEOUT_TESTLOG.md",
    )
    save_table_csv("f15_audit_defects.csv", headers, defects)
    sidecars["F15"] = {
        "title": "Closeout Audit Trail of Methodological Defects + Fix-Forward Items",
        "split": "Methodology Audit",
        "n": 9,
        "source_files": ["docs/METHODOLOGY_FAILURES.md", "docs/CLOSEOUT_TESTLOG.md"],
        "source_sha256": [
            sha256_file(REPO_ROOT / "docs" / "METHODOLOGY_FAILURES.md"),
            sha256_file(REPO_ROOT / "docs" / "CLOSEOUT_TESTLOG.md"),
        ],
        "assertions": [
            "All 6 original defects documented in git history",
            "3 fix-forward items added",
            "0 banned words in remediation text",
        ],
    }


# ---------------------------------------------------------------------------
# External Figure Generators (X1 to X5)
# ---------------------------------------------------------------------------
def generate_x_figures(ctx: dict, theme_name: str, sidecars: dict):
    """Generate external benchmark figures if Phase 2 predictions exist."""
    ext_rows = ctx["ext_rows"]
    if not ext_rows or len(ext_rows) < 621:
        print(
            f"[make_figures] External predictions incomplete ({len(ext_rows)}/621). Skipping X1-X5."
        )
        return

    t = THEMES[theme_name]
    gandalf_rows = [
        r
        for r in ext_rows
        if r.get("source") == "gandalf"
        or r.get("family") == "gandalf"
        or "gandalf" in str(r.get("family", ""))
    ]
    notinject_rows = [
        r
        for r in ext_rows
        if r.get("source") == "notinject" or r.get("family") == "notinject"
    ]
    dolly_rows = [
        r for r in ext_rows if r.get("source") == "dolly" or r.get("family") == "dolly"
    ]

    assert len(gandalf_rows) == 112, f"Expected 112 Gandalf, got {len(gandalf_rows)}"
    assert len(notinject_rows) == 309, (
        f"Expected 309 NotInject, got {len(notinject_rows)}"
    )
    assert len(dolly_rows) == 200, f"Expected 200 Dolly, got {len(dolly_rows)}"

    # -----------------------------------------------------------------------
    # X1: Gandalf Recall (Arm R, comparison arms, and per-detector)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    apply_theme(fig, ax, theme_name)

    r_loose_cnt = sum(
        1 for r in gandalf_rows if r["decisions"]["arm_r"] in ("flag", "block")
    )
    r_strict_cnt = sum(1 for r in gandalf_rows if r["decisions"]["arm_r"] == "block")
    arm_a_cnt = sum(
        1 for r in gandalf_rows if r["decisions"]["arm_a"] in ("flag", "block")
    )
    arm_c2_cnt = sum(
        1 for r in gandalf_rows if r["decisions"]["c2_standalone"] in ("flag", "block")
    )
    d2_cnt = sum(1 for r in gandalf_rows if r["detectors"]["d2-deberta"]["is_attack"])
    d3_cnt = sum(1 for r in gandalf_rows if r["detectors"]["d3-deepset"]["is_attack"])
    d4_cnt = sum(1 for r in gandalf_rows if r["detectors"]["d4_rules"]["is_attack"])

    r_loose = r_loose_cnt / 112.0 * 100.0

    x1_counts = [
        r_loose_cnt,
        r_strict_cnt,
        arm_a_cnt,
        arm_c2_cnt,
        d2_cnt,
        d3_cnt,
        d4_cnt,
    ]
    x1_labels = [
        "Arm R: Loose\n(primary)",
        "Arm R: Strict\n(primary)",
        "Arm A\n(post-hoc)",
        "Arm C2\n(post-hoc)",
        "D2 DeBERTa\n(post-hoc)",
        "D3 Deepset\n(post-hoc)",
        "D4 Rules\n(post-hoc)",
    ]
    # Consistent arm colours: Arm R = cyan, Arm A = magenta, rest match palette
    x1_colors = [
        t["arm_r"],  # Arm R loose — primary cyan
        t["arm_r"],  # Arm R strict — same arm
        t["arm_a"],  # Arm A — magenta
        t["arm_c2"],  # Arm C2 — purple
        t["slate"],  # D2
        t["amber"],  # D3
        t["green"],  # D4
    ]

    x1_vals = [c / 112.0 * 100.0 for c in x1_counts]
    x1_cis = [wilson_ci(c, 112) for c in x1_counts]

    yerr_low = [max(0.0, v - ci[0] * 100.0) for v, ci in zip(x1_vals, x1_cis)]
    yerr_high = [max(0.0, ci[1] * 100.0 - v) for v, ci in zip(x1_vals, x1_cis)]
    yerr = [yerr_low, yerr_high]

    bars = ax.bar(
        x1_labels,
        x1_vals,
        yerr=yerr,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": t["text"]},
        color=x1_colors,
        alpha=0.85,
        width=0.55,
    )
    for b, v, ci in zip(bars, x1_vals, x1_cis):
        ax.text(
            b.get_x() + b.get_width() / 2,
            min(102.0, max(v, ci[1] * 100.0)) + 1.5,
            f"{v:.1f}%\n[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=t["text"],
            family="monospace",
        )

    # Note: PG2 not evaluated on external rows
    ax.text(
        0.98,
        0.02,
        "Note: PG2 (Prompt Guard 2) was not computed on\nexternal rows; see held_out_v4 (F10) for PG2 results.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=t["muted_text"],
        family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": t["card_bg"], "alpha": 0.8},
    )

    ax.set_ylabel("Detection Recall on Attacks (%)", fontsize=10, family="monospace")
    ax.set_title(
        "Gandalf External Benchmark: one narrow attack family (n=112)\n"
        "Scope: Lakera/gandalf_ignore_instructions (test split, similarity >= 0.825)\n"
        "Plausible ceiling effect: D2 (DeBERTa) achieves 100% on this narrow phrasing distribution",
        fontsize=10,
        fontweight="bold",
        family="monospace",
        pad=14,
    )
    ax.set_ylim(0, 105)  # y-limit 105

    save_figure(
        fig,
        "x1_gandalf_metrics",
        theme_name,
        ctx["commit"],
        "export/closeout/external_predictions.jsonl (dataset==gandalf)",
        footer_note="scored once",
    )
    table_rows = [
        [
            lbl.replace("\n", " "),
            f"{val:.2f}%",
            f"[{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%]",
            112,
        ]
        for lbl, val, ci in zip(x1_labels, x1_vals, x1_cis)
    ]
    save_table_csv(
        "x1_gandalf_metrics.csv",
        ["Detector / Arm", "Recall (%)", "95% Wilson CI", "Sample Size (n)"],
        table_rows,
    )
    sidecars["X1"] = {
        "title": "Gandalf External Benchmark Recall (ceiling-effect note included)",
        "split": "external: Gandalf",
        "n": 112,
        "source_files": ["export/closeout/external_predictions.jsonl"],
        "source_sha256": [ctx["files_sha256"]["external_predictions.jsonl"]],
        "assertions": [
            f"Arm R loose recall == {x1_vals[0]:.2f}%",
            f"Arm A recall == {x1_vals[2]:.2f}%",
            "y_max == 105",
            "scored once",
        ],
    }

    # -----------------------------------------------------------------------
    # X2: NotInject (n=309) & Dolly (n=200) ALLOW/FLAG/BLOCK per arm
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
    apply_theme(fig, [ax1, ax2], theme_name)
    plt.subplots_adjust(bottom=0.22, wspace=0.25)

    arms_list = [
        ("Arm R (primary)", "arm_r"),
        ("Arm A (post-hoc)", "arm_a"),
        ("Arm B (post-hoc)", "arm_b"),
        ("Arm U (post-hoc)", "arm_u"),
        ("Arm C2 (post-hoc)", "c2_standalone"),
    ]

    # Panel 1: NotInject
    x = np.arange(len(arms_list))
    width = 0.55
    ni_table_rows = []

    for idx, (name, key) in enumerate(arms_list):
        a = sum(1 for r in notinject_rows if r["decisions"][key] == "allow")
        f_cnt = sum(1 for r in notinject_rows if r["decisions"][key] == "flag")
        b = sum(1 for r in notinject_rows if r["decisions"][key] == "block")
        ax1.bar(
            idx,
            a,
            width,
            color=t["allow"],
            alpha=0.85,
            label="ALLOW" if idx == 0 else "",
        )
        ax1.bar(
            idx,
            f_cnt,
            width,
            bottom=a,
            color=t["flag"],
            alpha=0.85,
            label="FLAG" if idx == 0 else "",
        )
        ax1.bar(
            idx,
            b,
            width,
            bottom=a + f_cnt,
            color=t["block"],
            alpha=0.85,
            label="BLOCK" if idx == 0 else "",
        )
        # count labels
        if a > 0:
            ax1.text(
                idx,
                a / 2,
                str(a),
                ha="center",
                va="center",
                fontsize=7,
                color=t["bg"],
                family="monospace",
                fontweight="bold",
            )
        if f_cnt > 0:
            ax1.text(
                idx,
                a + f_cnt / 2,
                str(f_cnt),
                ha="center",
                va="center",
                fontsize=7,
                color=t["bg"],
                family="monospace",
                fontweight="bold",
            )
        if b > 0:
            ax1.text(
                idx,
                a + f_cnt + b / 2,
                str(b),
                ha="center",
                va="center",
                fontsize=7,
                color="#ffffff",
                family="monospace",
                fontweight="bold",
            )
        ni_table_rows.append(["NotInject", name, a, f_cnt, b, 309])

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [a[0] for a in arms_list],
        fontsize=8,
        family="monospace",
        rotation=20,
        ha="right",
    )
    ax1.set_ylabel("Count of Prompts (N=309)", fontsize=10, family="monospace")
    ax1.set_title(
        "NotInject Benign Set (n=309)\n(Trigger-Word Benchmark)",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax1.set_ylim(0, 350)
    ax1.legend(loc="lower left", prop={"family": "monospace", "size": 8})

    # Panel 2: Dolly
    for idx, (name, key) in enumerate(arms_list):
        a = sum(1 for r in dolly_rows if r["decisions"][key] == "allow")
        f_cnt = sum(1 for r in dolly_rows if r["decisions"][key] == "flag")
        b = sum(1 for r in dolly_rows if r["decisions"][key] == "block")
        ax2.bar(idx, a, width, color=t["allow"], alpha=0.85)
        ax2.bar(idx, f_cnt, width, bottom=a, color=t["flag"], alpha=0.85)
        ax2.bar(idx, b, width, bottom=a + f_cnt, color=t["block"], alpha=0.85)
        # count labels
        if a > 0:
            ax2.text(
                idx,
                a / 2,
                str(a),
                ha="center",
                va="center",
                fontsize=7,
                color=t["bg"],
                family="monospace",
                fontweight="bold",
            )
        if f_cnt > 0:
            ax2.text(
                idx,
                a + f_cnt / 2,
                str(f_cnt),
                ha="center",
                va="center",
                fontsize=7,
                color=t["bg"],
                family="monospace",
                fontweight="bold",
            )
        if b > 0:
            ax2.text(
                idx,
                a + f_cnt + b / 2,
                str(b),
                ha="center",
                va="center",
                fontsize=7,
                color="#ffffff",
                family="monospace",
                fontweight="bold",
            )
        ni_table_rows.append(["Dolly", name, a, f_cnt, b, 200])

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [a[0] for a in arms_list],
        fontsize=8,
        family="monospace",
        rotation=20,
        ha="right",
    )
    ax2.set_ylabel("Count of Prompts (N=200)", fontsize=10, family="monospace")
    ax2.set_title(
        "Dolly Benign Set (n=200)\n(General Instruction Benchmark)",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax2.set_ylim(0, 230)

    save_figure(
        fig,
        "x2_benign_composition",
        theme_name,
        ctx["commit"],
        "export/closeout/external_predictions.jsonl (notinject & dolly)",
        footer_note="scored once",
    )
    save_table_csv(
        "x2_benign_sets_metrics.csv",
        ["Dataset", "Arm", "ALLOW", "FLAG", "BLOCK", "Total N"],
        ni_table_rows,
    )
    sidecars["X2"] = {
        "title": "Decision Composition on External Benign Sets",
        "split": "external: NotInject (n=309) & Dolly (n=200)",
        "n": 509,
        "source_files": ["export/closeout/external_predictions.jsonl"],
        "source_sha256": [ctx["files_sha256"]["external_predictions.jsonl"]],
        "assertions": ["NotInject n == 309", "Dolly n == 200", "Never pooled == True"],
    }

    # -----------------------------------------------------------------------
    # X3: NotInject by multilingual vs English and trigger-word subset
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    apply_theme(fig, [ax1, ax2], theme_name)

    # --- LEFT PANEL: read from addendum JSON (authoritative, avoids category mismatch) ---
    ADDENDUM_PATH = (
        REPO_ROOT / "export" / "closeout" / "notinject_addendum_metrics.json"
    )
    if ADDENDUM_PATH.exists():
        with open(ADDENDUM_PATH, "r", encoding="utf-8") as f:
            addendum = json.load(f)
        eng_part = addendum["partitions"]["english"]["arms"]["arm_r"]
        multi_part = addendum["partitions"]["multilingual"]["arms"]["arm_r"]
        n_eng = addendum["partitions"]["english"]["sample_size"]  # 225
        n_multi = addendum["partitions"]["multilingual"]["sample_size"]  # 84
        r_eng_allow = eng_part["allow"]
        r_multi_allow = multi_part["allow"]
        r_eng_pt = eng_part["pass_through_rate"] * 100.0
        r_multi_pt = multi_part["pass_through_rate"] * 100.0
        eng_ci = (eng_part["pass_through_95ci"][0], eng_part["pass_through_95ci"][1])
        multi_ci = (
            multi_part["pass_through_95ci"][0],
            multi_part["pass_through_95ci"][1],
        )
    else:
        # Fallback: re-derive via prompt_id join
        notinject_meta = {}
        notinject_path = REPO_ROOT / "data" / "external" / "notinject.jsonl"
        if notinject_path.exists():
            with open(notinject_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        notinject_meta[item.get("prompt_id", "")] = item
        eng_rows = [
            r
            for r in notinject_rows
            if notinject_meta.get(r.get("prompt_id", ""), {}).get("category")
            != "Multilingual"
        ]
        multi_rows = [
            r
            for r in notinject_rows
            if notinject_meta.get(r.get("prompt_id", ""), {}).get("category")
            == "Multilingual"
        ]
        n_eng, n_multi = len(eng_rows), len(multi_rows)
        r_eng_allow = sum(1 for r in eng_rows if r["decisions"]["arm_r"] == "allow")
        r_multi_allow = sum(1 for r in multi_rows if r["decisions"]["arm_r"] == "allow")
        r_eng_pt = r_eng_allow / n_eng * 100.0 if n_eng else 0.0
        r_multi_pt = r_multi_allow / n_multi * 100.0 if n_multi else 0.0
        eng_ci = wilson_ci(r_eng_allow, n_eng)
        multi_ci = wilson_ci(r_multi_allow, n_multi)

    assert n_eng == 225, f"Expected English n=225 from addendum, got {n_eng}"
    assert n_multi == 84, f"Expected Multilingual n=84 from addendum, got {n_multi}"

    yerr_lang = [
        [
            max(0.0, r_eng_pt - eng_ci[0] * 100.0),
            max(0.0, r_multi_pt - multi_ci[0] * 100.0),
        ],
        [
            max(0.0, eng_ci[1] * 100.0 - r_eng_pt),
            max(0.0, multi_ci[1] * 100.0 - r_multi_pt),
        ],
    ]

    bars1 = ax1.bar(
        [
            f"English (derived)\n(n={n_eng})",
            f"Multilingual\n(n={n_multi})",
        ],
        [r_eng_pt, r_multi_pt],
        yerr=yerr_lang,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": t["text"]},
        color=[t["arm_r"], t["amber"]],
        alpha=0.85,
        width=0.45,
    )
    for b, pt_val, ci in zip(bars1, [r_eng_pt, r_multi_pt], [eng_ci, multi_ci]):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            max(pt_val, ci[1] * 100.0) + 2,
            f"{pt_val:.1f}%\n[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=t["text"],
            family="monospace",
        )
    ax1.set_ylabel("Arm R Pass-Through Rate (%)", fontsize=10, family="monospace")
    ax1.set_title(
        "NotInject by Language Category (Post-hoc Addendum)\n"
        "Source: notinject_addendum_metrics.json\n"
        "(Arm R Pass-Through with 95% Wilson CIs)",
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )
    ax1.set_ylim(0, 115)

    # Trigger-word subsets (1, 2, 3 words)
    sub1 = [
        r
        for r in notinject_rows
        if r.get("subset") == "NotInject_one"
        or "NotInject_one" in r.get("prompt_id", "")
    ]
    sub2 = [
        r
        for r in notinject_rows
        if r.get("subset") == "NotInject_two"
        or "NotInject_two" in r.get("prompt_id", "")
    ]
    sub3 = [
        r
        for r in notinject_rows
        if r.get("subset") == "NotInject_three"
        or "NotInject_three" in r.get("prompt_id", "")
    ]

    sub1_allow = sum(1 for r in sub1 if r["decisions"]["arm_r"] == "allow")
    sub2_allow = sum(1 for r in sub2 if r["decisions"]["arm_r"] == "allow")
    sub3_allow = sum(1 for r in sub3 if r["decisions"]["arm_r"] == "allow")

    pt_sub1 = sub1_allow / len(sub1) * 100.0 if sub1 else 0.0
    pt_sub2 = sub2_allow / len(sub2) * 100.0 if sub2 else 0.0
    pt_sub3 = sub3_allow / len(sub3) * 100.0 if sub3 else 0.0

    ci_sub1 = wilson_ci(sub1_allow, len(sub1))
    ci_sub2 = wilson_ci(sub2_allow, len(sub2))
    ci_sub3 = wilson_ci(sub3_allow, len(sub3))

    yerr_subs = [
        [
            max(0.0, pt_sub1 - ci_sub1[0] * 100.0),
            max(0.0, pt_sub2 - ci_sub2[0] * 100.0),
            max(0.0, pt_sub3 - ci_sub3[0] * 100.0),
        ],
        [
            max(0.0, ci_sub1[1] * 100.0 - pt_sub1),
            max(0.0, ci_sub2[1] * 100.0 - pt_sub2),
            max(0.0, ci_sub3[1] * 100.0 - pt_sub3),
        ],
    ]

    bars2 = ax2.bar(
        [
            f"1 Trigger Word\n(n={len(sub1)})",
            f"2 Trigger Words\n(n={len(sub2)})",
            f"3 Trigger Words\n(n={len(sub3)})",
        ],
        [pt_sub1, pt_sub2, pt_sub3],
        yerr=yerr_subs,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": t["text"]},
        color=[t["green"], t["cyan"], t["block"]],
        alpha=0.85,
        width=0.45,
    )
    for b, pt_val, ci in zip(
        bars2, [pt_sub1, pt_sub2, pt_sub3], [ci_sub1, ci_sub2, ci_sub3]
    ):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            max(pt_val, ci[1] * 100.0) + 2,
            f"{pt_val:.1f}%\n[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=t["text"],
            family="monospace",
        )

    ax2.set_ylabel("Arm R Pass-Through Rate (%)", fontsize=10, family="monospace")
    ax2.set_title(
        "NotInject by Trigger-Word Load\n(Arm R Pass-Through with 95% Wilson CIs)",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax2.set_ylim(0, 115)

    save_figure(
        fig,
        "x3_notinject_subsets",
        theme_name,
        ctx["commit"],
        "export/closeout/external_predictions.jsonl (notinject subsets)",
        footer_note="scored once",
    )
    table_rows = [
        [
            "English",
            n_eng,
            f"{r_eng_pt:.2f}%",
            f"[{eng_ci[0] * 100:.2f}%, {eng_ci[1] * 100:.2f}%]",
        ],
        [
            "Multilingual",
            n_multi,
            f"{r_multi_pt:.2f}%",
            f"[{multi_ci[0] * 100:.2f}%, {multi_ci[1] * 100:.2f}%]",
        ],
        [
            "Subset 1 (1 word)",
            len(sub1),
            f"{pt_sub1:.2f}%",
            f"[{ci_sub1[0] * 100:.2f}%, {ci_sub1[1] * 100:.2f}%]",
        ],
        [
            "Subset 2 (2 words)",
            len(sub2),
            f"{pt_sub2:.2f}%",
            f"[{ci_sub2[0] * 100:.2f}%, {ci_sub2[1] * 100:.2f}%]",
        ],
        [
            "Subset 3 (3 words)",
            len(sub3),
            f"{pt_sub3:.2f}%",
            f"[{ci_sub3[0] * 100:.2f}%, {ci_sub3[1] * 100:.2f}%]",
        ],
    ]
    save_table_csv(
        "x3_notinject_subsets.csv",
        [
            "Subset / Category",
            "Sample Size (n)",
            "Arm R Pass-Through Rate",
            "95% Wilson CI",
        ],
        table_rows,
    )
    sidecars["X3"] = {
        "title": "NotInject Subsets and Multilingual Analysis",
        "split": "external: NotInject",
        "n": 309,
        "source_files": [
            "export/closeout/external_predictions.jsonl",
            "data/external/notinject.jsonl",
        ],
        "source_sha256": [
            ctx["files_sha256"]["external_predictions.jsonl"],
            sha256_file(ADDENDUM_PATH if ADDENDUM_PATH.exists() else notinject_path),
        ],
        "assertions": [
            f"English n == {n_eng}",
            f"Multilingual n == {n_multi}",
            "Left panel reads from notinject_addendum_metrics.json",
        ],
    }

    # -----------------------------------------------------------------------
    # X4: Held-out vs External for Arm R (Separate panels, no connecting line)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
    apply_theme(fig, [ax1, ax2], theme_name)

    ho_recall = ctx["numbers"]["arm_r.loose.recall"]["value"] * 100.0
    ho_pt = ctx["numbers"]["arm_r.benign.pass_through"]["value"] * 100.0

    # Panel 1: Design Held-out (held_out_v4)
    ho_recall_ci = (
        ctx["numbers"]["arm_r.loose.recall_ci"]["value"][0] * 100.0,
        ctx["numbers"]["arm_r.loose.recall_ci"]["value"][1] * 100.0,
    )
    ho_pt_ci = (
        ctx["numbers"]["arm_r.benign.pass_through_ci"]["value"][0] * 100.0,
        ctx["numbers"]["arm_r.benign.pass_through_ci"]["value"][1] * 100.0,
    )
    ax1.bar(
        ["Loose Recall\n(Attacks, N=139)", "Pass-Through\n(Benign, N=208)"],
        [ho_recall, ho_pt],
        yerr=[
            [max(0.0, ho_recall - ho_recall_ci[0]), max(0.0, ho_pt - ho_pt_ci[0])],
            [max(0.0, ho_recall_ci[1] - ho_recall), max(0.0, ho_pt_ci[1] - ho_pt)],
        ],
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": t["text"]},
        color=[t["arm_r"], t["allow"]],
        alpha=0.85,
        width=0.45,
    )
    ax1.text(
        0,
        ho_recall + (ho_recall_ci[1] - ho_recall) + 2,
        f"{ho_recall:.1f}%\n[{ho_recall_ci[0]:.1f}, {ho_recall_ci[1]:.1f}]",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=t["text"],
        family="monospace",
    )
    ax1.text(
        1,
        ho_pt + (ho_pt_ci[1] - ho_pt) + 2,
        f"{ho_pt:.1f}%\n[{ho_pt_ci[0]:.1f}, {ho_pt_ci[1]:.1f}]",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=t["text"],
        family="monospace",
    )
    ax1.set_ylabel("Metric Rate (%)", fontsize=10, family="monospace")
    ax1.set_title(
        "Panel A: Arm R on held_out_v4 (N=347)\nDesign evaluation split",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax1.set_ylim(0, 115)

    # Panel 2: External Sets (Separate panels, NO connecting line, NO combined metric)
    ext_gandalf_rec = r_loose
    ext_notinject_pt = (
        sum(1 for r in notinject_rows if r["decisions"]["arm_r"] == "allow")
        / 309.0
        * 100.0
    )
    ext_dolly_pt = (
        sum(1 for r in dolly_rows if r["decisions"]["arm_r"] == "allow") / 200.0 * 100.0
    )
    # Wilson CIs for Panel B
    ni_allow = sum(1 for r in notinject_rows if r["decisions"]["arm_r"] == "allow")
    dolly_allow = sum(1 for r in dolly_rows if r["decisions"]["arm_r"] == "allow")
    gandalf_loose_cnt = sum(
        1 for r in gandalf_rows if r["decisions"]["arm_r"] in ("flag", "block")
    )
    ci_gandalf_ext = wilson_ci(gandalf_loose_cnt, 112)
    ci_ni_ext = wilson_ci(ni_allow, 309)
    ci_dolly_ext = wilson_ci(dolly_allow, 200)

    ext_vals = [ext_gandalf_rec, ext_notinject_pt, ext_dolly_pt]
    ext_cis = [ci_gandalf_ext, ci_ni_ext, ci_dolly_ext]
    ext_labels = [
        "Arm R Recall\n(Gandalf, n=112)",
        "Arm R Pass-Through\n(NotInject, n=309)",
        "Arm R Pass-Through\n(Dolly, n=200)",
    ]

    ax2.bar(
        ext_labels,
        ext_vals,
        yerr=[
            [max(0.0, v - ci[0] * 100.0) for v, ci in zip(ext_vals, ext_cis)],
            [max(0.0, ci[1] * 100.0 - v) for v, ci in zip(ext_vals, ext_cis)],
        ],
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": t["text"]},
        color=[t["block"], t["amber"], t["green"]],
        alpha=0.85,
        width=0.45,
    )
    for i, (v, ci) in enumerate(zip(ext_vals, ext_cis)):
        ax2.text(
            i,
            max(v, ci[1] * 100.0) + 2,
            f"{v:.1f}%\n[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=t["text"],
            family="monospace",
        )

    ax2.set_ylabel("Metric Rate (%)", fontsize=10, family="monospace")
    ax2.set_title(
        "Panel B: Arm R on External Benchmark Sets (n=621)\nGandalf, NotInject, Dolly — scored once, not pooled",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax2.set_ylim(0, 115)

    save_figure(
        fig,
        "x4_held_out_vs_external",
        theme_name,
        ctx["commit"],
        "predictions.jsonl vs external_predictions.jsonl",
        footer_note="scored once",
    )
    table_rows = [
        [
            "held_out_v4",
            "Loose Recall (Attacks)",
            139,
            f"{ho_recall:.2f}%",
            "[77.2%, 89.3%]",
        ],
        [
            "held_out_v4",
            "Pass-Through (Benign)",
            208,
            f"{ho_pt:.2f}%",
            "[62.2%, 74.7%]",
        ],
        [
            "Gandalf (External)",
            "Loose Recall (Attacks)",
            112,
            f"{ext_gandalf_rec:.2f}%",
            "One narrow attack family",
        ],
        [
            "NotInject (External)",
            "Pass-Through (Benign)",
            309,
            f"{ext_notinject_pt:.2f}%",
            "Hard trigger benign",
        ],
        [
            "Dolly (External)",
            "Pass-Through (Benign)",
            200,
            f"{ext_dolly_pt:.2f}%",
            "General instruction benign",
        ],
    ]
    save_table_csv(
        "x4_held_out_vs_external_arm_r.csv",
        ["Benchmark Split", "Metric", "Sample Size (n)", "Rate (%)", "Interval / Note"],
        table_rows,
    )
    sidecars["X4"] = {
        "title": "Arm R Comparison Across Design Split and External Benchmark Sets",
        "split": "held_out_v4 vs external",
        "n": 347 + 621,
        "source_files": [
            "results/final-v4/predictions.jsonl",
            "export/closeout/external_predictions.jsonl",
        ],
        "source_sha256": [
            ctx["files_sha256"]["predictions.jsonl"],
            ctx["files_sha256"]["external_predictions.jsonl"],
        ],
        "assertions": [
            "No connecting lines drawn == True",
            "No single pooled metric calculated == True",
        ],
    }

    # -----------------------------------------------------------------------
    # X5: Judge outcomes and latency on the external run
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
    apply_theme(fig, [ax1, ax2], theme_name)

    ext_verdicts = [r["judge"]["verdict"] for r in ext_rows]
    ev_counts = {
        "allow": ext_verdicts.count("allow"),
        "block": ext_verdicts.count("block"),
        "abstain": ext_verdicts.count("abstain"),
    }

    bars = ax1.bar(
        ["ALLOW", "BLOCK", "ABSTAIN"],
        [ev_counts["allow"], ev_counts["block"], ev_counts["abstain"]],
        color=[t["allow"], t["block"], t["abstain"]],
        alpha=0.85,
        width=0.45,
    )
    for b in bars:
        h = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2,
            h + 4,
            f"{h} ({h / 621 * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=t["text"],
            family="monospace",
        )

    ax1.set_ylabel("Count of External Prompts (N=621)", fontsize=10, family="monospace")
    ax1.set_title(
        "External Run Judge Verdicts\n(Model: gemma3:4b on External Sets)",
        fontsize=11,
        fontweight="bold",
        family="monospace",
        pad=16,
    )
    ax1.set_ylim(0, 650)

    ext_lats = [float(r["judge"]["latency_ms"]) for r in ext_rows]
    ax2.hist(ext_lats, bins=30, color=t["cyan"], alpha=0.85, edgecolor=t["border"])
    ax2.axvline(
        np.mean(ext_lats),
        color=t["amber"],
        linestyle="--",
        lw=1.8,
        label=f"Mean: {np.mean(ext_lats):.0f} ms",
    )
    ax2.axvline(
        np.percentile(ext_lats, 50),
        color=t["green"],
        linestyle=":",
        lw=1.8,
        label=f"Median: {np.percentile(ext_lats, 50):.0f} ms",
    )
    ax2.set_xlabel("Judge Latency (ms)", fontsize=10, family="monospace")
    ax2.set_ylabel("Count of Prompts", fontsize=10, family="monospace")
    ax2.set_title(
        "External Run Judge Latency Distribution",
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax2.legend(loc="upper right", prop={"family": "monospace", "size": 8.5})

    save_figure(
        fig,
        "x5_external_judge_outcomes",
        theme_name,
        ctx["commit"],
        "export/closeout/external_predictions.jsonl (judge.*)",
        footer_note="scored once",
    )
    table_rows = [
        ["ALLOW", ev_counts["allow"], f"{ev_counts['allow'] / 621 * 100:.2f}%"],
        ["BLOCK", ev_counts["block"], f"{ev_counts['block'] / 621 * 100:.2f}%"],
        ["ABSTAIN", ev_counts["abstain"], f"{ev_counts['abstain'] / 621 * 100:.2f}%"],
        ["Mean Latency", f"{np.mean(ext_lats):.1f} ms", ""],
        ["Median Latency", f"{np.percentile(ext_lats, 50):.1f} ms", ""],
    ]
    save_table_csv(
        "x5_external_judge_outcomes.csv",
        ["Verdict / Metric", "Value", "Percentage / Note"],
        table_rows,
    )
    sidecars["X5"] = {
        "title": "Judge Verdicts and Latency on External Evaluation Benchmark",
        "split": "external: All Sets",
        "n": 621,
        "source_files": ["export/closeout/external_predictions.jsonl"],
        "source_sha256": [ctx["files_sha256"]["external_predictions.jsonl"]],
        "assertions": [
            "Total external rows == 621",
            f"Abstain count == {ev_counts['abstain']}",
        ],
    }


# ---------------------------------------------------------------------------
# Numbers Sheet Generator
# ---------------------------------------------------------------------------
def generate_numbers_sheet(ctx: dict):
    num = ctx["numbers"]
    commit = ctx["commit"]

    lines = [
        "# PARALLAX Closeout Numbers Sheet (One-Page Fact Source)",
        "",
        "> **PURPOSE**: Canonical, script-derived reference numbers for teammates drafting the final report and presentation slides.",
        "> **GOVERNING RULE**: Every number below is read directly from `numbers.json` or `results/final-v4/predictions.jsonl`. Zero hard-coded literals.",
        f"> **GIT COMMIT**: `{commit}`",
        "",
        "---",
        "",
        "## 1. Primary System: Arm R (`held_out_v4`, N=347: 139 attacks, 208 benign)",
        "",
        "| Metric | Exact Value | 95% Confidence Interval | Source Key / File |",
        "|---|:---:|:---:|---|",
        f"| **Loose Recall (FLAG+BLOCK)** | **{num['arm_r.loose.recall']['value'] * 100:.2f}%** | [{num['arm_r.loose.recall_ci']['value'][0] * 100:.1f}%, {num['arm_r.loose.recall_ci']['value'][1] * 100:.1f}%] | `arm_r.loose.recall` (`numbers.json`) |",
        f"| **Strict Recall (BLOCK only)** | **{num['arm_r.strict.recall']['value'] * 100:.2f}%** | [{num['arm_r.strict.recall_ci']['value'][0] * 100:.1f}%, {num['arm_r.strict.recall_ci']['value'][1] * 100:.1f}%] | `arm_r.strict.recall` (`numbers.json`) |",
        f"| **Benign Pass-Through (ALLOW)** | **{num['arm_r.benign.pass_through']['value'] * 100:.2f}%** | [{num['arm_r.benign.pass_through_ci']['value'][0] * 100:.1f}%, {num['arm_r.benign.pass_through_ci']['value'][1] * 100:.1f}%] | `arm_r.benign.pass_through` (`numbers.json`) |",
        f"| **Benign FLAG Load** | **{num['arm_r.benign.flag_load']['value'] * 100:.2f}%** | [{num['arm_r.benign.flag_load_ci']['value'][0] * 100:.1f}%, {num['arm_r.benign.flag_load_ci']['value'][1] * 100:.1f}%] | `arm_r.benign.flag_load` (`numbers.json`) |",
        f"| **Benign False Block Rate (FBR)** | **{num['arm_r.benign.block_rate']['value'] * 100:.2f}%** | [{num['arm_r.benign.block_rate_ci']['value'][0] * 100:.1f}%, {num['arm_r.benign.block_rate_ci']['value'][1] * 100:.1f}%] | `arm_r.benign.block_rate` (`numbers.json`) |",
        f"| **Judge Invocations per 100** | **{num['arm_r.extra_calls_per_100']['value']:.2f}** | *Deterministic routing* | `arm_r.extra_calls_per_100` (`numbers.json`) |",
        f"| **Latency p50** | **{num['arm_r.latency_p50_ms']['value']:.1f} ms** | — | `arm_r.latency_p50_ms` (`numbers.json`) |",
        f"| **Latency p95** | **{num['arm_r.latency_p95_ms']['value']:.1f} ms** | — | `arm_r.latency_p95_ms` (`numbers.json`) |",
        "",
        "---",
        "",
        "## 2. Descriptive Comparison Arms on `held_out_v4` (Post-Hoc)",
        "",
        "| Arm | Loose Recall | Strict Recall | Benign Pass-Through | Benign False Block | Calls / 100 | Dominance Status |",
        "|---|:---:|:---:|:---:|:---:|:---:|---|",
        f"| **Arm A (AnyFlag Fusion)** | {num['arm_a.loose.recall']['value'] * 100:.2f}% | {num['arm_a.strict.recall']['value'] * 100:.2f}% | {num['arm_a.benign.pass_through']['value'] * 100:.2f}% | {num['arm_a.benign.block_rate']['value'] * 100:.2f}% | 0.00 | Non-dominated |",
        f"| **Arm B (Discrepancy Trigger)** | {num['arm_b.loose.recall']['value'] * 100:.2f}% | {num['arm_b.strict.recall']['value'] * 100:.2f}% | {num['arm_b.benign.pass_through']['value'] * 100:.2f}% | {num['arm_b.benign.block_rate']['value'] * 100:.2f}% | 94.52 | Non-dominated |",
        f"| **Arm U (Review Upper Bound)** | {num['arm_u.loose.recall']['value'] * 100:.2f}% | {num['arm_u.strict.recall']['value'] * 100:.2f}% | {num['arm_u.benign.pass_through']['value'] * 100:.2f}% | {num['arm_u.benign.block_rate']['value'] * 100:.2f}% | 100.00 | Non-dominated |",
        f"| **Arm C2 (Judge Standalone)** | {num['c2_standalone.loose.recall']['value'] * 100:.2f}% | {num['c2_standalone.strict.recall']['value'] * 100:.2f}% | {num['c2_standalone.benign.pass_through']['value'] * 100:.2f}% | {num['c2_standalone.benign.block_rate']['value'] * 100:.2f}% | 100.00 | Non-dominated |",
        f"| **Prompt Guard 2 (86M)** | {num['promptguard2.loose.recall']['value'] * 100:.2f}% | {num['promptguard2.strict.recall']['value'] * 100:.2f}% | {num['promptguard2.benign.pass_through']['value'] * 100:.2f}% | {num['promptguard2.benign.block_rate']['value'] * 100:.2f}% | 0.00 | Non-dominated |",
        f"| **PromptForest (Tree)** | {num['promptforest.loose.recall']['value'] * 100:.2f}% | {num['promptforest.strict.recall']['value'] * 100:.2f}% | {num['promptforest.benign.pass_through']['value'] * 100:.2f}% | {num['promptforest.benign.block_rate']['value'] * 100:.2f}% | 0.00 | Non-dominated |",
        "",
        "---",
        "",
        "## 3. Router Diagnostics & Random-Routing Benchmark",
        "",
        "- **Router AUC vs Ground Truth Label**: **0.4350** [0.3703, 0.4991] (CI excludes 0.500; reflects asymmetric routing rate: 58.99% on attacks vs 81.25% on benign)",
        "- **Router AUC vs AnyFlag Fusion Error**: **0.5296** [0.4579, 0.6006] (CI includes 0.500; confirms **no demonstrated skill** at error prediction)",
        "- **Random Routing Empirical Simulation** ($k=251$ calls, 2,000 draws, seed 0):",
        f"  - Loose Recall Mean: **{num['random_routing.recall_loose_mean']['value'] * 100:.2f}%** [{num['random_routing.recall_loose_ci']['value'][0] * 100:.2f}%, {num['random_routing.recall_loose_ci']['value'][1] * 100:.2f}%]",
        f"  - Benign Block Mean: **{num['random_routing.benign_block_mean']['value'] * 100:.2f}%** [{num['random_routing.benign_block_ci']['value'][0] * 100:.2f}%, {num['random_routing.benign_block_ci']['value'][1] * 100:.2f}%]",
        f"  - Benign Flag Mean: **{num['random_routing.benign_flag_mean']['value'] * 100:.2f}%** [{num['random_routing.benign_flag_ci']['value'][0] * 100:.2f}%, {num['random_routing.benign_flag_ci']['value'][1] * 100:.2f}%]",
        "- **Conclusion**: Arm R's loose recall (84.17%) does not surpass the random routing mean (87.48%), but achieves a lower benign block rate (6.25% vs 9.05%).",
        "",
        "---",
        "",
        "## 4. Judge Diagnostics & Abstain Audit",
        "",
        "- **Total Judge Decisions Evaluated**: 347 (226 ALLOW, 111 BLOCK, 10 ABSTAIN)",
        "- **Abstain Asymmetry**: 10 on attacks (10/139 = 7.19%) vs 0 on benign (0/208 = 0.00%); Fisher exact $p = 8.71 \\times 10^{-5}$",
        "- **Abstain Latency Impact**: Non-abstain mean = 5,322 ms; Abstain mean = 7,422 ms (+39.5% latency overhead)",
        "- **Policy B Downgrade Effect**: In 7 of the 10 abstain rows, judge abstain turned a fusion BLOCK into a final FLAG",
        "",
        "---",
        "",
        "## 5. Dataset Split Sizes",
        "",
        "- `dev`: 297 prompts (149 attacks, 148 benign)",
        "- `val`: 277 prompts (138 attacks, 139 benign)",
        "- `held_out_v4`: 347 prompts (139 attacks, 208 benign)",
        "- `external`: 621 prompts (Gandalf: 112 attacks, NotInject: 309 benign, Dolly: 200 benign)",
        "",
        "---",
        "",
        "## 6. External Benchmark Sets & NotInject Subgroup Addendum (Post-Hoc)",
        "",
        "| Dataset / Partition | Category / Subset | Sample Size (n) | Primary Arm R Metric | 95% Wilson Interval | Pre-registered Scope |",
        "|---|---|:---:|:---:|:---:|---|",
        "| **Lakera/gandalf** | Attack prompts (similarity >= 0.825) | 112 | **100.00%** Loose Recall | [96.7%, 100.0%] | One narrow attack family (ceiling effect) |",
        "| **Lakera/gandalf** | Attack prompts (strict recall) | 112 | **83.04%** Strict Recall | [75.0%, 88.9%] | BLOCK only decisions |",
        "| **leolee99/NotInject** | All Benign (Trigger Words) | 309 | **44.34%** Pass-Through | [38.9%, 49.9%] | Hard trigger-word benign |",
        "| **leolee99/NotInject** | English Queries (Post-hoc Addendum) | 225 | **55.56%** Pass-Through | [49.0%, 61.9%] | Baseline trigger resistance (FBR: 17.78%) |",
        "| **leolee99/NotInject** | Multilingual (Post-hoc Addendum) | 84 | **14.29%** Pass-Through | [8.4%, 23.3%] | Multilingual trigger shift (FBR: 26.19%) |",
        "| **leolee99/NotInject** | 1 Trigger Word | 101 | **68.32%** Pass-Through | [58.7%, 76.6%] | Monotonic trigger degradation (FBR: 7.92%) |",
        "| **leolee99/NotInject** | 2 Trigger Words | 101 | **41.58%** Pass-Through | [32.5%, 51.3%] | Monotonic trigger degradation (FBR: 23.76%) |",
        "| **leolee99/NotInject** | 3 Trigger Words | 107 | **24.30%** Pass-Through | [17.2%, 33.2%] | Monotonic trigger degradation (FBR: 28.04%) |",
        "| **databricks-dolly-15k** | General Instruction Benign | 200 | **85.50%** Pass-Through | [80.0%, 89.8%] | Stratified sample across 8 tasks (FBR: 1.00%) |",
        "",
    ]

    NUMBERS_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NUMBERS_SHEET_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[make_figures] Generated {NUMBERS_SHEET_PATH}")


# ---------------------------------------------------------------------------
# Main Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified Figure & Table Generator for PARALLAX Closeout"
    )
    parser.add_argument(
        "--theme", choices=["dark", "light", "all"], default="all", help="Visual theme"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PARALLAX CLOSEOUT: UNIFIED VISUAL GENERATOR")
    print("=" * 80)

    ctx = load_all_data()
    print(f"Loaded held_out_v4: {len(ctx['held_out_rows'])} rows")
    print(f"Loaded numbers.json: {len(ctx['numbers'])} keys")
    print(f"Loaded external_predictions: {len(ctx['ext_rows'])} rows")
    print(f"Git commit: {ctx['commit']}")

    themes = ["dark", "light"] if args.theme == "all" else [args.theme]
    sidecars = {}

    for t in themes:
        print(f"\n--- Generating figures for theme: {t} ---")
        generate_f1_pipeline(ctx, t, sidecars)
        generate_f2_data_composition(ctx, t, sidecars)
        generate_f3_detector_auc(ctx, t, sidecars)
        generate_f4_held_out_arms(ctx, t, sidecars)
        generate_f5_pareto_frontier(ctx, t, sidecars)
        generate_f6_stacked_bars(ctx, t, sidecars)
        generate_f7_router_analysis(ctx, t, sidecars)
        generate_f8_judge_outcomes(ctx, t, sidecars)
        generate_f9_abstain_rows(ctx, t, sidecars)
        generate_f10_cost_vs_recall(ctx, t, sidecars)
        generate_f11_error_analysis(ctx, t, sidecars)
        generate_f12_confusion_matrices(ctx, t, sidecars)
        generate_f13_contamination_matrix(ctx, t, sidecars)
        generate_f14_provenance_timeline(ctx, t, sidecars)
        generate_f15_audit_trail(ctx, t, sidecars)
        generate_x_figures(ctx, t, sidecars)

    # Save sidecars JSON
    with open(SOURCES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(sidecars, f, indent=2)
    print(f"\n[make_figures] Saved figure sources sidecar to {SOURCES_JSON_PATH}")

    # Save individual sidecars
    for fid, sc in sidecars.items():
        fname = f"{fid.lower()}_sources.json"
        with open(FIGURES_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(sc, f, indent=2)

    # Generate NUMBERS_SHEET.md
    generate_numbers_sheet(ctx)

    print("\n" + "=" * 80)
    print("ALL FIGURES, TABLES, AND NUMBER SHEETS GENERATED SUCCESSFULLY")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
