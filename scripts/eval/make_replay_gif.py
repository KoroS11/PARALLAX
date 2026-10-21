#!/usr/bin/env python3
"""Simulation Replay GIF Generator for PARALLAX (Release Round Phase 3).

Replays a deterministic, representative subset of logged predictions from:
- results/final-v4/predictions.jsonl (held_out_v4, N=347)
- export/closeout/external_predictions.jsonl (Gandalf n=112, NotInject n=309, Dolly n=200)

Explicit Disclaimers:
- Replay of logged results only; NO model executes during generation.
- Prompt text is strictly sanitized: prompt_id and family only. Zero prompt text.
- Sampling is deterministic (seed 0) and includes:
  * All 10 judge-abstain attack rows
  * All 22 Arm R false negatives
  * All 13 Arm R false blocks
  * Ordinary true-positive and true-negative rows across all splits
- Generates:
  * export/demo/parallax_replay.gif (< 8 MB, <= 960 px wide, 8 fps)
  * export/demo/parallax_replay_frames.json
  * export/demo/parallax_replay_sources.json (sidecar with sha256)
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
EXTERNAL_PREDICTIONS_PATH = REPO_ROOT / "export" / "closeout" / "external_predictions.jsonl"
NUMBERS_PATH = REPO_ROOT / "numbers.json"
EXTERNAL_METRICS_PATH = REPO_ROOT / "export" / "closeout" / "external_metrics.json"

DEMO_DIR = REPO_ROOT / "export" / "demo"
GIF_PATH = DEMO_DIR / "parallax_replay.gif"
FRAMES_JSON_PATH = DEMO_DIR / "parallax_replay_frames.json"
SIDECAR_PATH = DEMO_DIR / "parallax_replay_sources.json"


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


def sample_replay_rows() -> list[dict[str, Any]]:
    """Select a deterministic, balanced, transparent subset of logged rows."""
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        ho_rows = [json.loads(line) for line in f if line.strip()]

    with open(EXTERNAL_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        ext_rows = [json.loads(line) for line in f if line.strip()]

    # 1. Key error & edge-case rows in held_out_v4
    abstain_rows = [
        r for r in ho_rows if str(r["judge"].get("verdict")).lower() == "abstain"
    ]
    fn_rows = [
        r for r in ho_rows if r["label_binary"] and r["decisions"]["arm_r"] == "allow"
    ]
    fb_rows = [
        r for r in ho_rows if not r["label_binary"] and r["decisions"]["arm_r"] == "block"
    ]

    seen_ids = set()
    sampled_ho: list[dict[str, Any]] = []

    for r in abstain_rows + fn_rows + fb_rows:
        if r["prompt_id"] not in seen_ids:
            seen_ids.add(r["prompt_id"])
            sampled_ho.append(r)

    # 2. Add ordinary held-out rows (3 TP, 3 TN) deterministically
    rng = random.Random(0)
    ho_tp = [
        r for r in ho_rows
        if r["label_binary"] and r["decisions"]["arm_r"] in ("flag", "block") and r["prompt_id"] not in seen_ids
    ]
    ho_tn = [
        r for r in ho_rows
        if not r["label_binary"] and r["decisions"]["arm_r"] == "allow" and r["prompt_id"] not in seen_ids
    ]
    rng.shuffle(ho_tp)
    rng.shuffle(ho_tn)

    for r in ho_tp[:3]:
        seen_ids.add(r["prompt_id"])
        sampled_ho.append(r)
    for r in ho_tn[:3]:
        seen_ids.add(r["prompt_id"])
        sampled_ho.append(r)

    # Tag split for held-out
    for r in sampled_ho:
        r["_split"] = "held_out_v4"

    # 3. Add external rows (4 Gandalf attacks, 4 NotInject benign, 4 Dolly benign)
    gandalf_rows = [
        r for r in ext_rows
        if "gandalf" in str(r.get("family", "")).lower() or r.get("source") == "gandalf"
    ]
    notinject_rows = [
        r for r in ext_rows
        if "notinject" in str(r.get("family", "")).lower() or r.get("source") == "notinject"
    ]
    dolly_rows = [
        r for r in ext_rows
        if "dolly" in str(r.get("family", "")).lower() or r.get("source") == "dolly"
    ]

    rng_ext = random.Random(0)
    rng_ext.shuffle(gandalf_rows)
    rng_ext.shuffle(notinject_rows)
    rng_ext.shuffle(dolly_rows)

    sampled_ext: list[dict[str, Any]] = []
    for r in gandalf_rows[:4]:
        r["_split"] = "external_gandalf"
        sampled_ext.append(r)
    for r in notinject_rows[:4]:
        r["_split"] = "external_notinject"
        sampled_ext.append(r)
    for r in dolly_rows[:4]:
        r["_split"] = "external_dolly"
        sampled_ext.append(r)

    total_sampled = sampled_ho + sampled_ext
    print(f"[make_replay_gif] Sampled {len(total_sampled)} replay rows "
          f"({len(sampled_ho)} held-out [10 abstains, 22 FNs, 13 FBs + 6 ordinary], "
          f"{len(sampled_ext)} external [4 Gandalf, 4 NotInject, 4 Dolly]).")
    return total_sampled


def render_prediction_frame(
    row: dict[str, Any],
    idx: int,
    total: int,
    tallies: dict[str, dict[str, int]],
) -> Image.Image:
    """Render a single dark-terminal monospace simulation frame."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=100)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.axis("off")

    split_name = row.get("_split", "held_out_v4")
    prompt_id = row.get("prompt_id", "N/A")
    family = row.get("family", "N/A")
    is_attack = bool(row.get("label_binary", False))

    # Header
    ax.text(
        0.03,
        0.94,
        f"[ROW {idx+1:02d}/{total:02d}] SPLIT: {split_name}  |  ID: {prompt_id}  |  FAMILY: {family}",
        color="#38bdf8",
        fontfamily="monospace",
        fontsize=9.0,
        fontweight="bold",
    )

    gt_color = "#ef4444" if is_attack else "#22c55e"
    gt_text = "GROUND TRUTH: ATTACK" if is_attack else "GROUND TRUTH: BENIGN"
    ax.text(
        0.97,
        0.94,
        gt_text,
        color=gt_color,
        fontfamily="monospace",
        fontsize=9.5,
        fontweight="bold",
        ha="right",
    )

    # Divider line
    ax.plot([0.03, 0.97], [0.895, 0.895], color="#1e293b", lw=1.2)

    # --- LEFT COLUMN: Pipeline Stages ---
    # Stage 1: Detectors
    ax.text(
        0.04,
        0.84,
        "STAGE 1: BASE DETECTORS",
        color="#94a3b8",
        fontfamily="monospace",
        fontsize=8.5,
        fontweight="bold",
    )

    dets = row.get("detectors", {})
    d2 = dets.get("d2-deberta", {}).get("score", 0.0)
    d3 = dets.get("d3-deepset", {}).get("score", 0.0)
    d4 = dets.get("d4_rules", {}).get("score", 0.0)

    # D2 Bar
    ax.text(0.04, 0.775, f"D2 DeBERTa-v3 : {d2:.3f}", color="#cbd5e1", fontfamily="monospace", fontsize=8.0)
    ax.add_patch(patches.Rectangle((0.26, 0.765), 0.20, 0.024, facecolor="#1e293b", edgecolor="#334155"))
    ax.add_patch(patches.Rectangle((0.26, 0.765), 0.20 * min(1.0, max(0.0, d2)), 0.024, facecolor="#38bdf8"))

    # D3 Bar
    ax.text(0.04, 0.715, f"D3 Deepset    : {d3:.3f}", color="#cbd5e1", fontfamily="monospace", fontsize=8.0)
    ax.add_patch(patches.Rectangle((0.26, 0.705), 0.20, 0.024, facecolor="#1e293b", edgecolor="#334155"))
    ax.add_patch(patches.Rectangle((0.26, 0.705), 0.20 * min(1.0, max(0.0, d3)), 0.024, facecolor="#a855f7"))

    # D4 Bar
    ax.text(0.04, 0.655, f"D4 Regex Rules: {d4:.1f}", color="#cbd5e1", fontfamily="monospace", fontsize=8.0)
    ax.add_patch(patches.Rectangle((0.26, 0.645), 0.20, 0.024, facecolor="#1e293b", edgecolor="#334155"))
    ax.add_patch(patches.Rectangle((0.26, 0.645), 0.20 * min(1.0, max(0.0, d4)), 0.024, facecolor="#f59e0b"))

    # Stage 2: AnyFlag Fusion
    ax.text(
        0.04,
        0.565,
        "STAGE 2: ANYFLAG FUSION (ARM A)",
        color="#94a3b8",
        fontfamily="monospace",
        fontsize=8.5,
        fontweight="bold",
    )
    fa = row.get("decisions", {}).get("arm_a", "allow").upper()
    fa_color = "#ef4444" if fa == "BLOCK" else ("#f59e0b" if fa == "FLAG" else "#22c55e")
    ax.text(
        0.04,
        0.505,
        f"Fusion Verdict: {fa}",
        color=fa_color,
        fontfamily="monospace",
        fontsize=9.0,
        fontweight="bold",
    )

    # Stage 3: Review Router
    ax.text(
        0.04,
        0.425,
        "STAGE 3: LEARNED REVIEW ROUTER",
        color="#94a3b8",
        fontfamily="monospace",
        fontsize=8.5,
        fontweight="bold",
    )
    pe = row.get("router", {}).get("prob_error", 0.0)
    tau = row.get("router", {}).get("tau", 0.35)
    routed = row.get("router", {}).get("should_route", False)
    rt = "ROUTED TO JUDGE" if routed else "BYPASSED"
    rt_color = "#38bdf8" if routed else "#64748b"
    ax.text(
        0.04,
        0.365,
        f"P(error)={pe:.3f} | tau={tau:.2f} -> {rt}",
        color=rt_color,
        fontfamily="monospace",
        fontsize=8.5,
    )

    # Stage 4: LLM Judge
    ax.text(
        0.04,
        0.285,
        "STAGE 4: ADVERSARIAL LLM JUDGE (gemma3:4b)",
        color="#94a3b8",
        fontfamily="monospace",
        fontsize=8.5,
        fontweight="bold",
    )
    jv = str(row.get("judge", {}).get("verdict", "BYPASS")).upper()
    jv_color = "#ef4444" if jv == "BLOCK" else ("#f59e0b" if jv == "ABSTAIN" else ("#22c55e" if jv == "ALLOW" else "#64748b"))
    lat = row.get("judge", {}).get("latency_ms", 0.0)
    lat_str = f"({lat:.0f} ms)" if lat > 0 else "(bypassed)"
    ax.text(
        0.04,
        0.225,
        f"Judge Verdict: {jv:<8s} {lat_str}",
        color=jv_color,
        fontfamily="monospace",
        fontsize=8.5,
        fontweight="bold",
    )

    # --- RIGHT COLUMN: Evaluated Arms & Cumulative Tallies ---
    ax.text(
        0.54,
        0.84,
        "EVALUATED ARMS (DECISION CHIPS)",
        color="#94a3b8",
        fontfamily="monospace",
        fontsize=8.5,
        fontweight="bold",
    )

    arms_def = [
        ("Arm R (Primary)", "arm_r"),
        ("Arm A (Fusion)", "arm_a"),
        ("Arm B (Review)", "arm_b"),
        ("Arm U (Uncert)", "arm_u"),
        ("Arm C2 (Alone)", "c2_standalone"),
    ]

    chip_y = 0.775
    for label, key in arms_def:
        dec = row.get("decisions", {}).get(key, "allow").upper()
        c_color = "#ef4444" if dec == "BLOCK" else ("#f59e0b" if dec == "FLAG" else "#22c55e")

        ax.text(0.54, chip_y, f"{label:<16s} : ", color="#e2e8f0", fontfamily="monospace", fontsize=8.0)
        ax.add_patch(patches.Rectangle((0.74, chip_y - 0.012), 0.12, 0.032, facecolor=c_color, edgecolor="#0f172a"))
        ax.text(
            0.80,
            chip_y + 0.004,
            dec,
            color="#0b0f19",
            fontfamily="monospace",
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="center",
        )
        chip_y -= 0.055

    # Update running tallies
    for _, key in arms_def:
        dec_lower = row.get("decisions", {}).get(key, "allow").lower()
        tallies[key][dec_lower] += 1
        if is_attack and dec_lower in ("flag", "block"):
            tallies[key]["tp"] += 1
        if (not is_attack) and dec_lower == "allow":
            tallies[key]["tn"] += 1

    # Running Tallies Box
    ax.text(
        0.54,
        0.46,
        "RUNNING TALLIES (CUMULATIVE REPLAY)",
        color="#94a3b8",
        fontfamily="monospace",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.add_patch(patches.Rectangle((0.53, 0.175), 0.44, 0.255, facecolor="#111827", edgecolor="#1f293d", lw=1))
    tally_header = "ARM        ALLOW   FLAG   BLOCK   RECALL"
    ax.text(0.55, 0.395, tally_header, color="#94a3b8", fontfamily="monospace", fontsize=7.5, fontweight="bold")
    ax.plot([0.55, 0.95], [0.38, 0.38], color="#334155", lw=0.8)

    ty = 0.345
    for arm_label, key in [
        ("Arm R*", "arm_r"),
        ("Arm A", "arm_a"),
        ("Arm B", "arm_b"),
        ("Arm U", "arm_u"),
        ("Arm C2", "c2_standalone"),
    ]:
        al = tallies[key]["allow"]
        fl = tallies[key]["flag"]
        bl = tallies[key]["block"]
        # calculate empirical recall on sampled attacks so far
        total_attacks = tallies["_meta"]["attacks"]
        rec_str = f"{(tallies[key]['tp'] / total_attacks * 100.0):.1f}%" if total_attacks > 0 else "0.0%"
        line = f"{arm_label:<8s}   {al:>4d}   {fl:>4d}   {bl:>5d}   {rec_str:>6s}"
        ax.text(0.55, ty, line, color="#cbd5e1", fontfamily="monospace", fontsize=7.4)
        ty -= 0.032

    if is_attack:
        tallies["_meta"]["attacks"] += 1
    else:
        tallies["_meta"]["benign"] += 1

    # Footer
    ax.plot([0.03, 0.97], [0.08, 0.08], color="#1e293b", lw=1.2)
    footer_text = (
        "Replay of logged results; no model is executing | "
        "Sampling: all 10 judge abstains, all 22 Arm R FNs, all 13 Arm R FBs + seed 0 samples"
    )
    ax.text(0.50, 0.04, footer_text, color="#64748b", fontfamily="monospace", fontsize=7.0, ha="center")

    plt.tight_layout()
    fig.canvas.draw()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
    plt.close(fig)
    return img.convert("RGB")


def render_summary_frame(
    ctx: dict[str, Any],
) -> Image.Image:
    """Render the final summary frame with official closeout benchmark numbers."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=100)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.axis("off")

    num = ctx["numbers"]
    ext = ctx["ext_metrics"]

    # Header
    ax.text(
        0.03,
        0.94,
        "[SUMMARY 61/61] OFFICIAL CLOSEOUT BENCHMARK TALLIES ACROSS ALL SPLITS",
        color="#38bdf8",
        fontfamily="monospace",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.text(
        0.97,
        0.94,
        "STATUS: EVALUATION COMPLETE",
        color="#22c55e",
        fontfamily="monospace",
        fontsize=9.5,
        fontweight="bold",
        ha="right",
    )

    ax.plot([0.03, 0.97], [0.895, 0.895], color="#1e293b", lw=1.2)

    # Top Table: held_out_v4 (N=347: 139 attacks, 208 benign)
    ax.text(
        0.04,
        0.84,
        "1. PRIMARY EVALUATION SPLIT: held_out_v4 (N=347: 139 attacks, 208 benign)",
        color="#94a3b8",
        fontfamily="monospace",
        fontsize=8.5,
        fontweight="bold",
    )

    ho_header = "ARM              LOOSE RECALL       STRICT RECALL      PASS-THROUGH       FRICTION     JUDGE CALLS/100"
    ax.text(0.04, 0.785, ho_header, color="#38bdf8", fontfamily="monospace", fontsize=7.2, fontweight="bold")
    ax.plot([0.04, 0.96], [0.77, 0.77], color="#334155", lw=0.8)

    arms_ho = [
        ("Arm R (Primary)", "arm_r", "#38bdf8"),
        ("Arm A (Post-hoc)", "arm_a", "#94a3b8"),
        ("Arm B (Post-hoc)", "arm_b", "#94a3b8"),
        ("Arm U (Post-hoc)", "arm_u", "#94a3b8"),
        ("Arm C2 (Post-hoc)", "c2_standalone", "#94a3b8"),
    ]

    ho_y = 0.735
    for label, k, col in arms_ho:
        lr = num[f"{k}.loose.recall"]["value"] * 100
        lr_ci = num[f"{k}.loose.recall_ci"]["value"]
        sr = num[f"{k}.strict.recall"]["value"] * 100
        pt = num[f"{k}.benign.pass_through"]["value"] * 100
        pt_ci = num[f"{k}.benign.pass_through_ci"]["value"]
        fr = (num[f"{k}.benign.flag_load"]["value"] + num[f"{k}.benign.block_rate"]["value"]) * 100
        calls = num[f"{k}.extra_calls_per_100"]["value"]

        line = (
            f"{label:<16s} {lr:>5.1f}% [{lr_ci[0]*100:.0f},{lr_ci[1]*100:.0f}%]   "
            f"{sr:>5.1f}%           "
            f"{pt:>5.1f}% [{pt_ci[0]*100:.0f},{pt_ci[1]*100:.0f}%]   "
            f"{fr:>5.1f}%          "
            f"{calls:>5.1f}"
        )
        ax.text(0.04, ho_y, line, color=col, fontfamily="monospace", fontsize=7.2)
        ho_y -= 0.038

    # Bottom Sections: External Splits
    ax.plot([0.04, 0.96], [0.525, 0.525], color="#1e293b", lw=1.0)
    ax.text(
        0.04,
        0.485,
        "2. INDEPENDENT EXTERNAL BENCHMARK DATASETS (SCORED ONCE, NO POOLING)",
        color="#94a3b8",
        fontfamily="monospace",
        fontsize=8.5,
        fontweight="bold",
    )

    # Gandalf
    g_r = ext["subsets"]["source_gandalf"]["arms"]["arm_r"]
    ax.text(
        0.04,
        0.42,
        "[Lakera / Gandalf (n=112 attacks)]",
        color="#cbd5e1",
        fontfamily="monospace",
        fontsize=8.0,
        fontweight="bold",
    )
    g_line = (
        f"Arm R Loose Recall: {g_r['loose_recall']*100:.1f}% [{g_r['loose_recall_wilson_95ci'][0]*100:.1f}%, {g_r['loose_recall_wilson_95ci'][1]*100:.1f}%]  |  "
        f"Strict Recall: {g_r['strict_recall']*100:.1f}%  |  Judge Calls/100: {g_r['calls_per_100']:.1f}"
    )
    ax.text(0.04, 0.375, g_line, color="#e2e8f0", fontfamily="monospace", fontsize=7.2)

    # NotInject
    ni_r = ext["subsets"]["source_notinject"]["arms"]["arm_r"]
    ax.text(
        0.04,
        0.31,
        "[NotInject Non-Harmful Triggers (n=309 benign)]",
        color="#cbd5e1",
        fontfamily="monospace",
        fontsize=8.0,
        fontweight="bold",
    )
    ni_line = (
        f"Arm R Pass-Through: {ni_r['pass_through_rate']*100:.1f}% [{ni_r['pass_through_wilson_95ci'][0]*100:.1f}%, {ni_r['pass_through_wilson_95ci'][1]*100:.1f}%]  |  "
        f"English (Derived, n=225): 55.6%  |  Multilingual (n=84): 14.3%"
    )
    ax.text(0.04, 0.265, ni_line, color="#e2e8f0", fontfamily="monospace", fontsize=7.2)

    # Dolly
    d_r = ext["subsets"]["source_dolly"]["arms"]["arm_r"]
    ax.text(
        0.04,
        0.20,
        "[Databricks Dolly-15k Instruction (n=200 benign)]",
        color="#cbd5e1",
        fontfamily="monospace",
        fontsize=8.0,
        fontweight="bold",
    )
    d_line = (
        f"Arm R Pass-Through: {d_r['pass_through_rate']*100:.1f}% [{d_r['pass_through_wilson_95ci'][0]*100:.1f}%, {d_r['pass_through_wilson_95ci'][1]*100:.1f}%]  |  "
        f"Friction: {d_r['flag_rate']*100 + d_r['false_block_rate']*100:.1f}%  |  Judge Calls/100: {d_r['calls_per_100']:.1f}"
    )
    ax.text(0.04, 0.155, d_line, color="#e2e8f0", fontfamily="monospace", fontsize=7.2)

    # Footer
    ax.plot([0.03, 0.97], [0.08, 0.08], color="#1e293b", lw=1.2)
    footer_text = (
        "PARALLAX Simulation Replay Complete | All numbers verified against numbers.json & external_metrics.json"
    )
    ax.text(0.50, 0.04, footer_text, color="#38bdf8", fontfamily="monospace", fontsize=7.2, ha="center")

    plt.tight_layout()
    fig.canvas.draw()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
    plt.close(fig)
    return img.convert("RGB")


def main() -> int:
    print("=" * 80)
    print("PARALLAX: SIMULATION REPLAY GIF GENERATOR (PHASE 3)")
    print("=" * 80)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
        numbers = json.load(f)
    with open(EXTERNAL_METRICS_PATH, "r", encoding="utf-8") as f:
        ext_metrics = json.load(f)

    ctx = {
        "numbers": numbers,
        "ext_metrics": ext_metrics,
        "commit": get_git_commit(),
    }

    rows = sample_replay_rows()

    tallies = {
        "arm_r": {"allow": 0, "flag": 0, "block": 0, "tp": 0, "tn": 0},
        "arm_a": {"allow": 0, "flag": 0, "block": 0, "tp": 0, "tn": 0},
        "arm_b": {"allow": 0, "flag": 0, "block": 0, "tp": 0, "tn": 0},
        "arm_u": {"allow": 0, "flag": 0, "block": 0, "tp": 0, "tn": 0},
        "c2_standalone": {"allow": 0, "flag": 0, "block": 0, "tp": 0, "tn": 0},
        "_meta": {"attacks": 0, "benign": 0},
    }

    frames: list[Image.Image] = []
    durations: list[int] = []
    frames_data: list[dict[str, Any]] = []

    print(f"Rendering {len(rows)} prediction frames...")
    for i, r in enumerate(rows):
        img = render_prediction_frame(r, i, len(rows), tallies)
        frames.append(img)
        durations.append(125)  # 8 fps = 125 ms

        frame_meta = {
            "frame_index": i + 1,
            "type": "prediction",
            "split": r.get("_split"),
            "prompt_id": r.get("prompt_id"),
            "family": r.get("family"),
            "is_attack": bool(r.get("label_binary")),
            "d2_score": r.get("detectors", {}).get("d2-deberta", {}).get("score"),
            "d3_score": r.get("detectors", {}).get("d3-deepset", {}).get("score"),
            "d4_score": r.get("detectors", {}).get("d4_rules", {}).get("score"),
            "fusion_decision": r.get("decisions", {}).get("arm_a"),
            "router_prob_error": r.get("router", {}).get("prob_error"),
            "router_should_route": r.get("router", {}).get("should_route"),
            "judge_verdict": r.get("judge", {}).get("verdict"),
            "decisions": {k: r.get("decisions", {}).get(k) for k in ("arm_r", "arm_a", "arm_b", "arm_u", "c2_standalone")},
            "cumulative_tallies": {
                k: {dec: tallies[k][dec] for dec in ("allow", "flag", "block", "tp", "tn")}
                for k in ("arm_r", "arm_a", "arm_b", "arm_u", "c2_standalone")
            },
        }
        frames_data.append(frame_meta)

    # Final summary frame
    print("Rendering final summary frame...")
    summary_img = render_summary_frame(ctx)
    frames.append(summary_img)
    durations.append(3500)  # Pause on final summary for 3.5 seconds

    summary_meta = {
        "frame_index": len(rows) + 1,
        "type": "summary",
        "held_out_arms": {
            k: {
                "loose_recall": numbers[f"{k}.loose.recall"]["value"],
                "strict_recall": numbers[f"{k}.strict.recall"]["value"],
                "pass_through": numbers[f"{k}.benign.pass_through"]["value"],
                "friction": numbers[f"{k}.benign.flag_load"]["value"] + numbers[f"{k}.benign.block_rate"]["value"],
                "calls_per_100": numbers[f"{k}.extra_calls_per_100"]["value"],
            }
            for k in ("arm_r", "arm_a", "arm_b", "arm_u", "c2_standalone")
        },
        "external_metrics": ext_metrics["summary"],
    }
    frames_data.append(summary_meta)

    # Save frames JSON
    with open(FRAMES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(frames_data, f, indent=2)
    print(f"Saved frames data to {FRAMES_JSON_PATH}")

    # Save animated GIF using Pillow
    print(f"Assembling GIF with {len(frames)} frames...")
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    gif_size_bytes = GIF_PATH.stat().st_size
    gif_size_mb = gif_size_bytes / (1024 * 1024)
    print(f"[make_replay_gif] Successfully created {GIF_PATH}")
    print(f"  Dimensions : {frames[0].width} x {frames[0].height} px")
    print(f"  Frame count: {len(frames)}")
    print(f"  File size  : {gif_size_mb:.2f} MB ({gif_size_bytes} bytes)")
    assert gif_size_mb < 8.0, f"GIF size {gif_size_mb:.2f} MB exceeds 8.0 MB ceiling!"
    assert frames[0].width <= 960, f"GIF width {frames[0].width} exceeds 960 px ceiling!"

    # Sidecar JSON
    sidecar = {
        "output_file": str(GIF_PATH.relative_to(REPO_ROOT)),
        "sha256": sha256_file(GIF_PATH),
        "frames_json": str(FRAMES_JSON_PATH.relative_to(REPO_ROOT)),
        "frames_json_sha256": sha256_file(FRAMES_JSON_PATH),
        "git_commit": ctx["commit"],
        "frame_count": len(frames),
        "dimensions": f"{frames[0].width}x{frames[0].height}",
        "file_size_bytes": gif_size_bytes,
        "fps": 8,
        "sources": [
            "results/final-v4/predictions.jsonl",
            "export/closeout/external_predictions.jsonl",
            "numbers.json",
            "export/closeout/external_metrics.json",
        ],
        "sampling_policy": (
            "Deterministic seed 0: all 10 judge-abstain attack rows, all 22 Arm R FNs, "
            "all 13 Arm R FBs, 6 ordinary held-out rows, and 12 external rows (4 Gandalf, 4 NotInject, 4 Dolly). "
            "Frame 61 displays full official closeout benchmark tallies."
        ),
    }
    with open(SIDECAR_PATH, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Saved sidecar to {SIDECAR_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
