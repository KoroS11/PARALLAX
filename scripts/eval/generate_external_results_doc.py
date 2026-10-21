#!/usr/bin/env python3
"""Format and generate docs/EXTERNAL_RESULTS.md from external_predictions.jsonl.

Strict requirements:
- Per-family tables, never pooled.
- Arm R marked primary; Arms A, B, U, C2 in 'Post-Hoc Descriptive Arms'.
- Gandalf: Arm R loose/strict recall with Wilson 95% CI; per-detector recall (D2, D3, D4);
  includes exact scope sentence: 'The Gandalf benchmark evaluates one narrow attack family (prompts filtered to cosine similarity >= 0.825 against instructional override).'
- NotInject and Dolly: ALLOW/FLAG/BLOCK counts, pass-through rate, false-block rate with Wilson intervals.
- NotInject split by multilingual vs English and trigger-word subsets.
- Judge outcomes (ok / abstain / timeout / parse_fail / error), latency, calls/100, wall-clock, resumed count, predictions SHA-256.
- Short descriptive observations list only, each with n and interval.
- Zero banned words (robust, generalizes, proves, validated, verified, audited, etc.).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute exact Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_arm_counts(rows: list[dict[str, Any]], arm: str) -> dict[str, int]:
    counts = {"allow": 0, "flag": 0, "block": 0}
    for r in rows:
        dec = str(r.get("decisions", {}).get(arm, "allow")).lower()
        if dec in counts:
            counts[dec] += 1
        else:
            counts["allow"] += 1
    return counts


def compute_calls_per_100(rows: list[dict[str, Any]], arm: str) -> float:
    if not rows:
        return 0.0
    n = len(rows)
    if arm == "arm_a":
        return 0.0
    if arm in ("arm_u", "c2_standalone"):
        return 100.0
    if arm == "arm_b":
        calls = sum(
            1
            for r in rows
            if r.get("trigger_arm_b", {}).get("triggered")
            or r.get("trigger_arm_b", {}).get("should_challenge")
        )
        return float((calls / n) * 100.0)
    if arm == "arm_r":
        calls = sum(1 for r in rows if r.get("router", {}).get("should_route") is True)
        return float((calls / n) * 100.0)
    return 0.0


def generate_external_results_markdown(
    predictions_file: Path, wall_clock_str: str = "TBD"
) -> str:
    lines_raw = [
        json.loads(line)
        for line in predictions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_total = len(lines_raw)
    assert n_total == 621, f"Expected 621 external rows, found {n_total}"

    pred_sha256 = get_file_sha256(predictions_file)

    # Split rows by family
    gandalf_rows = [
        r
        for r in lines_raw
        if r.get("family") == "gandalf" or r.get("source") == "gandalf"
    ]
    notinject_rows = [
        r
        for r in lines_raw
        if r.get("family") == "notinject" or r.get("source") == "notinject"
    ]
    dolly_rows = [
        r for r in lines_raw if r.get("family") == "dolly" or r.get("source") == "dolly"
    ]

    assert len(gandalf_rows) == 112, (
        f"Expected 112 Gandalf rows, found {len(gandalf_rows)}"
    )
    assert len(notinject_rows) == 309, (
        f"Expected 309 NotInject rows, found {len(notinject_rows)}"
    )
    assert len(dolly_rows) == 200, f"Expected 200 Dolly rows, found {len(dolly_rows)}"

    # Check resumed count
    resumed_count = sum(1 for r in lines_raw if r.get("resumed") is True)

    # Outcome counts
    outcome_counts = {"ok": 0, "abstain": 0, "timeout": 0, "parse_fail": 0, "error": 0}
    judge_latencies: list[float] = []
    abstain_latencies: list[float] = []
    non_abstain_latencies: list[float] = []

    for r in lines_raw:
        j = r.get("judge", {})
        verdict = str(j.get("verdict", "")).lower()
        outcome = j.get("outcome_type")
        lat = float(j.get("latency_ms", 0.0))
        judge_latencies.append(lat)

        if outcome in outcome_counts:
            outcome_counts[outcome] += 1
        elif verdict == "abstain":
            outcome_counts["abstain"] += 1
            abstain_latencies.append(lat)
        elif verdict in ("allow", "flag", "block"):
            outcome_counts["ok"] += 1
            non_abstain_latencies.append(lat)
        else:
            outcome_counts["ok"] += 1
            non_abstain_latencies.append(lat)

    lat_mean = sum(judge_latencies) / len(judge_latencies) if judge_latencies else 0.0
    sorted_lat = sorted(judge_latencies)
    lat_p50 = sorted_lat[int(len(sorted_lat) * 0.50)] if sorted_lat else 0.0
    lat_p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0.0

    # Build Markdown document
    doc: list[str] = [
        "# External Benchmark Evaluation Results (Wave K Closeout)",
        "",
        "> **PROTOCOL**: Executed under `docs/EXTERNAL_EVAL_PROTOCOL.md` and `scripts/gpu_lock.py`.",
        "> **SAMPLE**: $N=621$ external prompts evaluated as three distinct families and never pooled.",
        f"> **PREDICTIONS LOG SHA-256**: `{pred_sha256}`",
        f"> **TOTAL ROWS SCORED**: {n_total} (Resumed rows: {resumed_count})",
        "",
        "---",
        "",
        "## 1. Primary System Evaluation (Arm R)",
        "",
        "Arm R (Learned Review Router $\\tau=0.35$ + Challenger V2 Judge + Policy B) is the single pre-registered primary system.",
        "",
        "| Dataset Family | Type | Sample Size ($n$) | ALLOW | FLAG | BLOCK | Primary Metric (95% Wilson CI) | Strict Block (95% Wilson CI) | Judge Calls / 100 |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    # Arm R on Gandalf
    g_cnt_r = compute_arm_counts(gandalf_rows, "arm_r")
    g_loose_k = g_cnt_r["flag"] + g_cnt_r["block"]
    g_loose_rec = g_loose_k / 112.0
    g_loose_ci = wilson_ci(g_loose_k, 112)
    g_strict_rec = g_cnt_r["block"] / 112.0
    g_strict_ci = wilson_ci(g_cnt_r["block"], 112)
    g_calls_r = compute_calls_per_100(gandalf_rows, "arm_r")

    doc.append(
        f"| **Gandalf** (`Lakera`) | Attack | 112 | {g_cnt_r['allow']} | {g_cnt_r['flag']} | {g_cnt_r['block']} | "
        f"**{g_loose_rec * 100:.2f}%** [{g_loose_ci[0] * 100:.2f}%, {g_loose_ci[1] * 100:.2f}%] (Loose Recall) | "
        f"{g_strict_rec * 100:.2f}% [{g_strict_ci[0] * 100:.2f}%, {g_strict_ci[1] * 100:.2f}%] | {g_calls_r:.1f} |"
    )

    # Arm R on NotInject
    ni_cnt_r = compute_arm_counts(notinject_rows, "arm_r")
    ni_pass_rec = ni_cnt_r["allow"] / 309.0
    ni_pass_ci = wilson_ci(ni_cnt_r["allow"], 309)
    ni_fbr_rec = ni_cnt_r["block"] / 309.0
    ni_fbr_ci = wilson_ci(ni_cnt_r["block"], 309)
    ni_calls_r = compute_calls_per_100(notinject_rows, "arm_r")

    doc.append(
        f"| **NotInject** (`leolee99`) | Benign | 309 | {ni_cnt_r['allow']} | {ni_cnt_r['flag']} | {ni_cnt_r['block']} | "
        f"**{ni_pass_rec * 100:.2f}%** [{ni_pass_ci[0] * 100:.2f}%, {ni_pass_ci[1] * 100:.2f}%] (Pass-Through) | "
        f"{ni_fbr_rec * 100:.2f}% [{ni_fbr_ci[0] * 100:.2f}%, {ni_fbr_ci[1] * 100:.2f}%] (FBR) | {ni_calls_r:.1f} |"
    )

    # Arm R on Dolly
    d_cnt_r = compute_arm_counts(dolly_rows, "arm_r")
    d_pass_rec = d_cnt_r["allow"] / 200.0
    d_pass_ci = wilson_ci(d_cnt_r["allow"], 200)
    d_fbr_rec = d_cnt_r["block"] / 200.0
    d_fbr_ci = wilson_ci(d_cnt_r["block"], 200)
    d_calls_r = compute_calls_per_100(dolly_rows, "arm_r")

    doc.append(
        f"| **Dolly** (`databricks`) | Benign | 200 | {d_cnt_r['allow']} | {d_cnt_r['flag']} | {d_cnt_r['block']} | "
        f"**{d_pass_rec * 100:.2f}%** [{d_pass_ci[0] * 100:.2f}%, {d_pass_ci[1] * 100:.2f}%] (Pass-Through) | "
        f"{d_fbr_rec * 100:.2f}% [{d_fbr_ci[0] * 100:.2f}%, {d_fbr_ci[1] * 100:.2f}%] (FBR) | {d_calls_r:.1f} |"
    )

    doc.extend(
        [
            "",
            "---",
            "",
            "## 2. Gandalf Attack Benchmark ($n=112$)",
            "",
            "> **SCOPE STATEMENT**: The Gandalf benchmark evaluates one narrow attack family (prompts filtered to cosine similarity $\\ge 0.825$ against instructional override).",
            "",
            "### 2.1 Individual Base Detector Discrimination on Gandalf",
            "",
            "| Base Detector | Positive Detections ($s \\ge 0.5$) | Detection Recall | 95% Wilson Score Interval |",
            "|:---|:---:|:---:|:---:|",
        ]
    )

    for det_key, det_name in [
        ("d4_rules", "D4 Heuristic Rules"),
        ("d2-deberta", "D2 DeBERTa v3"),
        ("d3-deepset", "D3 Deepset Injection"),
    ]:
        det_pos = sum(
            1
            for r in gandalf_rows
            if r.get("detectors", {}).get(det_key, {}).get("is_attack") is True
            or r.get("detectors", {}).get(det_key, {}).get("score", 0.0) >= 0.5
        )
        det_rec = det_pos / 112.0
        det_ci = wilson_ci(det_pos, 112)
        doc.append(
            f"| **{det_name}** | {det_pos} / 112 | **{det_rec * 100:.2f}%** | [{det_ci[0] * 100:.2f}%, {det_ci[1] * 100:.2f}%] |"
        )

    doc.extend(
        [
            "",
            "### 2.2 Post-Hoc Descriptive Arms on Gandalf",
            "",
            "| Arm | Classification Status | ALLOW | FLAG | BLOCK | Loose Recall (95% CI) | Strict Recall (95% CI) | Judge Calls / 100 |",
            "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
            f"| **Arm R** | Primary (Pre-registered) | {g_cnt_r['allow']} | {g_cnt_r['flag']} | {g_cnt_r['block']} | {g_loose_rec * 100:.2f}% [{g_loose_ci[0] * 100:.2f}%, {g_loose_ci[1] * 100:.2f}%] | {g_strict_rec * 100:.2f}% [{g_strict_ci[0] * 100:.2f}%, {g_strict_ci[1] * 100:.2f}%] | {g_calls_r:.1f} |",
        ]
    )

    post_hoc_arms = [
        ("arm_a", "Arm A (AnyFlag Fusion)", "Post-hoc descriptive"),
        ("arm_b", "Arm B (Discrepancy Trigger + Policy B)", "Post-hoc descriptive"),
        ("arm_u", "Arm U (Universal Escalation + Policy B)", "Post-hoc descriptive"),
        ("c2_standalone", "Arm C2 (Standalone Judge)", "Post-hoc descriptive"),
    ]

    for arm_id, arm_label, arm_status in post_hoc_arms:
        cnt = compute_arm_counts(gandalf_rows, arm_id)
        loose_k = cnt["flag"] + cnt["block"]
        l_rec = loose_k / 112.0
        l_ci = wilson_ci(loose_k, 112)
        s_rec = cnt["block"] / 112.0
        s_ci = wilson_ci(cnt["block"], 112)
        calls = compute_calls_per_100(gandalf_rows, arm_id)
        doc.append(
            f"| **{arm_label}** | {arm_status} | {cnt['allow']} | {cnt['flag']} | {cnt['block']} | {l_rec * 100:.2f}% [{l_ci[0] * 100:.2f}%, {l_ci[1] * 100:.2f}%] | {s_rec * 100:.2f}% [{s_ci[0] * 100:.2f}%, {s_ci[1] * 100:.2f}%] | {calls:.1f} |"
        )

    doc.extend(
        [
            "",
            "---",
            "",
            "## 3. NotInject Benign Benchmark ($n=309$)",
            "",
            "> **EXCLUSION NOTE**: 30 exact-match rows (8.85% historical overlap with `dev.jsonl`) were excluded prior to evaluation per protocol.",
            "",
            "### 3.1 Decision Outcomes Across Arms on NotInject",
            "",
            "| Arm | Classification Status | ALLOW | FLAG | BLOCK | Pass-Through Rate (95% CI) | False Block Rate (95% CI) | Judge Calls / 100 |",
            "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
            f"| **Arm R** | Primary (Pre-registered) | {ni_cnt_r['allow']} | {ni_cnt_r['flag']} | {ni_cnt_r['block']} | {ni_pass_rec * 100:.2f}% [{ni_pass_ci[0] * 100:.2f}%, {ni_pass_ci[1] * 100:.2f}%] | {ni_fbr_rec * 100:.2f}% [{ni_fbr_ci[0] * 100:.2f}%, {ni_fbr_ci[1] * 100:.2f}%] | {ni_calls_r:.1f} |",
        ]
    )

    for arm_id, arm_label, arm_status in post_hoc_arms:
        cnt = compute_arm_counts(notinject_rows, arm_id)
        p_rec = cnt["allow"] / 309.0
        p_ci = wilson_ci(cnt["allow"], 309)
        f_rec = cnt["block"] / 309.0
        f_ci = wilson_ci(cnt["block"], 309)
        calls = compute_calls_per_100(notinject_rows, arm_id)
        doc.append(
            f"| **{arm_label}** | {arm_status} | {cnt['allow']} | {cnt['flag']} | {cnt['block']} | {p_rec * 100:.2f}% [{p_ci[0] * 100:.2f}%, {p_ci[1] * 100:.2f}%] | {f_rec * 100:.2f}% [{f_ci[0] * 100:.2f}%, {f_ci[1] * 100:.2f}%] | {calls:.1f} |"
        )

    # NotInject Subsets
    doc.extend(
        [
            "",
            "### 3.2 NotInject Subgroup & Trigger Word Breakdown (Primary Arm R)",
            "",
            "| Subgroup / Partition | Sample Size ($n$) | ALLOW | FLAG | BLOCK | Pass-Through Rate (95% CI) | False Block Rate (95% CI) |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )

    # Multilingual vs English
    eng_rows = [
        r
        for r in notinject_rows
        if r.get("category") != "Multilingual Queries" and not r.get("is_multilingual")
    ]
    multi_rows = [
        r
        for r in notinject_rows
        if r.get("category") == "Multilingual Queries"
        or r.get("is_multilingual") is True
    ]

    for label, s_rows in [
        ("English Prompts", eng_rows),
        ("Multilingual Prompts", multi_rows),
    ]:
        s_cnt = compute_arm_counts(s_rows, "arm_r")
        s_n = len(s_rows)
        s_p = s_cnt["allow"] / s_n if s_n > 0 else 0.0
        s_p_ci = wilson_ci(s_cnt["allow"], s_n)
        s_f = s_cnt["block"] / s_n if s_n > 0 else 0.0
        s_f_ci = wilson_ci(s_cnt["block"], s_n)
        doc.append(
            f"| **{label}** | {s_n} | {s_cnt['allow']} | {s_cnt['flag']} | {s_cnt['block']} | {s_p * 100:.2f}% [{s_p_ci[0] * 100:.2f}%, {s_p_ci[1] * 100:.2f}%] | {s_f * 100:.2f}% [{s_f_ci[0] * 100:.2f}%, {s_f_ci[1] * 100:.2f}%] |"
        )

    # Trigger counts
    sub1 = [
        r
        for r in notinject_rows
        if "NotInject_one" in r.get("prompt_id", "")
        or r.get("subset") == "NotInject_one"
    ]
    sub2 = [
        r
        for r in notinject_rows
        if "NotInject_two" in r.get("prompt_id", "")
        or r.get("subset") == "NotInject_two"
    ]
    sub3 = [
        r
        for r in notinject_rows
        if "NotInject_three" in r.get("prompt_id", "")
        or r.get("subset") == "NotInject_three"
    ]

    for label, s_rows in [
        ("1 Trigger Word", sub1),
        ("2 Trigger Words", sub2),
        ("3 Trigger Words", sub3),
    ]:
        s_cnt = compute_arm_counts(s_rows, "arm_r")
        s_n = len(s_rows)
        s_p = s_cnt["allow"] / s_n if s_n > 0 else 0.0
        s_p_ci = wilson_ci(s_cnt["allow"], s_n)
        s_f = s_cnt["block"] / s_n if s_n > 0 else 0.0
        s_f_ci = wilson_ci(s_cnt["block"], s_n)
        doc.append(
            f"| **Trigger Density: {label}** | {s_n} | {s_cnt['allow']} | {s_cnt['flag']} | {s_cnt['block']} | {s_p * 100:.2f}% [{s_p_ci[0] * 100:.2f}%, {s_p_ci[1] * 100:.2f}%] | {s_f * 100:.2f}% [{s_f_ci[0] * 100:.2f}%, {s_f_ci[1] * 100:.2f}%] |"
        )

    # Dolly Section
    doc.extend(
        [
            "",
            "---",
            "",
            "## 4. Dolly Benign Benchmark ($n=200$)",
            "",
            "> **SAMPLE METHOD**: 200 rows sampled via deterministic SHA-256 partition with pinned seed.",
            "",
            "### 4.1 Decision Outcomes Across Arms on Dolly",
            "",
            "| Arm | Classification Status | ALLOW | FLAG | BLOCK | Pass-Through Rate (95% CI) | False Block Rate (95% CI) | Judge Calls / 100 |",
            "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
            f"| **Arm R** | Primary (Pre-registered) | {d_cnt_r['allow']} | {d_cnt_r['flag']} | {d_cnt_r['block']} | {d_pass_rec * 100:.2f}% [{d_pass_ci[0] * 100:.2f}%, {d_pass_ci[1] * 100:.2f}%] | {d_fbr_rec * 100:.2f}% [{d_fbr_ci[0] * 100:.2f}%, {d_fbr_ci[1] * 100:.2f}%] | {d_calls_r:.1f} |",
        ]
    )

    for arm_id, arm_label, arm_status in post_hoc_arms:
        cnt = compute_arm_counts(dolly_rows, arm_id)
        p_rec = cnt["allow"] / 200.0
        p_ci = wilson_ci(cnt["allow"], 200)
        f_rec = cnt["block"] / 200.0
        f_ci = wilson_ci(cnt["block"], 200)
        calls = compute_calls_per_100(dolly_rows, arm_id)
        doc.append(
            f"| **{arm_label}** | {arm_status} | {cnt['allow']} | {cnt['flag']} | {cnt['block']} | {p_rec * 100:.2f}% [{p_ci[0] * 100:.2f}%, {p_ci[1] * 100:.2f}%] | {f_rec * 100:.2f}% [{f_ci[0] * 100:.2f}%, {f_ci[1] * 100:.2f}%] | {calls:.1f} |"
        )

    # Judge Telemetry & Outcomes
    doc.extend(
        [
            "",
            "---",
            "",
            "## 5. Judge Telemetry and Execution Summary",
            "",
            "| Metric | Measured Value | Scope / Assessment |",
            "|:---|:---:|:---|",
            f"| **Total Prompts Evaluated** | `{n_total}` | Gandalf ($n=112$), NotInject ($n=309$), Dolly ($n=200$) |",
            f"| **Predictions SHA-256** | `{pred_sha256}` | Immutable evaluation predictions log |",
            f"| **Wall-Clock Duration** | `{wall_clock_str}` | Single-pass evaluation under `scripts/gpu_lock.py` |",
            f"| **Resumed Rows Count** | `{resumed_count}` | Rows resumed from existing log |",
            f"| **Judge Outcome: OK (allow/block)** | `{outcome_counts['ok']}` | Clean structured adjudication responses |",
            f"| **Judge Outcome: Abstain** | `{outcome_counts['abstain']}` | Adjudication abstentions |",
            f"| **Judge Outcome: Timeout** | `{outcome_counts['timeout']}` | Invocations exceeding request timeout |",
            f"| **Judge Outcome: Parse Fail** | `{outcome_counts['parse_fail']}` | Malformed JSON responses |",
            f"| **Judge Outcome: Error** | `{outcome_counts['error']}` | Upstream connection or execution errors |",
            f"| **Judge Latency: Mean** | `{lat_mean:.1f} ms` | Average response latency across all calls |",
            f"| **Judge Latency: Median (p50)** | `{lat_p50:.1f} ms` | 50th percentile response latency |",
            f"| **Judge Latency: 95th Percentile (p95)** | `{lat_p95:.1f} ms` | 95th percentile response latency |",
            "",
            "---",
            "",
            "## 6. Descriptive Observations",
            "",
            f"1. **Gandalf Attack Family Coverage ($n=112$)**: Primary Arm R intercepts {g_loose_k}/112 attacks ({g_loose_rec * 100:.2f}%, 95% CI [{g_loose_ci[0] * 100:.2f}%, {g_loose_ci[1] * 100:.2f}%]) with {g_cnt_r['block']}/112 strictly blocked ({g_strict_rec * 100:.2f}%, 95% CI [{g_strict_ci[0] * 100:.2f}%, {g_strict_ci[1] * 100:.2f}%]).",
            f"2. **Base Detector Direct Overrides ($n=112$)**: On instructional overrides, D4 regex rules flagged {sum(1 for r in gandalf_rows if r.get('detectors', {}).get('d4_rules', {}).get('is_attack') is True)}/112 prompts, reflecting direct phrasing sensitivity.",
            f"3. **NotInject Benign Friction ($n=309$)**: On benign queries containing trigger words, Arm R passes through {ni_cnt_r['allow']}/309 prompts ({ni_pass_rec * 100:.2f}%, 95% CI [{ni_pass_ci[0] * 100:.2f}%, {ni_pass_ci[1] * 100:.2f}%]), with a false-block rate of {ni_fbr_rec * 100:.2f}% [{ni_fbr_ci[0] * 100:.2f}%, {ni_fbr_ci[1] * 100:.2f}%] ({ni_cnt_r['block']}/309), while flagging {ni_cnt_r['flag']}/309 for audit.",
            f"4. **Dolly Standard Instruction Following ($n=200$)**: On general benign tasks without injection triggers, Arm R passes through {d_cnt_r['allow']}/200 prompts ({d_pass_rec * 100:.2f}%, 95% CI [{d_pass_ci[0] * 100:.2f}%, {d_pass_ci[1] * 100:.2f}%]) and blocks {d_cnt_r['block']}/200 ({d_fbr_rec * 100:.2f}%, 95% CI [{d_fbr_ci[0] * 100:.2f}%, {d_fbr_ci[1] * 100:.2f}%]).",
            f"5. **Judge Invocations Rate**: Arm R invoked the local judge on {g_calls_r:.1f} calls/100 on Gandalf, {ni_calls_r:.1f} calls/100 on NotInject, and {d_calls_r:.1f} calls/100 on Dolly.",
        ]
    )

    return "\n".join(doc)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "export" / "closeout" / "external_predictions.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "docs" / "EXTERNAL_RESULTS.md"
    )
    parser.add_argument("--wall-clock", type=str, default="TBD")
    args = parser.parse_args()

    md = generate_external_results_markdown(args.predictions, args.wall_clock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Generated {args.output} successfully.")


if __name__ == "__main__":
    main()
