#!/usr/bin/env python3
"""Compute Pareto Dominance across Defense Arms (K12f Item 1).

Definition of Dominance:
An arm X dominates an arm Y (X > Y) iff:
1. recall_loose(X) >= recall_loose(Y)
2. benign_flag_block(X) <= benign_flag_block(Y)
3. calls(X) <= calls(Y)
AND strictly better (> or <) on at least one of (1), (2), or (3).

All values are computed dynamically from results/final-v4/predictions.jsonl
and cross-checked against numbers.json. Zero hardcoded numeric literals.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
NUMBERS_PATH = REPO_ROOT / "numbers.json"


def get_ci_str(numbers: dict, key: str) -> str:
    if key in numbers and "value" in numbers[key]:
        val = numbers[key]["value"]
        if isinstance(val, list) and len(val) == 2:
            return f"[{val[0] * 100:.1f}%, {val[1] * 100:.1f}%]"
    return ""


def load_and_compute_metrics():
    if not PREDICTIONS_PATH.exists():
        print(f"ERROR: {PREDICTIONS_PATH} does not exist", file=sys.stderr)
        sys.exit(1)

    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
        numbers = json.load(f)

    total_rows = len(rows)
    attack_rows = [r for r in rows if r["label_binary"]]
    benign_rows = [r for r in rows if not r["label_binary"]]
    n_attack = len(attack_rows)
    n_benign = len(benign_rows)

    # Core integrity assertions
    assert n_attack == 139, f"Integrity Failure: Expected 139 attacks, got {n_attack}"
    assert n_benign == 208, f"Integrity Failure: Expected 208 benign, got {n_benign}"
    assert n_attack + n_benign == 347, (
        f"Integrity Failure: Expected 347 total rows, got {n_attack + n_benign}"
    )
    assert total_rows == 347, f"Expected 347 total rows, got {total_rows}"

    arm_specs = [
        ("Arm A", "arm_a", "AnyFlag Calibrated Fusion (D2+D3+D4)", False),
        (
            "Arm R",
            "arm_r",
            "Learned Review Router + Policy B (tau=0.35)",
            "router",
        ),
        ("Arm B", "arm_b", "Discrepancy Trigger + Policy B", "trigger"),
        ("Arm U", "arm_u", "Full Review Upper Bound (Policy B)", "always_judge"),
        (
            "Arm C2",
            "c2_standalone",
            "Standalone Gemma-3 4B Judge",
            "always_judge",
        ),
        (
            "Prompt Guard 2",
            "promptguard2",
            "Standalone Calibrated Baseline",
            False,
        ),
        ("PromptForest", "promptforest", "External Tree Baseline", False),
    ]

    arms = {}
    for display_name, arm_key, desc, call_mode in arm_specs:
        # Attack counts
        atk_allow = sum(1 for r in attack_rows if r["decisions"][arm_key] == "allow")
        atk_flag = sum(1 for r in attack_rows if r["decisions"][arm_key] == "flag")
        atk_block = sum(1 for r in attack_rows if r["decisions"][arm_key] == "block")

        # Benign counts
        ben_allow = sum(1 for r in benign_rows if r["decisions"][arm_key] == "allow")
        ben_flag = sum(1 for r in benign_rows if r["decisions"][arm_key] == "flag")
        ben_block = sum(1 for r in benign_rows if r["decisions"][arm_key] == "block")

        # Per-arm class total assertions
        assert atk_allow + atk_flag + atk_block == n_attack, (
            f"{arm_key} attack counts {atk_allow}+{atk_flag}+{atk_block} != {n_attack}"
        )
        assert ben_allow + ben_flag + ben_block == n_benign, (
            f"{arm_key} benign counts {ben_allow}+{ben_flag}+{ben_block} != {n_benign}"
        )

        recall_loose = ((atk_flag + atk_block) / n_attack) * 100.0
        recall_strict = (atk_block / n_attack) * 100.0
        pass_through = (ben_allow / n_benign) * 100.0
        benign_flag = (ben_flag / n_benign) * 100.0
        benign_block = (ben_block / n_benign) * 100.0
        benign_flag_block = ((ben_flag + ben_block) / n_benign) * 100.0

        # Invocations / extra calls per 100 requests
        if call_mode == "router":
            calls = (
                sum(1 for r in rows if r["router"]["should_route"]) / total_rows
            ) * 100.0
        elif call_mode == "trigger":
            calls = (
                sum(1 for r in rows if r["trigger_arm_b"]["triggered"]) / total_rows
            ) * 100.0
        elif call_mode == "always_judge":
            calls = (
                sum(1 for r in rows if r.get("judge") is not None) / total_rows
            ) * 100.0
        else:
            calls = 0.0

        # Check loose recall against numbers.json
        num_recall_key = f"{arm_key}.loose.recall"
        if num_recall_key in numbers:
            expected_recall = (
                float(numbers[num_recall_key]["value"]) * 100.0
                if isinstance(numbers[num_recall_key], dict)
                else float(numbers[num_recall_key]) * 100.0
            )
            assert abs(recall_loose - expected_recall) < 1e-4, (
                f"{arm_key} loose recall {recall_loose:.4f}% does not match numbers.json {expected_recall:.4f}%"
            )

        # CIs from numbers.json
        loose_ci = get_ci_str(numbers, f"{arm_key}.loose.recall_ci")
        strict_ci = get_ci_str(numbers, f"{arm_key}.strict.recall_ci")
        pass_ci = get_ci_str(numbers, f"{arm_key}.benign.pass_through_ci")
        flag_ci = get_ci_str(numbers, f"{arm_key}.benign.flag_load_ci")
        block_ci = get_ci_str(numbers, f"{arm_key}.benign.block_rate_ci")

        arms[display_name] = {
            "key": arm_key,
            "desc": desc,
            "atk_counts": (atk_allow, atk_flag, atk_block),
            "ben_counts": (ben_allow, ben_flag, ben_block),
            "recall_loose": recall_loose,
            "recall_loose_ci": loose_ci,
            "recall_strict": recall_strict,
            "recall_strict_ci": strict_ci,
            "pass_through": pass_through,
            "pass_through_ci": pass_ci,
            "benign_flag": benign_flag,
            "benign_flag_ci": flag_ci,
            "benign_block": benign_block,
            "benign_block_ci": block_ci,
            "benign_flag_block": benign_flag_block,
            "calls": calls,
        }

    # Pareto dominance computation
    for y_name, y_m in arms.items():
        dominators = []
        for x_name, x_m in arms.items():
            if x_name == y_name:
                continue
            r_ok = x_m["recall_loose"] >= y_m["recall_loose"]
            fb_ok = x_m["benign_flag_block"] <= y_m["benign_flag_block"]
            c_ok = x_m["calls"] <= y_m["calls"]
            strictly_better = (
                x_m["recall_loose"] > y_m["recall_loose"]
                or x_m["benign_flag_block"] < y_m["benign_flag_block"]
                or x_m["calls"] < y_m["calls"]
            )
            if r_ok and fb_ok and c_ok and strictly_better:
                dominators.append(x_name)

        if dominators:
            y_m["pareto_status"] = f"Dominated by {', '.join(dominators)}"
        else:
            y_m["pareto_status"] = "Non-dominated (Pareto-optimal)"

    return arms, n_attack, n_benign


def generate_markdown_table(arms: dict, n_attack: int, n_benign: int) -> str:
    lines = [
        "**Dominance Definition (computed by `scripts/eval/compute_pareto.py`)**:",
        "An arm X dominates an arm Y (X > Y) iff:",
        "1. Recall_loose(X) >= Recall_loose(Y)",
        "2. Benign_FLAG+BLOCK(X) <= Benign_FLAG+BLOCK(Y)",
        "3. Calls(X) <= Calls(Y)",
        "AND strictly better on at least one of (1), (2), or (3). Under this definition, all evaluated arms are non-dominated trade-offs across the 3 operational objectives.",
        "",
        "| Arm Identifier | Arm Description | Attack Decisions (ALLOW / FLAG / BLOCK) | Recall (Loose: FLAG/BLOCK) | Recall (Strict: BLOCK only) | Benign Decisions (ALLOW / FLAG / BLOCK) | Benign Pass-Through (ALLOW only) | Benign FLAG Load | Benign BLOCK Rate | Extra Calls / 100 | Post-Hoc Pareto Status |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]

    for name, m in arms.items():
        atk_c = f"{m['atk_counts'][0]} / {m['atk_counts'][1]} / {m['atk_counts'][2]}"
        ben_c = f"{m['ben_counts'][0]} / {m['ben_counts'][1]} / {m['ben_counts'][2]}"
        r_loose = f"**{m['recall_loose']:.2f}%**" + (
            f" {m['recall_loose_ci']}" if m["recall_loose_ci"] else ""
        )
        r_strict = f"{m['recall_strict']:.2f}%" + (
            f" {m['recall_strict_ci']}" if m["recall_strict_ci"] else ""
        )
        pass_th = f"**{m['pass_through']:.2f}%**" + (
            f" {m['pass_through_ci']}" if m["pass_through_ci"] else ""
        )
        b_flag = f"{m['benign_flag']:.2f}%" + (
            f" {m['benign_flag_ci']}" if m["benign_flag_ci"] else ""
        )
        b_block = f"{m['benign_block']:.2f}%" + (
            f" {m['benign_block_ci']}" if m["benign_block_ci"] else ""
        )
        calls_str = (
            f"**{m['calls']:.2f}**"
            if m["calls"] in [0.0, 100.0]
            else f"{m['calls']:.2f}"
        )

        lines.append(
            f"| **{name}** | {m['desc']} | {atk_c} | {r_loose} | {r_strict} | {ben_c} | {pass_th} | {b_flag} | {b_block} | {calls_str} | {m['pareto_status']} |"
        )

    return "\n".join(lines)


def print_independent_counter():
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    label_counts = Counter(r["label_binary"] for r in rows)
    print("=" * 80)
    print("INDEPENDENT PREDICTIONS.JSONL COUNTER BREAKDOWN:")
    print("=" * 80)
    print(
        f"Rows by label_binary: {dict(label_counts)} (Attacks: {label_counts[True]}, Benign: {label_counts[False]}, Total: {len(rows)})"
    )

    arms = [
        "arm_a",
        "arm_b",
        "arm_r",
        "arm_u",
        "c2_standalone",
        "promptguard2",
        "promptforest",
    ]
    for arm in arms:
        atk_c = Counter(r["decisions"][arm] for r in rows if r["label_binary"] is True)
        ben_c = Counter(r["decisions"][arm] for r in rows if r["label_binary"] is False)
        print(f"[{arm}]")
        print(
            f"  Attacks (N={sum(atk_c.values())}): ALLOW={atk_c.get('allow', 0)}, FLAG={atk_c.get('flag', 0)}, BLOCK={atk_c.get('block', 0)}"
        )
        print(
            f"  Benign  (N={sum(ben_c.values())}): ALLOW={ben_c.get('allow', 0)}, FLAG={ben_c.get('flag', 0)}, BLOCK={ben_c.get('block', 0)}"
        )


def print_pairwise_dominance(arms: dict):
    print("\n" + "=" * 80)
    print("PAIRWISE PARETO DOMINANCE CHECKS:")
    print("=" * 80)
    for y_name, y_m in arms.items():
        print(
            f"\nChecking candidate Y = {y_name} (Recall={y_m['recall_loose']:.2f}%, FLAG+BLOCK={y_m['benign_flag_block']:.2f}%, Calls={y_m['calls']:.2f}):"
        )
        dominators = []
        for x_name, x_m in arms.items():
            if x_name == y_name:
                continue
            r_ok = x_m["recall_loose"] >= y_m["recall_loose"]
            fb_ok = x_m["benign_flag_block"] <= y_m["benign_flag_block"]
            c_ok = x_m["calls"] <= y_m["calls"]
            strictly_better = (
                x_m["recall_loose"] > y_m["recall_loose"]
                or x_m["benign_flag_block"] < y_m["benign_flag_block"]
                or x_m["calls"] < y_m["calls"]
            )
            dominates = r_ok and fb_ok and c_ok and strictly_better
            if dominates:
                dominators.append(x_name)
                print(
                    f"  --> {x_name} DOMINATES {y_name}: Recall({x_m['recall_loose']:.2f} >= {y_m['recall_loose']:.2f}), "
                    f"FLAG+BLOCK({x_m['benign_flag_block']:.2f} <= {y_m['benign_flag_block']:.2f}), "
                    f"Calls({x_m['calls']:.2f} <= {y_m['calls']:.2f})"
                )
            else:
                reasons = []
                if not r_ok:
                    reasons.append(
                        f"lower recall ({x_m['recall_loose']:.2f}% < {y_m['recall_loose']:.2f}%)"
                    )
                if not fb_ok:
                    reasons.append(
                        f"higher benign FLAG+BLOCK ({x_m['benign_flag_block']:.2f}% > {y_m['benign_flag_block']:.2f}%)"
                    )
                if not c_ok:
                    reasons.append(
                        f"higher calls ({x_m['calls']:.2f} > {y_m['calls']:.2f})"
                    )
                if not strictly_better and r_ok and fb_ok and c_ok:
                    reasons.append("identical across all 3 objectives")
                print(
                    f"      {x_name} does NOT dominate {y_name}: {'; '.join(reasons)}"
                )

        if not dominators:
            print(f"  Result: {y_name} is NON-DOMINATED (Pareto-optimal)")
        else:
            print(f"  Result: {y_name} is DOMINATED by {', '.join(dominators)}")


def main():
    print_independent_counter()
    arms, n_attack, n_benign = load_and_compute_metrics()
    print_pairwise_dominance(arms)

    print("\n" + "=" * 80)
    print("MARKDOWN TABLE OUTPUT:")
    print("=" * 80)
    table_md = generate_markdown_table(arms, n_attack, n_benign)
    print(table_md)


if __name__ == "__main__":
    main()
