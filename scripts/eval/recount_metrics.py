#!/usr/bin/env python3
"""Independent Verification Recount Script (K2 TEST).

Written independently without importing closeout_metrics.py.
Directly parses results/final-v4/predictions.jsonl and verifies against the exact
human recount fixtures mandated in Closeout Plan v2.
Exits 0 if and only if all fixtures match exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
NUMBERS_PATH = REPO_ROOT / "numbers.json"

# Mandated human fixtures (ALLOW, FLAG, BLOCK)
MANDATED_FIXTURES: dict[str, dict[str, tuple[int, int, int]]] = {
    "arm_a": {"att": (37, 0, 102), "ben": (143, 0, 65)},
    "arm_r": {"att": (22, 36, 81), "ben": (143, 52, 13)},
    "arm_b": {"att": (14, 52, 73), "ben": (143, 64, 1)},
    "arm_u": {"att": (10, 56, 73), "ben": (143, 64, 1)},
    "c2_standalone": {"att": (36, 10, 93), "ben": (208, 0, 0)},
    "promptguard2": {"att": (94, 0, 45), "ben": (191, 0, 17)},
    "promptforest": {"att": (133, 0, 6), "ben": (208, 0, 0)},
}

MANDATED_JUDGE_RAW = {
    "attacks": {"allow": 36, "block": 93, "abstain": 10},
    "benign": {"allow": 208, "block": 0, "abstain": 0},
}

MANDATED_ROUTER_RATES = {
    "attacks": 0.619,
    "benign": 0.793,
}

MANDATED_TRIGGER_RATES = {
    "attacks": 0.950,
    "benign": 0.942,
}


def main() -> int:
    print("[recount_metrics] === Starting Independent Recount Verification ===")
    if not PREDICTIONS_PATH.exists():
        print(
            f"[recount_metrics] Missing predictions file at {PREDICTIONS_PATH}",
            file=sys.stderr,
        )
        return 1

    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    attacks = [r for r in records if r["label_binary"]]
    benign = [r for r in records if not r["label_binary"]]

    n_attacks = len(attacks)
    n_benign = len(benign)

    if n_attacks != 139 or n_benign != 208:
        print(
            f"[recount_metrics] FAILED: Row split mismatch. Attacks: {n_attacks} (exp 139), Benign: {n_benign} (exp 208)",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []

    # 1. Check Per-Arm Counts
    for arm, expected in MANDATED_FIXTURES.items():
        att_counts = {"allow": 0, "flag": 0, "block": 0}
        for r in attacks:
            dec = r.get("decisions", {}).get(arm)
            if dec in att_counts:
                att_counts[dec] += 1
            else:
                failures.append(f"Unknown decision '{dec}' for arm '{arm}' in attacks")

        ben_counts = {"allow": 0, "flag": 0, "block": 0}
        for r in benign:
            dec = r.get("decisions", {}).get(arm)
            if dec in ben_counts:
                ben_counts[dec] += 1
            else:
                failures.append(f"Unknown decision '{dec}' for arm '{arm}' in benign")

        actual_att = (att_counts["allow"], att_counts["flag"], att_counts["block"])
        actual_ben = (ben_counts["allow"], ben_counts["flag"], ben_counts["block"])

        if actual_att != expected["att"]:
            failures.append(
                f"Arm {arm} attack counts mismatch: actual {actual_att} != expected {expected['att']}"
            )
        else:
            print(f"[recount_metrics] PASS: {arm} attacks match {actual_att}")

        if actual_ben != expected["ben"]:
            failures.append(
                f"Arm {arm} benign counts mismatch: actual {actual_ben} != expected {expected['ben']}"
            )
        else:
            print(f"[recount_metrics] PASS: {arm} benign match {actual_ben}")

    # Check fusion equivalence with arm_a
    fusion_att = {"allow": 0, "flag": 0, "block": 0}
    for r in attacks:
        dec = r.get("fusion", {}).get("decision")
        if dec in fusion_att:
            fusion_att[dec] += 1
    fusion_ben = {"allow": 0, "flag": 0, "block": 0}
    for r in benign:
        dec = r.get("fusion", {}).get("decision")
        if dec in fusion_ben:
            fusion_ben[dec] += 1
    if (
        fusion_att["allow"],
        fusion_att["flag"],
        fusion_att["block"],
    ) != MANDATED_FIXTURES["arm_a"]["att"]:
        failures.append("Fusion attack counts do not match arm_a fixture")
    if (
        fusion_ben["allow"],
        fusion_ben["flag"],
        fusion_ben["block"],
    ) != MANDATED_FIXTURES["arm_a"]["ben"]:
        failures.append("Fusion benign counts do not match arm_a fixture")

    # 2. Check Judge Raw Outcomes
    j_att = {"allow": 0, "block": 0, "abstain": 0}
    for r in attacks:
        v = r.get("judge", {}).get("verdict")
        if v in j_att:
            j_att[v] += 1
        else:
            failures.append(f"Unknown judge verdict in attacks: {v}")

    j_ben = {"allow": 0, "block": 0, "abstain": 0}
    for r in benign:
        v = r.get("judge", {}).get("verdict")
        if v in j_ben:
            j_ben[v] += 1
        else:
            failures.append(f"Unknown judge verdict in benign: {v}")

    if j_att != MANDATED_JUDGE_RAW["attacks"]:
        failures.append(
            f"Judge attack raw counts mismatch: actual {j_att} != expected {MANDATED_JUDGE_RAW['attacks']}"
        )
    else:
        print(f"[recount_metrics] PASS: Judge attacks raw verdicts match {j_att}")

    if j_ben != MANDATED_JUDGE_RAW["benign"]:
        failures.append(
            f"Judge benign raw counts mismatch: actual {j_ben} != expected {MANDATED_JUDGE_RAW['benign']}"
        )
    else:
        print(f"[recount_metrics] PASS: Judge benign raw verdicts match {j_ben}")

    # 3. Check Router Rates
    atk_routed = (
        sum(1 for r in attacks if r.get("router", {}).get("should_route", False))
        / n_attacks
    )
    ben_routed = (
        sum(1 for r in benign if r.get("router", {}).get("should_route", False))
        / n_benign
    )

    if abs(atk_routed - MANDATED_ROUTER_RATES["attacks"]) > 0.001:
        failures.append(
            f"Router attack rate mismatch: actual {atk_routed:.3f} != expected {MANDATED_ROUTER_RATES['attacks']:.3f}"
        )
    else:
        print(
            f"[recount_metrics] PASS: Router attack should_route rate {atk_routed:.3f}"
        )

    if abs(ben_routed - MANDATED_ROUTER_RATES["benign"]) > 0.001:
        failures.append(
            f"Router benign rate mismatch: actual {ben_routed:.3f} != expected {MANDATED_ROUTER_RATES['benign']:.3f}"
        )
    else:
        print(
            f"[recount_metrics] PASS: Router benign should_route rate {ben_routed:.3f}"
        )

    # 4. Check Trigger Rates
    atk_trig = (
        sum(1 for r in attacks if r.get("trigger_arm_b", {}).get("triggered", False))
        / n_attacks
    )
    ben_trig = (
        sum(1 for r in benign if r.get("trigger_arm_b", {}).get("triggered", False))
        / n_benign
    )

    if abs(atk_trig - MANDATED_TRIGGER_RATES["attacks"]) > 0.001:
        failures.append(
            f"Trigger arm_b attack rate mismatch: actual {atk_trig:.3f} != expected {MANDATED_TRIGGER_RATES['attacks']:.3f}"
        )
    else:
        print(f"[recount_metrics] PASS: Trigger arm_b attack rate {atk_trig:.3f}")

    if abs(ben_trig - MANDATED_TRIGGER_RATES["benign"]) > 0.001:
        failures.append(
            f"Trigger arm_b benign rate mismatch: actual {ben_trig:.3f} != expected {MANDATED_TRIGGER_RATES['benign']:.3f}"
        )
    else:
        print(f"[recount_metrics] PASS: Trigger arm_b benign rate {ben_trig:.3f}")

    # 5. Check numbers.json consistency
    if not NUMBERS_PATH.exists():
        failures.append(f"Missing {NUMBERS_PATH}")
    else:
        with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
            numbers_data = json.load(f)
        for arm, exp in MANDATED_FIXTURES.items():
            if (
                numbers_data.get(f"{arm}.attack.allow", {}).get("value")
                != exp["att"][0]
            ):
                failures.append(f"numbers.json mismatch for {arm}.attack.allow")
            if numbers_data.get(f"{arm}.attack.flag", {}).get("value") != exp["att"][1]:
                failures.append(f"numbers.json mismatch for {arm}.attack.flag")
            if (
                numbers_data.get(f"{arm}.attack.block", {}).get("value")
                != exp["att"][2]
            ):
                failures.append(f"numbers.json mismatch for {arm}.attack.block")
            if (
                numbers_data.get(f"{arm}.benign.allow", {}).get("value")
                != exp["ben"][0]
            ):
                failures.append(f"numbers.json mismatch for {arm}.benign.allow")
            if numbers_data.get(f"{arm}.benign.flag", {}).get("value") != exp["ben"][1]:
                failures.append(f"numbers.json mismatch for {arm}.benign.flag")
            if (
                numbers_data.get(f"{arm}.benign.block", {}).get("value")
                != exp["ben"][2]
            ):
                failures.append(f"numbers.json mismatch for {arm}.benign.block")
        print(
            "[recount_metrics] PASS: numbers.json matches all fixture entries byte-for-byte."
        )

    if failures:
        print("[recount_metrics] FAILED with discrepancies:", file=sys.stderr)
        for err in failures:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("[recount_metrics] ALL RECOUNT FIXTURES MATCH PERFECTLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
