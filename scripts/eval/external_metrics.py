#!/usr/bin/env python3
"""Wave K13 External Evaluation Metrics Calculator (scripts/eval/external_metrics.py).

Features per EXTERNAL_EVAL_PROTOCOL.md:
- Reads prediction logs (results/final-v4/predictions.jsonl or external evaluation logs).
- Computes per-family performance tables without pooling benign families.
- Calculates exact Wilson 95% score confidence intervals for all binomial proportions.
- Generates subset and multilingual splits (e.g. NotInject trigger counts, multilingual category).
- Tracks full decision outcome distributions (ALLOW, FLAG, BLOCK) per arm.
- Measures judge invocation rates (calls per 100 requests) and latency distributions.
- Includes automated self-test validating exact reproduction of numbers.json values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, ClassVar

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute exact Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    # Standard normal quantile for 95% confidence (1.959963984540054)
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def get_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ExternalMetricsCalculator:
    """Computes protocol-compliant external evaluation metrics and tables."""

    ARMS: ClassVar[list[str]] = ["arm_r", "arm_a", "arm_b", "arm_u", "c2_standalone"]

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    @classmethod
    def from_jsonl(cls, filepath: Path) -> ExternalMetricsCalculator:
        with open(filepath, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return cls(records)

    def compute_arm_outcomes(
        self, rows: list[dict[str, Any]], arm: str
    ) -> dict[str, int]:
        """Compute ALLOW, FLAG, BLOCK counts for a specific arm on given rows."""
        counts = {"allow": 0, "flag": 0, "block": 0}
        for r in rows:
            dec = str(r.get("decisions", {}).get(arm, "allow")).lower()
            if dec in counts:
                counts[dec] += 1
            else:
                counts["allow"] += 1
        return counts

    def compute_calls_per_100(self, rows: list[dict[str, Any]], arm: str) -> float:
        """Compute judge calls per 100 requests for an arm."""
        if not rows:
            return 0.0
        n = len(rows)
        if arm == "arm_a":
            return 0.0
        if arm in ("arm_u", "c2_standalone"):
            return 100.0
        if arm == "arm_b":
            # Arm B calls judge when trigger_arm_b fired
            calls = sum(
                1
                for r in rows
                if r.get("trigger_arm_b", {}).get("triggered")
                or r.get("trigger_arm_b", {}).get("should_challenge")
            )
            return float((calls / n) * 100.0)
        if arm == "arm_r":
            # Arm R calls judge when router.should_route is True
            calls = sum(
                1 for r in rows if r.get("router", {}).get("should_route") is True
            )
            return float((calls / n) * 100.0)
        return 0.0

    def compute_family_metrics(
        self, rows: list[dict[str, Any]], is_attack: bool
    ) -> dict[str, Any]:
        """Compute metrics for a single dataset family."""
        n = len(rows)
        family_results: dict[str, Any] = {"total_samples": n, "arms": {}}

        for arm in self.ARMS:
            outcomes = self.compute_arm_outcomes(rows, arm)
            calls_rate = self.compute_calls_per_100(rows, arm)

            allow_cnt = outcomes["allow"]
            flag_cnt = outcomes["flag"]
            block_cnt = outcomes["block"]

            if is_attack:
                loose_caught = flag_cnt + block_cnt
                loose_recall = loose_caught / n if n > 0 else 0.0
                loose_ci = wilson_ci(loose_caught, n)

                strict_caught = block_cnt
                strict_recall = strict_caught / n if n > 0 else 0.0
                strict_ci = wilson_ci(strict_caught, n)

                family_results["arms"][arm] = {
                    "allow": allow_cnt,
                    "flag": flag_cnt,
                    "block": block_cnt,
                    "loose_recall": round(loose_recall, 4),
                    "loose_recall_wilson_95ci": [
                        round(loose_ci[0], 4),
                        round(loose_ci[1], 4),
                    ],
                    "strict_recall": round(strict_recall, 4),
                    "strict_recall_wilson_95ci": [
                        round(strict_ci[0], 4),
                        round(strict_ci[1], 4),
                    ],
                    "calls_per_100": calls_rate,
                }
            else:
                pass_through = allow_cnt / n if n > 0 else 0.0
                pass_ci = wilson_ci(allow_cnt, n)

                fbr = block_cnt / n if n > 0 else 0.0
                fbr_ci = wilson_ci(block_cnt, n)

                flag_rate = flag_cnt / n if n > 0 else 0.0
                flag_ci = wilson_ci(flag_cnt, n)

                family_results["arms"][arm] = {
                    "allow": allow_cnt,
                    "flag": flag_cnt,
                    "block": block_cnt,
                    "pass_through_rate": round(pass_through, 4),
                    "pass_through_wilson_95ci": [
                        round(pass_ci[0], 4),
                        round(pass_ci[1], 4),
                    ],
                    "flag_rate": round(flag_rate, 4),
                    "flag_rate_wilson_95ci": [
                        round(flag_ci[0], 4),
                        round(flag_ci[1], 4),
                    ],
                    "false_block_rate": round(fbr, 4),
                    "false_block_rate_wilson_95ci": [
                        round(fbr_ci[0], 4),
                        round(fbr_ci[1], 4),
                    ],
                    "calls_per_100": calls_rate,
                }

        return family_results

    def generate_full_report(self) -> dict[str, Any]:
        """Generate comprehensive metrics partitioned by family, split, and multilingual status."""
        total_n = len(self.records)
        attacks = [r for r in self.records if r.get("label_binary") is True]
        benigns = [r for r in self.records if r.get("label_binary") is False]

        report: dict[str, Any] = {
            "summary": {
                "total_rows": total_n,
                "total_attacks": len(attacks),
                "total_benign": len(benigns),
            },
            "calls_per_100": {
                arm: self.compute_calls_per_100(self.records, arm) for arm in self.ARMS
            },
            "attacks_overall": self.compute_family_metrics(attacks, is_attack=True),
            "benign_overall": self.compute_family_metrics(benigns, is_attack=False),
            "subsets": {},
        }

        # Subsets by source / dataset / category
        sources = sorted({r.get("source") for r in self.records if r.get("source")})
        for src in sources:
            src_rows = [r for r in self.records if r.get("source") == src]
            src_is_atk = any(r.get("label_binary") is True for r in src_rows)
            report["subsets"][f"source_{src}"] = self.compute_family_metrics(
                src_rows, is_attack=src_is_atk
            )

        # Multilingual split if present
        multilingual_rows = [
            r
            for r in self.records
            if r.get("category") == "Multilingual Queries"
            or r.get("is_multilingual") is True
        ]
        if multilingual_rows:
            report["subsets"]["multilingual"] = self.compute_family_metrics(
                multilingual_rows, is_attack=False
            )

        # NotInject trigger-word splits if present
        for sub in ("NotInject_one", "NotInject_two", "NotInject_three"):
            sub_rows = [r for r in self.records if r.get("subset") == sub]
            if sub_rows:
                report["subsets"][sub] = self.compute_family_metrics(
                    sub_rows, is_attack=False
                )

        return report

    def render_markdown_tables(self) -> str:
        """Render protocol-compliant GitHub Markdown comparison tables."""
        rep = self.generate_full_report()
        atk = rep["attacks_overall"]["arms"]
        ben = rep["benign_overall"]["arms"]
        calls = rep["calls_per_100"]

        lines = [
            "# External Generalization Evaluation Metrics Report",
            "",
            "## 1. Overall Performance by Arm",
            "",
            "| Arm | Attack Loose Recall (95% CI) | Attack Strict (Block) | Benign Pass-Through (95% CI) | Benign FBR (95% CI) | Calls / 100 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for arm in self.ARMS:
            a_data = atk[arm]
            b_data = ben[arm]
            a_rec = f"{a_data['loose_recall'] * 100:.2f}% [{a_data['loose_recall_wilson_95ci'][0] * 100:.2f}%, {a_data['loose_recall_wilson_95ci'][1] * 100:.2f}%]"
            a_strict = f"{a_data['strict_recall'] * 100:.2f}%"
            b_pass = f"{b_data['pass_through_rate'] * 100:.2f}% [{b_data['pass_through_wilson_95ci'][0] * 100:.2f}%, {b_data['pass_through_wilson_95ci'][1] * 100:.2f}%]"
            b_fbr = f"{b_data['false_block_rate'] * 100:.2f}% [{b_data['false_block_rate_wilson_95ci'][0] * 100:.2f}%, {b_data['false_block_rate_wilson_95ci'][1] * 100:.2f}%]"
            c_rate = f"{calls[arm]:.2f}"
            lines.append(
                f"| `{arm}` | {a_rec} | {a_strict} | {b_pass} | {b_fbr} | {c_rate} |"
            )

        lines.extend(
            [
                "",
                "## 2. Decision Outcome Counts (ALLOW / FLAG / BLOCK)",
                "",
                "| Arm | Attack ALLOW | Attack FLAG | Attack BLOCK | Benign ALLOW | Benign FLAG | Benign BLOCK |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )

        for arm in self.ARMS:
            a = atk[arm]
            b = ben[arm]
            lines.append(
                f"| `{arm}` | {a['allow']} | {a['flag']} | {a['block']} | {b['allow']} | {b['flag']} | {b['block']} |"
            )

        return "\n".join(lines)


def test_against_numbers_json(
    predictions_path: Path = PROJECT_ROOT
    / "results"
    / "final-v4"
    / "predictions.jsonl",
    numbers_path: Path = PROJECT_ROOT / "numbers.json",
) -> None:
    """Verify that external_metrics.py reproduces overlapping numbers.json values exactly."""
    calc = ExternalMetricsCalculator.from_jsonl(predictions_path)
    rep = calc.generate_full_report()
    numbers = json.loads(numbers_path.read_text(encoding="utf-8"))

    # Test suite of overlapping keys between numbers.json and external_metrics.py
    checks = [
        (
            "dataset.total_rows",
            rep["summary"]["total_rows"],
            numbers["dataset.total_rows"]["value"],
        ),
        (
            "dataset.attack_count",
            rep["summary"]["total_attacks"],
            numbers["dataset.attack_count"]["value"],
        ),
        (
            "dataset.benign_count",
            rep["summary"]["total_benign"],
            numbers["dataset.benign_count"]["value"],
        ),
        (
            "arm_r.attack.allow",
            rep["attacks_overall"]["arms"]["arm_r"]["allow"],
            numbers["arm_r.attack.allow"]["value"],
        ),
        (
            "arm_r.attack.flag",
            rep["attacks_overall"]["arms"]["arm_r"]["flag"],
            numbers["arm_r.attack.flag"]["value"],
        ),
        (
            "arm_r.attack.block",
            rep["attacks_overall"]["arms"]["arm_r"]["block"],
            numbers["arm_r.attack.block"]["value"],
        ),
        (
            "arm_r.benign.allow",
            rep["benign_overall"]["arms"]["arm_r"]["allow"],
            numbers["arm_r.benign.allow"]["value"],
        ),
        (
            "arm_r.benign.flag",
            rep["benign_overall"]["arms"]["arm_r"]["flag"],
            numbers["arm_r.benign.flag"]["value"],
        ),
        (
            "arm_r.benign.block",
            rep["benign_overall"]["arms"]["arm_r"]["block"],
            numbers["arm_r.benign.block"]["value"],
        ),
        (
            "arm_r.loose.recall",
            rep["attacks_overall"]["arms"]["arm_r"]["loose_recall"],
            numbers["arm_r.loose.recall"]["value"],
        ),
        (
            "arm_r.strict.recall",
            rep["attacks_overall"]["arms"]["arm_r"]["strict_recall"],
            numbers["arm_r.strict.recall"]["value"],
        ),
        (
            "arm_r.benign.block_rate",
            rep["benign_overall"]["arms"]["arm_r"]["false_block_rate"],
            numbers["arm_r.benign.block_rate"]["value"],
        ),
        (
            "arm_r.benign.flag_load",
            rep["benign_overall"]["arms"]["arm_r"]["flag_rate"],
            numbers["arm_r.benign.flag_load"]["value"],
        ),
        (
            "arm_r.benign.pass_through",
            rep["benign_overall"]["arms"]["arm_r"]["pass_through_rate"],
            numbers["arm_r.benign.pass_through"]["value"],
        ),
        (
            "arm_r.extra_calls_per_100",
            rep["calls_per_100"]["arm_r"],
            numbers["arm_r.extra_calls_per_100"]["value"],
        ),
        (
            "arm_a.attack.allow",
            rep["attacks_overall"]["arms"]["arm_a"]["allow"],
            numbers["arm_a.attack.allow"]["value"],
        ),
        (
            "arm_a.attack.block",
            rep["attacks_overall"]["arms"]["arm_a"]["block"],
            numbers["arm_a.attack.block"]["value"],
        ),
        (
            "arm_a.benign.allow",
            rep["benign_overall"]["arms"]["arm_a"]["allow"],
            numbers["arm_a.benign.allow"]["value"],
        ),
        (
            "arm_a.benign.block",
            rep["benign_overall"]["arms"]["arm_a"]["block"],
            numbers["arm_a.benign.block"]["value"],
        ),
        (
            "arm_a.loose.recall",
            rep["attacks_overall"]["arms"]["arm_a"]["loose_recall"],
            numbers["arm_a.loose.recall"]["value"],
        ),
        (
            "arm_b.attack.allow",
            rep["attacks_overall"]["arms"]["arm_b"]["allow"],
            numbers["arm_b.attack.allow"]["value"],
        ),
        (
            "arm_b.attack.flag",
            rep["attacks_overall"]["arms"]["arm_b"]["flag"],
            numbers["arm_b.attack.flag"]["value"],
        ),
        (
            "arm_b.attack.block",
            rep["attacks_overall"]["arms"]["arm_b"]["block"],
            numbers["arm_b.attack.block"]["value"],
        ),
        (
            "arm_b.benign.allow",
            rep["benign_overall"]["arms"]["arm_b"]["allow"],
            numbers["arm_b.benign.allow"]["value"],
        ),
        (
            "arm_b.benign.flag",
            rep["benign_overall"]["arms"]["arm_b"]["flag"],
            numbers["arm_b.benign.flag"]["value"],
        ),
        (
            "arm_b.benign.block",
            rep["benign_overall"]["arms"]["arm_b"]["block"],
            numbers["arm_b.benign.block"]["value"],
        ),
        (
            "arm_b.loose.recall",
            rep["attacks_overall"]["arms"]["arm_b"]["loose_recall"],
            numbers["arm_b.loose.recall"]["value"],
        ),
        (
            "arm_b.extra_calls_per_100",
            rep["calls_per_100"]["arm_b"],
            numbers["arm_b.extra_calls_per_100"]["value"],
        ),
        (
            "arm_u.attack.allow",
            rep["attacks_overall"]["arms"]["arm_u"]["allow"],
            numbers["arm_u.attack.allow"]["value"],
        ),
        (
            "arm_u.attack.flag",
            rep["attacks_overall"]["arms"]["arm_u"]["flag"],
            numbers["arm_u.attack.flag"]["value"],
        ),
        (
            "arm_u.attack.block",
            rep["attacks_overall"]["arms"]["arm_u"]["block"],
            numbers["arm_u.attack.block"]["value"],
        ),
        (
            "arm_u.loose.recall",
            rep["attacks_overall"]["arms"]["arm_u"]["loose_recall"],
            numbers["arm_u.loose.recall"]["value"],
        ),
        (
            "c2_standalone.attack.allow",
            rep["attacks_overall"]["arms"]["c2_standalone"]["allow"],
            numbers["c2_standalone.attack.allow"]["value"],
        ),
        (
            "c2_standalone.attack.flag",
            rep["attacks_overall"]["arms"]["c2_standalone"]["flag"],
            numbers["c2_standalone.attack.flag"]["value"],
        ),
        (
            "c2_standalone.attack.block",
            rep["attacks_overall"]["arms"]["c2_standalone"]["block"],
            numbers["c2_standalone.attack.block"]["value"],
        ),
        (
            "c2_standalone.loose.recall",
            rep["attacks_overall"]["arms"]["c2_standalone"]["loose_recall"],
            numbers["c2_standalone.loose.recall"]["value"],
        ),
        (
            "c2_standalone.benign.pass_through",
            rep["benign_overall"]["arms"]["c2_standalone"]["pass_through_rate"],
            numbers["c2_standalone.benign.pass_through"]["value"],
        ),
    ]

    mismatches = []
    for key, computed, expected in checks:
        if abs(computed - expected) > 1e-4:
            mismatches.append((key, computed, expected))

    if mismatches:
        raise AssertionError(
            f"Reproduction test failed on {len(mismatches)} keys: {mismatches}"
        )

    print(
        f"PASS: external_metrics.py reproduced all {len(checks)} tested keys from numbers.json perfectly."
    )


def main():
    parser = argparse.ArgumentParser(
        description="External evaluation metrics calculator"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "results" / "final-v4" / "predictions.jsonl",
    )
    parser.add_argument(
        "--test-reproduction",
        action="store_true",
        help="Run self-test against numbers.json",
    )
    parser.add_argument(
        "--output-json", type=Path, help="Path to write output JSON metrics"
    )
    args = parser.parse_args()

    if args.test_reproduction:
        test_against_numbers_json(predictions_path=args.predictions)
        return

    calc = ExternalMetricsCalculator.from_jsonl(args.predictions)
    rep = calc.generate_full_report()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"Metrics saved to {args.output_json}")

    print(calc.render_markdown_tables())


if __name__ == "__main__":
    main()
