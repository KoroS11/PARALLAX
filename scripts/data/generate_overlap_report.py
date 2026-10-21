"""CLI script to generate train/val/test overlap and near-duplicate reports.

Uses MinHash LSH and n-gram Jaccard similarity to scan for split contamination,
enforces canonical cluster isolation, and writes split manifests.

Usage:
    uv run python -m scripts.data.generate_overlap_report [--limit 500] [--output docs/eval/overlap_report.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from parallax.data.loaders import load_technique_samples
from parallax.data.splits import (
    enforce_cluster_isolation,
    format_contamination_report_markdown,
    generate_contamination_report,
    generate_splits_manifest,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Generate train/val/test overlap and near-duplicate report."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples to load per split for faster scanning (default: all).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for near-duplicates (default: 0.85).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/eval/overlap_report.md",
        help="Output markdown file path (default: docs/eval/overlap_report.md).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/canonical_splits_manifest.json",
        help="Output manifest path (default: data/manifests/canonical_splits_manifest.json).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token if needed.",
    )
    parser.add_argument(
        "--raw-splits",
        action="store_true",
        help="Analyze raw splits without applying canonical cluster isolation.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PARALLAX Split Contamination and Overlap Analyzer")
    print("=" * 70)

    print(
        f"\n[1/4] Loading neuralchemy/prompt-injection-technique splits (limit={args.limit})..."
    )
    train_samples = load_technique_samples(
        split="train", limit=args.limit, token=args.token
    )
    val_samples = load_technique_samples(
        split="val", limit=args.limit, token=args.token
    )
    test_samples = load_technique_samples(
        split="test", limit=args.limit, token=args.token
    )

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }
    print(
        f"    Loaded -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}"
    )

    if not args.raw_splits:
        print(
            f"\n[2/4] Enforcing canonical split deduplication & cluster isolation (threshold={args.threshold})..."
        )
        splits = enforce_cluster_isolation(
            splits, threshold=args.threshold, n_gram=5, seed=42
        )
        print(
            f"    Canonical -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
        )
    else:
        print(
            "\n[2/4] Skipping canonical cluster isolation (--raw-splits specified)..."
        )

    print(
        f"\n[3/4] Computing exact overlaps and MinHash near-duplicates (threshold={args.threshold})..."
    )
    report = generate_contamination_report(splits, similarity_threshold=args.threshold)

    # Verify zero exact overlap between canonical train and any eval set
    exact_train_val = len(report["exact_matches_train_val"])
    exact_train_test = len(report["exact_matches_train_test"])

    if not args.raw_splits and (exact_train_val > 0 or exact_train_test > 0):
        print(
            f"\n[!] ERROR: Canonical train has exact overlaps with evaluation sets! "
            f"Train-Val: {exact_train_val}, Train-Test: {exact_train_test}"
        )
        return 1

    if not args.raw_splits:
        print(
            "    [✓] Zero exact overlap verified between canonical train and all eval sets."
        )

    md_output = format_contamination_report_markdown(report)
    print(md_output)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_output, encoding="utf-8")
        print(f"\n[4/4] Report successfully written to: {out_path.resolve()}")

    if args.manifest and not args.raw_splits:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = generate_splits_manifest(splits, output_path=manifest_path, seed=42)
        print(
            f"    [✓] Canonical splits manifest written to: {manifest_path.resolve()} (total: {manifest['total_samples']})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
