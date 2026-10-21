"""CLI script to download, verify, and summarize all PARALLAX datasets,
enforce canonical splits, verify zero exact overlap, and generate split manifest.

Usage:
    uv run python -m scripts.data.verify_datasets [--limit 50] [--token <HF_TOKEN>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from parallax.data.loaders import (
    BINARY_DATASET_ID,
    MINDGARD_DATASET_ID,
    NOTINJECT_DATASET_ID,
    PROMPTGUARD2_DATASET_ID,
    SIMSONSUN_DATASET_ID,
    TECHNIQUE_DATASET_ID,
    GatedDatasetAccessError,
    load_binary_samples,
    load_mindgard_samples,
    load_notinject_samples,
    load_promptguard2_samples,
    load_simsonsun_jailbreaks,
    load_technique_samples,
)
from parallax.data.schema import summarize_samples
from parallax.data.splits import (
    check_exact_overlaps,
    enforce_cluster_isolation,
    generate_splits_manifest,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Verify and download PARALLAX benchmark datasets."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to load per split (default: full split).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token for gated repos.",
    )
    parser.add_argument(
        "--strict-gated",
        action="store_true",
        help="Fail with an error if gated datasets (Mindgard, Prompt Guard 2) cannot be accessed.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/canonical_splits_manifest.json",
        help="Output manifest path (default: data/manifests/canonical_splits_manifest.json).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PARALLAX Dataset Verification Suite")
    print("=" * 70)

    datasets_to_check: list[dict[str, Any]] = [
        {
            "name": "Prompt Injection Technique (Multi-class)",
            "id": TECHNIQUE_DATASET_ID,
            "loader": lambda: load_technique_samples(
                limit=args.limit, token=args.token
            ),
            "gated": False,
        },
        {
            "name": "Prompt Injection Dataset (Binary)",
            "id": BINARY_DATASET_ID,
            "loader": lambda: load_binary_samples(limit=args.limit, token=args.token),
            "gated": False,
        },
        {
            "name": "Simsonsun Jailbreaks (OOD Test)",
            "id": SIMSONSUN_DATASET_ID,
            "loader": lambda: load_simsonsun_jailbreaks(
                limit=args.limit, token=args.token
            ),
            "gated": False,
        },
        {
            "name": "NotInject (Benign Over-Defense Control)",
            "id": NOTINJECT_DATASET_ID,
            "loader": lambda: load_notinject_samples(
                limit=args.limit, token=args.token
            ),
            "gated": False,
        },
        {
            "name": "Mindgard Evasion Samples (Gated OOD Test)",
            "id": MINDGARD_DATASET_ID,
            "loader": lambda: load_mindgard_samples(
                limit=args.limit, token=args.token, allow_missing=not args.strict_gated
            ),
            "gated": True,
        },
        {
            "name": "Prompt Guard 2 Benchmark (Gated D3)",
            "id": PROMPTGUARD2_DATASET_ID,
            "loader": lambda: load_promptguard2_samples(
                limit=args.limit, token=args.token, allow_missing=not args.strict_gated
            ),
            "gated": True,
        },
    ]

    has_failures = False
    technique_samples_loaded = None

    for ds in datasets_to_check:
        print(f"\n[+] Verifying: {ds['name']} ({ds['id']})")
        try:
            samples = ds["loader"]()
            if ds["id"] == TECHNIQUE_DATASET_ID:
                technique_samples_loaded = samples

            if not samples:
                if ds["gated"]:
                    print(
                        "    [*] GATED / RESTRICTED ACCESS (Handled gracefully: 0 samples loaded)"
                    )
                    print(
                        "        -> To download, request access at https://huggingface.co/"
                        + ds["id"]
                        + " and set HF_TOKEN"
                    )
                else:
                    print("    [!] Warning: 0 samples loaded.")
                continue

            summary = summarize_samples(samples)
            print(f"    - Loaded: {summary['total_samples']} samples")
            print(f"    - Split distribution: {summary['split_counts']}")
            print(f"    - Binary distribution: {summary['binary_counts']}")
            print(f'    - Sample text preview: "{samples[0].text[:75]}..."')
            print(
                f"    - Sample label: {samples[0].label_multiclass} (binary={samples[0].label_binary})"
            )
            print("    [*] Status: VERIFIED & NORMALIZED")

        except GatedDatasetAccessError as e:
            print(f"    [!] Gated Access Error: {e}")
            if args.strict_gated:
                has_failures = True
        except Exception as e:  # noqa: BLE001
            print(f"    [X] Verification Failed: {e}")
            has_failures = True

    # Canonical split generation & zero exact overlap verification
    print("\n" + "=" * 70)
    print("Canonical Splits Verification (D-1)")
    print("=" * 70)

    if technique_samples_loaded:
        train_s = [s for s in technique_samples_loaded if s.split == "train"]
        val_s = [s for s in technique_samples_loaded if s.split == "val"]
        test_s = [s for s in technique_samples_loaded if s.split == "test"]

        raw_splits = {"train": train_s, "val": val_s, "test": test_s}
        canonical_splits = enforce_cluster_isolation(
            raw_splits, threshold=0.8, n_gram=5, seed=42
        )

        # Check exact overlaps
        overlaps_val = check_exact_overlaps(
            canonical_splits["train"], canonical_splits["val"]
        )
        overlaps_test = check_exact_overlaps(
            canonical_splits["train"], canonical_splits["test"]
        )

        if overlaps_val or overlaps_test:
            print(
                f"[X] Exact overlap found! Train-Val: {len(overlaps_val)}, Train-Test: {len(overlaps_test)}"
            )
            has_failures = True
        else:
            print(
                "[✓] Zero exact overlap verified between canonical train and eval sets (val & test)."
            )

        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = generate_splits_manifest(
            canonical_splits, output_path=manifest_path, seed=42
        )
        print(f"[✓] Canonical splits manifest written to: {manifest_path.resolve()}")
        print(f"    - Sample counts: {manifest['sample_counts']}")
        print(f"    - Split SHA256: {manifest['split_sha256']}")
    else:
        print(
            "[!] Note: Technique samples were not loaded; skipping canonical splits generation."
        )

    print("\n" + "=" * 70)
    if has_failures:
        print("VERIFICATION FINISHED WITH ERRORS.")
        return 1
    print("VERIFICATION COMPLETED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
