#!/usr/bin/env python3
"""Compute exact SHA-256 hashes and synchronize across all closeout docs (K12c Item 1).

Every hash is computed directly via hashlib from the file on disk, never typed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MANIFEST_PATH = (
    REPO_ROOT / "eval" / "logs" / "manifests" / "final-v4-20261019-142005-8fb16e.json"
)
PREDICTIONS_PATH = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "held_out_v4.jsonl"


def get_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main():
    manifest_sha = get_hash(MANIFEST_PATH)
    pred_sha = get_hash(PREDICTIONS_PATH)
    split_sha = get_hash(SPLIT_PATH)

    print(f"[update_hashes] Manifest SHA-256:    {manifest_sha}")
    print(f"[update_hashes] Predictions SHA-256: {pred_sha}")
    print(f"[update_hashes] Split SHA-256:       {split_sha}")

    # Prior erroneous manifest hash to replace
    erroneous_manifest_hashes = [
        "d5e72719246d84aa7d4b0f9bf28135beea7842c070c0c6e0821c17ee1fa30a7b",
        "d5e72719246d",
    ]

    # Target documentation files
    docs_to_check = [
        REPO_ROOT / "docs" / "PROVENANCE.md",
        REPO_ROOT / "docs" / "STATUS.md",
        REPO_ROOT / "docs" / "REVIEW_PACKAGE.md",
        REPO_ROOT / "docs" / "SAFE_CLAIMS.md",
        REPO_ROOT / "HANDOFF_TEAMMATES.md",
        REPO_ROOT / "report" / "REPORT_SKELETON.md",
        REPO_ROOT / "results" / "final-v4" / "README.md",
    ]

    for doc in docs_to_check:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        orig_text = text
        for err_h in erroneous_manifest_hashes:
            if err_h in text:
                print(f"[update_hashes] Replacing {err_h} in {doc.name}")
                text = text.replace(err_h, manifest_sha)

        if text != orig_text:
            doc.write_text(text, encoding="utf-8")
            print(f"[update_hashes] Updated {doc.name}")


if __name__ == "__main__":
    main()
