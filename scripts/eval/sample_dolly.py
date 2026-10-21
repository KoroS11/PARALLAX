"""Dolly-15k Stratified Sampling Script (Phase B2).

Selects a deterministic stratified sample of n=200 prompts across documented categories
from databricks/databricks-dolly-15k (pinned revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a)
under random seed 0 per docs/EXTERNAL_EVAL_PROTOCOL.md.

Outputs:
- data/external/dolly_sample.jsonl: row_index, category, prompt_hash (committed to git)
- data/external/dolly_text.jsonl: row_index, category, prompt_hash, text (gitignored)
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


def get_existing_split_texts() -> set[str]:
    texts = set()
    for p in Path("data/splits").glob("*.jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                texts.add(r["text"].strip())
    return texts


def build_dolly_sample(seed: int = 0, sample_size: int = 200) -> list[dict[str, Any]]:
    pinned_rev = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
    ds = load_dataset(
        "databricks/databricks-dolly-15k", split="train", revision=pinned_rev
    )

    existing_texts = get_existing_split_texts()

    # 1. Extraction and length cap <= 2000 chars
    candidates_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_len = 0
    excluded_dedup = 0

    for idx, row in enumerate(ds):
        inst = (row.get("instruction") or "").strip()
        ctx = (row.get("context") or "").strip()
        prompt = inst if not ctx else f"{inst}\n\n{ctx}"

        if len(prompt) > 2000:
            excluded_len += 1
            continue
        if prompt in existing_texts:
            excluded_dedup += 1
            continue

        cat = row["category"]
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        candidates_by_cat[cat].append(
            {
                "row_index": idx,
                "category": cat,
                "prompt_hash": prompt_hash,
                "text": prompt,
            }
        )

    total_candidates = sum(len(v) for v in candidates_by_cat.values())
    print(f"Total Dolly raw rows: {len(ds)}")
    print(f"Excluded >2000 chars: {excluded_len}")
    print(f"Excluded exact catalog duplicate: {excluded_dedup}")
    print(f"Remaining candidate pool: {total_candidates}")

    # 2. Largest-remainder stratified proportional allocation
    cats = sorted(candidates_by_cat.keys())
    proportions = {c: len(candidates_by_cat[c]) / total_candidates for c in cats}
    allocations = {c: int(proportions[c] * sample_size) for c in cats}
    remainder = sample_size - sum(allocations.values())
    remainders = sorted(
        cats, key=lambda c: proportions[c] * sample_size - allocations[c], reverse=True
    )
    for i in range(remainder):
        allocations[remainders[i]] += 1

    print("\nDeterministic Category Allocation (Hamilton Proportional Stratification):")
    for c in cats:
        print(
            f"  {c:25s}: {allocations[c]:2d} rows (pool: {len(candidates_by_cat[c]):5d}, {proportions[c] * 100:.2f}%)"
        )
    print(f"  Total Allocation: {sum(allocations.values())} rows")

    # 3. Deterministic sampling within sorted categories using single RNG stream (seed 0)
    # Candidates in each category maintain natural row_index ascending order
    rng = random.Random(seed)
    sampled_rows: list[dict[str, Any]] = []

    for c in cats:
        cat_candidates = candidates_by_cat[c]
        sampled = rng.sample(cat_candidates, allocations[c])
        # Sort by row_index for clean deterministic ordering
        sampled.sort(key=lambda x: x["row_index"])
        sampled_rows.extend(sampled)

    sampled_rows.sort(key=lambda x: x["row_index"])
    return sampled_rows


def main() -> None:
    sample = build_dolly_sample(seed=0, sample_size=200)

    out_dir = Path("data/external")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. dolly_sample.jsonl: IDs and hashes only (committed to git)
    sample_file = out_dir / "dolly_sample.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        for r in sample:
            line = {
                "row_index": r["row_index"],
                "category": r["category"],
                "prompt_hash": r["prompt_hash"],
            }
            f.write(json.dumps(line) + "\n")
    print(f"\nWrote {len(sample)} rows to {sample_file}")

    # 2. dolly_text.jsonl: full text for local scoring (gitignored)
    text_file = out_dir / "dolly_text.jsonl"
    with open(text_file, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(sample)} rows to {text_file} (to be gitignored)")


if __name__ == "__main__":
    main()
