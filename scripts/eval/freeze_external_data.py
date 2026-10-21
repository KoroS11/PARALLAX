"""Builds and freezes the external evaluation benchmark datasets (Phase B2).

Frozen datasets:
1. Lakera/gandalf_ignore_instructions (test split): n=112
2. leolee99/NotInject (all 3 subsets minus 30 dev.jsonl public_catalog exact matches): n=309
   Columns: id, text, subset, category, word_list
3. databricks/databricks-dolly-15k (stratified n=200 sample, seed 0):
   - data/external/dolly_sample.jsonl: row_index, category, prompt_hash (committed)
   - data/external/dolly_text.jsonl: row_index, category, prompt_hash, text (gitignored)

Outputs:
- data/external/FROZEN_EXTERNAL.sha256: SHA-256 manifest of the frozen committed artifacts
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


def build_gandalf(out_dir: Path) -> Path:
    pinned_rev = "04737b65e90a6794ec227012e4a255a7def6344b"
    ds = load_dataset(
        "Lakera/gandalf_ignore_instructions", split="test", revision=pinned_rev
    )

    out_file = out_dir / "gandalf.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            rec = {
                "id": f"gandalf_{idx:03d}",
                "text": row["text"].strip(),
                "similarity": float(row["similarity"]),
                "source": "Lakera/gandalf_ignore_instructions",
                "source_split": "test",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} rows to {out_file}")
    return out_file


def build_notinject(out_dir: Path, existing_texts: set[str]) -> Path:
    pinned_rev = "847ae76cf8fea5ed325429e569ae8cfef022d2e0"
    ds = load_dataset("leolee99/NotInject", revision=pinned_rev)

    subsets = ["NotInject_one", "NotInject_two", "NotInject_three"]
    out_file = out_dir / "notinject.jsonl"

    all_rows: list[dict[str, Any]] = []
    excluded_count = 0

    for s in subsets:
        for idx, row in enumerate(ds[s]):
            text = row["prompt"].strip()
            if text in existing_texts:
                excluded_count += 1
                continue
            all_rows.append(
                {
                    "id": f"notinject_{s}_{idx:03d}",
                    "text": text,
                    "subset": s,
                    "category": row["category"],
                    "word_list": row.get("word_list", []),
                    "source": "leolee99/NotInject",
                }
            )

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in all_rows)

    print(
        f"Wrote {len(all_rows)} rows to {out_file} (excluded {excluded_count} exact matches against catalog)"
    )
    return out_file


def build_dolly(
    out_dir: Path, existing_texts: set[str], seed: int = 0, sample_size: int = 200
) -> tuple[Path, Path]:
    pinned_rev = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
    ds = load_dataset(
        "databricks/databricks-dolly-15k", split="train", revision=pinned_rev
    )

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
    cats = sorted(candidates_by_cat.keys())
    proportions = {c: len(candidates_by_cat[c]) / total_candidates for c in cats}
    allocations = {c: int(proportions[c] * sample_size) for c in cats}
    remainder = sample_size - sum(allocations.values())
    remainders = sorted(
        cats, key=lambda c: proportions[c] * sample_size - allocations[c], reverse=True
    )
    for i in range(remainder):
        allocations[remainders[i]] += 1

    rng = random.Random(seed)
    sampled_rows: list[dict[str, Any]] = []
    for c in cats:
        cat_candidates = candidates_by_cat[c]
        sampled = rng.sample(cat_candidates, allocations[c])
        sampled.sort(key=lambda x: x["row_index"])
        sampled_rows.extend(sampled)

    sampled_rows.sort(key=lambda x: x["row_index"])

    sample_file = out_dir / "dolly_sample.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        for r in sampled_rows:
            line = {
                "row_index": r["row_index"],
                "category": r["category"],
                "prompt_hash": r["prompt_hash"],
            }
            f.write(json.dumps(line) + "\n")

    text_file = out_dir / "dolly_text.jsonl"
    with open(text_file, "w", encoding="utf-8") as f:
        for r in sampled_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled_rows)} rows to {sample_file} (IDs and hashes only)")
    print(f"Wrote {len(sampled_rows)} rows to {text_file} (gitignored local text)")
    return sample_file, text_file


def compute_frozen_manifest(out_dir: Path) -> Path:
    # Files to include in frozen manifest (committed files only)
    committed_files = [
        out_dir / "gandalf.jsonl",
        out_dir / "notinject.jsonl",
        out_dir / "dolly_sample.jsonl",
    ]

    manifest_file = out_dir / "FROZEN_EXTERNAL.sha256"
    lines = []
    for p in committed_files:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{sha}  {p.name}")

    manifest_content = "\n".join(lines) + "\n"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(manifest_content)

    print(f"\nWrote manifest to {manifest_file}:")
    print(manifest_content)
    return manifest_file


def main() -> None:
    out_dir = Path("data/external")
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_texts = get_existing_split_texts()
    print(f"Existing unique catalog texts: {len(existing_texts)}")

    build_gandalf(out_dir)
    build_notinject(out_dir, existing_texts)
    build_dolly(out_dir, existing_texts, seed=0, sample_size=200)
    compute_frozen_manifest(out_dir)


if __name__ == "__main__":
    main()
