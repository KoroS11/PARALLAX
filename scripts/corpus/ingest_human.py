"""Human Batch Ingestion and Author-Clustered Partitioning Engine.

PARALLAX Wave H.3 Task T1.a:
Reads data/human_submissions/*.csv (columns: author, class, family, scenario, text, notes).
Validates schema and class parity per author.
Deduplicates (exact normalized match and char 3-gram cosine >= 0.90 across all classes and splits).
Applies neutral domain masking via eval/mask.py.
Partitions records into splits strictly by AUTHOR-HASHED cluster so that held_out_v4
receives whole authors' submissions untouched.

Zero-Touch Invariant:
Held-out submissions must never be read, scored, or summarized prior to pre-registered evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mask import mask_prompt_text

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HUMAN_SUBMISSIONS_DIR = REPO_ROOT / "data" / "human_submissions"

REQUIRED_COLUMNS = {"author", "class", "family", "scenario", "text", "notes"}
VALID_CLASSES = {"attack", "benign"}


def get_author_split(author: str) -> str:
    """Deterministically assign an author to a split using cryptographic hashing.

    Whole authors are allocated exclusively to one split to ensure that held_out_v4
    receives submissions from authors unseen in dev or val.
    """
    digest = hashlib.sha256(author.strip().lower().encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket < 4:
        return "dev"
    elif bucket < 7:
        return "val"
    else:
        return "held_out_v4"


def normalize_text(text: str) -> str:
    """Normalize text for exact deduplication."""
    return " ".join(text.strip().lower().split())


def load_and_validate_human_submissions(
    submissions_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Ingest, validate, and deduplicate human submissions from CSV files."""
    sdir = submissions_dir or HUMAN_SUBMISSIONS_DIR
    if not sdir.exists():
        logger.warning("Human submissions directory not found: %s", sdir)
        return []

    csv_files = sorted(sdir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", sdir)
        return []

    raw_records: list[dict[str, Any]] = []
    author_classes: dict[str, set[str]] = {}

    for fpath in csv_files:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(
                set(reader.fieldnames)
            ):
                raise ValueError(
                    f"CSV file {fpath} missing required columns {REQUIRED_COLUMNS}. Found: {reader.fieldnames}"
                )

            for line_idx, row in enumerate(reader, start=2):
                author = row["author"].strip()
                cls = row["class"].strip().lower()
                family = row["family"].strip()
                scenario = row["scenario"].strip()
                text = row["text"].strip()
                notes = row.get("notes", "").strip()

                if not author or not text:
                    continue

                if cls not in VALID_CLASSES:
                    raise ValueError(
                        f"Invalid class '{cls}' at line {line_idx} in {fpath}. Must be 'attack' or 'benign'."
                    )

                author_classes.setdefault(author, set()).add(cls)

                raw_records.append(
                    {
                        "author": author,
                        "class": cls,
                        "label_binary": (cls == "attack"),
                        "family": family,
                        "scenario": scenario,
                        "raw_text": text,
                        "notes": notes,
                        "source_file": fpath.name,
                    }
                )

    # Verify rule: Every author must submit BOTH classes
    for author, classes in author_classes.items():
        if len(classes) < 2:
            raise ValueError(
                f"Author '{author}' submitted only classes {classes}. "
                f"Rule: Every human author must write BOTH 'attack' and 'benign' prompts."
            )

    logger.info(
        "Loaded %d raw human submissions from %d authors across %d CSV files.",
        len(raw_records),
        len(author_classes),
        len(csv_files),
    )

    # 1. Exact text deduplication
    seen_exact: dict[str, dict[str, Any]] = {}
    exact_deduped: list[dict[str, Any]] = []
    for r in raw_records:
        norm = normalize_text(r["raw_text"])
        if norm in seen_exact:
            prev = seen_exact[norm]
            if prev["label_binary"] != r["label_binary"]:
                raise ValueError(
                    f"Label conflict in human submissions: text from {r['author']} ({r['class']}) "
                    f"matches text from {prev['author']} ({prev['class']}): '{norm[:60]}...'"
                )
            continue
        seen_exact[norm] = r
        exact_deduped.append(r)

    # 2. Near-duplicate deduplication (char 3-gram cosine >= 0.90)
    if len(exact_deduped) > 1:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
        X = vec.fit_transform([r["raw_text"] for r in exact_deduped])
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 0.0)

        drop_indices = set()
        for i in range(len(exact_deduped)):
            if i in drop_indices:
                continue
            high_sim = np.where(sim[i] >= 0.90)[0]
            for j in high_sim:
                if j > i and j not in drop_indices:
                    # Check for cross-label conflict
                    if (
                        exact_deduped[i]["label_binary"]
                        != exact_deduped[j]["label_binary"]
                    ):
                        raise ValueError(
                            f"Cross-label near-duplicate in human submissions: "
                            f"{exact_deduped[i]['author']} ({exact_deduped[i]['class']}) vs "
                            f"{exact_deduped[j]['author']} ({exact_deduped[j]['class']})"
                        )
                    drop_indices.add(j)

        filtered_records = [
            r for idx, r in enumerate(exact_deduped) if idx not in drop_indices
        ]
        logger.info(
            "Dropped %d near-duplicates (char 3-gram cosine >= 0.90). Remaining: %d",
            len(drop_indices),
            len(filtered_records),
        )
    else:
        filtered_records = exact_deduped

    # 3. Apply neutral masking and author-hashed split clustering
    processed: list[dict[str, Any]] = []
    for idx, r in enumerate(filtered_records, start=1):
        masked = mask_prompt_text(r["raw_text"])
        split = get_author_split(r["author"])
        template_id = f"tmpl_human_{r['author']}_{r['family']}_{idx:03d}"

        # Classify benign family
        benign_family = "none"
        if not r["label_binary"]:
            text_lower = masked.lower()
            if any(
                w in text_lower
                for w in [
                    "override",
                    "ignore",
                    "bypass",
                    "system",
                    "emergency",
                    "disregard",
                ]
            ):
                benign_family = "hard"
            else:
                benign_family = "easy"

        processed.append(
            {
                "id": f"human-{split}-{idx:04d}",
                "text": masked,
                "label_binary": r["label_binary"],
                "label_multiclass": r["family"],
                "technique": r["family"] if r["label_binary"] else "none",
                "family": r["family"]
                if r["label_binary"]
                else f"benign_{benign_family}",
                "split": split,
                "source": "human_submission",
                "source_tool": "human_batch",
                "author": r["author"],
                "author_split": split,
                "scenario": r["scenario"],
                "template_id": template_id,
                "benign_family": benign_family,
                "notes": r["notes"],
                "metadata": {
                    "goal": {
                        "canary": "",
                        "forbidden_tools": [],
                        "target_action": "safe_query"
                        if not r["label_binary"]
                        else "injection",
                    },
                    "author": r["author"],
                    "scenario": r["scenario"],
                    "source": "human_submission",
                    "benign_family": benign_family,
                    "d3_in_distribution": False,
                },
                "provenance": {
                    "source_tool": "human_batch",
                    "source_file": r["source_file"],
                    "license": "MIT",
                    "author": r["author"],
                },
            }
        )

    split_counts = {
        "dev": sum(1 for p in processed if p["split"] == "dev"),
        "val": sum(1 for p in processed if p["split"] == "val"),
        "held_out_v4": sum(1 for p in processed if p["split"] == "held_out_v4"),
    }
    logger.info("Human batch ingestion complete: %s", split_counts)
    return processed


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    records = load_and_validate_human_submissions()
    print(f"Loaded {len(records)} human submission records.")
    for sp in ["dev", "val", "held_out_v4"]:
        sp_recs = [r for r in records if r["split"] == sp]
        atk = sum(1 for r in sp_recs if r["label_binary"])
        print(
            f"  Split '{sp}': {len(sp_recs)} total ({atk} attacks, {len(sp_recs) - atk} benign)"
        )
