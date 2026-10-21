"""Data schema utilities, normalization helpers, and validation for normalized datasets.

Follows the DatasetSample contract specified in DESIGN.md Section 6 and
parallax.contracts.data.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from parallax.contracts.data import DatasetSample

# Canonical multi-class technique categories (8 classes including 'none')
CANONICAL_TECHNIQUES: tuple[str, ...] = (
    "none",
    "keyword_override",
    "persona_play",
    "encoding",
    "payload_splitting",
    "context_overflow",
    "few_shot_poisoning",
    "multilingual",
)

TECHNIQUE_TO_ID: dict[str, int] = {
    tech: i for i, tech in enumerate(CANONICAL_TECHNIQUES)
}
ID_TO_TECHNIQUE: dict[int, str] = {
    i: tech for i, tech in enumerate(CANONICAL_TECHNIQUES)
}

VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


def normalize_split_name(split: str) -> str:
    """Normalize split naming variations to canonical 'train', 'val', or 'test'.

    Args:
        split: Raw split identifier (e.g. 'validation', 'dev', 'training', 'val').

    Returns:
        One of 'train', 'val', or 'test'.

    Raises:
        ValueError: If split cannot be mapped to canonical names.
    """
    clean = split.strip().lower()
    if clean in ("train", "training"):
        return "train"
    if clean in ("val", "validation", "dev", "valid"):
        return "val"
    if clean in ("test", "testing", "eval", "evaluation"):
        return "test"
    raise ValueError(
        f"Unknown split name '{split}'. Must map to 'train', 'val', or 'test'."
    )


def normalize_text(text: str) -> str:
    """Normalize text whitespace and strip leading/trailing whitespace.
    Applies Unicode NFKC normalization.

    Args:
        text: Input prompt text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKC", text).strip()


def create_dataset_sample(
    text: str,
    label_multiclass: str,
    label_binary: bool,
    source: str,
    split: str,
    metadata: dict[str, Any] | None = None,
) -> DatasetSample:
    """Factory helper to construct and validate a DatasetSample.

    Normalizes text and split name before creating the Pydantic model.
    """
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        raise ValueError("Dataset sample text must not be empty.")

    normalized_split = normalize_split_name(split)
    clean_multiclass = str(label_multiclass).strip().lower()
    clean_binary = bool(label_binary)

    return DatasetSample(
        text=cleaned_text,
        label_multiclass=clean_multiclass,
        label_binary=clean_binary,
        source=str(source).strip(),
        split=normalized_split,
        metadata=dict(metadata) if metadata is not None else {},
    )


def validate_sample(sample: DatasetSample | dict[str, Any]) -> DatasetSample:
    """Validate a DatasetSample instance or dictionary.

    Args:
        sample: DatasetSample instance or dict matching the schema.

    Returns:
        Validated DatasetSample instance.
    """
    if isinstance(sample, DatasetSample):
        if sample.split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{sample.split}'. Must be one of {VALID_SPLITS}"
            )
        return sample
    if isinstance(sample, dict):
        return create_dataset_sample(
            text=sample["text"],
            label_multiclass=sample["label_multiclass"],
            label_binary=sample["label_binary"],
            source=sample["source"],
            split=sample["split"],
            metadata=sample.get("metadata", {}),
        )
    raise TypeError(f"Expected DatasetSample or dict, got {type(sample).__name__}")


def sample_to_dict(sample: DatasetSample) -> dict[str, Any]:
    """Convert DatasetSample to dictionary representation."""
    return sample.model_dump()


def sample_from_dict(d: dict[str, Any]) -> DatasetSample:
    """Convert dictionary to validated DatasetSample."""
    return validate_sample(d)


def samples_to_dataframe(samples: Sequence[DatasetSample]) -> pd.DataFrame:
    """Convert a sequence of DatasetSample objects to a pandas DataFrame."""
    records = [sample.model_dump() for sample in samples]
    return pd.DataFrame.from_records(records)


def dataframe_to_samples(df: pd.DataFrame) -> list[DatasetSample]:
    """Convert a pandas DataFrame back into a list of DatasetSample objects."""
    samples: list[DatasetSample] = []
    for _, row in df.iterrows():
        sample = create_dataset_sample(
            text=str(row["text"]),
            label_multiclass=str(row["label_multiclass"]),
            label_binary=bool(row["label_binary"]),
            source=str(row["source"]),
            split=str(row["split"]),
            metadata=row.get("metadata")
            if isinstance(row.get("metadata"), dict)
            else {},
        )
        samples.append(sample)
    return samples


def summarize_samples(samples: Iterable[DatasetSample]) -> dict[str, Any]:
    """Calculate summary statistics for an iterable of DatasetSample objects.

    Returns:
        Dictionary with total count, split distribution, binary distribution,
        and multi-class technique distribution.
    """
    total = 0
    split_counts: dict[str, int] = {}
    binary_counts: dict[str, int] = {"benign": 0, "attack": 0}
    multiclass_counts: dict[str, int] = {}
    sources: set[str] = set()

    for s in samples:
        total += 1
        split_counts[s.split] = split_counts.get(s.split, 0) + 1
        if s.label_binary:
            binary_counts["attack"] += 1
        else:
            binary_counts["benign"] += 1
        multiclass_counts[s.label_multiclass] = (
            multiclass_counts.get(s.label_multiclass, 0) + 1
        )
        sources.add(s.source)

    return {
        "total_samples": total,
        "split_counts": split_counts,
        "binary_counts": binary_counts,
        "multiclass_counts": multiclass_counts,
        "unique_sources": sorted(sources),
    }
