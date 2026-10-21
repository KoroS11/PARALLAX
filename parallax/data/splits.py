"""Dataset splitting, deduplication, and contamination check utilities.

Provides:
- Partition management and verification
- Normalized-hash exact deduplication within and across splits
- MinHash LSH near-duplicate clustering and isolation (5-gram Jaccard >= 0.8)
- Deterministic seeded canonical split generation
- Split manifest generation recording counts, distributions, and SHA256 hashes
- Comprehensive contamination reporting for train/val/test splits
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from parallax.contracts.data import DatasetSample
from parallax.data.schema import (
    normalize_split_name,
)


# ---------------------------------------------------------------------------
# Normalization & Normalized Hash Helpers
# ---------------------------------------------------------------------------
def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for deduplication: NFKC unicode normalization, lowercase, strip, and collapse whitespace."""
    if not isinstance(text, str):
        text = str(text)
    norm = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", norm)


def compute_normalized_hash(text: str) -> str:
    """Compute SHA-256 hash of dedup-normalized text."""
    norm = normalize_text_for_dedup(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def deduplicate_samples(
    samples: Sequence[DatasetSample],
) -> list[DatasetSample]:
    """Deduplicate a sequence of DatasetSamples by normalized text hash, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[DatasetSample] = []
    for s in samples:
        h = compute_normalized_hash(s.text)
        if h not in seen:
            seen.add(h)
            unique.append(s)
    return unique


def deduplicate_by_normalized_hash(
    splits: dict[str, Sequence[DatasetSample]],
) -> dict[str, list[DatasetSample]]:
    """Deduplicate samples within splits and across splits using normalized SHA256 hashes.

    Precedence rules:
    - Within each split, duplicate normalized hashes are dropped (first occurrence kept).
    - Cross-split precedence: test > val > train.
    - Duplicates that appear in test are dropped from val and train.
    - Duplicates that appear in val are dropped from train.

    Args:
        splits: Dictionary mapping split names ('train', 'val', 'test', etc.) to samples.

    Returns:
        Dictionary with deduplicated sample lists per split.
    """
    norm_splits: dict[str, Sequence[DatasetSample]] = {}
    for k, v in splits.items():
        norm_key = normalize_split_name(k)
        norm_splits[norm_key] = v

    # Deduplicate test first (highest precedence)
    seen_test: set[str] = set()
    dedup_test: list[DatasetSample] = []
    for s in norm_splits.get("test", []):
        h = compute_normalized_hash(s.text)
        if h not in seen_test:
            seen_test.add(h)
            dedup_test.append(s.model_copy(update={"split": "test"}))

    # Deduplicate val (cannot share hash with test)
    seen_val: set[str] = set()
    dedup_val: list[DatasetSample] = []
    for s in norm_splits.get("val", []):
        h = compute_normalized_hash(s.text)
        if h not in seen_test and h not in seen_val:
            seen_val.add(h)
            dedup_val.append(s.model_copy(update={"split": "val"}))

    # Deduplicate train (cannot share hash with test or val)
    seen_train: set[str] = set()
    dedup_train: list[DatasetSample] = []
    for s in norm_splits.get("train", []):
        h = compute_normalized_hash(s.text)
        if h not in seen_test and h not in seen_val and h not in seen_train:
            seen_train.add(h)
            dedup_train.append(s.model_copy(update={"split": "train"}))

    result: dict[str, list[DatasetSample]] = {}
    if "train" in norm_splits:
        result["train"] = dedup_train
    if "val" in norm_splits:
        result["val"] = dedup_val
    if "test" in norm_splits:
        result["test"] = dedup_test

    # Any custom/additional splits
    for k, v in norm_splits.items():
        if k not in ("train", "val", "test"):
            seen_k: set[str] = set()
            dedup_k: list[DatasetSample] = []
            for s in v:
                h = compute_normalized_hash(s.text)
                if h not in seen_test and h not in seen_val and h not in seen_k:
                    seen_k.add(h)
                    dedup_k.append(s.model_copy(update={"split": k}))
            result[k] = dedup_k

    return result


# ---------------------------------------------------------------------------
# Split Management & Partition Verification
# ---------------------------------------------------------------------------
def split_samples(
    samples: Sequence[DatasetSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify: bool = True,
    seed: int = 42,
) -> dict[str, list[DatasetSample]]:
    """Partition a sequence of DatasetSample objects into train, val, and test.

    Args:
        samples: Sequence of input DatasetSample objects.
        train_ratio: Proportion of training data (e.g., 0.8).
        val_ratio: Proportion of validation data (e.g., 0.1).
        test_ratio: Proportion of test data (e.g., 0.1).
        stratify: Whether to preserve class proportions across splits.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to list of DatasetSample.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    rng = random.Random(seed)

    if not stratify:
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set = [
            s.model_copy(update={"split": "train"}) for s in shuffled[:n_train]
        ]
        val_set = [
            s.model_copy(update={"split": "val"})
            for s in shuffled[n_train : n_train + n_val]
        ]
        test_set = [
            s.model_copy(update={"split": "test"}) for s in shuffled[n_train + n_val :]
        ]
        return {"train": train_set, "val": val_set, "test": test_set}

    # Stratified split by label_multiclass
    class_groups: dict[str, list[DatasetSample]] = {}
    for s in samples:
        class_groups.setdefault(s.label_multiclass, []).append(s)

    train_out: list[DatasetSample] = []
    val_out: list[DatasetSample] = []
    test_out: list[DatasetSample] = []

    for group in class_groups.values():
        shuffled_group = list(group)
        rng.shuffle(shuffled_group)
        k = len(shuffled_group)
        k_train = round(k * train_ratio)
        k_val = round(k * val_ratio)
        # Ensure we do not exceed total k
        k_train = min(k_train, k)
        k_val = min(k_val, max(0, k - k_train))

        train_out.extend(
            s.model_copy(update={"split": "train"}) for s in shuffled_group[:k_train]
        )
        val_out.extend(
            s.model_copy(update={"split": "val"})
            for s in shuffled_group[k_train : k_train + k_val]
        )
        test_out.extend(
            s.model_copy(update={"split": "test"})
            for s in shuffled_group[k_train + k_val :]
        )

    return {"train": train_out, "val": val_out, "test": test_out}


def verify_partition_integrity(
    splits: dict[str, Sequence[DatasetSample]],
) -> dict[str, Any]:
    """Verify that dataset partitions are valid, non-empty, and conform to contracts.

    Checks:
    - Splits have allowed names ('train', 'val', 'test').
    - No empty prompt text.
    - Consistency between sample.split attribute and partition key.
    - Returns distribution counts per partition.
    """
    results: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "partition_sizes": {},
        "class_distributions": {},
    }

    for split_key, partition in splits.items():
        norm_key = normalize_split_name(split_key)
        results["partition_sizes"][norm_key] = len(partition)
        class_counts: dict[str, int] = {}

        for idx, sample in enumerate(partition):
            if not sample.text.strip():
                results["valid"] = False
                results["errors"].append(
                    f"Empty text in {split_key} sample index {idx}"
                )

            if sample.split != norm_key:
                results["valid"] = False
                results["errors"].append(
                    f"Sample at {split_key}[{idx}] has mismatched split field '{sample.split}' (expected '{norm_key}')"
                )

            class_counts[sample.label_multiclass] = (
                class_counts.get(sample.label_multiclass, 0) + 1
            )

        results["class_distributions"][norm_key] = class_counts

    return results


# ---------------------------------------------------------------------------
# Contamination & Overlap Checking (Exact + n-gram Jaccard)
# ---------------------------------------------------------------------------
def tokenize_words(text: str) -> list[str]:
    """Lowercase and extract alphanumeric word tokens with unicode normalization."""
    norm = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"\w+", norm, re.UNICODE)


def get_ngrams(tokens: Sequence[str], n: int = 3) -> set[tuple[str, ...]]:
    """Extract set of n-grams from a token sequence."""
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def get_char_ngrams(text: str, n: int = 4) -> set[str]:
    """Extract character n-grams from text for fine-grained similarity."""
    cleaned = "".join(text.lower().split())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def jaccard_similarity(set_a: set[Any], set_b: set[Any]) -> float:
    """Compute standard Jaccard similarity between two sets: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a.intersection(set_b))
    union_len = len(set_a.union(set_b))
    return intersection_len / union_len if union_len > 0 else 0.0


def compute_ngram_jaccard(
    text1: str, text2: str, n: int = 3, use_words: bool = True
) -> float:
    """Compute n-gram Jaccard similarity between two texts.

    Args:
        text1: First text.
        text2: Second text.
        n: n-gram length (default 3 for words, 4 for characters).
        use_words: If True, uses word n-grams; if False, uses character n-grams.

    Returns:
        Float similarity score in [0.0, 1.0].
    """
    t1 = normalize_text_for_dedup(text1)
    t2 = normalize_text_for_dedup(text2)
    if t1 == t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    if use_words:
        tokens1 = tokenize_words(t1)
        tokens2 = tokenize_words(t2)
        if not tokens1 or not tokens2:
            return 0.0
        # If tokens are fewer than n, compare word sets directly
        if len(tokens1) < n or len(tokens2) < n:
            return jaccard_similarity(set(tokens1), set(tokens2))
        return jaccard_similarity(get_ngrams(tokens1, n=n), get_ngrams(tokens2, n=n))
    return jaccard_similarity(get_char_ngrams(t1, n=n), get_char_ngrams(t2, n=n))


def check_exact_overlaps(
    train_samples: Sequence[DatasetSample],
    eval_samples: Sequence[DatasetSample],
) -> list[dict[str, Any]]:
    """Find exact string matches between train and evaluation splits using normalized hashes.

    Returns:
        List of dict records for each overlap:
        {'train_index': int, 'eval_index': int, 'text': str, 'train_label': str, 'eval_label': str}
    """
    train_lookup: dict[str, list[int]] = {}
    for idx, s in enumerate(train_samples):
        key = normalize_text_for_dedup(s.text)
        train_lookup.setdefault(key, []).append(idx)

    overlaps: list[dict[str, Any]] = []
    for eval_idx, eval_s in enumerate(eval_samples):
        key = normalize_text_for_dedup(eval_s.text)
        if key in train_lookup:
            for train_idx in train_lookup[key]:
                train_s = train_samples[train_idx]
                overlaps.append(
                    {
                        "train_index": train_idx,
                        "eval_index": eval_idx,
                        "text": eval_s.text,
                        "train_label": train_s.label_multiclass,
                        "eval_label": eval_s.label_multiclass,
                        "train_binary": train_s.label_binary,
                        "eval_binary": eval_s.label_binary,
                    }
                )

    return overlaps


# ---------------------------------------------------------------------------
# MinHash & LSH Near-Duplicate Detection
# ---------------------------------------------------------------------------
class MinHashIndexer:
    """MinHash and Locality-Sensitive Hashing (LSH) index for fast near-duplicate search.

    Uses 64-bit integer hash seeds with universal hashing: (a * hash + b) % prime.
    """

    _PRIME: int = 4294967311  # Large 32-bit prime

    def __init__(
        self,
        num_perm: int = 64,
        threshold: float = 0.8,
        n_gram: int = 1,
        seed: int = 42,
    ) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self.n_gram = n_gram
        rng = random.Random(seed)
        # Generate random linear hash coefficients (a, b)
        self.a = [rng.randint(1, self._PRIME - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self._PRIME - 1) for _ in range(num_perm)]

        # Determine bands and rows for LSH: threshold ~ (1/b)^(1/r)
        self.num_bands, self.rows_per_band = self._find_lsh_bands(num_perm, threshold)
        self.buckets: list[dict[int, list[int]]] = [{} for _ in range(self.num_bands)]
        self.signatures: list[list[int]] = []
        self.texts: list[str] = []

    def _find_lsh_bands(self, num_perm: int, threshold: float) -> tuple[int, int]:
        """Find band count b and rows r such that b * r == num_perm and (1/b)^(1/r) ~ threshold."""
        best_b, best_r = 1, num_perm
        min_diff = float("inf")
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                t = (1.0 / b) ** (1.0 / r)
                diff = abs(t - threshold)
                if diff < min_diff:
                    min_diff = diff
                    best_b, best_r = b, r
        return best_b, best_r

    def compute_signature(self, text: str) -> list[int]:
        """Compute the MinHash signature of a text string."""
        tokens = tokenize_words(text)
        if len(tokens) >= self.n_gram:
            ngrams = get_ngrams(tokens, self.n_gram)
        elif tokens:
            ngrams = {(w,) for w in tokens}
        else:
            ngrams = {tuple(c) for c in get_char_ngrams(text, n=3)}

        if not ngrams:
            return [0] * self.num_perm

        # Hash each token to 32-bit int
        token_hashes = [
            int(
                hashlib.md5(
                    " ".join(gram).encode("utf-8"), usedforsecurity=False
                ).hexdigest()[:8],
                16,
            )
            for gram in ngrams
        ]

        sig = [self._PRIME] * self.num_perm
        for h in token_hashes:
            for i in range(self.num_perm):
                val = (self.a[i] * h + self.b[i]) % self._PRIME
                sig[i] = min(sig[i], val)
        return sig

    def index_samples(self, samples: Sequence[DatasetSample]) -> None:
        """Build MinHash LSH index for given samples."""
        self.signatures = []
        self.texts = []
        self.buckets = [{} for _ in range(self.num_bands)]

        for idx, sample in enumerate(samples):
            sig = self.compute_signature(sample.text)
            self.signatures.append(sig)
            self.texts.append(sample.text)

            # Insert into LSH buckets
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                band_subsig = tuple(sig[start : start + self.rows_per_band])
                bucket_key = hash(band_subsig)
                self.buckets[band_idx].setdefault(bucket_key, []).append(idx)

    def estimate_jaccard(self, sig1: list[int], sig2: list[int]) -> float:
        """Estimate Jaccard similarity as fraction of matching signature positions."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / self.num_perm

    def query_near_duplicates(
        self,
        query_text: str,
        threshold: float | None = None,
    ) -> list[tuple[int, float]]:
        """Find indexed candidate items that exceed near-duplicate similarity threshold.

        Returns:
            List of (indexed_sample_id, estimated_similarity).
        """
        target_thresh = threshold if threshold is not None else self.threshold
        query_sig = self.compute_signature(query_text)
        candidates: set[int] = set()

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band_subsig = tuple(query_sig[start : start + self.rows_per_band])
            bucket_key = hash(band_subsig)
            if bucket_key in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][bucket_key])

        results: list[tuple[int, float]] = []
        for cand_idx in candidates:
            cand_sig = self.signatures[cand_idx]
            sim = self.estimate_jaccard(query_sig, cand_sig)
            if sim >= target_thresh:
                results.append((cand_idx, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


def check_near_duplicates(
    train_samples: Sequence[DatasetSample],
    eval_samples: Sequence[DatasetSample],
    threshold: float = 0.8,
    num_perm: int = 64,
    n_gram: int = 1,
) -> list[dict[str, Any]]:
    """Scan eval_samples against train_samples using MinHash LSH for near-duplicates.

    Returns:
        List of match records with train_index, eval_index, similarity, and texts.
    """
    indexer = MinHashIndexer(num_perm=num_perm, threshold=threshold, n_gram=n_gram)
    indexer.index_samples(train_samples)

    matches: list[dict[str, Any]] = []
    for eval_idx, eval_s in enumerate(eval_samples):
        cand_matches = indexer.query_near_duplicates(eval_s.text, threshold=threshold)
        for train_idx, sim in cand_matches:
            # Confirm with exact n-gram Jaccard
            confirmed_sim = compute_ngram_jaccard(
                eval_s.text, train_samples[train_idx].text, n=n_gram
            )
            if confirmed_sim >= threshold or sim >= threshold:
                matches.append(
                    {
                        "train_index": train_idx,
                        "eval_index": eval_idx,
                        "estimated_sim": round(sim, 3),
                        "ngram_jaccard": round(confirmed_sim, 3),
                        "eval_text": eval_s.text,
                        "train_text": train_samples[train_idx].text,
                        "train_label": train_samples[train_idx].label_multiclass,
                        "eval_label": eval_s.label_multiclass,
                    }
                )
    return matches


# ---------------------------------------------------------------------------
# Near-Duplicate Clustering & Cross-Split Isolation (D-1)
# ---------------------------------------------------------------------------
def cluster_samples(
    samples: Sequence[DatasetSample],
    threshold: float = 0.8,
    n_gram: int = 5,
    seed: int = 42,
) -> list[list[int]]:
    """Cluster samples into connected components of near-duplicates (n-gram Jaccard >= threshold).

    Uses MinHash LSH for candidate pairing, then verifies each pair with exact n-gram Jaccard.
    Forms an undirected graph where an edge connects any two samples with Jaccard >= threshold,
    and returns connected components as clusters of sample indices.

    Args:
        samples: Sequence of DatasetSample objects.
        threshold: Jaccard similarity threshold for an edge (default 0.8).
        n_gram: Size of n-grams (default 5).
        seed: Random seed for MinHash indexing.

    Returns:
        List of clusters, where each cluster is a list of integer indices in samples.
    """
    n = len(samples)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Index all samples with MinHash
    candidate_pairs: set[tuple[int, int]] = set()
    if n <= 500:
        # Check all pairs for small sets to guarantee 100% precision and recall
        for i in range(n):
            for j in range(i + 1, n):
                candidate_pairs.add((i, j))
    else:
        lsh_thresh = max(0.4, threshold - 0.15)
        indexer = MinHashIndexer(
            num_perm=64, threshold=lsh_thresh, n_gram=n_gram, seed=seed
        )
        indexer.index_samples(samples)
        for bucket_dict in indexer.buckets:
            for idx_list in bucket_dict.values():
                if len(idx_list) > 1:
                    for i in range(len(idx_list)):
                        for j in range(i + 1, len(idx_list)):
                            u, v = idx_list[i], idx_list[j]
                            if u > v:
                                u, v = v, u
                            candidate_pairs.add((u, v))

    # Verify candidate pairs with exact 5-gram Jaccard
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in candidate_pairs:
        sim = compute_ngram_jaccard(samples[u].text, samples[v].text, n=n_gram)
        if sim >= threshold:
            adj[u].append(v)
            adj[v].append(u)

    # Find connected components (BFS)
    visited: set[int] = set()
    clusters: list[list[int]] = []

    for i in range(n):
        if i not in visited:
            component = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            component.sort()
            clusters.append(component)

    return clusters


def enforce_cluster_isolation(
    splits: dict[str, Sequence[DatasetSample]],
    threshold: float = 0.8,
    n_gram: int = 5,
    seed: int = 42,
) -> dict[str, list[DatasetSample]]:
    """Ensure clusters of near-duplicate prompts (5-gram Jaccard >= threshold) NEVER straddle splits.

    Rules:
    1. Exact normalized-hash deduplication: within-split duplicates are removed;
       cross-split exact duplicates are dropped with precedence: test > val > train.
    2. Near-duplicate clustering: all samples across splits are clustered using
       MinHash LSH and n-gram Jaccard >= threshold.
    3. Cross-split cluster resolution: If a cluster contains samples in multiple splits:
       - Cross-split duplicates are ALWAYS dropped from train.
       - If a cluster spans test and val, samples are dropped from val (test takes precedence).
       - Therefore, no near-duplicate cluster straddles across splits.

    Args:
        splits: Dictionary mapping split names (e.g. 'train', 'val', 'test') to samples.
        threshold: Jaccard similarity threshold for near-duplicates (default 0.8).
        n_gram: Word shingle size (default 5).
        seed: Random seed for deterministic indexing.

    Returns:
        New splits dictionary where no near-duplicate cluster straddles any split boundary.
    """
    # Step 1: Exact deduplication with test > val > train precedence
    deduped_splits = deduplicate_by_normalized_hash(splits)

    # Step 2: Pool all samples with tracking of origin (split, index)
    all_samples: list[DatasetSample] = []
    sample_origin: list[tuple[str, int]] = []
    ordered_splits = ["test", "val", "train"] + [
        k for k in deduped_splits if k not in ("test", "val", "train")
    ]
    for s_name in ordered_splits:
        if s_name in deduped_splits:
            for idx, sample in enumerate(deduped_splits[s_name]):
                all_samples.append(sample)
                sample_origin.append((s_name, idx))

    # Step 3: Cluster all samples
    clusters = cluster_samples(
        all_samples, threshold=threshold, n_gram=n_gram, seed=seed
    )

    # Step 4: Detect cross-split clusters and drop lower precedence samples
    dropped_origins: set[tuple[str, int]] = set()
    for cluster in clusters:
        cluster_splits = {sample_origin[i][0] for i in cluster}
        if len(cluster_splits) > 1:
            # Cross-split straddling detected!
            # Precedence: test > val > train
            if "test" in cluster_splits:
                for i in cluster:
                    s_name, idx = sample_origin[i]
                    if s_name != "test":
                        dropped_origins.add((s_name, idx))
            elif "val" in cluster_splits:
                for i in cluster:
                    s_name, idx = sample_origin[i]
                    if s_name != "val":
                        dropped_origins.add((s_name, idx))
            else:
                winner = min(cluster_splits)
                for i in cluster:
                    s_name, idx = sample_origin[i]
                    if s_name != winner:
                        dropped_origins.add((s_name, idx))

    # Step 5: Construct output splits
    output_splits: dict[str, list[DatasetSample]] = {}
    for s_name, samples_list in deduped_splits.items():
        output_splits[s_name] = [
            s
            for idx, s in enumerate(samples_list)
            if (s_name, idx) not in dropped_origins
        ]

    return output_splits


def generate_canonical_splits(
    samples: Sequence[DatasetSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify: bool = True,
    seed: int = 42,
    threshold: float = 0.8,
    n_gram: int = 5,
) -> dict[str, list[DatasetSample]]:
    """Deterministically generate canonical splits from a pool of samples.

    Guarantees:
    - Normalized hash deduplication (exact duplicates removed).
    - Near-duplicate clustering: entire clusters are assigned to a single split,
      ensuring clusters NEVER straddle across splits.
    - Deterministic and reproducible from seed.
    - Preserves class stratification by cluster.

    Args:
        samples: Sequence of DatasetSample objects.
        train_ratio: Target proportion for train split.
        val_ratio: Target proportion for val split.
        test_ratio: Target proportion for test split.
        stratify: Whether to preserve class proportions by cluster.
        seed: Random seed for deterministic generation.
        threshold: Jaccard similarity threshold for clustering (default 0.8).
        n_gram: Word shingle size (default 5).

    Returns:
        Dictionary mapping 'train', 'val', 'test' to list of DatasetSample.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    unique_samples = deduplicate_samples(samples)
    if not unique_samples:
        return {"train": [], "val": [], "test": []}

    clusters = cluster_samples(
        unique_samples, threshold=threshold, n_gram=n_gram, seed=seed
    )

    # For deterministic assignment, sort cluster indices
    cluster_items: list[tuple[str, list[DatasetSample]]] = []
    for c in clusters:
        c_samples = [unique_samples[i] for i in c]
        # Find majority label
        label_counts: dict[str, int] = {}
        for s in c_samples:
            label_counts[s.label_multiclass] = (
                label_counts.get(s.label_multiclass, 0) + 1
            )
        maj_label = min(label_counts.items(), key=lambda x: (-x[1], x[0]))[0]
        cluster_items.append((maj_label, c_samples))

    if not stratify:
        rng = random.Random(seed)
        shuffled = list(cluster_items)
        shuffled.sort(key=lambda item: compute_split_sha256(item[1]))
        rng.shuffle(shuffled)

        total_items = sum(len(c_s) for _, c_s in shuffled)
        target_train = round(total_items * train_ratio)
        target_val = round(total_items * val_ratio)

        train_out: list[DatasetSample] = []
        val_out: list[DatasetSample] = []
        test_out: list[DatasetSample] = []

        curr_train = 0
        curr_val = 0
        for _, c_samples in shuffled:
            sz = len(c_samples)
            if curr_train + sz <= target_train or (
                curr_train == 0 and target_train > 0
            ):
                train_out.extend(
                    s.model_copy(update={"split": "train"}) for s in c_samples
                )
                curr_train += sz
            elif curr_val + sz <= target_val or (curr_val == 0 and target_val > 0):
                val_out.extend(s.model_copy(update={"split": "val"}) for s in c_samples)
                curr_val += sz
            else:
                test_out.extend(
                    s.model_copy(update={"split": "test"}) for s in c_samples
                )

        return {"train": train_out, "val": val_out, "test": test_out}

    # Stratified by cluster majority label
    groups: dict[str, list[list[DatasetSample]]] = {}
    for maj_label, c_samples in cluster_items:
        groups.setdefault(maj_label, []).append(c_samples)

    train_out = []
    val_out = []
    test_out = []

    rng = random.Random(seed)
    for label in sorted(groups.keys()):
        c_list = groups[label]
        c_list.sort(key=lambda cs: compute_split_sha256(cs))
        rng.shuffle(c_list)

        n_group_samples = sum(len(cs) for cs in c_list)
        t_train = round(n_group_samples * train_ratio)
        t_val = round(n_group_samples * val_ratio)

        cur_tr = 0
        cur_vl = 0
        for cs in c_list:
            sz = len(cs)
            if (cur_tr + sz <= t_train and t_train > 0) or (
                cur_tr == 0 and t_train > 0
            ):
                train_out.extend(s.model_copy(update={"split": "train"}) for s in cs)
                cur_tr += sz
            elif (cur_vl + sz <= t_val and t_val > 0) or (cur_vl == 0 and t_val > 0):
                val_out.extend(s.model_copy(update={"split": "val"}) for s in cs)
                cur_vl += sz
            else:
                test_out.extend(s.model_copy(update={"split": "test"}) for s in cs)

    return {"train": train_out, "val": val_out, "test": test_out}


# ---------------------------------------------------------------------------
# Split Manifest Generation
# ---------------------------------------------------------------------------
def compute_split_sha256(samples: Sequence[DatasetSample]) -> str:
    """Compute deterministic SHA256 hash for a split of DatasetSamples.

    Sorts samples deterministically by (normalized_text, label_multiclass, label_binary, source)
    and computes SHA256 of concatenated canonical records.
    """
    hasher = hashlib.sha256()
    sorted_samples = sorted(
        samples,
        key=lambda s: (
            normalize_text_for_dedup(s.text),
            s.label_multiclass,
            s.label_binary,
            s.source,
        ),
    )
    for s in sorted_samples:
        line = f"{normalize_text_for_dedup(s.text)}|{s.label_multiclass}|{int(s.label_binary)}|{s.source}\n"
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def generate_splits_manifest(
    splits: dict[str, Sequence[DatasetSample]],
    output_path: str | Path | None = "data/manifests/canonical_splits_manifest.json",
    seed: int = 42,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate manifest recording sample counts, class distributions, and SHA256 hashes of splits.

    Args:
        splits: Dictionary mapping split names ('train', 'val', 'test') to sequences of DatasetSample.
        output_path: Optional path to write manifest JSON file (e.g. data/manifests/canonical_splits_manifest.json).
        seed: Random seed used to generate splits.
        metadata: Optional extra metadata dictionary.

    Returns:
        Dictionary of the generated manifest.
    """
    sample_counts: dict[str, int] = {}
    class_distributions: dict[str, dict[str, int]] = {}
    binary_distributions: dict[str, dict[str, int]] = {}
    split_sha256: dict[str, str] = {}
    all_samples: list[DatasetSample] = []

    for split_name in sorted(splits.keys()):
        partition = splits[split_name]
        sample_counts[split_name] = len(partition)
        mc_counts: dict[str, int] = {}
        bin_counts: dict[str, int] = {"benign": 0, "attack": 0}
        for s in partition:
            mc_counts[s.label_multiclass] = mc_counts.get(s.label_multiclass, 0) + 1
            if s.label_binary:
                bin_counts["attack"] += 1
            else:
                bin_counts["benign"] += 1
            all_samples.append(s)

        class_distributions[split_name] = dict(sorted(mc_counts.items()))
        binary_distributions[split_name] = bin_counts
        split_sha256[split_name] = compute_split_sha256(partition)

    overall_sha256 = compute_split_sha256(all_samples)
    total_samples = sum(sample_counts.values())

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "seed": seed,
        "total_samples": total_samples,
        "sample_counts": sample_counts,
        "class_distributions": class_distributions,
        "binary_distributions": binary_distributions,
        "split_sha256": split_sha256,
        "overall_sha256": overall_sha256,
        "dedup_policy": {
            "method": "MinHash LSH + 5-gram Jaccard",
            "threshold": 0.8,
            "precedence": "test > val > train",
            "normalization": "NFKC",
        },
    }
    if metadata:
        manifest["metadata"] = metadata

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return manifest


# ---------------------------------------------------------------------------
# Contamination Report Generator
# ---------------------------------------------------------------------------
def generate_contamination_report(
    splits: dict[str, Sequence[DatasetSample]],
    similarity_threshold: float = 0.85,
    n_gram: int = 1,
) -> dict[str, Any]:
    """Generate comprehensive overlap and contamination metrics across splits.

    Args:
        splits: Dictionary containing 'train', 'val', and optionally 'test' splits.
        similarity_threshold: Jaccard threshold for near-duplicate detection.
        n_gram: Word shingle size for near-duplicate calculation (default 1).

    Returns:
        Structured report dictionary with summary counts, exact overlaps,
        and near-duplicate records.
    """
    train = splits.get("train", [])
    val = splits.get("val", [])
    test = splits.get("test", [])

    report: dict[str, Any] = {
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "exact_matches_train_val": [],
        "exact_matches_train_test": [],
        "near_duplicates_train_val": [],
        "near_duplicates_train_test": [],
        "is_contaminated": False,
    }

    if train and val:
        report["exact_matches_train_val"] = check_exact_overlaps(train, val)
        report["near_duplicates_train_val"] = check_near_duplicates(
            train, val, threshold=similarity_threshold, n_gram=n_gram
        )

    if train and test:
        report["exact_matches_train_test"] = check_exact_overlaps(train, test)
        report["near_duplicates_train_test"] = check_near_duplicates(
            train, test, threshold=similarity_threshold, n_gram=n_gram
        )

    exact_count = len(report["exact_matches_train_val"]) + len(
        report["exact_matches_train_test"]
    )
    near_count = len(report["near_duplicates_train_val"]) + len(
        report["near_duplicates_train_test"]
    )

    report["total_exact_leaks"] = exact_count
    report["total_near_leaks"] = near_count
    report["is_contaminated"] = exact_count > 0 or near_count > 0

    return report


def format_contamination_report_markdown(report: dict[str, Any]) -> str:
    """Format the contamination report into human-readable Markdown."""
    lines: list[str] = [
        "# Dataset Overlap and Split Contamination Report",
        "",
        "## 1. Split Partition Counts",
        f"- **Train Samples:** {report['train_count']}",
        f"- **Validation Samples:** {report['val_count']}",
        f"- **Test Samples:** {report['test_count']}",
        "",
        "## 2. Contamination Findings",
        f"- **Exact Matches (Train vs Val):** {len(report['exact_matches_train_val'])}",
        f"- **Exact Matches (Train vs Test):** {len(report['exact_matches_train_test'])}",
        f"- **Near-Duplicates (Train vs Val):** {len(report['near_duplicates_train_val'])}",
        f"- **Near-Duplicates (Train vs Test):** {len(report['near_duplicates_train_test'])}",
        f"- **Contamination Status:** {'FLAGGED' if report['is_contaminated'] else 'CLEAN'}",
        "",
    ]

    if report["exact_matches_train_val"]:
        lines.append("### Exact Leaks (Train vs Val)")
        for item in report["exact_matches_train_val"][:5]:
            snippet = item["text"][:80].replace("\n", " ")
            lines.append(
                f'- Train[{item["train_index"]}] vs Val[{item["eval_index"]}]: "{snippet}..."'
            )
        lines.append("")

    if report["near_duplicates_train_val"]:
        lines.append("### Near-Duplicate Pairs (Train vs Val)")
        for item in report["near_duplicates_train_val"][:5]:
            t_snippet = item["train_text"][:60].replace("\n", " ")
            e_snippet = item["eval_text"][:60].replace("\n", " ")
            lines.append(
                f'- Sim {item["ngram_jaccard"]}: "{t_snippet}" vs "{e_snippet}"'
            )
        lines.append("")

    return "\n".join(lines)
