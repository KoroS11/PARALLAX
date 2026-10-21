"""Dataset loaders for HuggingFace and local benchmark datasets.

All loaders yield/return normalized DatasetSample objects (text, label_multiclass,
label_binary, source, split, metadata) conforming to the frozen contract.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub.errors import GatedRepoError

from parallax.contracts.data import DatasetSample
from parallax.data.schema import (
    CANONICAL_TECHNIQUES,
    create_dataset_sample,
    normalize_split_name,
)

logger = logging.getLogger(__name__)

# Repository IDs on Hugging Face
TECHNIQUE_DATASET_ID = "neuralchemy/prompt-injection-technique"
BINARY_DATASET_ID = "neuralchemy/Prompt-injection-dataset"
MINDGARD_DATASET_ID = "Mindgard/evaded-prompt-injection-and-jailbreak-samples"
SIMSONSUN_DATASET_ID = "Simsonsun/JailbreakPrompts"
NOTINJECT_DATASET_ID = "leolee99/NotInject"
PROMPTGUARD2_DATASET_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class GatedDatasetAccessError(RuntimeError):
    """Raised when access to a gated dataset is denied or credentials are missing."""


class DatasetLoadError(RuntimeError):
    """Raised when loading a dataset fails."""


# ---------------------------------------------------------------------------
# Loader: neuralchemy/prompt-injection-technique (Multi-class + binary)
# ---------------------------------------------------------------------------
def load_technique_samples(
    split: str | None = None,
    limit: int | None = None,
    dataset_name_or_path: str = TECHNIQUE_DATASET_ID,
    hf_dataset: Dataset | DatasetDict | None = None,
    token: str | None = None,
) -> list[DatasetSample]:
    """Load normalized samples from neuralchemy/prompt-injection-technique.

    Features:
        text (str): Prompt text.
        technique (str): Name of technique (e.g. 'keyword_override', 'none').
        technique_label (int): Integer label (0..7).
        source (str): Synthetic or origin source.

    Args:
        split: Canonical split ('train', 'val', 'test') or None for all.
        limit: Max number of samples to return per split (or total).
        dataset_name_or_path: HF repo id or local path.
        hf_dataset: Pre-loaded Dataset or DatasetDict for testing/mocking.
        token: HuggingFace access token.

    Returns:
        List of normalized DatasetSample objects.
    """
    samples: list[DatasetSample] = []
    splits_to_load = (
        [normalize_split_name(split)] if split else ["train", "val", "test"]
    )

    for cur_split in splits_to_load:
        hf_split_name = "validation" if cur_split == "val" else cur_split
        if hf_dataset is not None:
            data = (
                hf_dataset[hf_split_name]
                if isinstance(hf_dataset, DatasetDict)
                else hf_dataset
            )
        else:
            try:
                data = load_dataset(
                    dataset_name_or_path, split=hf_split_name, token=token
                )  # nosec B615
            except Exception as e:
                raise DatasetLoadError(
                    f"Failed to load {dataset_name_or_path} split {cur_split}: {e}"
                ) from e

        count = 0
        for row in data:
            if limit is not None and count >= limit:
                break

            text = row.get("text", "")
            if not text or not str(text).strip():
                continue

            technique = str(row.get("technique", "none")).strip().lower()
            if technique not in CANONICAL_TECHNIQUES:
                technique = "none"

            # technique 'none' or label 0 is benign (label_binary = False)
            technique_label = row.get("technique_label", 0)
            is_attack = technique != "none" and technique_label != 0

            sample = create_dataset_sample(
                text=str(text),
                label_multiclass=technique,
                label_binary=is_attack,
                source=TECHNIQUE_DATASET_ID,
                split=cur_split,
                metadata={
                    "technique_label": technique_label,
                    "original_source": row.get("source", "neuralchemy"),
                },
            )
            samples.append(sample)
            count += 1

    return samples


# ---------------------------------------------------------------------------
# Loader: neuralchemy/Prompt-injection-dataset (Binary baseline)
# ---------------------------------------------------------------------------
def load_binary_samples(
    split: str | None = None,
    limit: int | None = None,
    dataset_name_or_path: str = BINARY_DATASET_ID,
    hf_dataset: Dataset | DatasetDict | None = None,
    token: str | None = None,
) -> list[DatasetSample]:
    """Load normalized samples from neuralchemy/Prompt-injection-dataset.

    Features:
        text (str): Prompt text.
        label (int): 0 for benign, 1 for injection.
        category (str): Attack category or 'benign'.

    Args:
        split: Canonical split ('train', 'val', 'test') or None for all.
        limit: Max samples to return.
        dataset_name_or_path: HF repo id or local path.
        hf_dataset: Pre-loaded Dataset or DatasetDict for testing/mocking.
        token: HuggingFace access token.

    Returns:
        List of normalized DatasetSample objects.
    """
    samples: list[DatasetSample] = []
    splits_to_load = (
        [normalize_split_name(split)] if split else ["train", "val", "test"]
    )

    for cur_split in splits_to_load:
        hf_split_name = "validation" if cur_split == "val" else cur_split
        if hf_dataset is not None:
            data = (
                hf_dataset[hf_split_name]
                if isinstance(hf_dataset, DatasetDict)
                else hf_dataset
            )
        else:
            try:
                data = load_dataset(
                    dataset_name_or_path, split=hf_split_name, token=token
                )  # nosec B615
            except Exception as e:
                raise DatasetLoadError(
                    f"Failed to load {dataset_name_or_path} split {cur_split}: {e}"
                ) from e

        count = 0
        for row in data:
            if limit is not None and count >= limit:
                break

            text = row.get("text", "")
            if not text or not str(text).strip():
                continue

            raw_label = row.get("label", 0)
            is_attack = bool(raw_label == 1 or raw_label == "1")
            category = (
                str(
                    row.get("category", "none" if not is_attack else "prompt_injection")
                )
                .strip()
                .lower()
            )
            label_multiclass = "none" if not is_attack else category

            metadata: dict[str, Any] = {
                "category": category,
                "original_source": row.get("source", ""),
                "severity": row.get("severity", ""),
                "group_id": row.get("group_id", ""),
                "augmented": row.get("augmented", False),
                "tags": row.get("tags", []),
            }

            sample = create_dataset_sample(
                text=str(text),
                label_multiclass=label_multiclass,
                label_binary=is_attack,
                source=BINARY_DATASET_ID,
                split=cur_split,
                metadata=metadata,
            )
            samples.append(sample)
            count += 1

    return samples


def load_fixture_samples(
    fixture_name_or_dataset_id: str,
    split: str = "test",
) -> list[DatasetSample]:
    """Load normalized samples from tiny local offline fixtures.

    Provides offline fallbacks for gated datasets (Mindgard, Prompt Guard 2)
    to ensure unit tests and local runs never crash.
    """
    canonical_split = normalize_split_name(split)
    target = fixture_name_or_dataset_id.lower().strip()
    if "mindgard" in target:
        path = FIXTURES_DIR / "mindgard_fixture.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        samples: list[DatasetSample] = []
        for row in data:
            samples.append(
                create_dataset_sample(
                    text=row["modified_prompt"],
                    label_multiclass=row["attack_name"],
                    label_binary=True,
                    source=MINDGARD_DATASET_ID,
                    split=canonical_split,
                    metadata={
                        "attack_name": row["attack_name"],
                        "original_prompt": row.get("original_prompt", ""),
                        "is_evaded": row.get("evaded", True),
                    },
                )
            )
        return samples

    if any(
        k in target
        for k in ("promptguard", "prompt-guard", "prompt_guard", "meta-llama")
    ):
        path = FIXTURES_DIR / "promptguard2_fixture.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = []
        for row in data:
            raw_label = row.get("label", 0)
            is_attack = bool(raw_label == 1 or raw_label == "1")
            tech = str(
                row.get("technique") or ("keyword_override" if is_attack else "none")
            )
            samples.append(
                create_dataset_sample(
                    text=row["text"],
                    label_multiclass=tech,
                    label_binary=is_attack,
                    source=PROMPTGUARD2_DATASET_ID,
                    split=canonical_split,
                    metadata={"technique": tech},
                )
            )
        return samples
    return []


# ---------------------------------------------------------------------------
# Loader: Mindgard/evaded-prompt-injection-and-jailbreak-samples (Gated)
# ---------------------------------------------------------------------------
def load_mindgard_samples(
    limit: int | None = None,
    dataset_name_or_path: str = MINDGARD_DATASET_ID,
    hf_dataset: Dataset | DatasetDict | None = None,
    token: str | None = None,
    allow_missing: bool = True,
    use_fixture: bool = False,
    split: str = "test",
) -> list[DatasetSample]:
    """Load normalized samples from Mindgard/evaded-prompt-injection-and-jailbreak-samples.

    Handles gated access gracefully: if the token is missing or access is rejected,
    returns an empty list when allow_missing=True with a logged warning, or raises
    GatedDatasetAccessError when allow_missing=False. If use_fixture=True, loads
    from local offline fixture without network access.

    Features:
        modified_prompt / prompt: Attack prompt text.
        attack_name: Specific evasion technique name.
        original_prompt: Baseline prompt before perturbation.

    Args:
        limit: Max samples to load.
        dataset_name_or_path: HF repo id or local path.
        hf_dataset: Pre-loaded Dataset or DatasetDict for testing/mocking.
        token: HuggingFace access token.
        allow_missing: If True, returns [] on access error; if False, raises error.
        use_fixture: If True, loads local fixture without external API calls.
        split: Default split to assign (defaults to 'test' for OOD evaluation).

    Returns:
        List of normalized DatasetSample objects.
    """
    if use_fixture:
        fixture_samples = load_fixture_samples("mindgard", split=split)
        return fixture_samples[:limit] if limit is not None else fixture_samples

    samples: list[DatasetSample] = []
    canonical_split = normalize_split_name(split)

    if hf_dataset is not None:
        data = (
            hf_dataset["train"]
            if isinstance(hf_dataset, DatasetDict) and "train" in hf_dataset
            else hf_dataset
        )
        if isinstance(data, DatasetDict):
            data = data[next(iter(data.keys()))]
    else:
        try:
            data = load_dataset(dataset_name_or_path, token=token)  # nosec B615
            if isinstance(data, DatasetDict):
                data = data[next(iter(data.keys()))]
        except (GatedRepoError, PermissionError) as e:
            msg = (
                f"Access to gated dataset '{dataset_name_or_path}' is restricted. "
                f"Request access on HuggingFace and set HF_TOKEN. Details: {e}"
            )
            logger.warning(msg)
            if not allow_missing:
                raise GatedDatasetAccessError(msg) from e
            return []
        except Exception as e:
            err_str = str(e).lower()
            if (
                "gated" in err_str
                or "403" in err_str
                or "401" in err_str
                or "authenticated" in err_str
            ):
                msg = (
                    f"Access to gated dataset '{dataset_name_or_path}' requires authentication. "
                    f"Set HF_TOKEN or request access on Hugging Face. Details: {e}"
                )
                logger.warning(msg)
                if not allow_missing:
                    raise GatedDatasetAccessError(msg) from e
                return []
            raise DatasetLoadError(f"Failed to load {dataset_name_or_path}: {e}") from e

    count = 0
    for row in data:
        if limit is not None and count >= limit:
            break

        # Modified prompt contains the evasive attack payload
        text = row.get("modified_prompt") or row.get("prompt") or row.get("text", "")
        if not text or not str(text).strip():
            continue

        attack_name = str(
            row.get("attack_name") or row.get("technique") or "evasion_attack"
        ).strip()

        sample = create_dataset_sample(
            text=str(text),
            label_multiclass=attack_name,
            label_binary=True,  # All samples in Mindgard are evasion attacks/jailbreaks
            source=MINDGARD_DATASET_ID,
            split=canonical_split,
            metadata={
                "attack_name": attack_name,
                "original_prompt": row.get("original_prompt", ""),
                "is_evaded": row.get("evaded", True),
            },
        )
        samples.append(sample)
        count += 1

    return samples


# ---------------------------------------------------------------------------
# Loader: Simsonsun/JailbreakPrompts (Independent jailbreaks)
# ---------------------------------------------------------------------------
def load_simsonsun_jailbreaks(
    limit: int | None = None,
    dataset_name_or_path: str = SIMSONSUN_DATASET_ID,
    hf_dataset: Dataset | DatasetDict | None = None,
    token: str | None = None,
    split: str = "test",
) -> list[DatasetSample]:
    """Load normalized jailbreak samples from Simsonsun/JailbreakPrompts.

    Features:
        Prompt: The adversarial jailbreak text.
        Source: Origin source label (e.g. 'In-the-Wild', 'Nannazi').

    Args:
        limit: Max samples to load.
        dataset_name_or_path: HF repo id or local path.
        hf_dataset: Pre-loaded Dataset or DatasetDict for testing/mocking.
        token: HuggingFace access token.
        split: Default split to assign (defaults to 'test' for OOD evaluation).

    Returns:
        List of normalized DatasetSample objects.
    """
    samples: list[DatasetSample] = []
    canonical_split = normalize_split_name(split)

    if hf_dataset is not None:
        ds_dict = (
            hf_dataset
            if isinstance(hf_dataset, DatasetDict)
            else {"default": hf_dataset}
        )
    else:
        try:
            ds_dict = load_dataset(dataset_name_or_path, token=token)  # nosec B615
            if not isinstance(ds_dict, DatasetDict):
                ds_dict = {"default": ds_dict}
        except Exception as e:
            raise DatasetLoadError(f"Failed to load {dataset_name_or_path}: {e}") from e

    count = 0
    for split_key, partition in ds_dict.items():
        for row in partition:
            if limit is not None and count >= limit:
                return samples

            text = row.get("Prompt") or row.get("prompt") or row.get("text", "")
            if not text or not str(text).strip():
                continue

            source_tag = str(row.get("Source", "simsonsun_jailbreak")).strip()

            sample = create_dataset_sample(
                text=str(text),
                label_multiclass="jailbreak",
                label_binary=True,  # All prompts in Simsonsun are jailbreak attacks
                source=SIMSONSUN_DATASET_ID,
                split=canonical_split,
                metadata={
                    "jailbreak_source": source_tag,
                    "hf_partition": split_key,
                },
            )
            samples.append(sample)
            count += 1

    return samples


# ---------------------------------------------------------------------------
# Loader: NotInject (False Positive Rate / Over-defense check)
# ---------------------------------------------------------------------------
def load_notinject_samples(
    limit: int | None = None,
    dataset_name_or_path: str = NOTINJECT_DATASET_ID,
    hf_dataset: Dataset | DatasetDict | None = None,
    token: str | None = None,
    split: str = "test",
) -> list[DatasetSample]:
    """Load normalized benign samples from NotInject (leolee99/NotInject).

    Designed to evaluate False Positive Rate (FPR) / over-defense where prompts
    look suspicious or discuss security/translation without containing real attacks.

    Features:
        prompt: Benign text.
        word_list: List of trigger words present.
        category: Query category.

    Args:
        limit: Max samples to load.
        dataset_name_or_path: HF repo id or local path.
        hf_dataset: Pre-loaded Dataset or DatasetDict for testing/mocking.
        token: HuggingFace access token.
        split: Default split to assign (defaults to 'test' for E3 evaluation).

    Returns:
        List of normalized DatasetSample objects (all label_binary=False, label_multiclass='none').
    """
    samples: list[DatasetSample] = []
    canonical_split = normalize_split_name(split)

    if hf_dataset is not None:
        ds_dict = (
            hf_dataset
            if isinstance(hf_dataset, DatasetDict)
            else {"default": hf_dataset}
        )
    else:
        try:
            ds_dict = load_dataset(dataset_name_or_path, token=token)  # nosec B615
            if not isinstance(ds_dict, DatasetDict):
                ds_dict = {"default": ds_dict}
        except Exception as e:
            raise DatasetLoadError(f"Failed to load {dataset_name_or_path}: {e}") from e

    count = 0
    for split_key, partition in ds_dict.items():
        for row in partition:
            if limit is not None and count >= limit:
                return samples

            text = row.get("prompt") or row.get("text", "")
            if not text or not str(text).strip():
                continue

            category = str(row.get("category", "benign")).strip()
            word_list = row.get("word_list", [])

            sample = create_dataset_sample(
                text=str(text),
                label_multiclass="none",  # Benign control sample
                label_binary=False,  # False positive benchmark control
                source=NOTINJECT_DATASET_ID,
                split=canonical_split,
                metadata={
                    "category": category,
                    "word_list": word_list,
                    "hf_partition": split_key,
                },
            )
            samples.append(sample)
            count += 1

    return samples


# ---------------------------------------------------------------------------
# Loader: meta-llama/Llama-Prompt-Guard-2-86M (Gated Benchmark)
# ---------------------------------------------------------------------------
def load_promptguard2_samples(
    limit: int | None = None,
    dataset_name_or_path: str = PROMPTGUARD2_DATASET_ID,
    hf_dataset: Dataset | DatasetDict | None = None,
    token: str | None = None,
    allow_missing: bool = True,
    use_fixture: bool = False,
    split: str = "test",
) -> list[DatasetSample]:
    """Load normalized samples from meta-llama/Llama-Prompt-Guard-2-86M benchmark.

    Handles gated access gracefully: if token is missing or access is rejected,
    returns an empty list when allow_missing=True with a logged warning, or raises
    GatedDatasetAccessError when allow_missing=False.
    If use_fixture=True, loads tiny local fixture offline.

    Args:
        limit: Max samples to load.
        dataset_name_or_path: HF repo id or local path.
        hf_dataset: Pre-loaded Dataset or DatasetDict for testing/mocking.
        token: HuggingFace access token.
        allow_missing: If True, returns [] on access error; if False, raises error.
        use_fixture: If True, loads local fixture without external API calls.
        split: Default split to assign (defaults to 'test').

    Returns:
        List of normalized DatasetSample objects.
    """
    if use_fixture:
        fixture_samples = load_fixture_samples("promptguard2", split=split)
        return fixture_samples[:limit] if limit is not None else fixture_samples

    canonical_split = normalize_split_name(split)
    if hf_dataset is not None:
        data = (
            hf_dataset["test"]
            if isinstance(hf_dataset, DatasetDict) and "test" in hf_dataset
            else (
                hf_dataset["train"]
                if isinstance(hf_dataset, DatasetDict) and "train" in hf_dataset
                else hf_dataset
            )
        )
        if isinstance(data, DatasetDict):
            data = data[next(iter(data.keys()))]
    else:
        try:
            data = load_dataset(dataset_name_or_path, token=token)  # nosec B615
            if isinstance(data, DatasetDict):
                data = data[next(iter(data.keys()))]
        except (GatedRepoError, PermissionError) as e:
            msg = (
                f"Access to gated model/dataset '{dataset_name_or_path}' is restricted. "
                f"Request access on Hugging Face (https://huggingface.co/{dataset_name_or_path}) "
                f"and set the HF_TOKEN environment variable. Details: {e}"
            )
            logger.warning(msg)
            if not allow_missing:
                raise GatedDatasetAccessError(msg) from e
            return []
        except Exception as e:
            err_str = str(e).lower()
            if any(
                k in err_str
                for k in ("gated", "403", "401", "authenticated", "restricted")
            ):
                msg = (
                    f"Access to gated model/dataset '{dataset_name_or_path}' requires authentication. "
                    f"Set HF_TOKEN or request access on Hugging Face (https://huggingface.co/{dataset_name_or_path}). "
                    f"Details: {e}"
                )
                logger.warning(msg)
                if not allow_missing:
                    raise GatedDatasetAccessError(msg) from e
                return []
            raise DatasetLoadError(f"Failed to load {dataset_name_or_path}: {e}") from e

    samples: list[DatasetSample] = []
    count = 0
    for row in data:
        if limit is not None and count >= limit:
            break
        text = row.get("text") or row.get("prompt") or ""
        if not text or not str(text).strip():
            continue
        raw_label = row.get("label", 0)
        is_attack = bool(
            raw_label == 1
            or raw_label == "1"
            or str(raw_label).upper() in ("MALICIOUS", "ATTACK", "INJECTION")
        )
        technique = (
            str(row.get("technique") or ("keyword_override" if is_attack else "none"))
            .strip()
            .lower()
        )
        if technique not in CANONICAL_TECHNIQUES:
            technique = "keyword_override" if is_attack else "none"

        sample = create_dataset_sample(
            text=str(text),
            label_multiclass=technique,
            label_binary=is_attack,
            source=PROMPTGUARD2_DATASET_ID,
            split=canonical_split,
            metadata={"original_label": raw_label},
        )
        samples.append(sample)
        count += 1
    return samples


# ---------------------------------------------------------------------------
# Unified Registry Loader
# ---------------------------------------------------------------------------
LOADER_REGISTRY: dict[str, Callable[..., list[DatasetSample]]] = {
    TECHNIQUE_DATASET_ID: load_technique_samples,
    "technique": load_technique_samples,
    BINARY_DATASET_ID: load_binary_samples,
    "binary": load_binary_samples,
    MINDGARD_DATASET_ID: load_mindgard_samples,
    "mindgard": load_mindgard_samples,
    SIMSONSUN_DATASET_ID: load_simsonsun_jailbreaks,
    "simsonsun": load_simsonsun_jailbreaks,
    NOTINJECT_DATASET_ID: load_notinject_samples,
    "notinject": load_notinject_samples,
    PROMPTGUARD2_DATASET_ID: load_promptguard2_samples,
    "promptguard2": load_promptguard2_samples,
    "prompt-guard-2": load_promptguard2_samples,
    "prompt_guard_2": load_promptguard2_samples,
}


def load_dataset_samples(
    name: str,
    split: str | None = None,
    limit: int | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> list[DatasetSample]:
    """Unified entry point to load samples by dataset identifier or short name.

    Args:
        name: Name or HF identifier of dataset.
        split: Split to load ('train', 'val', 'test', or None for all).
        limit: Max samples to load.
        token: HuggingFace access token.
        **kwargs: Additional parameters passed to the specific loader.

    Returns:
        List of normalized DatasetSample objects.
    """
    key = name.strip()
    if key in LOADER_REGISTRY:
        loader = LOADER_REGISTRY[key]
        if key in (
            MINDGARD_DATASET_ID,
            "mindgard",
            SIMSONSUN_DATASET_ID,
            "simsonsun",
            NOTINJECT_DATASET_ID,
            "notinject",
            PROMPTGUARD2_DATASET_ID,
            "promptguard2",
            "prompt-guard-2",
            "prompt_guard_2",
        ):
            # These sets are primarily evaluation / OOD test sets
            target_split = split or "test"
            return loader(limit=limit, token=token, split=target_split, **kwargs)
        return loader(split=split, limit=limit, token=token, **kwargs)

    raise ValueError(
        f"Unknown dataset name '{name}'. Available loaders: {sorted(LOADER_REGISTRY.keys())}"
    )
