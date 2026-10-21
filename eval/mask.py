"""Prompt masking and synthetic marker neutralization engine for PARALLAX Wave H.

Implements the fundamental evaluation principle:
"The detector, judge, and router inputs must carry NO class-separating synthetic artifacts.
Attack SUCCESS is decided from sandbox metadata and agent traces, never from strings in the prompt."

Provides:
- `mask_prompt_text(text: str) -> str`: Replaces raw canary tokens, mock tool syntax,
  UUIDs, and harness markers with neutral domain placeholders.
- `contains_synthetic_markers(text: str) -> bool`: Detects raw synthetic marker patterns.
- `get_evaluation_text(record: dict, masked: bool = True) -> str`: Retrieves evaluation
  text from a JSONL record, masked by default, logging `input_masked=true`.
- `MaskedDetector`: Transparent wrapper around any Detector conforming to the Detector
  protocol, executing prediction on masked inputs and logging `input_masked=true`.
- `wrap_detector(detector, masked: bool = True)`: Factory helper for detector wrapping.
- `mask_dataset(records, masked: bool = True)`: Deep-copies and masks dataset records.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any

from parallax.contracts.detectors import Detector, DetectorResult

logger = logging.getLogger(__name__)

# Standard neutral domain placeholders
CANARY_PLACEHOLDER: str = "[CANARY_TOKEN]"
ACTION_PLACEHOLDER: str = "[AUTHORIZED_ACTION]"
HARNESS_PLACEHOLDER: str = "[TEST_HARNESS]"

# Regex patterns for raw synthetic markers
_CANARY_TOKEN_RE = re.compile(
    r"\bCANARY-[A-Za-z0-9_-]+\b|\bCANARY_(?!TOKEN\b)(?:HEX_)?[A-Za-z0-9_-]+\b",
    re.IGNORECASE,
)

_MOCK_CALL_RE = re.compile(
    r"\bmock_\w+\s*\([^)]*\)",
    re.IGNORECASE,
)

_MOCK_STANDALONE_RE = re.compile(
    r"\bmock_\w+\b",
    re.IGNORECASE,
)

_UUID_TOKEN_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

_ACTION_WRAPPER_RE = re.compile(
    r"\[Action:\s*(?:mock_|CANARY-)[^\]]+\]",
    re.IGNORECASE,
)

_HTML_HARNESS_COMMENT_RE = re.compile(
    r"<!--\s*(?:ADMIN_EXEC|TEST_HARNESS|SYNTHETIC_HARNESS|SYSTEM_OVERRIDE):\s*.*?-->",
    re.DOTALL | re.IGNORECASE,
)

_SYNTHETIC_PREFIX_RE = re.compile(
    r"\b(?:HARNESS|MOCK|SYNTHETIC)_[A-Za-z0-9_-]+\b",
    re.IGNORECASE,
)

_HARNESS_BRACKET_RE = re.compile(
    r"\[(?:TEST_HARNESS|SYNTHETIC_TEST)_[^\]]*\]|\[(?:TEST_HARNESS|SYNTHETIC_TEST)\s*:[^\]]*\]",
    re.IGNORECASE,
)

# Combined detection patterns to test for raw synthetic markers
_DETECTION_PATTERNS = [
    _CANARY_TOKEN_RE,
    _MOCK_CALL_RE,
    _MOCK_STANDALONE_RE,
    _ACTION_WRAPPER_RE,
    _HTML_HARNESS_COMMENT_RE,
    _SYNTHETIC_PREFIX_RE,
    _HARNESS_BRACKET_RE,
    _UUID_TOKEN_RE,
]


def contains_synthetic_markers(text: str) -> bool:
    """Check if the provided text contains any raw synthetic marker patterns.

    Returns True if raw CANARY-*, mock_*(), synthetic harness tags, or UUIDs
    are present. Does NOT match neutral placeholders like `[CANARY_TOKEN]`.
    """
    if not text:
        return False

    for pat in _DETECTION_PATTERNS:
        if pat.search(text):
            return True
    return False


def mask_prompt_text(text: str) -> str:
    """Neutralize synthetic markers in prompt text with domain placeholders.

    Replaces:
    - `CANARY-\\d+`, `CANARY-[A-Z0-9_-]+`, and `CANARY_<hex>` with `[CANARY_TOKEN]`
    - `mock_\\w+\\(.*?\\)` call syntax with `[AUTHORIZED_ACTION]`
    - Standalone `mock_\\w+` identifiers with `[AUTHORIZED_ACTION]`
    - Standard UUID tokens with `[CANARY_TOKEN]`
    - Action/comment wrapper syntax like `[Action: mock_...]` or `<!-- ADMIN_EXEC: ... -->`

    Preserves legitimate sentence semantics, words (e.g. 'canary', 'mockery'), and SCADA syntax.
    """
    if not text:
        return text

    masked = text

    # 1. Action wrapper syntax e.g. [Action: mock_trip_breaker() CANARY-1099]
    # Neutralize to [Action: [AUTHORIZED_ACTION]]
    masked = _ACTION_WRAPPER_RE.sub(f"[Action: {ACTION_PLACEHOLDER}]", masked)

    # 2. HTML-style test harness / admin comments
    masked = _HTML_HARNESS_COMMENT_RE.sub(f"<!-- {ACTION_PLACEHOLDER} -->", masked)

    # 3. Explicit harness bracket tokens
    masked = _HARNESS_BRACKET_RE.sub(HARNESS_PLACEHOLDER, masked)

    # 4. Mock tool call syntax with parameters e.g. mock_trip_breaker() or mock_set_chlorine(ppm=999)
    masked = _MOCK_CALL_RE.sub(ACTION_PLACEHOLDER, masked)

    # 5. Standalone mock tool identifiers e.g. "mock_emergency_vent"
    masked = _MOCK_STANDALONE_RE.sub(ACTION_PLACEHOLDER, masked)

    # 6. Canary tokens e.g. CANARY-1039, CANARY-EXT-001, CANARY_HEX_...
    masked = _CANARY_TOKEN_RE.sub(CANARY_PLACEHOLDER, masked)

    # 7. UUID-like tokens
    masked = _UUID_TOKEN_RE.sub(CANARY_PLACEHOLDER, masked)

    # 8. Synthetic prefix constants e.g. HARNESS_ID_001
    masked = _SYNTHETIC_PREFIX_RE.sub(HARNESS_PLACEHOLDER, masked)

    return masked


def get_evaluation_text(record: dict[str, Any], masked: bool = True) -> str:
    """Retrieve evaluation prompt text from a JSONL record, masked by default.

    Logs `input_masked=true` when masked=True, or `input_masked=false` when masked=False.
    Supports either 'text' or 'prompt' keys in the record.
    """
    raw_text = str(record.get("text") or record.get("prompt", ""))
    sample_id = str(record.get("id", "unknown"))

    if masked:
        logger.info("input_masked=true sample_id=%s", sample_id)
        return mask_prompt_text(raw_text)

    logger.info("input_masked=false sample_id=%s", sample_id)
    return raw_text


def mask_dataset(
    records: list[dict[str, Any]], masked: bool = True
) -> list[dict[str, Any]]:
    """Return a deep copy of dataset records with text masked according to `masked` setting."""
    masked_records: list[dict[str, Any]] = []
    for r in records:
        rec_copy = copy.deepcopy(r)
        rec_copy["text"] = get_evaluation_text(rec_copy, masked=masked)
        rec_copy["input_masked"] = masked
        masked_records.append(rec_copy)
    return masked_records


class MaskedDetector:
    """Transparent proxy wrapper for any Detector protocol implementation.

    Intercepts calls to `predict` and `predict_sync`, applying `mask_prompt_text`
    by default and logging `input_masked=true`.
    """

    def __init__(self, detector: Detector, masked: bool = True) -> None:
        self.detector = detector
        self.masked = masked
        self.name: str = getattr(detector, "name", "unknown_detector")

    async def predict(self, text: str) -> DetectorResult:
        """Run prediction on masked (or unmasked) text."""
        effective_text = mask_prompt_text(text) if self.masked else text
        logger.info(
            "Predicting with detector=%s input_masked=%s",
            self.name,
            str(self.masked).lower(),
        )
        return await self.detector.predict(effective_text)

    def predict_sync(self, text: str) -> DetectorResult:
        """Run synchronous prediction if supported by underlying detector."""
        effective_text = mask_prompt_text(text) if self.masked else text
        logger.info(
            "Predicting sync with detector=%s input_masked=%s",
            self.name,
            str(self.masked).lower(),
        )
        if hasattr(self.detector, "predict_sync") and callable(
            self.detector.predict_sync
        ):
            return self.detector.predict_sync(effective_text)  # type: ignore[no-any-return]
        import asyncio

        return asyncio.run(self.detector.predict(effective_text))


def wrap_detector(detector: Detector, masked: bool = True) -> Detector:
    """Wrap any detector instance to enforce input masking by default."""
    return MaskedDetector(detector=detector, masked=masked)  # type: ignore[return-value]
