"""Unified preprocessing pipeline for PARALLAX (Stage D0).

Produces normalized, decoded, and chunked views of prompt inputs along with
security anomaly flags and truncation indicators for downstream parallel detectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from parallax.preprocess.chunker import DEFAULT_STEP, DEFAULT_WINDOW_SIZE, chunk_text
from parallax.preprocess.decoders import (
    MAX_DECODED_LENGTH,
    MAX_RECURSION_DEPTH,
    decode_rot13,
    recursive_decode,
)
from parallax.preprocess.normalizer import normalize, normalize_with_flags

KNOWN_ENCODING_FLAGS: frozenset[str] = frozenset(
    {
        "base64_decoded",
        "hex_decoded",
        "url_decoded",
        "rot13_decoded",
        "zero_width_stripped",
        "homoglyphs_folded",
    }
)


@dataclass
class PreprocessResult:
    """Result of preprocessing an input prompt.

    Attributes
    ----------
    views : dict[str, str]
        Named views of the prompt (always includes 'raw' and 'normalized',
        plus 'decoded' and/or 'chunk_<i>' when applicable).
    flags : list[str]
        List of anomaly flags detected (e.g. 'zero_width_stripped',
        'homoglyphs_folded', 'base64_decoded', 'chunked', etc.).
    truncated : bool
        True if the raw input or decoded output was capped to prevent
        resource exhaustion.
    encoding_present : bool
        True if any encoding or obfuscation (base64, hex, rot13, url,
        homoglyphs, zero-width) was stripped or decoded.
    encoding_types : list[str]
        Specific encoding anomaly flags identified.
    """

    views: dict[str, str] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    truncated: bool = False
    encoding_present: bool = False
    encoding_types: list[str] = field(default_factory=list)


def preprocess(
    text: str,
    max_length: int = MAX_DECODED_LENGTH,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP,
    include_rot13: bool = False,
) -> PreprocessResult:
    """Preprocess input text into multiple views with anomaly detection flags.

    Parameters
    ----------
    text : str
        The incoming raw prompt text.
    max_length : int, default=16384 (16 KB)
        Maximum allowed character length for raw input and decoded views.
    window_size : int, default=512
        Window width for sliding-window chunking.
    step : int, default=256
        Stride between sliding windows.
    include_rot13 : bool, default=False
        If True, always include a ROT13 view regardless of whether ROT13
        heuristics fired.

    Returns
    -------
    PreprocessResult
        Views, anomaly flags, and truncation state.
    """
    flags: list[str] = []
    truncated = False

    # 1. Enforce bounded input size
    if len(text) > max_length:
        raw_text = text[:max_length]
        truncated = True
        flags.append("input_truncated")
    else:
        raw_text = text

    views: dict[str, str] = {"raw": raw_text}

    # 2. Normalization
    normalized, norm_flags = normalize_with_flags(raw_text)
    views["normalized"] = normalized
    flags.extend(norm_flags)

    # 3. Bounded recursive decoding
    decoded, decode_flags, dec_truncated = recursive_decode(
        normalized,
        max_depth=MAX_RECURSION_DEPTH,
        max_length=max_length,
    )
    if dec_truncated:
        truncated = True

    if decode_flags:
        flags.extend(decode_flags)
        views["decoded"] = decoded
        # If decoding revealed hidden characters or homoglyphs, normalize decoded view
        norm_decoded = normalize(decoded)
        if norm_decoded != decoded:
            views["decoded_normalized"] = norm_decoded

    # 4. ROT13 view handling
    if include_rot13 or "rot13_decoded" in flags:
        r13 = decode_rot13(normalized)
        views["rot13"] = r13
        if "rot13_decoded" in flags:
            views["decoded_rot13"] = r13

    # 5. Sliding-window chunking for long inputs
    if len(normalized) > window_size:
        chunks = chunk_text(normalized, window_size=window_size, step=step)
        if len(chunks) > 1:
            flags.append("chunked")
            for idx, chunk in enumerate(chunks):
                views[f"chunk_{idx}"] = chunk

    # Deduplicate flags while preserving order
    deduped_flags: list[str] = []
    seen: set[str] = set()
    for f in flags:
        if f not in seen:
            seen.add(f)
            deduped_flags.append(f)

    encoding_types = [f for f in deduped_flags if f in KNOWN_ENCODING_FLAGS]
    encoding_present = len(encoding_types) > 0

    return PreprocessResult(
        views=views,
        flags=deduped_flags,
        truncated=truncated,
        encoding_present=encoding_present,
        encoding_types=encoding_types,
    )
