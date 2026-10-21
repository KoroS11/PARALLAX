"""PARALLAX Preprocessing Package (Stage D0).

Provides input normalization, bounded multi-layer decoders, sliding-window
chunking, and a unified preprocessing pipeline producing multiple views and
security flags for downstream detectors.
"""

from parallax.preprocess.chunker import (
    DEFAULT_STEP,
    DEFAULT_WINDOW_SIZE,
    chunk_text,
    chunk_views,
)
from parallax.preprocess.decoders import (
    decode_base64,
    decode_hex,
    decode_rot13,
    decode_url,
    recursive_decode,
)
from parallax.preprocess.normalizer import (
    HOMOGLYPH_MAP,
    ZERO_WIDTH_CHARS,
    normalize,
    normalize_with_flags,
)
from parallax.preprocess.pipeline import PreprocessResult, preprocess

__all__ = [
    "DEFAULT_STEP",
    "DEFAULT_WINDOW_SIZE",
    "HOMOGLYPH_MAP",
    "ZERO_WIDTH_CHARS",
    "PreprocessResult",
    "chunk_text",
    "chunk_views",
    "decode_base64",
    "decode_hex",
    "decode_rot13",
    "decode_url",
    "normalize",
    "normalize_with_flags",
    "preprocess",
    "recursive_decode",
]
