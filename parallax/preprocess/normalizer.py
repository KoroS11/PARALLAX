"""Normalization stage for input prompts in PARALLAX.

Applies Unicode NFKC normalization, zero-width and invisible character stripping,
control character removal, basic Cyrillic and Greek homoglyph folding to Latin,
and whitespace normalization and collapse.
"""

from __future__ import annotations

import re
import unicodedata

# Explicit collection of zero-width, invisible, and format control characters
ZERO_WIDTH_CHARS: frozenset[str] = frozenset(
    {
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\ufeff",  # Zero-width no-break space (BOM)
        "\u00ad",  # Soft hyphen
        "\u200e",  # Left-to-right mark
        "\u200f",  # Right-to-left mark
        "\u202a",  # Left-to-right embedding
        "\u202b",  # Right-to-left embedding
        "\u202c",  # Pop directional formatting
        "\u202d",  # Left-to-right override
        "\u202e",  # Right-to-left override
        "\u2060",  # Word joiner
        "\u2061",  # Function application
        "\u2062",  # Invisible times
        "\u2063",  # Invisible separator
        "\u2064",  # Invisible plus
        "\u2066",  # Left-to-right isolate
        "\u2067",  # Right-to-left isolate
        "\u2068",  # First strong isolate
        "\u2069",  # Pop directional isolate
        "\u180e",  # Mongolian vowel separator
        "\ufff9",  # Interlinear annotation anchor
        "\ufffa",  # Interlinear annotation separator
        "\ufffb",  # Interlinear annotation terminator
    }
)

# Cyrillic and Greek lookalikes mapped to Latin ASCII
HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic uppercase lookalikes
    "\u0410": "A",  # А
    "\u0412": "B",  # В
    "\u0415": "E",  # Е
    "\u0405": "S",  # Ѕ
    "\u0406": "I",  # І
    "\u0408": "J",  # Ј
    "\u041a": "K",  # К
    "\u041c": "M",  # М
    "\u041d": "H",  # Н
    "\u041e": "O",  # О
    "\u0420": "P",  # Р
    "\u0421": "C",  # С
    "\u0422": "T",  # Т
    "\u0423": "Y",  # У
    "\u0425": "X",  # Х
    # Cyrillic lowercase lookalikes
    "\u0430": "a",  # а
    "\u0435": "e",  # е
    "\u043e": "o",  # о
    "\u0440": "p",  # р
    "\u0441": "c",  # с
    "\u0443": "y",  # у
    "\u0445": "x",  # х
    "\u0455": "s",  # ѕ
    "\u0456": "i",  # і
    "\u0458": "j",  # ј
    "\u043a": "k",  # к
    # Greek uppercase lookalikes
    "\u0391": "A",  # Α
    "\u0392": "B",  # Β
    "\u0395": "E",  # Ε
    "\u0396": "Z",  # Ζ
    "\u0397": "H",  # Η
    "\u0399": "I",  # Ι
    "\u039a": "K",  # Κ
    "\u039c": "M",  # Μ
    "\u039d": "N",  # Ν
    "\u039f": "O",  # Ο
    "\u03a1": "P",  # Ρ
    "\u03a4": "T",  # Τ
    "\u03a5": "Y",  # Υ
    "\u03a7": "X",  # Χ
    # Greek lowercase lookalikes
    "\u03b1": "a",  # α
    "\u03b5": "e",  # ε
    "\u03b9": "i",  # ι
    "\u03ba": "k",  # κ
    "\u03bf": "o",  # ο
    "\u03c1": "p",  # ρ
    "\u03c2": "s",  # ς
    "\u03c5": "u",  # υ
    "\u03bd": "v",  # ν
    "\u03c7": "x",  # χ
    "\u03f5": "e",  # ϵ
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize input text.

    Pipeline:
    1. Unicode NFKC normalization
    2. Homoglyph folding for Cyrillic and Greek lookalikes
    3. Removal of zero-width and invisible format characters
    4. Removal of non-whitespace control characters
    5. Whitespace normalization and collapse
    6. Canonical NFKC normalization to guarantee idempotence

    Guaranteed property: normalize(normalize(text)) == normalize(text).
    """
    normalized, _ = normalize_with_flags(text)
    return normalized


def normalize_with_flags(text: str) -> tuple[str, list[str]]:
    """Normalize text and return detection flags for any anomalies stripped or folded.

    Returns
    -------
    tuple[str, list[str]]
        The normalized string and a list of observed anomaly flags.
    """
    if not text:
        return "", []

    flags: list[str] = []

    # Step 1: Unicode NFKC
    nfkc_text = unicodedata.normalize("NFKC", text)
    if nfkc_text != text:
        flags.append("unicode_nfkc")

    has_homoglyphs = False
    has_zero_width = False
    has_control_chars = False

    cleaned_chars: list[str] = []
    for ch in nfkc_text:
        # Check zero-width / invisible format characters
        cat = unicodedata.category(ch)
        if ch in ZERO_WIDTH_CHARS or cat == "Cf":
            has_zero_width = True
            continue

        # Check control characters (Cc) and surrogates (Cs), unless whitespace
        if cat in ("Cc", "Cs"):
            if ch.isspace() or ch in ("\t", "\n", "\r"):
                cleaned_chars.append(" ")
            else:
                has_control_chars = True
            continue

        # Fold homoglyphs
        if ch in HOMOGLYPH_MAP:
            cleaned_chars.append(HOMOGLYPH_MAP[ch])
            has_homoglyphs = True
            continue

        cleaned_chars.append(ch)

    if has_zero_width:
        flags.append("zero_width_stripped")
    if has_control_chars:
        flags.append("control_chars_removed")
    if has_homoglyphs:
        flags.append("homoglyphs_folded")

    intermediate = "".join(cleaned_chars)

    # Step 5: Whitespace normalization and collapse
    collapsed = _WHITESPACE_RE.sub(" ", intermediate).strip()
    if collapsed != intermediate:
        flags.append("whitespace_collapsed")

    # Step 6: Final NFKC to ensure strict idempotence
    final_text = unicodedata.normalize("NFKC", collapsed)

    return final_text, flags
