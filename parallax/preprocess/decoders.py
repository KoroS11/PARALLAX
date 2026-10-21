"""Bounded multi-layer decoders for obfuscated prompt inputs in PARALLAX.

Supports Base64, Hex (\\x.. and 0x..), URL-encoding (%..), and ROT13.
Guarantees:
- Max recursion depth <= 2
- Max decoded output length capped at 16 KB (16,384 characters)
- Safe handling of malformed or binary data without unhandled exceptions
- Pure text transformations: NEVER evaluates or executes decoded content.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import urllib.parse

MAX_DECODED_LENGTH = 16384  # 16 KB
MAX_RECURSION_DEPTH = 2

# Common English words used to score ROT13 heuristic
_COMMON_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "that",
        "have",
        "with",
        "this",
        "from",
        "they",
        "will",
        "would",
        "there",
        "their",
        "ignore",
        "system",
        "prompt",
        "instruction",
        "instructions",
        "password",
        "canary",
        "secret",
        "never",
        "always",
        "rules",
        "payload",
        "access",
        "bypass",
        "execute",
        "print",
        "delete",
        "disable",
        "admin",
    }
)

_ROT13_TRIGGER_RE = re.compile(r"\b(?:rot13|rot-13|caesar|cipher)\b", re.IGNORECASE)
_HEX_SLASH_X_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
_HEX_ZERO_X_COMMA_RE = re.compile(
    r"\b0x[0-9a-fA-F]{1,2}(?:\s*,\s*0x[0-9a-fA-F]{1,2})+\b"
)
_HEX_ZERO_X_SPACE_RE = re.compile(r"\b0x[0-9a-fA-F]{2}(?:\s+0x[0-9a-fA-F]{2})+\b")
_HEX_ZERO_X_CONT_RE = re.compile(r"\b0x([0-9a-fA-F]{4,})\b")
_RAW_HEX_RE = re.compile(r"^[0-9a-fA-F]{8,}$")
_BASE64_TOKEN_RE = re.compile(
    r"(?<![-A-Za-z0-9+/])([A-Za-z0-9+/\-_]{8,}={0,2})(?![-A-Za-z0-9+/=])"
)
_BASE64_PREFIX_RE = re.compile(
    r"(?:base64|b64)[\s:=]+([A-Za-z0-9+/_-]{4,}={0,2})", re.IGNORECASE
)
_URL_ENCODED_RE = re.compile(r"%[0-9a-fA-F]{2}")


def _is_safe_printable_text(text: str) -> bool:
    """Check if string looks like valid printable text (not binary garbage)."""
    if not text or "\x00" in text:
        return False
    printable_count = sum(1 for c in text if c.isprintable() or c in ("\n", "\t", "\r"))
    return (printable_count / len(text)) >= 0.80


def decode_url(text: str) -> tuple[str, bool]:
    """Decode percent-encoded URL sequences if present."""
    if not text or not _URL_ENCODED_RE.search(text):
        return text, False
    try:
        decoded = urllib.parse.unquote(text)
        if decoded != text:
            return decoded, True
    except (ValueError, UnicodeError):
        return text, False
    return text, False


def _try_decode_hex_bytes(hex_str: str) -> str | None:
    """Safely convert a clean hex string to UTF-8 text."""
    try:
        raw_bytes = bytes.fromhex(hex_str)
        decoded = raw_bytes.decode("utf-8", errors="strict")
        if _is_safe_printable_text(decoded):
            return decoded
    except (ValueError, UnicodeError):
        return None
    return None


def decode_hex(text: str) -> tuple[str, bool]:
    """Decode escaped hex (\\x..) or prefixed hex (0x..) representations."""
    if not text:
        return text, False

    changed = False
    result = text

    # Case 1: \x48\x65\x6c\x6c\x6f
    def _replace_slash_x(match: re.Match[str]) -> str:
        nonlocal changed
        matched_str = match.group(0)
        clean_hex = matched_str.replace("\\x", "")
        decoded = _try_decode_hex_bytes(clean_hex)
        if decoded is not None:
            changed = True
            return decoded
        return matched_str

    result = _HEX_SLASH_X_RE.sub(_replace_slash_x, result)

    # Case 2: 0x48, 0x65, 0x6c or 0x48 0x65
    def _replace_0x_list(match: re.Match[str]) -> str:
        nonlocal changed
        matched_str = match.group(0)
        clean_hex = re.sub(r"0x|[\s,]", "", matched_str)
        if len(clean_hex) % 2 != 0:
            return matched_str
        decoded = _try_decode_hex_bytes(clean_hex)
        if decoded is not None:
            changed = True
            return decoded
        return matched_str

    result = _HEX_ZERO_X_COMMA_RE.sub(_replace_0x_list, result)
    result = _HEX_ZERO_X_SPACE_RE.sub(_replace_0x_list, result)

    # Case 3: 0x48656c6c6f
    def _replace_0x_cont(match: re.Match[str]) -> str:
        nonlocal changed
        clean_hex = match.group(1)
        if len(clean_hex) % 2 != 0:
            return match.group(0)
        decoded = _try_decode_hex_bytes(clean_hex)
        if decoded is not None:
            changed = True
            return decoded
        return match.group(0)

    result = _HEX_ZERO_X_CONT_RE.sub(_replace_0x_cont, result)

    # Case 4: standalone raw hex string
    stripped = result.strip()
    if (
        _RAW_HEX_RE.match(stripped)
        and len(stripped) % 2 == 0
        and not stripped.isdigit()
    ):
        decoded = _try_decode_hex_bytes(stripped)
        if decoded is not None and decoded != stripped:
            return decoded, True

    return result, changed


def _try_b64decode_string(candidate: str) -> str | None:
    """Attempt safe Base64 decoding of a candidate string."""
    try:
        clean = candidate.strip()
        # Add required padding
        missing_padding = (4 - len(clean) % 4) % 4
        clean_padded = clean + ("=" * missing_padding)

        # Decode using standard or urlsafe base64
        try:
            raw_bytes = base64.b64decode(clean_padded, validate=False)
        except (binascii.Error, ValueError):
            try:
                raw_bytes = base64.urlsafe_b64decode(clean_padded)
            except (binascii.Error, ValueError):
                return None

        if not raw_bytes:
            return None

        try:
            decoded = raw_bytes.decode("utf-8", errors="strict")
        except UnicodeError:
            return None

        if not _is_safe_printable_text(decoded):
            return None

        # Avoid false positives where an ASCII word decodes into another short string
        if decoded.strip() == candidate.strip():
            return None

        # If candidate consists exclusively of letters with no padding, check length and text quality
        if candidate.isalpha() and not candidate.endswith("=") and len(candidate) < 12:
            return None

        return decoded
    except (binascii.Error, ValueError, UnicodeError):
        return None


def decode_base64(text: str) -> tuple[str, bool]:
    """Decode standalone or embedded Base64 strings."""
    if not text:
        return text, False

    changed = False

    # Check whole string first
    stripped = text.strip()
    if len(stripped) >= 8 and (
        stripped.endswith("=")
        or "+" in stripped
        or "/" in stripped
        or "-" in stripped
        or "_" in stripped
        or any(c.isdigit() for c in stripped)
    ):
        decoded = _try_b64decode_string(stripped)
        if decoded is not None:
            return decoded, True

    # Check explicit base64: <payload> prefix
    def _replace_prefix_match(match: re.Match[str]) -> str:
        nonlocal changed
        token = match.group(1)
        decoded = _try_b64decode_string(token)
        if decoded is not None:
            changed = True
            return decoded
        return match.group(0)

    result = _BASE64_PREFIX_RE.sub(_replace_prefix_match, text)
    if changed:
        return result, True

    # Check embedded tokens
    def _replace_token_match(match: re.Match[str]) -> str:
        nonlocal changed
        token = match.group(1)
        decoded = _try_b64decode_string(token)
        if decoded is not None:
            changed = True
            return decoded
        return match.group(0)

    result = _BASE64_TOKEN_RE.sub(_replace_token_match, result)
    return result, changed


def decode_rot13(text: str) -> str:
    """Decode text using ROT13 cipher."""
    try:
        return codecs.decode(text, "rot_13")
    except (ValueError, UnicodeError):
        return text


def try_decode_rot13(text: str) -> tuple[str, bool]:
    """Detect and decode ROT13 encoded text using heuristic analysis and keyword triggers."""
    if not text or not any(c.isalpha() for c in text):
        return text, False

    has_trigger = bool(_ROT13_TRIGGER_RE.search(text))

    # Evaluate English word density of candidate vs decoded
    decoded = decode_rot13(text)
    if decoded == text:
        return text, False

    orig_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
    rot13_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", decoded.lower()))

    orig_score = len(orig_words & _COMMON_WORDS)
    rot13_score = len(rot13_words & _COMMON_WORDS)

    # Fire if trigger keywords are present, or ROT13 produces noticeably more common English words
    if has_trigger or (rot13_score >= 2 and rot13_score > orig_score):
        return decoded, True

    return text, False


def decode_step(
    text: str,
    max_length: int = MAX_DECODED_LENGTH,
) -> tuple[str, list[str], bool]:
    """Execute one decoding pass across all decoders (URL, Hex, Base64, ROT13).

    Returns
    -------
    tuple[str, list[str], bool]
        (decoded_text, flags, truncated)
    """
    flags: list[str] = []
    current = text
    truncated = False

    # 1. URL decoding
    current, url_decoded = decode_url(current)
    if url_decoded:
        flags.append("url_decoded")

    # 2. Hex decoding
    current, hex_decoded = decode_hex(current)
    if hex_decoded:
        flags.append("hex_decoded")

    # 3. Base64 decoding
    current, b64_decoded = decode_base64(current)
    if b64_decoded:
        flags.append("base64_decoded")

    # 4. ROT13 decoding
    current, r13_decoded = try_decode_rot13(current)
    if r13_decoded:
        flags.append("rot13_decoded")

    # Bounded length enforcement
    if len(current) > max_length:
        current = current[:max_length]
        truncated = True
        flags.append("decoded_truncated")

    return current, flags, truncated


def recursive_decode(
    text: str,
    max_depth: int = MAX_RECURSION_DEPTH,
    max_length: int = MAX_DECODED_LENGTH,
) -> tuple[str, list[str], bool]:
    """Recursively decode text up to max_depth (bounded <= 2) and cap output at max_length.

    Returns
    -------
    tuple[str, list[str], bool]
        (final_decoded_text, accumulated_flags, truncated)
    """
    current = text
    accumulated_flags: list[str] = []
    truncated = False

    for _depth in range(max_depth):
        next_text, step_flags, step_truncated = decode_step(
            current, max_length=max_length
        )
        if step_truncated:
            truncated = True

        if not step_flags or next_text == current:
            break

        accumulated_flags.extend(step_flags)
        current = next_text

        if truncated or len(current) >= max_length:
            break

    return current, accumulated_flags, truncated
