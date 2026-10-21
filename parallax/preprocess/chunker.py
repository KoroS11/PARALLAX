"""Sliding-window chunker for prompt texts and views in PARALLAX.

Ensures prompt injections in long inputs are not diluted across large context
windows by breaking inputs into overlapping sliding windows (default 512 chars,
step 256 chars).
"""

from __future__ import annotations

DEFAULT_WINDOW_SIZE = 512
DEFAULT_STEP = 256


def chunk_text(
    text: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP,
) -> list[str]:
    """Split text into overlapping sliding-window chunks.

    Parameters
    ----------
    text : str
        The input text to chunk.
    window_size : int, default=512
        Character width of each window.
    step : int, default=256
        Stride/step between successive windows.

    Returns
    -------
    list[str]
        List of text chunks. Returns empty list if text is empty.
    """
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive integers")

    if not text:
        return []

    text_len = len(text)
    if text_len <= window_size:
        return [text]

    chunks: list[str] = []
    i = 0
    while i < text_len:
        chunks.append(text[i : i + window_size])
        if i + window_size >= text_len:
            break
        i += step

    return chunks


def chunk_views(
    views: dict[str, str],
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP,
) -> dict[str, list[str]]:
    """Apply sliding-window chunking to all views in a view dictionary.

    Parameters
    ----------
    views : dict[str, str]
        Mapping from view name to view text.
    window_size : int, default=512
        Character width of each window.
    step : int, default=256
        Stride/step between successive windows.

    Returns
    -------
    dict[str, list[str]]
        Mapping from view name to list of chunk strings.
    """
    return {
        name: chunk_text(content, window_size=window_size, step=step)
        for name, content in views.items()
    }
