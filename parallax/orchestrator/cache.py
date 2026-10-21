"""Normalized-text prompt cache (in-process LRU and protocol)."""

import hashlib
import threading
import unicodedata
from collections import OrderedDict
from typing import Protocol, runtime_checkable

from parallax.contracts.api import ScanResponse


def normalize_text(text: str) -> str:
    """Canonical prompt normalization.

    Performs Unicode NFKC normalization, lowercasing, and collapses
    consecutive whitespace characters into a single space.
    """
    if not text:
        return ""
    # Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)
    # Lowercase
    normalized = normalized.lower()
    # Collapse multiple whitespace characters (including newlines and tabs)
    return " ".join(normalized.split())


def compute_prompt_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized prompt text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@runtime_checkable
class ScanCache(Protocol):
    """Protocol for scan response caches."""

    def get(self, prompt: str) -> ScanResponse | None:
        """Lookup cached scan response by prompt text."""
        ...

    def set(self, prompt: str, response: ScanResponse) -> None:
        """Store scan response for prompt text."""
        ...

    def delete(self, prompt: str) -> bool:
        """Delete prompt entry from cache. Returns True if deleted."""
        ...

    def clear(self) -> None:
        """Evict all entries from cache."""
        ...

    def __len__(self) -> int:
        """Return number of cached items."""
        ...


class LRUCache:
    """In-process thread-safe LRU cache keyed by normalized prompt hash."""

    def __init__(self, max_size: int = 1000) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max_size = max_size
        self._cache: OrderedDict[str, ScanResponse] = OrderedDict()
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def evictions(self) -> int:
        return self._evictions

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get(self, prompt: str) -> ScanResponse | None:
        """Retrieve cached scan response, or None on cache miss."""
        key = compute_prompt_hash(prompt)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                stored = self._cache[key]
                # Return a copy marked as cached
                return stored.model_copy(update={"cached": True})
            self._misses += 1
            return None

    def set(self, prompt: str, response: ScanResponse) -> None:
        """Store scan response for prompt, evicting LRU item if max_size reached."""
        key = compute_prompt_hash(prompt)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = response
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def delete(self, prompt: str) -> bool:
        """Remove prompt entry from cache."""
        key = compute_prompt_hash(prompt)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Evict all cached entries and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def __contains__(self, prompt: str) -> bool:
        key = compute_prompt_hash(prompt)
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
