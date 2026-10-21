"""Detector protocol and result contract."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class DetectorResult:
    """Standardized output produced by any detector."""

    name: str
    is_attack: bool | None = None  # None = abstained (timeout/error)
    score: float | None = None  # 0..1 attack probability
    attack_class: str | None = None  # technique label, if detector can produce one
    latency_ms: float = 0.0
    error: str | None = None


@runtime_checkable
class Detector(Protocol):
    """Common interface for all detectors in PARALLAX."""

    name: str

    async def predict(self, text: str) -> DetectorResult:
        """Run prediction on the given prompt text."""
        ...
