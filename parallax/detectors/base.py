"""Base class and common helpers for detector implementations."""

import time
from abc import ABC, abstractmethod

from parallax.contracts.detectors import DetectorResult


class BaseDetector(ABC):
    """Abstract base class for all PARALLAX detectors adhering to Detector Protocol."""

    name: str

    @abstractmethod
    async def predict(self, text: str) -> DetectorResult:
        """Run prediction on the given prompt text.

        Parameters
        ----------
        text : str
            The input prompt text to inspect.

        Returns
        -------
        DetectorResult
            Detection decision, score, technique class, latency, and error state.
        """
        ...

    @staticmethod
    def measure_latency_ms(start_time: float) -> float:
        """Calculate elapsed time in milliseconds from start_time."""
        return (time.perf_counter() - start_time) * 1000.0
