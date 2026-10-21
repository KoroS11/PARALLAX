"""D2: protectai/deberta-v3-base-prompt-injection-v2 binary detector."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

from parallax.contracts.detectors import Detector, DetectorResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
DEFAULT_THRESHOLD = 0.5

# Recognized labels for prompt injection / attack
INJECTION_LABELS = {
    "INJECTION",
    "MALICIOUS",
    "LABEL_1",
    "1",
    "ATTACK",
    "JAILBREAK",
}
# Recognized labels for benign / safe prompts
SAFE_LABELS = {
    "SAFE",
    "BENIGN",
    "LABEL_0",
    "0",
}


class DeBERTaDetector(Detector):
    """DeBERTa-v3 prompt injection binary detector (D2).

    Uses `protectai/deberta-v3-base-prompt-injection-v2` for binary classification.
    Per DESIGN.md Section 3, D2 is a pretrained binary detector and outputs
    `is_attack: bool`, `score: float`, and `attack_class: None`.
    """

    name: str = "d2-deberta"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        threshold: float = DEFAULT_THRESHOLD,
        pipeline: Callable[..., Any] | None = None,
        device: str | int | None = None,
        max_length: int = 512,
        name: str | None = None,
    ) -> None:
        """Initialize the DeBERTa detector.

        Args:
            model_name: Hugging Face model repository identifier.
            threshold: Probability threshold for is_attack verdict. Default 0.5.
            pipeline: Optional injected pipeline or callable for dependency injection/testing.
            device: Compute device identifier (e.g. 0, "cuda", "cpu", or -1).
            max_length: Token truncation limit.
            name: Optional detector name override (defaults to "d2-deberta").
        """
        if name is not None:
            self.name = name
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = pipeline
        self.device = device
        self.max_length = max_length
        self._lock = asyncio.Lock()

    def _ensure_pipeline(self) -> Any:
        """Lazily load the Hugging Face text-classification pipeline if not injected."""
        if self.pipeline is not None:
            return self.pipeline

        from transformers import pipeline

        device = self.device
        if device is None:
            try:
                import torch

                device = 0 if torch.cuda.is_available() else -1
            except (ImportError, RuntimeError, OSError):
                device = -1

        self.pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=device,
            truncation=True,
            max_length=self.max_length,
            top_k=None,
        )
        return self.pipeline

    def _parse_prediction(self, raw_output: Any) -> tuple[bool, float]:
        """Extract attack probability score and binary verdict from model output.

        Handles:
        - List of dicts with all label scores: [{'label': 'INJECTION', 'score': 0.9}, ...]
        - Single dict: {'label': 'INJECTION', 'score': 0.9}
        - Direct probability float or int
        """
        score: float | None = None

        if isinstance(raw_output, (int, float)):
            score = float(raw_output)
        elif isinstance(raw_output, dict):
            if "score" in raw_output and "label" in raw_output:
                lbl = str(raw_output["label"]).strip().upper()
                s = float(raw_output["score"])
                if lbl in INJECTION_LABELS:
                    score = s
                elif lbl in SAFE_LABELS:
                    score = 1.0 - s
                else:
                    score = s
            elif "score" in raw_output:
                score = float(raw_output["score"])
        elif isinstance(raw_output, list) and len(raw_output) > 0:
            items = raw_output
            # Nested list: e.g. [[{'label': ..., 'score': ...}, ...]]
            if isinstance(items[0], list):
                items = items[0]

            if all(isinstance(x, dict) for x in items):
                found_injection = False
                found_safe = False
                safe_score = 0.0
                for item in items:
                    lbl = str(item.get("label", "")).strip().upper()
                    s = float(item.get("score", 0.0))
                    if lbl in INJECTION_LABELS:
                        score = s
                        found_injection = True
                        break
                    elif lbl in SAFE_LABELS:
                        found_safe = True
                        safe_score = s

                if not found_injection and found_safe:
                    score = 1.0 - safe_score
                elif not found_injection and not found_safe and items:
                    if "score" in items[0]:
                        score = float(items[0]["score"])
                    else:
                        raise ValueError(
                            f"Unrecognized detector output structure: {raw_output}"
                        )
            elif isinstance(items[0], (int, float)):
                score = float(items[0])

        if score is None:
            raise ValueError(f"Unrecognized detector output structure: {raw_output}")

        score = max(0.0, min(1.0, score))
        is_attack = score >= self.threshold
        return is_attack, score

    async def predict(self, text: str) -> DetectorResult:
        """Run prediction on the given prompt text.

        Returns DetectorResult with:
        - is_attack: bool | None (None on abstained/error)
        - score: float | None (0..1 attack probability)
        - attack_class: None (D2 does not output technique labels)
        - latency_ms: float
        - error: None on success, error message on exception
        """
        start = time.perf_counter()
        if not isinstance(text, str) or not text.strip():
            latency_ms = (time.perf_counter() - start) * 1000.0
            return DetectorResult(
                name=self.name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=round(latency_ms, 3),
                error="Empty or invalid input prompt",
            )

        try:
            async with self._lock:
                pipe = self._ensure_pipeline()

            if inspect.iscoroutinefunction(pipe):
                raw_output = await pipe(text)
            else:
                raw_output = await asyncio.to_thread(pipe, text)

            if inspect.isawaitable(raw_output):
                raw_output = await raw_output

            is_attack, score = self._parse_prediction(raw_output)
            latency_ms = (time.perf_counter() - start) * 1000.0

            return DetectorResult(
                name=self.name,
                is_attack=is_attack,
                score=score,
                attack_class=None,
                latency_ms=latency_ms,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("Detector %s failed: %s", self.name, exc)
            return DetectorResult(
                name=self.name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=latency_ms,
                error=str(exc),
            )
