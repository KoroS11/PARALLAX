"""D3: meta-llama/Llama-Prompt-Guard-2-86M binary detector."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from parallax.contracts.detectors import Detector, DetectorResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "meta-llama/Llama-Prompt-Guard-2-86M"
DEFAULT_THRESHOLD = 0.5

# Recognized labels for malicious / injection / attack prompts
INJECTION_LABELS = {
    "LABEL_1",
    "1",
    "MALICIOUS",
    "INJECTION",
    "ATTACK",
    "JAILBREAK",
}
# Recognized labels for benign / safe prompts
SAFE_LABELS = {
    "LABEL_0",
    "0",
    "BENIGN",
    "SAFE",
}

GATED_ERROR_MSG = "Gated access required / HF token missing"


def check_hf_access(model_name: str = DEFAULT_MODEL_NAME) -> tuple[bool, str | None]:
    """Check whether Hugging Face token and access to the gated model are available.

    Returns:
        tuple[bool, str | None]: (True, None) if accessible, or (False, error_reason).
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except (ImportError, RuntimeError, OSError):
            token = None

    if not token:
        return False, GATED_ERROR_MSG

    # Verify access to repository files via huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

        try:
            hf_hub_download(  # nosec B615
                repo_id=model_name,
                filename="config.json",
                token=token,
            )
            return True, None
        except GatedRepoError:
            return False, GATED_ERROR_MSG
        except RepositoryNotFoundError:
            return False, f"Model repository not found: {model_name}"
    except Exception as exc:  # noqa: BLE001
        err_str = str(exc)
        if (
            "401" in err_str
            or "gated" in err_str.lower()
            or "restricted" in err_str.lower()
        ):
            return False, GATED_ERROR_MSG
        return False, f"HF access check error: {exc}"


class PromptGuard2Detector(Detector):
    """Prompt Guard 2 binary detector (D3).

    Uses `meta-llama/Llama-Prompt-Guard-2-86M` for binary prompt injection / jailbreak detection.
    Per DESIGN.md Section 3, D3 is a pretrained binary detector and outputs
    `is_attack: bool`, `score: float`, and `attack_class: None`.

    Note: Prompt Guard 2 is a gated model on Hugging Face. If HF access or token is missing,
    it gracefully returns DetectorResult with error="Gated access required / HF token missing"
    and is_attack=None (or raises if raise_on_error=True).
    """

    name: str = "d3-promptguard2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        threshold: float = DEFAULT_THRESHOLD,
        pipeline: Callable[..., Any] | None = None,
        device: str | int | None = None,
        max_length: int = 512,
        raise_on_error: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize PromptGuard2Detector.

        Args:
            model_name: Hugging Face model repository identifier (gated).
            threshold: Probability threshold for is_attack verdict. Default 0.5.
            pipeline: Optional injected pipeline or callable for dependency injection/testing.
            device: Compute device identifier (e.g. 0, "cuda", "cpu", or -1).
            max_length: Token truncation limit.
            raise_on_error: If True, re-raise exceptions instead of returning DetectorResult with error.
            name: Optional detector name override (defaults to "d3-promptguard2").
        """
        if name is not None:
            self.name = name
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = pipeline
        self.device = device
        self.max_length = max_length
        self.raise_on_error = raise_on_error
        self._lock = asyncio.Lock()

    def check_access(self) -> tuple[bool, str | None]:
        """Check if Hugging Face gated access is available for this model."""
        return check_hf_access(self.model_name)

    def _ensure_pipeline(self) -> Any:
        """Lazily load the Hugging Face text-classification pipeline if not injected."""
        if self.pipeline is not None:
            return self.pipeline

        # Verify access before attempting download
        has_access, reason = self.check_access()
        if not has_access:
            raise RuntimeError(reason or GATED_ERROR_MSG)

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
        - List of dicts: [{'label': 'LABEL_1', 'score': 0.9}, ...]
        - Single dict: {'label': 'MALICIOUS', 'score': 0.9}
        - Direct float or int probability
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
        - attack_class: None (D3 does not output technique labels)
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
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            err_msg = str(exc)
            if (
                GATED_ERROR_MSG in err_msg
                or "401" in err_msg
                or "gated" in err_msg.lower()
                or "restricted" in err_msg.lower()
            ):
                err_msg = GATED_ERROR_MSG

            if self.raise_on_error:
                raise RuntimeError(err_msg) from exc

            logger.warning("Detector %s failed: %s", self.name, err_msg)
            return DetectorResult(
                name=self.name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=latency_ms,
                error=err_msg,
            )
