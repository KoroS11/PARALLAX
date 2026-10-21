"""Third Detector HTTP service daemon for PARALLAX Wave G.

Serves deepset/deberta-v3-base-injection on 127.0.0.1:8766.
Integrates with PromptForestAdapter to restore a full 3-detector ensemble (N=3).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [third_detector_service] %(message)s",
)
logger = logging.getLogger("third_detector_service")

DEFAULT_MODEL_NAME: str = "deepset/deberta-v3-base-injection"
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8766
DEFAULT_THRESHOLD: float = 0.5
MAX_LENGTH: int = 512


class AnalyzeRequest(BaseModel):
    """Analyze request schema compatible with PromptForest and PARALLAX."""

    prompt: str | None = Field(default=None, description="Input prompt text")
    text: str | None = Field(default=None, description="Alternative text parameter")


class ThirdDetectorServer:
    """Manages model loading and inference for the third detector."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        pipeline: Any = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = pipeline
        self.model = None
        self.tokenizer = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.loaded = pipeline is not None

    def load_model(self) -> None:
        """Load tokenizer and model weights if not injected."""
        if self.loaded:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading model %s onto device: %s...", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        self.loaded = True
        logger.info("Model %s successfully loaded on %s.", self.model_name, self.device)

    def predict(self, text: str) -> tuple[bool, float, float]:
        """Run prediction on prompt text returning (is_attack, score, latency_ms)."""
        start = time.perf_counter()

        if not text or not text.strip():
            latency = (time.perf_counter() - start) * 1000.0
            return False, 0.0, latency

        if self.pipeline is not None:
            raw = self.pipeline(text)
            latency = (time.perf_counter() - start) * 1000.0
            score: float | None = None
            if isinstance(raw, (int, float)):
                score = float(raw)
            elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
                found_inj = False
                found_safe = False
                safe_s = 0.0
                for item in raw:
                    lbl = str(item.get("label", "")).strip().upper()
                    s = float(item.get("score", 0.0))
                    if lbl in (
                        "INJECTION",
                        "ATTACK",
                        "LABEL_1",
                        "1",
                        "MALICIOUS",
                        "JAILBREAK",
                    ):
                        score = s
                        found_inj = True
                        break
                    elif lbl in ("SAFE", "BENIGN", "LABEL_0", "0", "LEGIT"):
                        found_safe = True
                        safe_s = s
                if not found_inj and found_safe:
                    score = 1.0 - safe_s
                elif not found_inj and not found_safe and raw:
                    score = float(raw[0].get("score", 0.0))
            elif isinstance(raw, dict):
                lbl = str(raw.get("label", "")).strip().upper()
                s = float(raw.get("score", 0.0))
                if lbl in ("SAFE", "BENIGN", "LABEL_0", "0", "LEGIT"):
                    score = 1.0 - s
                else:
                    score = s
            if score is None:
                score = 0.0
            score = max(0.0, min(1.0, score))
            return score >= self.threshold, round(score, 4), latency

        if not self.loaded:
            self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        latency = (time.perf_counter() - start) * 1000.0
        # Positive injection class is index 1 for deepset/deberta-v3-base-injection
        score = float(probs[1]) if len(probs) > 1 else float(probs[0])
        is_attack = score >= self.threshold
        return is_attack, round(score, 4), round(latency, 2)


def create_app(
    detector: ThirdDetectorServer | None = None,
    lazy_load: bool = True,
) -> FastAPI:
    """Create FastAPI application for the third detector service."""
    app = FastAPI(
        title="PARALLAX Third Detector Service",
        description="Local HTTP inference service for deepset/deberta-v3-base-injection",
        version="1.0.0",
    )

    det = detector or ThirdDetectorServer()

    if not lazy_load:
        det.load_model()

    @app.get("/")
    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint conforming to PromptForest service standard."""
        return {
            "status": "ok",
            "service": "third_detector",
            "model": det.model_name,
            "device": det.device,
            "loaded": det.loaded,
        }

    @app.post("/analyze")
    @app.post("/predict")
    async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
        """Analyze prompt for prompt injection returning standard telemetry."""
        prompt = req.prompt if req.prompt is not None else req.text
        if prompt is None:
            prompt = ""

        try:
            is_malicious, score, latency_ms = det.predict(prompt)
            return {
                "status": "ok",
                "is_malicious": is_malicious,
                "malicious_score": score,
                "details": {
                    "deepset": score,
                },
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            logger.exception("Inference failure")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="PARALLAX Third Detector HTTP Daemon")
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Bind port (default: 8766)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME, help="Hugging Face model ID"
    )
    parser.add_argument("--device", default=None, help="Device (cpu or cuda)")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold"
    )
    args = parser.parse_args()

    detector = ThirdDetectorServer(
        model_name=args.model,
        device=args.device,
        threshold=args.threshold,
    )
    app = create_app(detector=detector, lazy_load=False)
    logger.info(
        "Starting third detector daemon on http://%s:%d...", args.host, args.port
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
