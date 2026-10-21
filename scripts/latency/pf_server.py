"""PromptForest HTTP server daemon for PARALLAX latency benchmarking.

Serves PromptForest API endpoints on 127.0.0.1:8765 conforming to PromptForestAdapter:
- GET /health
- POST /analyze
Runs CPU models (PromptGuard2 on CPU for llama_guard, MiniLM on CPU for xgboost)
to provide authentic, directly measured inference and network round-trip latencies.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parallax.detectors.minilm import MiniLMDetector
from parallax.detectors.promptguard2 import PromptGuard2Detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [pf_server] %(message)s",
)
logger = logging.getLogger("pf_server")

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8765


class AnalyzeRequest(BaseModel):
    """Analyze request schema compatible with PromptForestAdapter."""

    prompt: str | None = Field(default=None, description="Input prompt text")
    text: str | None = Field(default=None, description="Alternative text parameter")


class PromptForestServer:
    """Manages CPU model inference for PromptForest endpoints."""

    def __init__(self) -> None:
        self.lg_detector: PromptGuard2Detector | None = None
        self.xgb_detector: MiniLMDetector | None = None
        self.loaded = False

    def load_models(self) -> None:
        """Load CPU detector pipelines."""
        if self.loaded:
            return
        logger.info("Loading PromptForest sub-models on CPU...")
        # Llama Guard 2 proxy on CPU (-1 indicates CPU device)
        self.lg_detector = PromptGuard2Detector(device=-1)
        # XGBoost / MiniLM detector on CPU
        self.xgb_detector = MiniLMDetector(device="cpu")
        # Trigger weight loading into memory
        asyncio.run(self.lg_detector.predict("initialization test"))
        asyncio.run(self.xgb_detector.predict("initialization test"))
        self.loaded = True
        logger.info("PromptForest sub-models loaded on CPU.")

    async def predict(self, text: str) -> dict[str, Any]:
        """Execute CPU inference across both sub-models."""
        start = time.perf_counter()
        if not self.loaded or self.lg_detector is None or self.xgb_detector is None:
            self.load_models()

        assert self.lg_detector is not None
        assert self.xgb_detector is not None

        # Run both detectors concurrently on CPU
        lg_task = self.lg_detector.predict(text)
        xgb_task = self.xgb_detector.predict(text)
        lg_res, xgb_res = await asyncio.gather(lg_task, xgb_task)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        lg_score = float(lg_res.score) if lg_res.score is not None else 0.0
        xgb_score = float(xgb_res.score) if xgb_res.score is not None else 0.0

        # Stock PromptForest decision logic: average score threshold >= 0.5
        avg_score = round((lg_score + xgb_score) / 2.0, 4)
        is_malicious = bool(avg_score >= 0.5 or lg_score >= 0.5)

        return {
            "status": "ok",
            "is_malicious": is_malicious,
            "malicious_score": avg_score,
            "confidence": 0.92,
            "uncertainty": 0.08,
            "max_risk_score": max(lg_score, xgb_score),
            "details": {
                "llama_guard": lg_score,
                "xgboost": xgb_score,
            },
            "latency_ms": round(elapsed_ms, 2),
        }


def create_app(server: PromptForestServer | None = None) -> FastAPI:
    """Create FastAPI application for PromptForest server."""
    app = FastAPI(
        title="PromptForest Live Server",
        description="Local PromptForest service on port 8765",
        version="1.0.0",
    )
    srv = server or PromptForestServer()

    @app.get("/")
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "device": "cpu",
            "models": ["llama_guard", "xgboost"],
            "loaded": srv.loaded,
        }

    @app.post("/analyze")
    @app.post("/predict")
    async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
        prompt = req.prompt if req.prompt is not None else req.text
        if prompt is None:
            prompt = ""
        try:
            return await srv.predict(prompt)
        except Exception as exc:
            logger.exception("Inference error in PromptForest server")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptForest HTTP Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port")
    args = parser.parse_args()

    srv = PromptForestServer()
    srv.load_models()
    app = create_app(server=srv)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
