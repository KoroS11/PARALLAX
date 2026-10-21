"""D1: sentence-transformers/all-MiniLM-L6-v2 + Two-stage technique classifier.

Stage 1: Binary classifier (calibrated attack vs none probability).
Stage 2: Multi-class technique classifier trained on attack samples.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from parallax.contracts.detectors import DetectorResult
from parallax.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ARTIFACT_PATH = "artifacts/d1_minilm/classifier.joblib"

DEFAULT_TECHNIQUE_CLASSES: list[str] = [
    "none",
    "keyword_override",
    "persona_play",
    "encoding",
    "payload_splitting",
    "context_overflow",
    "few_shot_poisoning",
    "multilingual",
]

DEFAULT_ATTACK_CLASSES: list[str] = [
    "keyword_override",
    "persona_play",
    "encoding",
    "payload_splitting",
    "context_overflow",
    "few_shot_poisoning",
    "multilingual",
]

DEFAULT_LABEL_TO_TECHNIQUE: dict[int, str] = {
    0: "none",
    1: "keyword_override",
    2: "persona_play",
    3: "encoding",
    4: "payload_splitting",
    5: "context_overflow",
    6: "few_shot_poisoning",
    7: "multilingual",
}


class MockFallbackEncoder:
    """Deterministic, fast offline mock encoder for uninitialized fallback mode."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, text: str | list[str]) -> np.ndarray:
        if isinstance(text, (list, tuple)):
            return np.vstack([self._encode_single(str(t)) for t in text])
        return self._encode_single(str(text))

    def _encode_single(self, text: str) -> np.ndarray:
        seed = sum(ord(c) for c in text) % 10000 if text else 0
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)


class MockFallbackStage1:
    """Mock stage 1 binary classifier for fallback mode."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.classes_ = np.array([0, 1])
        self.threshold = threshold

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        probas = np.zeros((n, 2), dtype=float)
        for i in range(n):
            val = float(np.mean(x[i]))
            attack_prob = 1.0 / (1.0 + np.exp(-val * 10.0))
            attack_prob = float(np.clip(attack_prob, 0.0, 1.0))
            probas[i] = [1.0 - attack_prob, attack_prob]
        return probas


class MockFallbackStage2:
    """Mock stage 2 technique classifier for fallback mode."""

    def __init__(self, attack_classes: list[str] | None = None) -> None:
        self.classes_ = np.array(attack_classes or DEFAULT_ATTACK_CLASSES)

    def predict(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        preds = []
        for i in range(n):
            idx = int(abs(sum(x[i][:5])) * 100) % len(self.classes_)
            preds.append(self.classes_[idx])
        return np.array(preds)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        num_c = len(self.classes_)
        uniform = np.full((n, num_c), 1.0 / num_c)
        return uniform


class MiniLMDetector(BaseDetector):
    """MiniLM embedding + Two-stage technique detector (D1).

    Conforms to PARALLAX Detector Protocol:
      name: str = "d1-minilm"
      async def predict(self, text: str) -> DetectorResult

    Two-stage architecture:
      - Stage 1: Binary classifier (calibrated probability: attack vs none).
      - Stage 2: Technique classification head trained on attack samples.
      - Decision threshold: tuned on validation such that FPR on benign <= 0.05.

    Supports:
      - Multiple preprocess views scoring (returning the maximum attack score).
      - Offline fallback / mock mode when weights are uninitialized.
      - Dependency injection for unit tests without downloading weights.
    """

    def __init__(
        self,
        name: str = "d1-minilm",
        model_name: str = DEFAULT_MODEL_NAME,
        artifact_path: str | Path | None = None,
        encoder: Any = None,
        classifier: Any = None,
        stage1: Any = None,
        stage2: Any = None,
        classes: list[str] | None = None,
        attack_classes: list[str] | None = None,
        threshold: float = 0.5,
        device: str | None = "cpu",
        fallback: bool = True,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.artifact_path = Path(artifact_path or DEFAULT_ARTIFACT_PATH)
        self.threshold = float(threshold)
        self.device = device
        self.fallback = fallback

        self._encoder = encoder
        self._classifier = classifier
        self._stage1 = stage1
        self._stage2 = stage2
        self._classes = (
            classes if classes is not None else list(DEFAULT_TECHNIQUE_CLASSES)
        )
        self._attack_classes = (
            attack_classes
            if attack_classes is not None
            else list(DEFAULT_ATTACK_CLASSES)
        )
        self._loaded: bool = encoder is not None and (
            classifier is not None or (stage1 is not None and stage2 is not None)
        )
        self.is_mock_fallback: bool = False

    def _setup_mock_fallback(self) -> None:
        """Initialize lightweight mock encoder and classifier for offline execution."""
        logger.info(
            "MiniLMDetector: Artifact not found and models not injected. "
            "Initializing lightweight offline mock/fallback mode."
        )
        self._encoder = MockFallbackEncoder(dim=384)
        self._stage1 = MockFallbackStage1(threshold=self.threshold)
        self._stage2 = MockFallbackStage2(attack_classes=self._attack_classes)
        self._classes = list(DEFAULT_TECHNIQUE_CLASSES)
        self._attack_classes = list(DEFAULT_ATTACK_CLASSES)
        self.is_mock_fallback = True
        self._loaded = True

    def _ensure_loaded(self) -> None:
        """Lazy load encoder model and classifier head artifacts if not injected."""
        if self._loaded:
            return

        has_injected = (
            self._encoder is not None
            or self._classifier is not None
            or self._stage1 is not None
        )

        if not has_injected and not self.artifact_path.exists():
            if self.fallback:
                self._setup_mock_fallback()
                return
            raise FileNotFoundError(
                f"Classifier artifact not found at {self.artifact_path}. "
                "Train the classifier using scripts/train_minilm.py or inject a mock classifier."
            )

        if self._encoder is None:
            if not self.artifact_path.exists() and self.fallback:
                self._encoder = MockFallbackEncoder(dim=384)
            else:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.model_name, device=self.device)

        if self._classifier is None and (self._stage1 is None or self._stage2 is None):
            if not self.artifact_path.exists():
                if self.fallback:
                    self._setup_mock_fallback()
                    return
                raise FileNotFoundError(
                    f"Classifier artifact not found at {self.artifact_path}."
                )

            loaded = joblib.load(self.artifact_path)
            if isinstance(loaded, dict):
                # Check for two-stage architecture
                if "stage1" in loaded and "stage2" in loaded:
                    self._stage1 = loaded["stage1"]
                    self._stage2 = loaded["stage2"]
                elif "classifier" in loaded:
                    self._classifier = loaded["classifier"]
                else:
                    self._classifier = loaded

                if "classes" in loaded:
                    self._classes = [str(c) for c in loaded["classes"]]
                if "attack_classes" in loaded:
                    self._attack_classes = [str(c) for c in loaded["attack_classes"]]
                if "threshold" in loaded and self.threshold == 0.5:
                    self.threshold = float(loaded["threshold"])
            else:
                self._classifier = loaded

        self._loaded = True

    def _resolve_classes(self) -> list[str]:
        """Resolve multi-class technique names from classifier or configured classes."""
        clf = self._classifier or self._stage2
        if clf is not None and hasattr(clf, "classes_"):
            raw_classes = clf.classes_
            resolved = []
            for c in raw_classes:
                if isinstance(c, (int, np.integer)):
                    resolved.append(DEFAULT_LABEL_TO_TECHNIQUE.get(int(c), str(c)))
                else:
                    resolved.append(str(c))
            return resolved
        return self._classes

    def _score_single_embedding(self, emb: np.ndarray) -> tuple[float, str]:
        """Compute calibrated attack probability score and technique class for an embedding vector."""
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Path A: Two-stage architecture
        if self._stage1 is not None and self._stage2 is not None:
            # Stage 1: Calibrated binary classification
            if hasattr(self._stage1, "predict_proba"):
                s1_proba = np.asarray(self._stage1.predict_proba(emb)[0], dtype=float)
                # Find attack probability (class 1 or class != 'none')
                if hasattr(self._stage1, "classes_"):
                    c1 = list(self._stage1.classes_)
                    if "none" in c1:
                        none_i = c1.index("none")
                        attack_score = 1.0 - s1_proba[none_i]
                    elif 1 in c1:
                        attack_score = s1_proba[c1.index(1)]
                    else:
                        attack_score = s1_proba[-1]
                else:
                    attack_score = s1_proba[-1]
            elif hasattr(self._stage1, "decision_function"):
                df = float(self._stage1.decision_function(emb)[0])
                attack_score = 1.0 / (1.0 + np.exp(-df))
            elif hasattr(self._stage1, "predict"):
                pred1 = self._stage1.predict(emb)[0]
                attack_score = 1.0 if pred1 in (1, "1", "attack", True) else 0.0
            else:
                raise TypeError(
                    "Stage 1 classifier must implement predict_proba, decision_function, or predict"
                )

            attack_score = float(np.clip(attack_score, 0.0, 1.0))

            # Stage 2: Technique classification head
            tech_class = "none"
            if hasattr(self._stage2, "predict_proba"):
                s2_proba = np.asarray(self._stage2.predict_proba(emb)[0], dtype=float)
                s2_classes = (
                    list(self._stage2.classes_)
                    if hasattr(self._stage2, "classes_")
                    else self._attack_classes
                )
                top_idx = int(np.argmax(s2_proba))
                tech_class = str(s2_classes[top_idx])
            elif hasattr(self._stage2, "predict"):
                tech_class = str(self._stage2.predict(emb)[0])

            return attack_score, tech_class

        # Path B: Single-stage multi-class classifier (backward compatibility)
        if self._classifier is not None:
            classes = self._resolve_classes()
            if hasattr(self._classifier, "predict_proba"):
                proba_matrix = self._classifier.predict_proba(emb)
                proba = np.asarray(proba_matrix[0], dtype=float)
            elif hasattr(self._classifier, "decision_function"):
                df_arr = np.asarray(
                    self._classifier.decision_function(emb)[0], dtype=float
                )
                if df_arr.ndim == 0 or len(df_arr) == 1:
                    sig = 1.0 / (1.0 + np.exp(-float(df_arr)))
                    proba = np.array([1.0 - sig, sig])
                else:
                    exp_df = np.exp(df_arr - np.max(df_arr))
                    proba = exp_df / np.sum(exp_df)
            elif hasattr(self._classifier, "predict"):
                pred = self._classifier.predict(emb)[0]
                proba = np.zeros(len(classes), dtype=float)
                matched_idx = 0
                for i, c in enumerate(classes):
                    if str(c).lower() == str(pred).lower():
                        matched_idx = i
                        break
                proba[matched_idx] = 1.0
            else:
                raise TypeError(
                    "Classifier must implement predict_proba, decision_function, or predict"
                )

            none_idx = next(
                (i for i, c in enumerate(classes) if str(c).lower() == "none"), None
            )
            if none_idx is not None and none_idx < len(proba):
                attack_score = float(np.clip(1.0 - proba[none_idx], 0.0, 1.0))
            else:
                attack_score = float(np.max(proba))

            attack_candidates = [
                i
                for i, c in enumerate(classes)
                if str(c).lower() != "none" and i < len(proba)
            ]
            if attack_candidates:
                top_attack_idx = max(attack_candidates, key=lambda idx: proba[idx])
                tech_class = str(classes[top_attack_idx])
            else:
                tech_class = "keyword_override"

            return attack_score, tech_class

        raise RuntimeError(
            "Neither two-stage nor single-stage classifier is configured."
        )

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode prompt text into normalized 1D embedding."""
        if hasattr(self._encoder, "encode"):
            emb = self._encoder.encode(text)
        elif callable(self._encoder):
            emb = self._encoder(text)
        else:
            raise TypeError("Encoder must have an 'encode' method or be callable")

        if hasattr(emb, "cpu") and hasattr(emb, "numpy"):
            emb = emb.cpu().numpy()
        emb = np.asarray(emb)
        if emb.ndim > 1:
            emb = emb.squeeze(0)
        return emb

    def predict_sync(
        self,
        text: str | Sequence[str],
        views: Sequence[str] | None = None,
    ) -> DetectorResult:
        """Synchronously execute detection on prompt text or multiple preprocess views.

        Args:
            text: Primary prompt text string or sequence of prompt strings.
            views: Optional additional preprocess views (e.g. raw, normalized, chunks).

        Returns:
            DetectorResult conforming to PARALLAX protocol.
        """
        start_time = time.perf_counter()

        # Input validation:
        if not isinstance(text, str) or not text.strip():
            if views:
                candidate_views = [v for v in views if isinstance(v, str) and v.strip()]
            else:
                candidate_views = []
        else:
            candidate_views = [text]
            if views:
                for v in views:
                    if isinstance(v, str) and v.strip() and v not in candidate_views:
                        candidate_views.append(v)

        # Abstain on empty or invalid input
        if not candidate_views:
            latency_ms = self.measure_latency_ms(start_time)
            return DetectorResult(
                name=self.name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=round(latency_ms, 3),
                error="Empty or invalid input prompt",
            )

        try:
            self._ensure_loaded()

            # Score each view and aggregate using maximum attack score
            best_score: float = -1.0
            best_technique: str = "none"

            for view_text in candidate_views:
                emb = self._encode_text(view_text)
                view_score, view_technique = self._score_single_embedding(emb)
                if view_score > best_score:
                    best_score = view_score
                    best_technique = view_technique

            attack_score = float(np.clip(best_score, 0.0, 1.0))
            is_attack = bool(attack_score >= self.threshold)
            attack_class = best_technique if is_attack else "none"

            latency_ms = self.measure_latency_ms(start_time)
            return DetectorResult(
                name=self.name,
                is_attack=is_attack,
                score=round(attack_score, 4),
                attack_class=attack_class,
                latency_ms=round(latency_ms, 3),
                error=None,
            )

        except Exception as e:  # noqa: BLE001
            latency_ms = self.measure_latency_ms(start_time)
            return DetectorResult(
                name=self.name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=round(latency_ms, 3),
                error=str(e),
            )

    async def predict(
        self,
        text: str | Sequence[str],
        views: Sequence[str] | None = None,
    ) -> DetectorResult:
        """Asynchronously run prediction on prompt text or multiple views."""
        return await asyncio.to_thread(self.predict_sync, text, views)
