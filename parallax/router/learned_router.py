"""Learned Review Router for PARALLAX (Wave F Stream 5).

Predicts when the primary detector ensemble verdict is likely to be incorrect
based on multi-detector score signals, score discrepancy, ensemble confidence,
preprocessing encoding anomaly flags, and input prompt length.

Prompts predicted as error-prone (or selected via random audit sampling) are
routed to the adversarial Challenger LLM for secondary evaluation.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult
from parallax.orchestrator.fusion import FusionResult
from parallax.preprocess.pipeline import preprocess

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "d2",
    "d3",
    "d4",
    "discrepancy",
    "confidence",
    "enc_present",
    "len",
]


@dataclass
class RouterFeatures:
    """Extracted numeric feature representations for router inference."""

    features_dict: dict[str, float]
    feature_vector: np.ndarray


@dataclass
class RouterDecision:
    """Routing verdict indicating whether Challenger LLM review is required.

    Attributes
    ----------
    should_route : bool
        True if the prompt should be escalated to the Challenger LLM.
    route_reason : str
        Reason for escalation ('learned_error_prediction', 'random_audit_sample', 'none').
    prob_error : float
        Predicted probability that the primary ensemble verdict is wrong.
    is_audit_sample : bool
        True if routing was triggered by stochastic random audit sampling.
    threshold : float
        Decision threshold applied to prob_error.
    """

    should_route: bool
    route_reason: str
    prob_error: float
    is_audit_sample: bool = False
    threshold: float = 0.35


def extract_router_features(
    prompt: str,
    results: list[DetectorResult],
    fused_result: FusionResult | None = None,
    feature_names: list[str] | None = None,
) -> RouterFeatures:
    """Extract standard routing features from prompt text, detector results, and fusion state.

    Wave H.3: Extracts features from calibrated D2 (DeBERTa), D3 (Deepset), D4 (Rules),
    omitting dropped D1 (MiniLM).

    Parameters
    ----------
    prompt : str
        Input prompt string.
    results : list[DetectorResult]
        Individual detector evaluation outcomes.
    fused_result : FusionResult, optional
        Pre-computed fusion outcome. If None, computes simple fallback metrics.
    feature_names : list[str], optional
        Active feature subset to order vector. Defaults to FEATURE_NAMES.

    Returns
    -------
    RouterFeatures
        Named dictionary and 1D numpy vector of features.
    """
    valid_scores: list[float] = []
    d1_raw: float | None = None
    d2_raw: float | None = None
    d3_raw: float | None = None
    rules_raw: float | None = None

    for r in results:
        if r.score is not None:
            valid_scores.append(float(r.score))
        name_lower = (r.name or "").lower()
        if "deepset" in name_lower or "d3" in name_lower or "third" in name_lower:
            if r.score is not None:
                d3_raw = float(r.score)
        elif "minilm" in name_lower or "d1" in name_lower:
            if r.score is not None:
                d1_raw = float(r.score)
        elif "rule" in name_lower or "d4" in name_lower:
            # Rules detector: binary indicator 1.0 if attack triggered, else 0.0
            if r.is_attack is True or (r.score is not None and float(r.score) > 0.5):
                rules_raw = 1.0
            elif r.score is not None:
                rules_raw = float(r.score)
            else:
                rules_raw = 0.0
        elif "deberta" in name_lower or "d2" in name_lower or "protectai" in name_lower:
            if r.score is not None:
                d2_raw = float(r.score)
        elif "promptforest" in name_lower or "pf" in name_lower:
            if (
                hasattr(r, "sub_scores")
                and isinstance(r.sub_scores, dict)
                and "deepset" in r.sub_scores
            ):
                d3_raw = float(r.sub_scores["deepset"])
            elif r.score is not None:
                d3_raw = float(r.score)

    if valid_scores:
        s_max = max(valid_scores)
        s_min = min(valid_scores)
        s_mean = float(np.mean(valid_scores))
        s_std = float(np.std(valid_scores))
        discrepancy = s_max - s_min
    else:
        s_max = 0.0
        s_min = 0.0
        s_mean = 0.5
        s_std = 0.0
        discrepancy = 0.0

    # Graceful handling for missing/abstaining detectors (sentinel/mean + indicator flag)
    d3_missing = 1.0 if d3_raw is None else 0.0
    d3_val = d3_raw if d3_raw is not None else s_mean
    d1_val = d1_raw if d1_raw is not None else s_mean
    d2_val = d2_raw if d2_raw is not None else s_mean
    d4_val = rules_raw if rules_raw is not None else 0.0

    if fused_result is not None and fused_result.confidence is not None:
        confidence = float(fused_result.confidence)
        is_block = 1.0 if fused_result.decision == Decision.BLOCK else 0.0
    else:
        confidence = abs(s_max - 0.5) * 2.0
        is_block = 1.0 if s_max >= 0.8 else 0.0

    # Preprocessing anomaly inspection
    prep = preprocess(prompt)
    enc_present = 1.0 if prep.encoding_present else 0.0
    num_enc_flags = float(len(prep.encoding_types))

    char_len = float(len(prompt))
    log_char_len = math.log1p(char_len)
    est_tokens = char_len / 4.0

    feat_dict: dict[str, float] = {
        "d2": d2_val,
        "d3": d3_val,
        "d4": d4_val,
        "discrepancy": discrepancy,
        "confidence": confidence,
        "enc_present": enc_present,
        "len": char_len,
        # Backwards compatibility and auxiliary telemetry
        "d1": d1_val,
        "d3_missing": d3_missing,
        "score_rules": d4_val,
        "score_minilm": d1_val,
        "score_deberta": d2_val,
        "score_deepset": d3_val,
        "score_promptforest": d3_val,
        "char_length": char_len,
        "encoding_present": enc_present,
        "num_encoding_flags": num_enc_flags,
        "log_char_length": log_char_len,
        "estimated_tokens": est_tokens,
        "score_max": s_max,
        "score_min": s_min,
        "score_mean": s_mean,
        "score_std": s_std,
        "fused_decision_is_block": is_block,
    }

    active_names = feature_names if feature_names is not None else FEATURE_NAMES
    vec = np.array([feat_dict[k] for k in active_names], dtype=float)
    return RouterFeatures(features_dict=feat_dict, feature_vector=vec)


class LearnedReviewRouter:
    """Machine learning review router predicting when the primary ensemble verdict is wrong."""

    def __init__(
        self,
        threshold: float = 0.35,
        random_audit_rate: float = 0.05,
        random_seed: int = 42,
        model: LogisticRegression | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.random_audit_rate = float(random_audit_rate)
        self.random_seed = int(random_seed)
        self._rng = random.Random(self.random_seed)
        self.feature_names = (
            list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        )

        if model is not None:
            self.model = model
            self._is_fitted = True
        else:
            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=500,
                            random_state=self.random_seed,
                        ),
                    ),
                ]
            )
            self._is_fitted = False

    def fit(
        self, X: np.ndarray | list[list[float]], y_wrong: np.ndarray | list[int]
    ) -> LearnedReviewRouter:
        """Fit the router on feature matrix X and binary target y_wrong (1 if ensemble error, 0 if correct)."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y_wrong, dtype=int)

        unique_labels = np.unique(y_arr)
        if len(unique_labels) < 2:
            # Handle edge case where all training samples are correct or all wrong
            logger.warning(
                "Training data contains only 1 class (%s). Creating fallback balanced synthetic data.",
                unique_labels,
            )
            synthetic_row = np.zeros((1, X_arr.shape[1]))
            synthetic_row[0, 4] = 0.8  # high discrepancy implies error
            X_arr = np.vstack([X_arr, synthetic_row])
            y_arr = np.concatenate([y_arr, [1 - unique_labels[0]]])

        self.model.fit(X_arr, y_arr)
        self._is_fitted = True
        return self

    def predict_proba_error(
        self,
        prompt: str,
        results: list[DetectorResult],
        fused_result: FusionResult | None = None,
    ) -> float:
        """Compute the estimated probability that the primary ensemble decision is wrong."""
        feat = extract_router_features(
            prompt, results, fused_result, feature_names=self.feature_names
        )
        if not self._is_fitted:
            # Heuristic fallback before fitting: high discrepancy or high encoding flags implies higher error risk
            discrepancy = feat.features_dict["discrepancy"]
            encoding = feat.features_dict["encoding_present"]
            return float(min(1.0, max(0.0, 0.2 + 0.5 * discrepancy + 0.3 * encoding)))

        vec = np.array([feat.features_dict[k] for k in self.feature_names], dtype=float)
        X = vec.reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        # Resolve class indices
        classes = getattr(self.model, "classes_", None)
        if classes is None and hasattr(self.model, "named_steps"):
            clf_step = self.model.named_steps.get("clf")
            classes = getattr(clf_step, "classes_", None)

        if classes is not None and 1 in classes:
            idx_wrong = list(classes).index(1)
            return float(proba[idx_wrong])
        return float(proba[-1])

    def route(
        self,
        prompt: str,
        results: list[DetectorResult],
        fused_result: FusionResult | None = None,
    ) -> RouterDecision:
        """Evaluate whether to route the prompt to the Challenger LLM.

        Parameters
        ----------
        prompt : str
            Input prompt string.
        results : list[DetectorResult]
            Individual detector results.
        fused_result : FusionResult, optional
            Primary ensemble fusion result.

        Returns
        -------
        RouterDecision
            Actionable routing decision with audit trail.
        """
        prob_error = self.predict_proba_error(prompt, results, fused_result)

        # 1. Learned error prediction threshold
        if prob_error >= self.threshold:
            return RouterDecision(
                should_route=True,
                route_reason="learned_error_prediction",
                prob_error=round(prob_error, 4),
                is_audit_sample=False,
                threshold=self.threshold,
            )

        # 2. Stochastic random audit sampling on allowed/low-risk prompts
        is_allowed = (fused_result is None) or (fused_result.decision != Decision.BLOCK)
        if (
            is_allowed
            and self.random_audit_rate > 0.0
            and self._rng.random() < self.random_audit_rate
        ):
            return RouterDecision(
                should_route=True,
                route_reason="random_audit_sample",
                prob_error=round(prob_error, 4),
                is_audit_sample=True,
                threshold=self.threshold,
            )

        return RouterDecision(
            should_route=False,
            route_reason="none",
            prob_error=round(prob_error, 4),
            is_audit_sample=False,
            threshold=self.threshold,
        )

    def save(self, path: str | Path) -> None:
        """Save fitted router model and parameters to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threshold": self.threshold,
            "random_audit_rate": self.random_audit_rate,
            "random_seed": self.random_seed,
            "is_fitted": self._is_fitted,
            "model": self.model if self._is_fitted else None,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, p)

    @classmethod
    def load(cls, path: str | Path) -> LearnedReviewRouter:
        """Load fitted router from serialized artifact."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Router artifact not found: {p}")
        payload = joblib.load(p)
        router = cls(
            threshold=payload["threshold"],
            random_audit_rate=payload["random_audit_rate"],
            random_seed=payload["random_seed"],
            model=payload.get("model"),
            feature_names=payload.get("feature_names", FEATURE_NAMES),
        )
        router._is_fitted = bool(payload.get("is_fitted", False))
        return router


# Alias for concise invocation
LearnedRouter = LearnedReviewRouter
