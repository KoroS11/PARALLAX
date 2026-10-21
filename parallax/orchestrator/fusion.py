"""Fusion strategies for combining multi-detector predictions with abstain-awareness."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult


@dataclass
class FusionResult:
    """Aggregated decision and metadata output from fusion layer."""

    decision: Decision
    final_technique: str | None = None
    disagreement: bool = False
    confidence: float | None = None
    results: list[DetectorResult] = field(default_factory=list)
    degraded: bool = False


@runtime_checkable
class FusionStrategy(Protocol):
    """Protocol for fusion algorithms."""

    def fuse(
        self,
        results: list[DetectorResult],
        prompt: str | None = None,
    ) -> FusionResult:
        """Combine individual detector outputs into a single fusion verdict."""
        ...


class BaseFusion(ABC):
    """Abstract base class with shared fusion utilities and fail-open/closed policies.

    Parameters
    ----------
    flag_threshold : float, default=0.5
        Threshold above which prompt is flagged.
    block_threshold : float, default=0.8
        Threshold above which prompt is blocked.
    fail_closed : bool, default=False
        If True, returns fail_closed_decision when fewer than min_successful_detectors succeed.
        If False (fail-open), allows or computes verdict with remaining detectors.
    min_successful_detectors : int, default=1
        Minimum number of non-abstained detector results required before triggering policy.
    fail_closed_decision : Decision, default=Decision.FLAG
        Decision returned when fail_closed policy triggers.
    """

    def __init__(
        self,
        flag_threshold: float = 0.5,
        block_threshold: float = 0.8,
        fail_closed: bool = False,
        min_successful_detectors: int = 1,
        fail_closed_decision: Decision = Decision.FLAG,
    ) -> None:
        if flag_threshold > block_threshold:
            raise ValueError(
                f"flag_threshold ({flag_threshold}) cannot exceed block_threshold ({block_threshold})"
            )
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold
        self.fail_closed = fail_closed
        self.min_successful_detectors = min_successful_detectors
        self.fail_closed_decision = fail_closed_decision

    @abstractmethod
    def fuse(
        self,
        results: list[DetectorResult],
        prompt: str | None = None,
    ) -> FusionResult:
        """Fuse detector outputs into a unified verdict."""
        ...

    def check_abstentions(
        self,
        results: list[DetectorResult],
    ) -> tuple[bool, list[DetectorResult], FusionResult | None]:
        """Check for abstentions and evaluate fail-closed / fail-open policies.

        Returns (is_degraded, valid_results, early_fusion_result).
        """
        valid = [r for r in results if r.is_attack is not None and r.error is None]
        is_degraded = len(valid) < len(results) or any(
            r.is_attack is None for r in results
        )

        if len(valid) < self.min_successful_detectors:
            if self.fail_closed:
                early_res = FusionResult(
                    decision=self.fail_closed_decision,
                    final_technique=None,
                    disagreement=False,
                    confidence=1.0,
                    results=results,
                    degraded=True,
                )
                return True, valid, early_res
            if not valid:
                early_res = FusionResult(
                    decision=Decision.ALLOW,
                    final_technique=None,
                    disagreement=False,
                    confidence=0.0,
                    results=results,
                    degraded=True,
                )
                return True, valid, early_res

        return is_degraded, valid, None

    def extract_technique(
        self,
        results: list[DetectorResult],
        is_attack: bool,
    ) -> str | None:
        """Extract multi-class attack technique attribution."""
        if not is_attack:
            return None

        # Priority 1: Check D1 (MiniLM multi-class specialist)
        for r in results:
            name_lower = r.name.lower()
            if "d1" in name_lower and r.attack_class:
                cls_clean = r.attack_class.strip().lower()
                if cls_clean not in ("none", "benign", "clean", ""):
                    return r.attack_class

        # Priority 2: Check other detectors that produce an attack_class (e.g. D5 judge)
        for r in results:
            if r.attack_class:
                cls_clean = r.attack_class.strip().lower()
                if cls_clean not in ("none", "benign", "clean", ""):
                    return r.attack_class

        return None

    def compute_disagreement(self, results: list[DetectorResult]) -> bool:
        """Compute whether individual non-abstained detectors disagreed."""
        valid = [r for r in results if r.is_attack is not None]
        if len(valid) < 2:
            return False
        has_attack = any(r.is_attack is True for r in valid)
        has_benign = any(r.is_attack is False for r in valid)
        return has_attack and has_benign

    def determine_decision(
        self,
        is_attack: bool,
        confidence: float | None,
    ) -> Decision:
        """Map binary attack status and continuous confidence to Decision enum."""
        if not is_attack:
            return Decision.ALLOW
        if confidence is not None and confidence >= self.block_threshold:
            return Decision.BLOCK
        return Decision.FLAG


class AnyFlagFusion(BaseFusion):
    """Flags if any detector flags; confidence is maximum score observed."""

    def fuse(
        self,
        results: list[DetectorResult],
        prompt: str | None = None,
    ) -> FusionResult:
        is_degraded, valid, early_res = self.check_abstentions(results)
        if early_res is not None:
            return early_res

        disagreement = self.compute_disagreement(results)

        if not valid:
            return FusionResult(
                decision=Decision.ALLOW,
                final_technique=None,
                disagreement=False,
                confidence=0.0,
                results=results,
                degraded=is_degraded,
            )

        flagged = [r for r in valid if r.is_attack is True]
        all_scores = [r.score for r in valid if r.score is not None]

        if flagged:
            is_attack = True
            flagged_scores = [r.score for r in flagged if r.score is not None]
            if flagged_scores:
                confidence = max(flagged_scores)
            elif all_scores:
                confidence = max(all_scores)
            else:
                confidence = 1.0
        else:
            is_attack = False
            confidence = max(all_scores) if all_scores else 0.0

        decision = self.determine_decision(is_attack, confidence)
        technique = self.extract_technique(results, is_attack)

        return FusionResult(
            decision=decision,
            final_technique=technique,
            disagreement=disagreement,
            confidence=round(confidence, 4) if confidence is not None else None,
            results=results,
            degraded=is_degraded,
        )


class MajorityVoteFusion(BaseFusion):
    """Flags if >= 50% of non-abstained detectors flag attack."""

    def fuse(
        self,
        results: list[DetectorResult],
        prompt: str | None = None,
    ) -> FusionResult:
        is_degraded, valid, early_res = self.check_abstentions(results)
        if early_res is not None:
            return early_res

        disagreement = self.compute_disagreement(results)

        if not valid:
            return FusionResult(
                decision=Decision.ALLOW,
                final_technique=None,
                disagreement=False,
                confidence=0.0,
                results=results,
                degraded=is_degraded,
            )

        flagged_count = sum(1 for r in valid if r.is_attack is True)
        total_count = len(valid)
        vote_ratio = flagged_count / total_count

        is_attack = vote_ratio >= 0.5

        if is_attack:
            flagged_scores = [
                r.score for r in valid if r.is_attack is True and r.score is not None
            ]
            if flagged_scores:
                confidence = sum(flagged_scores) / len(flagged_scores)
            else:
                confidence = vote_ratio
        else:
            all_scores = [r.score for r in valid if r.score is not None]
            if all_scores:
                confidence = sum(all_scores) / len(all_scores)
            else:
                confidence = vote_ratio

        decision = self.determine_decision(is_attack, confidence)
        technique = self.extract_technique(results, is_attack)

        return FusionResult(
            decision=decision,
            final_technique=technique,
            disagreement=disagreement,
            confidence=round(confidence, 4),
            results=results,
            degraded=is_degraded,
        )


DEFAULT_F1_WEIGHTS: dict[str, float] = {
    "d1": 0.85,
    "d1-minilm": 0.85,
    "d2": 0.90,
    "d2-deberta": 0.90,
    "d3": 0.88,
    "d3-promptguard": 0.88,
    "d4": 0.75,
    "d4-rules": 0.75,
    "d5": 0.82,
    "d5-judge": 0.82,
}


class F1WeightedVoteFusion(BaseFusion):
    """Weights detector predictions by validation F1 scores."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        default_weight: float = 0.80,
        flag_threshold: float = 0.5,
        block_threshold: float = 0.8,
        fail_closed: bool = False,
        min_successful_detectors: int = 1,
        fail_closed_decision: Decision = Decision.FLAG,
    ) -> None:
        super().__init__(
            flag_threshold=flag_threshold,
            block_threshold=block_threshold,
            fail_closed=fail_closed,
            min_successful_detectors=min_successful_detectors,
            fail_closed_decision=fail_closed_decision,
        )
        self.weights = (
            dict(weights) if weights is not None else dict(DEFAULT_F1_WEIGHTS)
        )
        self.default_weight = default_weight

    def get_weight(self, detector_name: str) -> float:
        """Resolve weight for a detector name using exact or prefix match."""
        if detector_name in self.weights:
            return self.weights[detector_name]
        lower_name = detector_name.lower()
        for k, w in self.weights.items():
            if k.lower() == lower_name or k.lower() in lower_name:
                return w
        return self.default_weight

    def fuse(
        self,
        results: list[DetectorResult],
        prompt: str | None = None,
    ) -> FusionResult:
        is_degraded, valid, early_res = self.check_abstentions(results)
        if early_res is not None:
            return early_res

        disagreement = self.compute_disagreement(results)

        if not valid:
            return FusionResult(
                decision=Decision.ALLOW,
                final_technique=None,
                disagreement=False,
                confidence=0.0,
                results=results,
                degraded=is_degraded,
            )

        attack_weight = 0.0
        total_weight = 0.0
        weighted_score_sum = 0.0

        for r in valid:
            w = self.get_weight(r.name)
            total_weight += w
            if r.is_attack is True:
                attack_weight += w

            score_val = (
                r.score if r.score is not None else (1.0 if r.is_attack else 0.0)
            )
            weighted_score_sum += w * score_val

        if total_weight > 0:
            vote_ratio = attack_weight / total_weight
            confidence = weighted_score_sum / total_weight
        else:
            vote_ratio = 0.0
            confidence = 0.0

        is_attack = vote_ratio >= 0.5
        decision = self.determine_decision(is_attack, confidence)
        technique = self.extract_technique(results, is_attack)

        return FusionResult(
            decision=decision,
            final_technique=technique,
            disagreement=disagreement,
            confidence=round(confidence, 4),
            results=results,
            degraded=is_degraded,
        )


class StackedMetaClassifier(BaseFusion):
    """LogisticRegression meta-classifier trained on detector score matrix."""

    def __init__(
        self,
        detector_names: list[str] | None = None,
        model: LogisticRegression | None = None,
        impute_value: float = 0.0,
        flag_threshold: float = 0.5,
        block_threshold: float = 0.8,
        fail_closed: bool = False,
        min_successful_detectors: int = 1,
        fail_closed_decision: Decision = Decision.FLAG,
    ) -> None:
        super().__init__(
            flag_threshold=flag_threshold,
            block_threshold=block_threshold,
            fail_closed=fail_closed,
            min_successful_detectors=min_successful_detectors,
            fail_closed_decision=fail_closed_decision,
        )
        self.detector_names = detector_names or ["d1", "d2", "d3"]
        self.impute_value = impute_value

        if model is not None:
            self.model = model
        else:
            self.model = self._build_default_model(len(self.detector_names))

    def _build_default_model(self, num_detectors: int) -> LogisticRegression:
        """Create a default fitted LogisticRegression model for out-of-the-box operation."""
        x_low = (
            np.zeros((4, num_detectors)) + np.array([0.02, 0.05, 0.15, 0.25])[:, None]
        )
        y_low = np.zeros(4, dtype=int)

        x_high = (
            np.zeros((4, num_detectors)) + np.array([0.65, 0.75, 0.85, 0.95])[:, None]
        )
        y_high = np.ones(4, dtype=int)

        X = np.vstack([x_low, x_high])
        y = np.concatenate([y_low, y_high])

        clf = LogisticRegression(C=1.0)
        clf.fit(X, y)
        return clf

    def fit(self, X: Any, y: Any) -> StackedMetaClassifier:
        """Fit or retrain the meta-classifier on validation scores matrix X and labels y."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        self.model = LogisticRegression(C=1.0)
        self.model.fit(X_arr, y_arr)
        return self

    def train_from_matrix(
        self,
        scores: np.ndarray | list[list[float]],
        labels: np.ndarray | list[int],
    ) -> StackedMetaClassifier:
        """Train meta-classifier directly from a 2D scores matrix and label vector."""
        return self.fit(scores, labels)

    def train_from_file(
        self,
        filepath: str | Path,
        feature_cols: list[str] | None = None,
        label_col: str = "label",
    ) -> StackedMetaClassifier:
        """Train meta-classifier loaded from a CSV, JSON, or NPZ file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Training scores file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".npz":
            data = np.load(path)
            X = data["X"] if "X" in data else data["scores"]
            y = data["y"] if "y" in data else data["labels"]
            return self.fit(X, y)

        if suffix == ".json":
            with open(path, encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, dict) and "X" in content and "y" in content:
                X = np.array(content["X"], dtype=float)
                y = np.array(content["y"], dtype=int)
                return self.fit(X, y)
            if isinstance(content, list):
                # List of record dicts
                rows = content
                cols = feature_cols or self.detector_names
                X_list = [
                    [float(row.get(c, self.impute_value)) for c in cols] for row in rows
                ]
                y_list = [int(row[label_col]) for row in rows]
                return self.fit(X_list, y_list)
            raise ValueError(f"Unsupported JSON format in {path}")

        # Default: CSV reader
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV file: {path}")

            cols = feature_cols or [
                c
                for c in reader.fieldnames
                if c != label_col and c in self.detector_names
            ]
            if not cols:
                # Fallback to all non-label columns
                cols = [c for c in reader.fieldnames if c != label_col]

            X_list = []
            y_list = []
            for row in reader:
                feat = [
                    float(row[c]) if row[c] != "" else self.impute_value for c in cols
                ]
                X_list.append(feat)
                y_list.append(int(float(row[label_col])))

            self.detector_names = cols
            return self.fit(X_list, y_list)

    def save(self, filepath: str | Path) -> None:
        """Persist trained meta-classifier and metadata to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "detector_names": self.detector_names,
            "impute_value": self.impute_value,
            "flag_threshold": self.flag_threshold,
            "block_threshold": self.block_threshold,
            "fail_closed": self.fail_closed,
            "min_successful_detectors": self.min_successful_detectors,
            "fail_closed_decision": self.fail_closed_decision,
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, filepath: str | Path) -> StackedMetaClassifier:
        """Load a persisted meta-classifier from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Classifier file not found: {path}")
        bundle = joblib.load(path)
        instance = cls(
            detector_names=bundle["detector_names"],
            model=bundle["model"],
            impute_value=bundle["impute_value"],
            flag_threshold=bundle["flag_threshold"],
            block_threshold=bundle["block_threshold"],
            fail_closed=bundle.get("fail_closed", False),
            min_successful_detectors=bundle.get("min_successful_detectors", 1),
            fail_closed_decision=bundle.get("fail_closed_decision", Decision.FLAG),
        )
        return instance

    def extract_features(self, results: list[DetectorResult]) -> np.ndarray:
        """Extract ordered feature vector from detector results matching detector_names."""
        features: list[float] = []

        result_map: dict[str, DetectorResult] = {}
        for r in results:
            result_map[r.name.lower()] = r

        for expected in self.detector_names:
            exp_lower = expected.lower()
            matched_result: DetectorResult | None = None
            if exp_lower in result_map:
                matched_result = result_map[exp_lower]
            else:
                for k, v in result_map.items():
                    if exp_lower in k:
                        matched_result = v
                        break

            if matched_result is not None:
                if matched_result.score is not None:
                    val = matched_result.score
                elif matched_result.is_attack is True:
                    val = 1.0
                elif matched_result.is_attack is False:
                    val = 0.0
                else:
                    val = self.impute_value
            else:
                val = self.impute_value

            features.append(val)

        return np.array(features, dtype=float)

    def fuse(
        self,
        results: list[DetectorResult],
        prompt: str | None = None,
    ) -> FusionResult:
        is_degraded, valid, early_res = self.check_abstentions(results)
        if early_res is not None:
            return early_res

        disagreement = self.compute_disagreement(results)

        if not valid:
            return FusionResult(
                decision=Decision.ALLOW,
                final_technique=None,
                disagreement=False,
                confidence=0.0,
                results=results,
                degraded=is_degraded,
            )

        features = self.extract_features(results)
        proba = self.model.predict_proba(features.reshape(1, -1))[0]

        if 1 in self.model.classes_:
            attack_idx = list(self.model.classes_).index(1)
            confidence = float(proba[attack_idx])
        else:
            confidence = float(proba[-1])

        is_attack = confidence >= self.flag_threshold
        decision = self.determine_decision(is_attack, confidence)
        technique = self.extract_technique(results, is_attack)

        return FusionResult(
            decision=decision,
            final_technique=technique,
            disagreement=disagreement,
            confidence=round(confidence, 4),
            results=results,
            degraded=is_degraded,
        )


class MeanFusion(BaseFusion):
    """Computes mean continuous attack score across non-abstained detectors.

    Flags attack if mean score >= flag_threshold, blocks if mean score >= block_threshold.
    """

    def fuse(
        self,
        results: list[DetectorResult],
        prompt: str | None = None,
    ) -> FusionResult:
        is_degraded, valid, early_res = self.check_abstentions(results)
        if early_res is not None:
            return early_res

        disagreement = self.compute_disagreement(results)

        if not valid:
            return FusionResult(
                decision=Decision.ALLOW,
                final_technique=None,
                disagreement=False,
                confidence=0.0,
                results=results,
                degraded=is_degraded,
            )

        scores = [
            r.score if r.score is not None else (1.0 if r.is_attack else 0.0)
            for r in valid
        ]
        mean_score = sum(scores) / len(scores) if scores else 0.0

        is_attack = mean_score >= self.flag_threshold
        decision = self.determine_decision(is_attack, mean_score)
        technique = self.extract_technique(results, is_attack)

        return FusionResult(
            decision=decision,
            final_technique=technique,
            disagreement=disagreement,
            confidence=round(mean_score, 4),
            results=results,
            degraded=is_degraded,
        )


def get_fusion_strategy(name: str = "any", **kwargs: Any) -> BaseFusion:
    """Factory to retrieve a configured fusion strategy by name."""
    norm = name.strip().lower().replace("_", "-")
    if norm in ("any", "any-flag", "max"):
        return AnyFlagFusion(**kwargs)
    if norm in ("majority", "majority-vote", "vote"):
        return MajorityVoteFusion(**kwargs)
    if norm in ("mean", "average", "avg"):
        return MeanFusion(**kwargs)
    if norm in ("f1", "f1-weighted", "f1-vote"):
        return F1WeightedVoteFusion(**kwargs)
    if norm in ("stacked", "meta", "stacked-meta", "stacker"):
        return StackedMetaClassifier(**kwargs)
    raise ValueError(f"Unknown fusion strategy: {name}")
