"""D4: Rule/regex heuristics detector (adapted from prompt-shield).

Fast non-ML baseline evaluating regular expressions across multiple preprocess views
(raw, normalized, decoded) with strict ReDoS timeouts and technique categorization.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from parallax.contracts.detectors import DetectorResult
from parallax.detectors.base import BaseDetector
from parallax.detectors.patterns import ALL_RULES, Rule, RuleMatch


class RuleDetector(BaseDetector):
    """Rule-based heuristic prompt-injection detector (D4).

    Evaluates prompt text against categorized regex heuristics adapted from prompt-shield.
    Supports multi-view evaluation (raw, normalized, decoded views) and enforces strict
    ReDoS safety via bounded regex lengths and execution timeouts.

    Parameters
    ----------
    name : str, default="d4_rules"
        Detector identifier conforming to PARALLAX protocol.
    rules : list[Rule] | None, default=None
        Custom ruleset. If None, defaults to `ALL_RULES` catalog.
    threshold : float, default=0.5
        Confidence score threshold above which `is_attack` is True.
    rule_timeout : float, default=0.05
        Timeout in seconds allocated per individual regex evaluation (prevents ReDoS).
    max_eval_length : int, default=16384
        Maximum character length evaluated per view to prevent resource exhaustion.
    use_preprocess : bool, default=True
        Whether to generate normalized/decoded views via `preprocess()` if none are supplied.
    score_aggregation : str, default="max"
        Method for combining matched rule weights: 'max' (highest weight) or
        'probabilistic' (noisy-OR combination).
    """

    def __init__(
        self,
        name: str = "d4_rules",
        rules: list[Rule] | None = None,
        threshold: float = 0.5,
        rule_timeout: float = 0.05,
        max_eval_length: int = 16384,
        use_preprocess: bool = True,
        score_aggregation: str = "max",
    ) -> None:
        self.name = name
        self.rules: list[Rule] = list(rules) if rules is not None else list(ALL_RULES)
        self.threshold = float(threshold)
        self.rule_timeout = float(rule_timeout)
        self.max_eval_length = int(max_eval_length)
        self.use_preprocess = bool(use_preprocess)
        self.score_aggregation = score_aggregation.lower()

        # Precompile all rule regexes with strict length and case-insensitivity
        self._compiled_rules: list[tuple[Rule, Any]] = [
            (rule, rule.compile()) for rule in self.rules
        ]

    def _get_views(
        self, text: str, views: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Resolve evaluation views from explicit input or preprocessing pipeline."""
        if views is not None:
            resolved = dict(views)
            if "raw" not in resolved:
                resolved["raw"] = text
            return resolved

        if self.use_preprocess:
            try:
                from parallax.preprocess.pipeline import preprocess

                pre = preprocess(text, max_length=self.max_eval_length)
                return pre.views
            except Exception:  # noqa: BLE001
                # Fallback to raw text view if preprocess fails
                return {"raw": text}

        return {"raw": text}

    def evaluate_views(
        self, views: dict[str, str]
    ) -> tuple[float, str, list[RuleMatch]]:
        """Evaluate rules against a set of prompt views.

        Parameters
        ----------
        views : dict[str, str]
            Mapping of view names (e.g. 'raw', 'normalized', 'decoded') to strings.

        Returns
        -------
        tuple[float, str, list[RuleMatch]]
            (aggregated_score, predicted_technique, list_of_matches)
        """
        matches: list[RuleMatch] = []

        for view_name, view_text in views.items():
            if not isinstance(view_text, str) or not view_text.strip():
                continue

            # Strict character length bound per view to guarantee ReDoS defense
            bounded_text = view_text[: self.max_eval_length]

            for rule, pattern in self._compiled_rules:
                try:
                    # Enforce ReDoS timeout per pattern execution
                    match = pattern.search(bounded_text, timeout=self.rule_timeout)
                    if match:
                        matches.append(
                            RuleMatch(
                                rule_id=rule.rule_id,
                                technique=rule.technique,
                                weight=rule.weight,
                                view=view_name,
                                matched_text=match.group(0),
                                start=match.start(),
                                end=match.end(),
                                explanation=rule.explanation,
                            )
                        )
                except TimeoutError:
                    # Regex timed out on pathological input; skip to prevent hanging
                    continue

        if not matches:
            return 0.0, "none", []

        # Deduplicate matched rules by rule_id for scoring
        unique_matches: dict[str, RuleMatch] = {m.rule_id: m for m in matches}

        if self.score_aggregation == "probabilistic":
            prod = math.prod(1.0 - m.weight for m in unique_matches.values())
            score = 1.0 - prod
        else:  # default "max"
            score = max(m.weight for m in unique_matches.values())

        score = float(max(0.0, min(1.0, score)))

        # Attribute technique to the category with highest maximum rule weight
        technique_weights: dict[str, float] = {}
        for m in matches:
            if (
                m.technique not in technique_weights
                or m.weight > technique_weights[m.technique]
            ):
                technique_weights[m.technique] = m.weight

        best_technique = max(technique_weights.items(), key=lambda item: item[1])[0]

        return score, best_technique, matches

    def evaluate_detailed(
        self, text: str, views: dict[str, str] | None = None
    ) -> tuple[DetectorResult, list[RuleMatch]]:
        """Run detection and return both the DetectorResult and individual matches."""
        start_time = time.perf_counter()

        if not isinstance(text, str) or not text.strip():
            latency_ms = self.measure_latency_ms(start_time)
            return (
                DetectorResult(
                    name=self.name,
                    is_attack=None,
                    score=None,
                    attack_class=None,
                    latency_ms=round(latency_ms, 3),
                    error="Empty or invalid input prompt",
                ),
                [],
            )

        try:
            eval_views = self._get_views(text, views)
            score, technique, matches = self.evaluate_views(eval_views)

            is_attack = bool(score >= self.threshold)
            attack_class = technique if is_attack else "none"
            latency_ms = self.measure_latency_ms(start_time)

            result = DetectorResult(
                name=self.name,
                is_attack=is_attack,
                score=round(score, 4),
                attack_class=attack_class,
                latency_ms=round(latency_ms, 3),
                error=None,
            )
            return result, matches

        except Exception as e:  # noqa: BLE001
            latency_ms = self.measure_latency_ms(start_time)
            return (
                DetectorResult(
                    name=self.name,
                    is_attack=None,
                    score=None,
                    attack_class=None,
                    latency_ms=round(latency_ms, 3),
                    error=str(e),
                ),
                [],
            )

    def predict_sync(
        self, text: str, views: dict[str, str] | None = None
    ) -> DetectorResult:
        """Synchronously execute rule-based heuristic prediction."""
        result, _ = self.evaluate_detailed(text, views)
        return result

    async def predict(
        self, text: str, views: dict[str, str] | None = None
    ) -> DetectorResult:
        """Asynchronously execute rule-based prediction adhering to Detector Protocol."""
        return await asyncio.to_thread(self.predict_sync, text, views)


__all__ = ["RuleDetector"]
