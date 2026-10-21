"""Rule-based trigger for activating adversarial challenger review.

Evaluates detector disagreements, discrepancy scores, borderline fused scores,
and low ensemble confidence to determine whether a second-opinion challenger
should be invoked.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from parallax.challenger.schemas import TriggerReason, TriggerResult
from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult

logger = logging.getLogger(__name__)


@dataclass
class TriggerConfig:
    """Configurable thresholds for triggering adversarial review."""

    # Discrepancy threshold: Delta = max(score) - min(score)
    discrepancy_threshold: float = 0.35
    # Borderline interval: fused score in [borderline_low, borderline_high]
    borderline_low: float = 0.35
    borderline_high: float = 0.75
    # Minimum confidence threshold: if confidence < min_confidence_threshold
    min_confidence_threshold: float = 0.60
    # Binary split check: triggers if any detector flags attack while another flags benign
    check_binary_split: bool = True
    # Trigger on detected encoding or obfuscation (base64, hex, rot13, url, homoglyphs)
    trigger_on_encoding: bool = True
    # High-confidence allow ceiling: if fused_score <= this AND confidence >= high_confidence_threshold, gated out
    high_confidence_allow_threshold: float = 0.20
    # High-confidence block floor: if fused_score >= this AND confidence >= high_confidence_threshold, gated out
    high_confidence_block_threshold: float = 0.85
    # High confidence threshold for gating
    high_confidence_threshold: float = 0.70
    # Enforce uncertainty gating: only invoke challenger when uncertainty is flagged
    uncertainty_gating: bool = True
    # Router error probability threshold for learned router integration
    router_error_threshold: float = 0.35
    # Low-confidence-but-allowed margin: triggers if ALLOW but fused_score in [0.50 - margin, 0.50)
    low_confidence_allow_margin: float = 0.15
    # Random audit sample rate for ALLOWed prompts (default 0.0 for calibrated operational gating)
    random_audit_rate: float = 0.0
    # Random seed for reproducible audits
    random_audit_seed: int = 42
    # Only challenge when original decision is not BLOCK (kept False so uncertain BLOCK can be downgraded)
    skip_on_block: bool = False
    # Enabled toggle
    enabled: bool = True


class ChallengerTrigger:
    """Evaluates whether an incoming scan requires adversarial challenger escalation."""

    def __init__(self, config: TriggerConfig | None = None) -> None:
        self.config = config or TriggerConfig()
        self._rng = random.Random(self.config.random_audit_seed)
        self.telemetry: dict[str, int] = {
            "total_evaluations": 0,
            "total_challenges": 0,
            "discrepancy": 0,
            "borderline": 0,
            "low_confidence": 0,
            "binary_split": 0,
            "encoding_detected": 0,
            "low_confidence_allowed": 0,
            "random_audit": 0,
            "router_uncertainty": 0,
        }

    def evaluate(
        self,
        detector_results: Sequence[DetectorResult],
        fused_score: float | None = None,
        confidence: float | None = None,
        original_decision: Decision | None = None,
        encoding_present: bool = False,
        encoding_types: Sequence[str] | None = None,
        router_prob_error: float | None = None,
        router_should_route: bool | None = None,
    ) -> TriggerResult:
        """Evaluate trigger criteria against individual detector outputs and fusion state.

        Args:
            detector_results: List of outputs from individual detectors.
            fused_score: Continuous fused attack probability from ensemble.
            confidence: Ensemble confidence score.
            original_decision: The initial decision (ALLOW, FLAG, BLOCK).
            encoding_present: Whether preprocessor detected obfuscation or encodings.
            encoding_types: List of detected encoding types.
            router_prob_error: Probability of ensemble error predicted by learned router.
            router_should_route: Direct routing recommendation from learned router.

        Returns:
            TriggerResult detailing whether to challenge and the matching trigger reasons.
        """
        self.telemetry["total_evaluations"] += 1

        if not self.config.enabled:
            return TriggerResult(should_challenge=False, reasons=[TriggerReason.NONE])

        # If already blocked and skip_on_block is True, avoid wasting review compute
        if self.config.skip_on_block and original_decision == Decision.BLOCK:
            return TriggerResult(
                should_challenge=False,
                reasons=[],
                details={"reason": "already_blocked_and_immutable"},
            )

        reasons: list[TriggerReason] = []
        details: dict[str, object] = {}

        # 1. Extract valid continuous scores
        valid_scores = [
            r.score
            for r in detector_results
            if r.score is not None and r.error is None and not (r.is_attack is None)
        ]

        min_score = min(valid_scores) if valid_scores else 0.0
        max_score = max(valid_scores) if valid_scores else 0.0
        discrepancy_score = max_score - min_score if len(valid_scores) >= 2 else 0.0
        eff_fused = (
            fused_score
            if fused_score is not None
            else (sum(valid_scores) / len(valid_scores) if valid_scores else 0.0)
        )
        # Decision confidence semantics:
        # If confidence is not provided, or caller mistakenly passed raw attack score (confidence == fused_score)
        # on an ALLOW decision where attack probability is low (<= 0.5), compute true decision confidence as (1.0 - eff_fused).
        if confidence is not None:
            if (
                original_decision == Decision.ALLOW
                and fused_score is not None
                and abs(confidence - fused_score) < 1e-6
                and fused_score <= 0.5
            ):
                eff_conf = 1.0 - eff_fused
            else:
                eff_conf = confidence
        else:
            eff_conf = (
                (1.0 - eff_fused)
                if original_decision == Decision.ALLOW
                else (
                    eff_fused
                    if original_decision in (Decision.FLAG, Decision.BLOCK)
                    else 1.0
                )
            )

        details["valid_scores_count"] = len(valid_scores)
        details["min_score"] = min_score
        details["max_score"] = max_score

        # Check router uncertainty
        router_uncertain = (router_should_route is True) or (
            router_prob_error is not None
            and router_prob_error >= self.config.router_error_threshold
        )

        # 2. Uncertainty Gating: NEVER invoke on high-confidence allow or high-confidence block
        if self.config.uncertainty_gating:
            # High-confidence ALLOW gating:
            any_attack_flag = any(
                r.is_attack is True
                for r in detector_results
                if r.is_attack is not None and r.error is None
            )
            is_allow_traffic = original_decision == Decision.ALLOW or (
                original_decision is None
                and eff_fused <= self.config.high_confidence_allow_threshold
            )
            if (
                is_allow_traffic
                and eff_fused <= self.config.high_confidence_allow_threshold
                and eff_conf >= self.config.high_confidence_threshold
                and not any_attack_flag
                and not encoding_present
                and not router_uncertain
                and self.config.random_audit_rate == 0.0
            ):
                return TriggerResult(
                    should_challenge=False,
                    reasons=[TriggerReason.NONE],
                    discrepancy_score=round(discrepancy_score, 4),
                    min_score=round(min_score, 4),
                    max_score=round(max_score, 4),
                    fused_score=round(eff_fused, 4),
                    confidence=round(eff_conf, 4),
                    details={"gated": "high_confidence_allow"},
                )

            # High-confidence BLOCK gating:
            valid_dets = [
                r
                for r in detector_results
                if r.is_attack is not None and r.error is None
            ]
            all_attack_flags = len(valid_dets) > 0 and all(
                r.is_attack is True for r in valid_dets
            )
            if (
                original_decision == Decision.BLOCK
                and eff_fused >= self.config.high_confidence_block_threshold
                and eff_conf >= self.config.high_confidence_threshold
                and all_attack_flags
                and discrepancy_score < self.config.discrepancy_threshold
                and not router_uncertain
            ):
                return TriggerResult(
                    should_challenge=False,
                    reasons=[TriggerReason.NONE],
                    discrepancy_score=round(discrepancy_score, 4),
                    min_score=round(min_score, 4),
                    max_score=round(max_score, 4),
                    fused_score=round(eff_fused, 4),
                    confidence=round(eff_conf, 4),
                    details={"gated": "high_confidence_block"},
                )

        # Check Discrepancy Score: Delta >= tau_disc
        if (
            len(valid_scores) >= 2
            and discrepancy_score >= self.config.discrepancy_threshold
        ):
            reasons.append(TriggerReason.DISCREPANCY)
            details["discrepancy_exceeded"] = True
            self.telemetry["discrepancy"] += 1

        # Check Binary Split: any detector says True and any says False
        if self.config.check_binary_split:
            attacks = [
                r.is_attack
                for r in detector_results
                if r.is_attack is not None and r.error is None
            ]
            if True in attacks and False in attacks:
                reasons.append(TriggerReason.BINARY_SPLIT)
                details["binary_split"] = True
                self.telemetry["binary_split"] += 1

        # Check Borderline Fused Score: fused_score in [borderline_low, borderline_high]
        if self.config.borderline_low <= eff_fused <= self.config.borderline_high:
            reasons.append(TriggerReason.BORDERLINE)
            details["borderline_fused"] = eff_fused
            self.telemetry["borderline"] += 1

        # Check Learned Router Uncertainty
        if router_uncertain:
            reasons.append(TriggerReason.ROUTER_UNCERTAINTY)
            details["router_uncertainty"] = True
            details["router_prob_error"] = router_prob_error
            details["router_should_route"] = router_should_route
            self.telemetry["router_uncertainty"] += 1

        # Check Low Confidence: confidence < min_confidence_threshold
        eff_conf = confidence if confidence is not None else 1.0
        if eff_conf < self.config.min_confidence_threshold:
            reasons.append(TriggerReason.LOW_CONFIDENCE)
            details["low_confidence"] = eff_conf
            self.telemetry["low_confidence"] += 1

        # Check Encoding Detected: if preprocessor unpacked hidden payload
        if self.config.trigger_on_encoding and encoding_present:
            reasons.append(TriggerReason.ENCODING_DETECTED)
            details["encoding_detected"] = True
            details["encoding_types"] = list(encoding_types or [])
            self.telemetry["encoding_detected"] += 1

        # Check Low Confidence but Allowed (boundary defense)
        if (
            original_decision == Decision.ALLOW
            and self.config.low_confidence_allow_margin > 0.0
            and (0.50 - self.config.low_confidence_allow_margin) <= eff_fused < 0.50
        ):
            reasons.append(TriggerReason.LOW_CONFIDENCE_ALLOWED)
            details["low_confidence_allowed"] = True
            self.telemetry["low_confidence_allowed"] += 1

        # Check Random Audit Sample (to measure correlated blind spots on ALLOWed traffic)
        if (
            original_decision == Decision.ALLOW
            and self.config.random_audit_rate > 0.0
            and len(reasons) == 0
            and self._rng.random() < self.config.random_audit_rate
        ):
            reasons.append(TriggerReason.RANDOM_AUDIT)
            details["random_audit"] = True
            self.telemetry["random_audit"] += 1

        should_challenge = len(reasons) > 0
        if should_challenge:
            self.telemetry["total_challenges"] += 1

        return TriggerResult(
            should_challenge=should_challenge,
            reasons=reasons,
            discrepancy_score=round(discrepancy_score, 4),
            min_score=round(min_score, 4),
            max_score=round(max_score, 4),
            fused_score=round(eff_fused, 4),
            confidence=round(eff_conf, 4),
            details=details,
        )

    def get_telemetry_report(self) -> dict[str, Any]:
        """Return firing rates and counts for each trigger reason."""
        tot = self.telemetry["total_evaluations"]
        return {
            "total_evaluations": tot,
            "total_challenges": self.telemetry["total_challenges"],
            "challenge_rate": (
                (self.telemetry["total_challenges"] / tot) if tot > 0 else 0.0
            ),
            "counts": dict(self.telemetry),
            "rates": {
                k: (v / tot if tot > 0 else 0.0)
                for k, v in self.telemetry.items()
                if k not in ("total_evaluations", "total_challenges")
            },
        }
