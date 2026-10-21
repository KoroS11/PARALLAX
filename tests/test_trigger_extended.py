"""Unit tests for extended Challenger trigger criteria and telemetry (Wave E1.2)."""

from parallax.challenger.schemas import TriggerReason
from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig
from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult


def test_trigger_fires_on_encoding_detected() -> None:
    """Verify ChallengerTrigger fires when preprocessor reports encoding_present."""
    trigger = ChallengerTrigger(TriggerConfig(trigger_on_encoding=True))
    dets = [
        DetectorResult(name="d1", score=0.10, is_attack=False, latency_ms=1.0),
        DetectorResult(name="d4", score=0.05, is_attack=False, latency_ms=1.0),
    ]

    res = trigger.evaluate(
        detector_results=dets,
        fused_score=0.08,
        original_decision=Decision.ALLOW,
        encoding_present=True,
        encoding_types=["base64_decoded"],
    )

    assert res.should_challenge is True
    assert TriggerReason.ENCODING_DETECTED in res.reasons
    assert res.details.get("encoding_detected") is True


def test_trigger_fires_on_low_confidence_allowed() -> None:
    """Verify ChallengerTrigger fires when decision is ALLOW but fused score is close to threshold."""
    trigger = ChallengerTrigger(TriggerConfig(low_confidence_allow_margin=0.15))
    dets = [
        DetectorResult(name="d1", score=0.45, is_attack=False, latency_ms=1.0),
        DetectorResult(name="d4", score=0.42, is_attack=False, latency_ms=1.0),
    ]

    res = trigger.evaluate(
        detector_results=dets,
        fused_score=0.44,
        original_decision=Decision.ALLOW,
    )

    assert res.should_challenge is True
    assert TriggerReason.LOW_CONFIDENCE_ALLOWED in res.reasons


def test_trigger_random_audit_sampling() -> None:
    """Verify ChallengerTrigger random audit samples a fraction of ALLOWed queries."""
    trigger = ChallengerTrigger(
        TriggerConfig(
            trigger_on_encoding=False,
            low_confidence_allow_margin=0.0,
            discrepancy_threshold=0.99,
            random_audit_rate=1.0,  # 100% audit rate for deterministic test
        )
    )
    dets = [
        DetectorResult(name="d1", score=0.05, is_attack=False, latency_ms=1.0),
        DetectorResult(name="d4", score=0.05, is_attack=False, latency_ms=1.0),
    ]

    res = trigger.evaluate(
        detector_results=dets,
        fused_score=0.05,
        original_decision=Decision.ALLOW,
    )

    assert res.should_challenge is True
    assert TriggerReason.RANDOM_AUDIT in res.reasons


def test_trigger_telemetry_reporting() -> None:
    """Verify ChallengerTrigger tracks firing statistics per reason."""
    trigger = ChallengerTrigger(TriggerConfig(trigger_on_encoding=True))
    dets = [
        DetectorResult(name="d1", score=0.10, is_attack=False, latency_ms=1.0),
        DetectorResult(name="d4", score=0.10, is_attack=False, latency_ms=1.0),
    ]

    trigger.evaluate(
        detector_results=dets,
        fused_score=0.10,
        original_decision=Decision.ALLOW,
        encoding_present=True,
    )

    report = trigger.get_telemetry_report()
    assert report["total_evaluations"] == 1
    assert report["total_challenges"] == 1
    assert report["counts"]["encoding_detected"] == 1
    assert report["challenge_rate"] == 1.0
