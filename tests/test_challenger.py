"""Tests for Wave D Adversarial Challenger Subsystem (Hermetic, no live models)."""

import asyncio
from typing import Any

import pytest

from parallax.challenger.challenger import ChallengerReviewer
from parallax.challenger.schemas import (
    ChallengerVerdict,
    JudgeResult,
    TriggerReason,
    map_challenger_verdict_to_decision,
)
from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig
from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult


class FakeJudge:
    """Hermetic fake judge for unit tests."""

    def __init__(
        self,
        verdict: ChallengerVerdict = ChallengerVerdict.BLOCK,
        confidence: float = 0.95,
        delay: float = 0.0,
        should_fail: bool = False,
    ) -> None:
        self.verdict = verdict
        self.confidence = confidence
        self.delay = delay
        self.should_fail = should_fail
        self.call_count = 0
        self.last_prompt = ""
        self.last_hypothesis = ""

    async def evaluate(
        self,
        prompt: str,
        counter_evidence_hypothesis: str,
        detector_summary: dict[str, Any],
    ) -> JudgeResult:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_hypothesis = counter_evidence_hypothesis

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise RuntimeError("Synthetic judge connection crash")

        return JudgeResult(
            verdict=self.verdict,
            confidence=self.confidence,
            reasoning="Fake reasoning identifying synthetic injection pattern",
            attack_class="mock_jailbreak",
            model="fake_qwen",
            latency_ms=12.5,
            extra_calls=1,
        )


def test_trigger_discrepancy() -> None:
    """Test trigger fires when discrepancy exceeds threshold."""
    config = TriggerConfig(discrepancy_threshold=0.4)
    trigger = ChallengerTrigger(config)

    # Det 1 says 0.85, Det 2 says 0.15 -> Delta = 0.70 >= 0.40
    results = [
        DetectorResult(name="d1", is_attack=True, score=0.85),
        DetectorResult(name="d2", is_attack=False, score=0.15),
    ]

    res = trigger.evaluate(results, fused_score=0.45)
    assert res.should_challenge is True
    assert TriggerReason.DISCREPANCY in res.reasons
    assert res.discrepancy_score == 0.70


def test_trigger_borderline_fused() -> None:
    """Test trigger fires when fused score is in borderline band [0.4, 0.65]."""
    trigger = ChallengerTrigger(TriggerConfig(discrepancy_threshold=0.9))

    results = [
        DetectorResult(name="d1", is_attack=False, score=0.48),
        DetectorResult(name="d2", is_attack=False, score=0.52),
    ]

    res = trigger.evaluate(results, fused_score=0.50)
    assert res.should_challenge is True
    assert TriggerReason.BORDERLINE in res.reasons


def test_trigger_low_confidence() -> None:
    """Test trigger fires when confidence drops below threshold."""
    trigger = ChallengerTrigger(TriggerConfig(min_confidence_threshold=0.6))

    results = [
        DetectorResult(name="d1", is_attack=False, score=0.2),
        DetectorResult(name="d2", is_attack=False, score=0.2),
    ]

    res = trigger.evaluate(results, fused_score=0.2, confidence=0.4)
    assert res.should_challenge is True
    assert TriggerReason.LOW_CONFIDENCE in res.reasons


def test_trigger_skip_on_immutable_block() -> None:
    """Test trigger skips evaluation if decision is already BLOCK (immutable under asymmetry)."""
    trigger = ChallengerTrigger(TriggerConfig(skip_on_block=True))

    results = [
        DetectorResult(name="d1", is_attack=True, score=0.99),
        DetectorResult(name="d2", is_attack=False, score=0.01),
    ]

    res = trigger.evaluate(results, original_decision=Decision.BLOCK)
    assert res.should_challenge is False


def test_asymmetric_verdict_mapping_block_invariant() -> None:
    """Invariant: BLOCK may NEVER be downgraded to ALLOW or FLAG."""
    # Even if judge explicitly says ALLOW, BLOCK stays BLOCK
    assert (
        map_challenger_verdict_to_decision(ChallengerVerdict.ALLOW, Decision.BLOCK)
        == Decision.BLOCK
    )
    assert (
        map_challenger_verdict_to_decision(ChallengerVerdict.ABSTAIN, Decision.BLOCK)
        == Decision.BLOCK
    )
    assert (
        map_challenger_verdict_to_decision(ChallengerVerdict.ESCALATE, Decision.BLOCK)
        == Decision.BLOCK
    )
    assert map_challenger_verdict_to_decision(None, Decision.BLOCK) == Decision.BLOCK


def test_asymmetric_verdict_mapping_escalations() -> None:
    """Test that Challenger can escalate ALLOW -> BLOCK or FLAG."""
    # ALLOW + Judge BLOCK -> Decision.BLOCK
    assert (
        map_challenger_verdict_to_decision(ChallengerVerdict.BLOCK, Decision.ALLOW)
        == Decision.BLOCK
    )

    # ALLOW + Judge ESCALATE -> Decision.BLOCK (when escalate_to_block=True)
    assert (
        map_challenger_verdict_to_decision(
            ChallengerVerdict.ESCALATE, Decision.ALLOW, escalate_to_block=True
        )
        == Decision.BLOCK
    )

    # ALLOW + Judge ESCALATE -> Decision.FLAG (when escalate_to_block=False)
    assert (
        map_challenger_verdict_to_decision(
            ChallengerVerdict.ESCALATE, Decision.ALLOW, escalate_to_block=False
        )
        == Decision.FLAG
    )

    # ALLOW + Judge ALLOW -> Decision.ALLOW
    assert (
        map_challenger_verdict_to_decision(ChallengerVerdict.ALLOW, Decision.ALLOW)
        == Decision.ALLOW
    )

    # ALLOW + Judge ABSTAIN -> Decision.ALLOW (fallback to original)
    assert (
        map_challenger_verdict_to_decision(ChallengerVerdict.ABSTAIN, Decision.ALLOW)
        == Decision.ALLOW
    )


@pytest.mark.asyncio
async def test_challenger_escalates_borderline_allow() -> None:
    """Verify ChallengerReviewer invokes judge and escalates contested ALLOW to BLOCK."""
    fake_judge = FakeJudge(verdict=ChallengerVerdict.BLOCK, confidence=0.98)
    reviewer = ChallengerReviewer(
        trigger=ChallengerTrigger(TriggerConfig(discrepancy_threshold=0.3)),
        judge=fake_judge,
    )

    results = [
        DetectorResult(
            name="d1_deberta", is_attack=True, score=0.75, attack_class="roleplay"
        ),
        DetectorResult(name="d2_minilm", is_attack=False, score=0.20),
    ]

    record = await reviewer.review(
        prompt="Synthesize a mock protocol override",
        detector_results=results,
        fused_score=0.48,
        original_decision=Decision.ALLOW,
    )

    assert fake_judge.call_count == 1
    assert record.is_escalated is True
    assert record.original_decision == Decision.ALLOW
    assert record.final_decision == Decision.BLOCK
    assert record.judge_verdict == ChallengerVerdict.BLOCK
    assert record.extra_model_calls == 1
    assert "ADVERSARIAL CHALLENGER COUNTER-EVIDENCE" in record.counter_evidence


@pytest.mark.asyncio
async def test_challenger_timeout_fallback() -> None:
    """Verify on judge timeout, Challenger preserves original verdict and logs error."""
    # Delay 0.2s with timeout 0.05s
    slow_judge = FakeJudge(delay=0.20)
    reviewer = ChallengerReviewer(
        trigger=ChallengerTrigger(),
        judge=slow_judge,
        timeout_seconds=0.05,
    )

    results = [
        DetectorResult(name="d1", is_attack=True, score=0.8),
        DetectorResult(name="d2", is_attack=False, score=0.1),
    ]

    record = await reviewer.review(
        prompt="Test timeout prompt",
        detector_results=results,
        fused_score=0.45,
        original_decision=Decision.ALLOW,
    )

    assert record.is_escalated is False
    assert record.final_decision == Decision.ALLOW  # Preserved original!
    assert record.error is not None
    assert "timed out" in record.error.lower()
    assert record.judge_verdict == ChallengerVerdict.ABSTAIN


@pytest.mark.asyncio
async def test_challenger_crash_fallback() -> None:
    """Verify on judge exception, Challenger preserves original verdict and logs error."""
    broken_judge = FakeJudge(should_fail=True)
    reviewer = ChallengerReviewer(
        trigger=ChallengerTrigger(),
        judge=broken_judge,
    )

    results = [
        DetectorResult(name="d1", is_attack=True, score=0.9),
        DetectorResult(name="d2", is_attack=False, score=0.1),
    ]

    record = await reviewer.review(
        prompt="Crash test prompt",
        detector_results=results,
        fused_score=0.5,
        original_decision=Decision.ALLOW,
    )

    assert record.is_escalated is False
    assert record.final_decision == Decision.ALLOW
    assert record.error is not None
    assert "Synthetic judge connection crash" in record.error
    assert record.judge_verdict == ChallengerVerdict.ABSTAIN


@pytest.mark.asyncio
async def test_challenger_skips_when_no_trigger() -> None:
    """Fast path: when detectors agree clearly, judge is never called."""
    fake_judge = FakeJudge()
    reviewer = ChallengerReviewer(judge=fake_judge)

    # Both clear benign: low scores, no discrepancy, high confidence
    results = [
        DetectorResult(name="d1", is_attack=False, score=0.05),
        DetectorResult(name="d2", is_attack=False, score=0.08),
    ]

    record = await reviewer.review(
        prompt="Standard harmless query",
        detector_results=results,
        fused_score=0.06,
        confidence=0.95,
        original_decision=Decision.ALLOW,
    )

    assert fake_judge.call_count == 0
    assert record.extra_model_calls == 0
    assert record.final_decision == Decision.ALLOW
    assert record.trigger_result.should_challenge is False
