"""Hermetic unit tests for Baseline Evaluation suite (Stream 6)."""

import pytest

from eval.baselines.runner import (
    BaselineEvaluationResult,
    check_promptguard2_baseline,
    evaluate_detector_on_samples,
    generate_baseline_markdown,
    load_benign_samples,
    load_held_out_attacks,
)
from parallax.contracts.detectors import Detector, DetectorResult


class MockBaselineDetector(Detector):
    name = "mock_promptforest"

    def __init__(self, attack_recall: float = 0.85, fpr: float = 0.05) -> None:
        self.attack_recall = attack_recall
        self.fpr = fpr
        self.loaded_detectors = ["llama_guard", "xgboost"]

    async def predict(self, text: str) -> DetectorResult:
        is_attack = "canary" in text.lower() or "override" in text.lower()
        score = 0.90 if is_attack else 0.10
        return DetectorResult(
            name=self.name,
            is_attack=is_attack,
            score=score,
            latency_ms=15.0,
            error=None,
        )


def test_load_held_out_attacks() -> None:
    """Verify loading held-out canary attacks returns valid attack entries."""
    attacks = load_held_out_attacks()
    assert len(attacks) > 0
    assert "text" in attacks[0]
    assert "canary" in attacks[0]


def test_load_benign_samples() -> None:
    """Verify loading benign operational task samples."""
    benign = load_benign_samples()
    assert len(benign) >= 16
    assert benign[0]["is_attack"] is False


@pytest.mark.asyncio
async def test_evaluate_detector_on_samples() -> None:
    """Verify metrics calculation over attacks and benign sets."""
    detector = MockBaselineDetector()
    attacks = [{"text": "Execute CANARY-9988 override"}] * 10
    benign = [{"text": "Read standard temperature telemetry"}] * 10

    metrics, latencies, loaded = await evaluate_detector_on_samples(
        detector, attacks, benign
    )

    assert metrics["recall"] == 1.0
    assert metrics["asr"] == 0.0
    assert len(latencies) == 20
    assert len(loaded) >= 2


def test_promptguard2_access_reporting() -> None:
    """Verify Prompt Guard 2 without token reports as unavailable with clear explanation."""
    res = check_promptguard2_baseline()
    assert res.name.startswith("Prompt Guard 2")
    if res.status == "unavailable":
        assert "access does not exist" in res.notes.lower()
        assert "gated" in res.notes.lower()


def test_generate_baseline_markdown() -> None:
    """Verify markdown table generation."""
    results = [
        BaselineEvaluationResult(
            name="PromptForest Alone",
            status="degraded",
            total_samples=200,
            attack_samples=120,
            benign_samples=80,
            recall=0.75,
            precision=0.88,
            f1=0.81,
            fpr=0.08,
            asr=0.25,
            p50_latency_ms=45.2,
            p95_latency_ms=88.1,
            notes="Degraded: vijil dome model 404",
        )
    ]
    md = generate_baseline_markdown(results)
    assert "# Baseline Guard Evaluation Report" in md
    assert "PromptForest Alone" in md
    assert "75.0%" in md
    assert "25.0%" in md
