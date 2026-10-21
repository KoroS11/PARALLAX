"""Unit tests for RealEvaluatorEnsemble (hermetic fakes only, zero live models/GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parallax.contracts.detectors import Detector, DetectorResult
from parallax.eval.real_ensemble import RealEvaluatorEnsemble


class FakeDetector(Detector):
    """Hermetic mock detector tracking invocations."""

    def __init__(self, name: str, score: float = 0.85, is_attack: bool = True) -> None:
        self.name = name
        self.score = score
        self.is_attack = is_attack
        self.calls = 0

    async def predict(self, text: str) -> DetectorResult:
        self.calls += 1
        return DetectorResult(
            name=self.name,
            is_attack=self.is_attack,
            score=self.score,
            attack_class="mock_class",
            latency_ms=1.5,
            error=None,
        )


@pytest.fixture
def temp_cache_file(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "detector_scores.json"


@pytest.mark.asyncio
async def test_real_ensemble_cache_miss_and_hit(temp_cache_file: Path) -> None:
    d1 = FakeDetector("d1-minilm", score=0.90)
    d2 = FakeDetector("d2-deberta", score=0.80)
    d4 = FakeDetector("d4_rules", score=0.95)
    pf = FakeDetector("promptforest", score=0.70)

    ensemble = RealEvaluatorEnsemble(
        cache_path=temp_cache_file,
        detectors=[d4, d1, d2, pf],
        auto_load_detectors=False,
    )

    prompt = "Test attack prompt CANARY-9999"

    # 1. Cache Miss: detectors are evaluated
    results = await ensemble.evaluate_sample_async(prompt)
    assert len(results) == 4
    assert [r.name for r in results] == [
        "d4_rules",
        "d1-minilm",
        "d2-deberta",
        "promptforest",
    ]
    assert d1.calls == 1
    assert d2.calls == 1
    assert d4.calls == 1
    assert pf.calls == 1

    # Cache file was written to disk
    assert temp_cache_file.exists()
    with open(temp_cache_file, encoding="utf-8") as f:  # noqa: ASYNC230
        data = json.load(f)
    prompt_hash = RealEvaluatorEnsemble.compute_prompt_hash(prompt)
    assert prompt_hash in data
    assert data[prompt_hash]["prompt"] == prompt

    # 2. Cache Hit: second call should use cache, calls do NOT increment
    results_hit = await ensemble.evaluate_sample_async(prompt)
    assert len(results_hit) == 4
    assert d1.calls == 1  # Still 1
    assert d2.calls == 1  # Still 1
    assert d4.calls == 1  # Still 1
    assert pf.calls == 1  # Still 1

    # Verify score integrity from cache
    assert results_hit[0].score == 0.95
    assert results_hit[1].score == 0.90
    assert results_hit[2].score == 0.80
    assert results_hit[3].score == 0.70


def test_real_ensemble_synchronous_evaluation(temp_cache_file: Path) -> None:
    d1 = FakeDetector("d1-minilm", score=0.55)
    d4 = FakeDetector("d4_rules", score=0.10, is_attack=False)

    ensemble = RealEvaluatorEnsemble(
        cache_path=temp_cache_file,
        detectors=[d4, d1],
        auto_load_detectors=False,
    )

    prompt = "Benign operational query: check pump 4 status."
    results = ensemble.evaluate_sample(prompt)

    assert len(results) == 2
    assert results[0].name == "d4_rules"
    assert results[0].is_attack is False
    assert results[1].name == "d1-minilm"
    assert results[1].is_attack is True

    # Call again synchronously (cache hit)
    results2 = ensemble.evaluate_sample(prompt)
    assert len(results2) == 2
    assert d1.calls == 1
    assert d4.calls == 1


@pytest.mark.asyncio
async def test_real_ensemble_promptforest_degraded_label(temp_cache_file: Path) -> None:
    class FakePFWithoutClass(Detector):
        name = "promptforest"

        async def predict(self, text: str) -> DetectorResult:
            return DetectorResult(
                name="promptforest",
                is_attack=False,
                score=0.10,
                attack_class=None,
                latency_ms=5.0,
            )

    ensemble = RealEvaluatorEnsemble(
        cache_path=temp_cache_file,
        detectors=[FakePFWithoutClass()],
        auto_load_detectors=False,
    )

    results = await ensemble.evaluate_sample_async("Routine system status")
    assert len(results) == 1
    # Enforces degraded ensemble label
    assert results[0].attack_class == "degraded ensemble"


@pytest.mark.asyncio
async def test_real_ensemble_refuses_stale_cache_on_version_mismatch(
    temp_cache_file: Path,
) -> None:
    """Verify that score cache is refused when model version changes."""

    class VersionedDetector(Detector):
        def __init__(self, name: str, version: str, score: float) -> None:
            self.name = name
            self.version = version
            self.score = score
            self.calls = 0

        async def predict(self, text: str) -> DetectorResult:
            self.calls += 1
            return DetectorResult(name=self.name, score=self.score, is_attack=True)

    d_v1 = VersionedDetector("d2-deberta", version="v1.0.0", score=0.99)
    ensemble_v1 = RealEvaluatorEnsemble(
        cache_path=temp_cache_file,
        detectors=[d_v1],
        auto_load_detectors=False,
    )

    prompt = "Sample prompt for cache validation"
    res1 = await ensemble_v1.evaluate_sample_async(prompt, run_id="run-v1")
    assert d_v1.calls == 1
    assert res1[0].score == 0.99

    # Now create new ensemble with updated detector version v2.0.0
    d_v2 = VersionedDetector("d2-deberta", version="v2.0.0", score=0.88)
    ensemble_v2 = RealEvaluatorEnsemble(
        cache_path=temp_cache_file,
        detectors=[d_v2],
        auto_load_detectors=False,
    )

    # Cache should be REFUSED because v1.0.0 != v2.0.0; d_v2 should be called
    res2 = await ensemble_v2.evaluate_sample_async(prompt, run_id="run-v2")
    assert d_v2.calls == 1
    assert res2[0].score == 0.88
