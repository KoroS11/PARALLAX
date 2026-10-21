"""Comprehensive tests for PARALLAX evaluation framework.

Covers:
- Hand-computed golden value tests for precision, recall, F1, ECE, and fixed-FPR recall.
- Determinism and bounds of bootstrap confidence intervals under fixed seed.
- Run manifest generator, dataset hash computation, and JSON/Markdown exporters.
- E1 through E7 experiment runners with fake detectors conforming to Detector protocol.
- Edge cases: single-class inputs, zero division, empty sequences, abstentions, and errors.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from parallax.contracts.data import DatasetSample
from parallax.contracts.detectors import DetectorResult
from parallax.eval.metrics import (
    BinaryMetrics,
    BootstrapCI,
    ClassificationReport,
    ClassMetrics,
    ConfusionMatrixResult,
    FixedFPRResult,
    compute_binary_metrics,
    compute_bootstrap_ci,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_ece,
    compute_pr_auc,
    compute_recall_at_fpr,
    compute_roc_auc,
)
from parallax.eval.reports import (
    compute_dataset_hash,
    export_results_json,
    generate_markdown_report,
    generate_run_manifest,
    plot_confusion_matrix,
    plot_latency_comparison,
    plot_pr_curve,
    plot_roc_curve,
    save_run_manifest,
)
from parallax.eval.runners import (
    EvalRunResult,
    evaluate_detector,
    evaluate_detector_sync,
    get_default_synthetic_samples,
    run_e1_multiclass,
    run_e2_ood_robustness,
    run_e3_over_defense,
    run_e4_fusion_comparison,
    run_e5_latency,
    run_e6_asr,
    run_e7_disagreement,
)

# ===========================================================================
# 1. Hand-computed Golden Value Tests
# ===========================================================================


def test_golden_values_precision_recall_f1_binary() -> None:
    """Hand-computed verification of binary precision, recall, F1, and accuracy.

    Hand Calculation:
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 1, 1]

    Positives (1):
      TP = 3, FN = 1, FP = 2, TN = 2
      Precision = 3 / (3 + 2) = 3/5 = 0.60
      Recall    = 3 / (3 + 1) = 3/4 = 0.75
      F1        = 2 * (0.6 * 0.75) / (0.6 + 0.75) = 2/3 =~ 0.666667
      Accuracy  = (3 + 2) / 8 = 5/8 = 0.625

    Negatives (0):
      TP = 2, FN = 2, FP = 1, TN = 3
      Precision = 2 / (2 + 1) = 2/3 =~ 0.666667
      Recall    = 2 / (2 + 2) = 2/4 = 0.50
      F1        = 2 * ( (2/3) * 0.5 ) / (2/3 + 0.5) = 4/7 =~ 0.571429

    Macro F1 = (2/3 + 4/7) / 2 = 13/21 =~ 0.619048
    """
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 1, 1]

    # Binary metrics API
    bm = compute_binary_metrics(y_true, y_pred)
    assert bm.precision == pytest.approx(0.60, rel=1e-5)
    assert bm.recall == pytest.approx(0.75, rel=1e-5)
    assert bm.f1 == pytest.approx(2.0 / 3.0, rel=1e-5)
    assert bm.accuracy == pytest.approx(0.625, rel=1e-5)
    assert bm.support_positive == 4
    assert bm.support_negative == 4

    # Multi-class classification report API
    report = compute_classification_metrics(y_true, y_pred, labels=["0", "1"])
    assert report.accuracy == pytest.approx(0.625, rel=1e-5)
    assert report.macro_f1 == pytest.approx(13.0 / 21.0, rel=1e-5)

    c1 = report.per_class["1"]
    assert c1.precision == pytest.approx(0.60, rel=1e-5)
    assert c1.recall == pytest.approx(0.75, rel=1e-5)
    assert c1.f1 == pytest.approx(2.0 / 3.0, rel=1e-5)
    assert c1.support == 4

    c0 = report.per_class["0"]
    assert c0.precision == pytest.approx(2.0 / 3.0, rel=1e-5)
    assert c0.recall == pytest.approx(0.50, rel=1e-5)
    assert c0.f1 == pytest.approx(4.0 / 7.0, rel=1e-5)
    assert c0.support == 4


def test_golden_values_multiclass_three_classes() -> None:
    """Hand-computed verification for 3-class classification metrics.

    y_true = ["A", "A", "B", "B", "C", "C"]
    y_pred = ["A", "B", "B", "C", "C", "C"]

    For "A": TP=1, FP=0, FN=1 -> prec=1.0, rec=0.5, f1=2/3
    For "B": TP=1, FP=1, FN=1 -> prec=0.5, rec=0.5, f1=0.5
    For "C": TP=2, FP=1, FN=0 -> prec=2/3, rec=1.0, f1=0.8
    Macro F1 = (2/3 + 1/2 + 4/5) / 3 = (20/30 + 15/30 + 24/30) / 3 = 59/90 =~ 0.655556
    """
    y_true = ["A", "A", "B", "B", "C", "C"]
    y_pred = ["A", "B", "B", "C", "C", "C"]

    report = compute_classification_metrics(y_true, y_pred, labels=["A", "B", "C"])
    assert report.total_support == 6
    assert report.accuracy == pytest.approx(4.0 / 6.0, rel=1e-5)

    assert report.per_class["A"].precision == pytest.approx(1.0, rel=1e-5)
    assert report.per_class["A"].recall == pytest.approx(0.5, rel=1e-5)
    assert report.per_class["A"].f1 == pytest.approx(2.0 / 3.0, rel=1e-5)

    assert report.per_class["B"].precision == pytest.approx(0.5, rel=1e-5)
    assert report.per_class["B"].recall == pytest.approx(0.5, rel=1e-5)
    assert report.per_class["B"].f1 == pytest.approx(0.5, rel=1e-5)

    assert report.per_class["C"].precision == pytest.approx(2.0 / 3.0, rel=1e-5)
    assert report.per_class["C"].recall == pytest.approx(1.0, rel=1e-5)
    assert report.per_class["C"].f1 == pytest.approx(0.8, rel=1e-5)

    assert report.macro_f1 == pytest.approx(59.0 / 90.0, rel=1e-5)


def test_golden_values_ece() -> None:
    """Hand-computed verification of Expected Calibration Error (ECE).

    y_true  = [1, 0, 1, 0]
    y_score = [0.2, 0.1, 0.8, 0.9]
    n_bins  = 2 (bins: [0.0, 0.5), [0.5, 1.0])

    Bin 0 ([0.0, 0.5)):
      scores: [0.2, 0.1] -> mean conf = 0.15
      labels: [1, 0]     -> mean acc  = 0.50
      count = 2, weight = 2/4 = 0.5
      error = 0.5 * |0.50 - 0.15| = 0.5 * 0.35 = 0.175

    Bin 1 ([0.5, 1.0]):
      scores: [0.8, 0.9] -> mean conf = 0.85
      labels: [1, 0]     -> mean acc  = 0.50
      count = 2, weight = 2/4 = 0.5
      error = 0.5 * |0.50 - 0.85| = 0.5 * 0.35 = 0.175

    Total ECE = 0.175 + 0.175 = 0.350
    """
    y_true = [1, 0, 1, 0]
    y_score = [0.2, 0.1, 0.8, 0.9]
    ece = compute_ece(y_true, y_score, n_bins=2)
    assert ece == pytest.approx(0.35, abs=1e-6)

    # Perfect calibration test
    perfect_true = [0, 0, 1, 1]
    perfect_score = [0.0, 0.0, 1.0, 1.0]
    assert compute_ece(perfect_true, perfect_score, n_bins=2) == pytest.approx(0.0)


def test_golden_values_fixed_fpr_recall() -> None:
    """Hand-computed verification of Recall at fixed False Positive Rate (FPR).

    Negative items: 5 items with scores [0.1, 0.2, 0.3, 0.4, 0.5]
    Positive items: 5 items with scores [0.35, 0.6, 0.7, 0.8, 0.9]

    At max_fpr <= 0.0:
      Negative threshold must be > 0.5 (0 negatives pass, achieved_fpr = 0.0).
      Positives passing threshold > 0.5: [0.6, 0.7, 0.8, 0.9] -> 4 positives.
      Recall = 4/5 = 0.80.

    At max_fpr <= 0.20:
      Allow 1 negative to pass (score 0.5).
      Operating threshold = 0.5.
      Positives with score >= 0.5: [0.6, 0.7, 0.8, 0.9] -> 4 positives.
      Recall = 4/5 = 0.80.

    At max_fpr <= 0.60:
      Allow 3 negatives to pass (scores 0.3, 0.4, 0.5).
      Operating threshold = 0.35.
      All 5 positives pass (0.35, 0.6, 0.7, 0.8, 0.9).
      Recall = 5/5 = 1.00.
    """
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.35, 0.6, 0.7, 0.8, 0.9]

    # Target FPR 0.0: only 4 positives have score > max(negative)
    res_zero_fpr = compute_recall_at_fpr(y_true, y_score, max_fpr=0.0)
    assert res_zero_fpr.achieved_fpr == pytest.approx(0.0)
    assert res_zero_fpr.recall == pytest.approx(0.80)

    # Target FPR 0.60: all 5 positives caught
    res_high_fpr = compute_recall_at_fpr(y_true, y_score, max_fpr=0.60)
    assert res_high_fpr.achieved_fpr <= 0.60
    assert res_high_fpr.recall == pytest.approx(1.00)


def test_roc_auc_and_pr_auc_golden() -> None:
    """Verify ROC-AUC and PR-AUC calculations against known values."""
    # Perfectly separated data
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    roc = compute_roc_auc(y_true, y_score)
    pr = compute_pr_auc(y_true, y_score)
    assert roc == pytest.approx(1.0)
    assert pr == pytest.approx(1.0)

    # Inverted data (worst detector)
    inv_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert compute_roc_auc(y_true, inv_score) == pytest.approx(0.0)


# ===========================================================================
# 2. Determinism and Properties of Bootstrap Confidence Intervals
# ===========================================================================


def test_bootstrap_ci_determinism() -> None:
    """Ensure bootstrap confidence intervals are bitwise identical under fixed seed."""
    y_true = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]

    def accuracy_fn(yt: Sequence[int], yp: Sequence[int]) -> float:
        return sum(1 for t, p in zip(yt, yp, strict=True) if t == p) / len(yt)

    # Run twice with the exact same seed (42)
    ci_1 = compute_bootstrap_ci(accuracy_fn, y_true, y_pred, n_bootstraps=200, seed=42)
    ci_2 = compute_bootstrap_ci(accuracy_fn, y_true, y_pred, n_bootstraps=200, seed=42)

    assert ci_1.point_estimate == ci_2.point_estimate
    assert ci_1.lower == ci_2.lower
    assert ci_1.upper == ci_2.upper
    assert ci_1.mean == ci_2.mean
    assert ci_1.std == ci_2.std
    assert ci_1.seed == 42

    # Bounds validity: lower <= point_estimate <= upper (within reasonable margin)
    assert ci_1.lower <= ci_1.upper
    assert ci_1.point_estimate >= ci_1.lower - 0.1
    assert ci_1.point_estimate <= ci_1.upper + 0.1

    # Run with different seed (999) -> different resamples
    ci_diff = compute_bootstrap_ci(
        accuracy_fn, y_true, y_pred, n_bootstraps=200, seed=999
    )
    assert (
        ci_diff.point_estimate == ci_1.point_estimate
    )  # Analytical point estimate unchanged
    assert (ci_diff.mean != ci_1.mean) or (ci_diff.std != ci_1.std)


# ===========================================================================
# 3. Run Manifest Writer and Dataset Hash Tests
# ===========================================================================


def test_run_manifest_generation_and_save(tmp_path: Path) -> None:
    """Test run manifest creation, hardware/library discovery, and JSON saving."""
    dataset_hashes = {
        "synthetic_test": "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"
    }
    manifest = generate_run_manifest(
        run_id="run-test-001",
        experiment_id="E1",
        seed=1337,
        dataset_hashes=dataset_hashes,
        extra_metadata={"author": "eval-stream"},
    )

    assert manifest["run_id"] == "run-test-001"
    assert manifest["experiment_id"] == "E1"
    assert manifest["random_seed"] == 1337
    assert "timestamp_utc" in manifest
    assert "git" in manifest
    assert "git_sha" in manifest["git"]
    assert "git_branch" in manifest["git"]
    assert "hardware" in manifest
    assert "cpu_architecture" in manifest["hardware"]
    assert "gpu" in manifest["hardware"]
    assert "libraries" in manifest["environment"]
    assert "scikit-learn" in manifest["environment"]["libraries"]
    assert (
        manifest["dataset_hashes"]["synthetic_test"] == dataset_hashes["synthetic_test"]
    )
    assert manifest["metadata"]["author"] == "eval-stream"

    # Save to disk
    manifest_path = tmp_path / "manifest.json"
    saved = save_run_manifest(manifest, manifest_path)
    assert saved.exists()

    with open(saved, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["run_id"] == "run-test-001"


def test_dataset_hash_computation(tmp_path: Path) -> None:
    """Test dataset fingerprint computation on files and in-memory samples."""
    samples = [
        DatasetSample(
            text="Test injection prompt",
            label_multiclass="direct",
            label_binary=True,
            source="test_src",
            split="test",
        ),
        DatasetSample(
            text="Harmless prompt",
            label_multiclass="none",
            label_binary=False,
            source="test_src",
            split="test",
        ),
    ]

    h1 = compute_dataset_hash(samples)
    h2 = compute_dataset_hash(samples)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex string

    # File hash
    test_file = tmp_path / "data.txt"
    test_file.write_text("dataset contents for hashing", encoding="utf-8")
    fh1 = compute_dataset_hash(test_file)
    fh2 = compute_dataset_hash(str(test_file))
    assert fh1 == fh2
    assert len(fh1) == 64


# ===========================================================================
# 4. Mock Detectors for Runner Testing
# ===========================================================================


class FakeKeywordDetector:
    """Protocol-compliant detector using simple heuristics for testing."""

    def __init__(self, name: str = "fake_keyword_detector") -> None:
        self.name = name

    async def predict(self, text: str) -> DetectorResult:
        lower = text.lower()
        is_attack = any(
            kw in lower
            for kw in [
                "ignore",
                "canary",
                "dan",
                "override",
                "failsafe",
                "base64",
                "swdub",
                "exploit",
            ]
        )
        score = 0.95 if is_attack else 0.05

        attack_class = "none"
        if "dan" in lower:
            attack_class = "roleplay"
        elif "base64" in lower or "swdub" in lower:
            attack_class = "obfuscation"
        elif "failsafe" in lower:
            attack_class = "cps_mock_tool"
        elif is_attack:
            attack_class = "direct_instruction"

        return DetectorResult(
            name=self.name,
            is_attack=is_attack,
            score=score,
            attack_class=attack_class,
            latency_ms=1.2,
            error=None,
        )


class AbstainingDetector:
    """Mock detector that simulates timeouts / abstentions."""

    def __init__(self, name: str = "abstained_detector") -> None:
        self.name = name

    async def predict(self, text: str) -> DetectorResult:
        return DetectorResult(
            name=self.name,
            is_attack=None,
            score=None,
            attack_class=None,
            latency_ms=0.0,
            error="Detector timeout (2000 ms)",
        )


class FailingDetector:
    """Mock detector that raises an unexpected exception during predict."""

    def __init__(self, name: str = "failing_detector") -> None:
        self.name = name

    async def predict(self, text: str) -> DetectorResult:
        raise RuntimeError("GPU out of memory simulation")


# ===========================================================================
# 5. E1 - E7 Runner Tests with Fake Detectors
# ===========================================================================


@pytest.mark.asyncio
async def test_generic_evaluate_detector_with_fake() -> None:
    """Verify generic evaluate_detector runner across valid predictions and abstentions."""
    detector = FakeKeywordDetector()
    samples = get_default_synthetic_samples()

    result = await evaluate_detector(
        detector=detector,
        dataset=samples,
        experiment_id="test_run",
        include_sample_results=True,
        compute_cis=True,
    )

    assert isinstance(result, EvalRunResult)
    assert result.detector_name == "fake_keyword_detector"
    assert result.total_samples == len(samples)
    assert result.evaluated_samples == len(samples)
    assert result.abstained_samples == 0
    assert result.error_count == 0
    assert result.binary_metrics is not None
    assert result.binary_metrics.f1 > 0.0
    assert result.multiclass_report is not None
    assert "direct_instruction" in result.per_technique_recall
    assert result.latency_stats["p50_ms"] > 0.0
    assert len(result.sample_results) == len(samples)

    # Sync wrapper
    sync_res = evaluate_detector_sync(
        detector=detector,
        dataset=samples[:2],
        experiment_id="sync_test",
    )
    assert sync_res.total_samples == 2


@pytest.mark.asyncio
async def test_evaluate_detector_with_abstentions_and_failures() -> None:
    """Verify runner handles abstentions and errors gracefully without crashing."""
    samples = get_default_synthetic_samples()[:3]

    abstained_det = AbstainingDetector()
    res_abstain = await evaluate_detector(abstained_det, samples)
    assert res_abstain.abstained_samples == 3
    assert res_abstain.error_count == 3
    assert res_abstain.abstain_rate == 1.0
    assert res_abstain.binary_metrics is None

    failing_det = FailingDetector()
    res_fail = await evaluate_detector(failing_det, samples)
    assert res_fail.abstained_samples == 3
    assert res_fail.error_count == 3
    assert res_fail.error_rate == 1.0


@pytest.mark.asyncio
async def test_e1_multiclass_runner() -> None:
    """Test E1 in-distribution multi-class evaluation runner."""
    detector = FakeKeywordDetector()
    result = await run_e1_multiclass(detector)
    assert result.experiment_id == "E1"
    assert result.multiclass_report is not None
    assert result.multiclass_report.macro_f1 >= 0.0
    assert "confusion_matrix" in result.to_dict()["binary_metrics"]


@pytest.mark.asyncio
async def test_e2_ood_robustness_runner() -> None:
    """Test E2 OOD robustness evaluation runner."""
    detector = FakeKeywordDetector()
    result = await run_e2_ood_robustness(detector)
    assert result.experiment_id == "E2"
    assert "evasion_rate" in result.metadata
    assert 0.0 <= result.metadata["evasion_rate"] <= 1.0


@pytest.mark.asyncio
async def test_e3_over_defense_runner() -> None:
    """Test E3 over-defense evaluation runner."""
    detector = FakeKeywordDetector()
    result = await run_e3_over_defense(detector)
    assert result.experiment_id == "E3"
    assert "false_positive_rate" in result.metadata
    assert 0.0 <= result.metadata["false_positive_rate"] <= 1.0


@pytest.mark.asyncio
async def test_e4_fusion_comparison_runner() -> None:
    """Test E4 fusion strategies comparison runner."""
    detectors = [
        FakeKeywordDetector("det_a"),
        FakeKeywordDetector("det_b"),
    ]
    results = await run_e4_fusion_comparison(detectors)
    assert "any_flag" in results
    assert "majority_vote" in results
    assert "score_average" in results
    for strat, res in results.items():
        assert res.experiment_id == f"E4_{strat}"
        assert res.binary_metrics is not None


@pytest.mark.asyncio
async def test_e5_latency_runner() -> None:
    """Test E5 latency and throughput benchmark runner."""
    detectors = [
        FakeKeywordDetector("det_1"),
        FakeKeywordDetector("det_2"),
    ]
    lat_res = await run_e5_latency(detectors, num_trials=2)
    assert lat_res["experiment_id"] == "E5"
    assert "sequential" in lat_res
    assert "parallel" in lat_res
    assert "p50_speedup" in lat_res
    assert lat_res["sequential"]["p50_ms"] >= 0.0
    assert lat_res["parallel"]["p50_ms"] >= 0.0


@pytest.mark.asyncio
async def test_e6_asr_runner() -> None:
    """Test E6 end-to-end Attack Success Rate runner."""
    guard = FakeKeywordDetector()
    asr_res = await run_e6_asr(guard_detector=guard)
    assert asr_res["experiment_id"] == "E6"
    assert "asr_guard_off" in asr_res
    assert "asr_guard_on" in asr_res
    assert "asr_reduction" in asr_res
    assert asr_res["asr_reduction"] >= 0.0


@pytest.mark.asyncio
async def test_e7_disagreement_runner() -> None:
    """Test E7 detector disagreement analysis runner."""
    detectors = [
        FakeKeywordDetector("d1"),
        AbstainingDetector("d2"),
    ]
    disagree_res = await run_e7_disagreement(detectors)
    assert disagree_res["experiment_id"] == "E7"
    assert "disagreement_rate" in disagree_res
    assert "split_prompts" in disagree_res


# ===========================================================================
# 6. Reporting and Visualization Tests
# ===========================================================================


def test_markdown_and_json_reports(tmp_path: Path) -> None:
    """Test Markdown table generation and JSON exporter."""
    det = FakeKeywordDetector()
    samples = get_default_synthetic_samples()[:4]
    res = evaluate_detector_sync(det, samples, experiment_id="E1")

    manifest = generate_run_manifest("test_run", "E1", seed=42)

    # Markdown report
    md = generate_markdown_report(res, title="Test Report", manifest=manifest)
    assert "# Test Report" in md
    assert "## Environment & Provenance" in md
    assert "## Evaluation Results Summary" in md
    assert "E1" in md
    assert "fake_keyword_detector" in md

    # JSON export
    json_path = tmp_path / "results.json"
    export_results_json(res, json_path, manifest=manifest)
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["manifest"]["run_id"] == "test_run"
    assert data["results"]["experiment_id"] == "E1"


def test_plot_generators(tmp_path: Path) -> None:
    """Test generation of confusion matrix, ROC curve, PR curve, and latency plots."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix plot
    cm = ConfusionMatrixResult(
        labels=["benign", "attack"],
        matrix=[[40, 2], [3, 45]],
    )
    cm_path = plot_confusion_matrix(
        cm,
        output_path=reports_dir / "cm.png",
        normalize=True,
    )
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0

    # ROC curve plot
    y_true = [0, 0, 1, 1, 0, 1]
    y_scores = {
        "MiniLM": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
        "DeBERTa": [0.05, 0.15, 0.85, 0.95, 0.25, 0.75],
    }
    roc_path = plot_roc_curve(
        y_true,
        y_scores,
        output_path=reports_dir / "roc.png",
    )
    assert roc_path.exists()
    assert roc_path.stat().st_size > 0

    # PR curve plot
    pr_path = plot_pr_curve(
        y_true,
        y_scores,
        output_path=reports_dir / "pr.png",
    )
    assert pr_path.exists()
    assert pr_path.stat().st_size > 0

    # Latency comparison plot
    lat_data = {
        "Sequential": [12.0, 13.5, 11.8, 14.2],
        "Parallel": [5.2, 5.8, 5.1, 6.0],
    }
    lat_path = plot_latency_comparison(
        lat_data,
        output_path=reports_dir / "latency.png",
    )
    assert lat_path.exists()
    assert lat_path.stat().st_size > 0


# ===========================================================================
# 7. Edge Cases and Defensive Guards
# ===========================================================================


def test_edge_cases_empty_and_single_class() -> None:
    """Test edge cases: empty lists, single class, zero division."""
    # Empty inputs
    assert compute_roc_auc([], []) is None
    assert compute_pr_auc([], []) is None
    assert compute_ece([], []) == 0.0
    empty_rep = compute_classification_metrics([], [])
    assert empty_rep.accuracy == 0.0
    assert empty_rep.macro_f1 == 0.0

    # Single class
    assert compute_roc_auc([1, 1, 1], [0.8, 0.9, 0.7]) is None
    assert compute_pr_auc([0, 0, 0], [0.1, 0.2, 0.3]) is None
    fixed_res = compute_recall_at_fpr([1, 1, 1], [0.8, 0.9, 0.7], max_fpr=0.01)
    assert fixed_res.recall == 0.0

    # Length mismatch errors
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_classification_metrics([1, 0], [1])

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_confusion_matrix([1, 0], [1])

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_roc_auc([1, 0], [0.5])

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_pr_auc([1, 0], [0.5])

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_ece([1, 0], [0.5])

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_recall_at_fpr([1, 0], [0.5])

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_bootstrap_ci(lambda yt, yp: 0.0, [1, 0], [1])

    with pytest.raises(ValueError, match="empty sequences"):
        compute_bootstrap_ci(lambda yt, yp: 0.0, [], [])

    with pytest.raises(ValueError, match="n_bins must be greater than 0"):
        compute_ece([1, 0], [0.5, 0.2], n_bins=0)


def test_metric_dataclass_serializations() -> None:
    """Test to_dict and to_numpy serialization methods."""
    cm = ConfusionMatrixResult(labels=["0", "1"], matrix=[[1, 2], [3, 4]])
    assert isinstance(cm.to_numpy(), np.ndarray)
    assert cm.to_dict()["labels"] == ["0", "1"]

    cm_metrics = ClassMetrics(precision=0.8, recall=0.7, f1=0.75, support=10)
    assert cm_metrics.to_dict()["f1"] == 0.75

    fpr_res = FixedFPRResult(
        target_fpr=0.01, achieved_fpr=0.005, recall=0.9, threshold=0.85
    )
    assert fpr_res.to_dict()["recall"] == 0.9

    boot_ci = BootstrapCI(
        point_estimate=0.8,
        lower=0.7,
        upper=0.9,
        mean=0.8,
        std=0.05,
        alpha=0.05,
        seed=42,
    )
    assert boot_ci.to_dict()["point_estimate"] == 0.8

    report = ClassificationReport(
        per_class={"1": cm_metrics},
        macro_precision=0.8,
        macro_recall=0.7,
        macro_f1=0.75,
        weighted_f1=0.75,
        accuracy=0.8,
        total_support=10,
        confusion_matrix=cm,
    )
    rep_dict = report.to_dict()
    assert rep_dict["macro_f1"] == 0.75

    bm = BinaryMetrics(
        precision=0.8,
        recall=0.7,
        f1=0.75,
        accuracy=0.8,
        support_positive=5,
        support_negative=5,
    )
    assert bm.to_dict()["precision"] == 0.8


@pytest.mark.asyncio
async def test_experiment_skeletons_with_none_args() -> None:
    """Test E4, E5, E6, E7 default fallback branches when no detectors or datasets are provided."""
    # E4 with None
    e4_res = await run_e4_fusion_comparison(detectors=None, dataset=None)
    assert "any_flag" in e4_res

    # E5 with None
    e5_res = await run_e5_latency(detectors=None, prompts=None, num_trials=1)
    assert e5_res["experiment_id"] == "E5"

    # E6 with None
    e6_res = await run_e6_asr(guard_detector=None, episodes=None)
    assert e6_res["experiment_id"] == "E6"

    # E7 with None
    e7_res = await run_e7_disagreement(detectors=None, dataset=None)
    assert e7_res["experiment_id"] == "E7"


def test_reporting_edge_cases(tmp_path: Path) -> None:
    """Test reporting utility functions with various input formats."""
    # export_results_json with a raw dict
    dict_path = tmp_path / "dict_res.json"
    export_results_json({"test_key": "test_val"}, dict_path)
    assert dict_path.exists()

    # export_results_json with a list of EvalRunResult
    det = FakeKeywordDetector()
    samples = get_default_synthetic_samples()[:2]
    res1 = evaluate_detector_sync(det, samples, experiment_id="E1")
    res2 = evaluate_detector_sync(det, samples, experiment_id="E2")
    list_path = tmp_path / "list_res.json"
    export_results_json([res1, res2], list_path)
    assert list_path.exists()

    # generate_markdown_report with list
    md_list = generate_markdown_report([res1, res2], title="Multi Report")
    assert "Multi Report" in md_list
    assert "E1" in md_list
    assert "E2" in md_list

    # compute_dataset_hash with raw strings/objects
    str_hash = compute_dataset_hash(["sample_a", "sample_b"])
    assert len(str_hash) == 64
    obj_hash = compute_dataset_hash(12345)
    assert len(obj_hash) == 64
