"""Unit tests for Wave F Final Evaluation script (eval-final)."""

from scripts.eval.eval_final import compute_metrics_with_cis, load_previous_evidence


def test_load_previous_evidence():
    """Verify loading of previously produced empirical runs."""
    evidence = load_previous_evidence()
    assert isinstance(evidence, dict)
    assert "baselines" in evidence
    assert "router" in evidence


def test_compute_metrics_with_cis():
    """Verify point estimates and cluster bootstrap CI computations."""
    y_true = [True, True, True, False, False]
    y_pred = [True, True, False, False, False]
    clusters = ["c1", "c1", "c2", "c3", "c3"]

    metrics = compute_metrics_with_cis(y_true, y_pred, clusters)
    assert "recall" in metrics
    assert "recall_ci_95" in metrics
    assert "fpr" in metrics
    assert "asr" in metrics
    assert 0.0 <= metrics["recall"] <= 1.0
    assert len(metrics["recall_ci_95"]) == 2
