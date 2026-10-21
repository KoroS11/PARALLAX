"""Unit tests for gate.py metric traceability validator (Wave E0)."""

import pytest

from scripts.gate import check_report_metrics_traceability


def test_gate_metrics_traceability_passes() -> None:
    """Verify that current docs/REPORT.md passes traceability check against results JSON."""
    check_report_metrics_traceability()


def test_gate_metrics_traceability_fails_on_unsupported_claim(
    monkeypatch, tmp_path
) -> None:
    """Verify that check_report_metrics_traceability exits with 1 if REPORT.md quotes unbacked claims."""
    fake_report = tmp_path / "REPORT.md"
    fake_report.write_text(
        "Arm C achieves 3.3% residual ASR in the benchmark.", encoding="utf-8"
    )

    import scripts.gate as gate_mod

    monkeypatch.setattr(gate_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "REPORT.md").write_text(
        fake_report.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(SystemExit) as exc_info:
        check_report_metrics_traceability()
    assert exc_info.value.code == 1
