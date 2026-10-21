"""Unit tests for sandbox utility evaluation and expanded CPS scenarios (Amendment D-10)."""

from __future__ import annotations

import pytest

from parallax.sandbox.scenarios import (
    ALL_CPS_SCENARIOS,
    EXPANDED_CPS_SCENARIOS,
    get_scenario,
    list_expanded_scenarios,
    list_scenarios,
)
from parallax.sandbox.utility import (
    BENIGN_TASK_CATALOG,
    ASRvsUtilityCurve,
    UtilityEvaluator,
    bootstrap_ci,
    explain_prior_asr_inflation,
)


def test_scenario_counts_core_and_expanded() -> None:
    # Core catalog remains strictly 12 for backwards compatibility
    assert len(ALL_CPS_SCENARIOS) == 12
    assert len(list_scenarios(expanded=False)) == 12

    # Expanded catalog contains exactly 16 scenarios (Amendment D-10)
    assert len(EXPANDED_CPS_SCENARIOS) == 16
    assert len(list_scenarios(expanded=True)) == 16
    assert len(list_expanded_scenarios()) == 16


def test_retrieval_of_all_16_scenarios() -> None:
    expected_ids = [
        "smart_grid_substation",
        "chemical_reactor",
        "robotic_surgery_arm",
        "autonomous_drone_fleet",
        "water_treatment_plant",
        "nuclear_cooling_loop",
        "railway_signaling",
        "smart_traffic_grid",
        "warehouse_agv",
        "pipeline_pressure",
        "hvac_life_support",
        "power_plant_turbine",
        "microgrid_bess",
        "pharma_cold_chain",
        "mining_conveyor",
        "satellite_attitude",
    ]
    for sc_id in expected_ids:
        sc = get_scenario(sc_id)
        assert sc.scenario_id == sc_id
        assert len(sc.safe_tools) >= 2
        assert len(sc.forbidden_tools) >= 2


def test_benign_task_catalog_coverage() -> None:
    assert len(BENIGN_TASK_CATALOG) == 16
    total_tasks = 0
    for sc_id, tasks in BENIGN_TASK_CATALOG.items():
        assert len(tasks) >= 5, f"Scenario {sc_id} has fewer than 5 benign tasks"
        total_tasks += len(tasks)
        for t in tasks:
            assert t.scenario_id == sc_id
            assert len(t.prompt) > 10
            assert len(t.expected_safe_tool) > 0
    assert total_tasks >= 80


def test_bootstrap_ci_calculation() -> None:
    data = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    low, high = bootstrap_ci(data, n_boot=500, ci=0.95, seed=42)
    assert 0.0 <= low <= high <= 1.0
    mean_val = sum(data) / len(data)
    assert low <= mean_val <= high

    # Edge cases
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([1.0]) == (1.0, 1.0)


@pytest.mark.asyncio
async def test_utility_evaluator_benign_pass() -> None:
    evaluator = UtilityEvaluator()

    # Mock guard that never blocks (returns False / 0.0)
    def benign_guard(prompt: str) -> float:
        return 0.05

    metrics = await evaluator.evaluate_guard(benign_guard, threshold=0.5)
    assert metrics.total_tasks == len(evaluator.tasks)
    assert metrics.passed_tasks == len(evaluator.tasks)
    assert metrics.blocked_tasks == 0
    assert metrics.utility == 1.0
    assert metrics.false_block_rate == 0.0
    assert metrics.utility_ci[0] == 1.0
    assert metrics.fbr_ci[1] == 0.0


@pytest.mark.asyncio
async def test_utility_evaluator_false_blocking() -> None:
    evaluator = UtilityEvaluator()

    # Mock guard that blocks tasks with 'temperature' or 'pressure' in prompt
    def aggressive_guard(prompt: str) -> float:
        if "temperature" in prompt.lower() or "pressure" in prompt.lower():
            return 0.95
        return 0.10

    metrics = await evaluator.evaluate_guard(aggressive_guard, threshold=0.5)
    assert metrics.blocked_tasks > 0
    assert metrics.passed_tasks > 0
    assert metrics.utility + metrics.false_block_rate == pytest.approx(1.0)
    assert 0.0 < metrics.false_block_rate < 1.0


def test_asr_vs_utility_curve_computation() -> None:
    # 20 attack scores and 20 benign scores
    attack_scores = [0.9, 0.85, 0.95, 0.4, 0.7, 0.99, 0.88, 0.92, 0.3, 0.8] * 2
    benign_scores = [0.05, 0.1, 0.02, 0.45, 0.15, 0.08, 0.12, 0.01, 0.2, 0.04] * 2

    curve = ASRvsUtilityCurve(attack_scores, benign_scores)
    points = curve.compute_curve()
    assert len(points) == 17

    for pt in points:
        assert 0.0 <= pt.asr <= 1.0
        assert 0.0 <= pt.utility <= 1.0
        assert 0.0 <= pt.false_block_rate <= 1.0

    optimal_thresh = curve.find_pareto_optimal_threshold()
    assert 0.1 <= optimal_thresh <= 0.9


def test_explain_prior_asr_inflation() -> None:
    explanation = explain_prior_asr_inflation()
    assert "Keyword Homogeneity" in explanation
    assert "Multi-Turn" in explanation
    assert "Operational Noise" in explanation
    assert "0.0%" in explanation
