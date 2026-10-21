# Code State Audit vs Producing Commit 959a99a

This document audits all changes introduced in working trees or intermediate branches relative to producing commit `959a99a`. Each diff hunk has been reviewed and classified:
- **`LOGGING_ONLY`**: Telemetry, outcome recording, and non-semantic log tracking.
- **`TESTS_ONLY`**: Unit tests, synthetic fixtures, test assertions, or gating verification scripts.
- **`DOCS_ONLY`**: Documentation, schema definitions, markdown references, and export metadata.
- **`TOOLING_NOT_EVALUATED`**: Corpus preprocessing, dataset ingestion, and offline audit tooling not part of the runtime evaluation engine.
- **`BEHAVIOR_AFFECTING`**: Any change that can change a detector score, routing decision, policy decision, judge prompt, judge parameter, timeout, abstain mapping, or threshold.

> **Status of Evaluated System**: The evaluated system for this closeout is strictly the code committed at `959a99a` on branch `closeout`. All uncommitted/experimental changes have been safely committed and parked on branch `wip-amendment1-steps` (patch: `plans/parked/working_tree_vs_959a99a.patch`). They are completely excluded from the evaluated system.

---

## Summary of Reviewed Files

| File | Classification | Primary Rationale & Disposition |
|---|---|---|
| `docs/H3_TESTLOG.md` | `DOCS_ONLY` | Wave H.3 test execution log documentation. |
| `docs/METRICS.md` | `DOCS_ONLY` | Narrative documentation defining five outcomes and universal abstain rule. |
| `docs/VIZ_SCHEMA.md` | `DOCS_ONLY` | Schema documentation for visualization exports. |
| `eval/export/pilot_viz_data.json` | `DOCS_ONLY` | Exported visualization JSON data formatting. |
| `parallax/challenger/judge.py` | `LOGGING_ONLY` | Records `outcome` field on `JudgeResult` ("ok", "timeout", "error", "parse_fail", "model_abstain"). |
| `parallax/challenger/schemas.py` | `BEHAVIOR_AFFECTING` | **Parked; excluded from the evaluated system**. Modifies `map_challenger_verdict_to_decision` to force `flag_on_abstain=True` and unconditionally map `ABSTAIN` to `FLAG` (downgrading `BLOCK` to `FLAG`). Introduces `judge_result_to_decision`. Evaluated system uses `959a99a`. |
| `scripts/corpus/ingest_human.py` | `TOOLING_NOT_EVALUATED` | Offline corpus ingestion and validation tooling. Not executed during held-out evaluation. |
| `scripts/corpus/shortcut_audit.py` | `TOOLING_NOT_EVALUATED` | Offline statistical dataset audit tooling. **Note**: The modified metric in the parked version can change audit numbers; closeout reports strictly use the committed (`959a99a`) version. |
| `scripts/gate.py` | `TESTS_ONLY` | Static repository sanity and power checks. |
| `tests/test_ingest_pipeline.py` | `TESTS_ONLY` | Unit tests for synthetic human submission ingestion. |
| `check_final4.py` | `TOOLING_NOT_EVALUATED` | Standalone check utility parked on `wip-amendment1-steps`. |
| `scripts/eval/eval_detectors_s5.py` | `TOOLING_NOT_EVALUATED` | Standalone S5 detector evaluation script parked on `wip-amendment1-steps`. |
| `scripts/eval/validate_viz_export.py` | `TOOLING_NOT_EVALUATED` | Standalone visualization schema validator parked on `wip-amendment1-steps`. |
| `tests/test_eval_detectors_s5.py` | `TESTS_ONLY` | Unit tests for S5 evaluation parked on `wip-amendment1-steps`. |
| `tests/test_gate_v3_synthetic.py` | `TESTS_ONLY` | Synthetic test fixture for gate v3 parked on `wip-amendment1-steps`. |
| `tests/test_judge_timeout_abstain.py` | `TESTS_ONLY` | Unit tests for judge abstain mapping parked on `wip-amendment1-steps`. |

---

## Detailed Hunk Analysis

### 1. `parallax/challenger/schemas.py`
- **Classification**: **`BEHAVIOR_AFFECTING`** *(Parked, excluded from the evaluated system)*
- **Hunk Analysis**:
  - Hunk 1 (Lines 118-124): Adds `outcome` field (`LOGGING_ONLY`).
  - Hunk 2 (Lines 172-225): Modifies default parameter `flag_on_abstain: bool = True` (was `False`) and injects an unconditional `if verdict == ChallengerVerdict.ABSTAIN: return Decision.FLAG` rule preceding the asymmetry invariant. This alters policy decisions, causes existing unit tests in `test_challenger.py` to fail, and changes system behavior on judge abstentions.
  - Hunk 3 (Lines 226-241): Introduces `judge_result_to_decision(result: JudgeResult) -> Decision`.
- **System Disposition**: Excluded from evaluated system. Parked on branch `wip-amendment1-steps`. Evaluated code is at `959a99a`.

### 2. `scripts/corpus/shortcut_audit.py`
- **Classification**: **`TOOLING_NOT_EVALUATED`**
- **Hunk Analysis**: In `audit_split`, within-pair partners sharing a `pair_id` were excluded from the cross-label nearest-neighbor cosine similarity mask (`opp_mask[i, j] = False`).
- **Audit Note**: This modification changes the calculation of `max_cross_sim` and nearest-neighbor opposite-class statistics. Because this modified metric can change reported audit numbers, all closeout reports and analyses strictly use the committed `959a99a` code.

### 3. `scripts/corpus/ingest_human.py`
- **Classification**: **`TOOLING_NOT_EVALUATED`**
- **Hunk Analysis**: Alters duplicate validation logic (raising `ValueError` on exact duplicates and non-twin near-duplicates) and fills missing schema columns with defaults. This script is offline preprocessing tooling and is not invoked during runtime evaluation.

### 4. `parallax/challenger/judge.py`
- **Classification**: **`LOGGING_ONLY`**
- **Hunk Analysis**: Assigns outcome strings ("ok", "timeout", "error", "parse_fail", "model_abstain") to `outcome` attribute on `JudgeResult`. Does not change model inputs, temperatures, prompts, seeds, or decisions.

### 5. Documentation, Test, and Tooling Files
- `docs/H3_TESTLOG.md`, `docs/METRICS.md`, `docs/VIZ_SCHEMA.md`, `eval/export/pilot_viz_data.json`: Classified as **`DOCS_ONLY`**.
- `scripts/gate.py`, `tests/test_ingest_pipeline.py`, `tests/test_eval_detectors_s5.py`, `tests/test_gate_v3_synthetic.py`, `tests/test_judge_timeout_abstain.py`: Classified as **`TESTS_ONLY`**.
- `check_final4.py`, `scripts/eval/eval_detectors_s5.py`, `scripts/eval/validate_viz_export.py`: Classified as **`TOOLING_NOT_EVALUATED`**.

---

## Behavior-Affecting Paths for Future Same-System Checks

For future differential or same-system checks (including K13 Step B3), the following paths are classified as **`BEHAVIOR_AFFECTING`** and must have zero non-logging diffs relative to commit `959a99a`:

1. **Contracts & Enums**:
   - `parallax/contracts/`
2. **Detector Implementations & Ensembles**:
   - `parallax/detectors/` (all detector modules: `promptforest.py`, `promptguard2.py`, `deberta.py`, etc.)
3. **Thresholds, Configs, and Calibrations**:
   - `configs/`
   - All threshold constants (`tau`, Platt scaling parameters, decision cutoffs)
4. **Routing, Fusion, and Policy Composition**:
   - `parallax/fusion/`
   - `parallax/router/`
   - `parallax/challenger/` (`judge.py`, `schemas.py`, `challenger.py`, judge prompt templates)
   - `parallax/orchestrator/`
5. **Evaluation Runner Scripts Composing Arms and Policies**:
   - `scripts/eval/eval_final.py` (executes Policy A, B, and C; composes arms A, B, R, U, C2; calculates discrepancies)
   - `scripts/eval/run_adaptive_attack.py` (adaptive benchmark harness against defensive arms)
   - `scripts/eval/run_fusion_router_eval.py`
   - `scripts/eval/closeout_metrics.py` (ground-truth metric and confusion matrix extraction)
