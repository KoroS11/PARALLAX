# PARALLAX: Full Project Execution Run Report

**Project Title**: Parallel Adversarial-Prompt Recognition and Layered Aggregation for LLM eXploits  
**Status**: **ALL WAVES COMPLETED (100% SUCCESS)**  
**Date**: October 11, 2026  

---

## 1. Wave Execution Chronology

### Wave 0: Foundations & Project Baseline
- **Commit**: `b878a6b`
- **Scope**: Established project dependency configuration with `uv`, CI quality gate (`scripts/gate.py`), hardware concurrency manager (`scripts/gpu_lock.py`), application settings, and immutable contracts in `parallax/contracts/`.
- **Milestone Outcome**: 100% passing baseline, contracts frozen at commit `35a04ab`.

### Wave A: Core Subsystems Build
- **Commit**: `35a04ab` $\to$ `3503816`
- **Scope**:
  - Stream A1: Data schema, loaders, and split generation (`parallax/data/`).
  - Stream A2: Multi-view preprocessing, decoders, and normalizer (`parallax/preprocess/`).
  - Stream A3: FastAPI application gateway, middlewares, and settings (`parallax/api/`).
  - Stream A4: SQLite/PostgreSQL storage layer with write-behind queue (`parallax/storage/`).
  - Stream A5: Evaluation metrics framework and runners (`parallax/eval/`).
- **Milestone Outcome**: 5 parallel streams merged cleanly, 225 unit tests passing, 89.61% test coverage.

### Wave A QA: Quality Assurance & Hardening
- **Commit**: `dea1139`
- **Scope**: Adversarial fuzzing test suites, property-based hypothesis testing, independent architecture review report (`docs/reviews/wave-a.md`).
- **Milestone Outcome**: 336 unit tests passing, 89.90% test coverage, Wave A QA certified PASS.

### Wave B: Advanced Detectors & Orchestration v2
- **Commit**: `1d05eb6` $\to$ `b09c819`
- **Scope**:
  - Stream B1: Trained D1 MiniLM+Logistic Regression multi-class head (`artifacts/d1_minilm/classifier.joblib`, 239 KB).
  - Stream B2: Orchestrator v2 with circuit breakers, micro-batching, and fast-path cascade mode (`parallax/orchestrator/`).
  - Stream B3: D4 ReDoS-safe regex heuristics rules engine (`parallax/detectors/rules.py`).
  - Stream B4: D5 Local LLM Judge detector with structured JSON reasoning (`parallax/detectors/judge.py`).
  - Stream B5: Sandbox v2 with 12 cyber-physical operational scenarios and mock actuators (`parallax/sandbox/`).
- **Milestone Outcome**: 5 streams merged, 519 unit tests passing, 91.20% test coverage.

### Wave B QA: Chaos Engineering & Golden Verification
- **Commit**: `199404e`
- **Scope**:
  - Chaos test suite (`tests/chaos/test_chaos.py`): Circuit breaker state transitions, timeout isolation, queue backpressure.
  - Golden integration suite (`tests/golden/test_golden_eval.py`): 40 curated prompts evaluating end-to-end defense.
  - Wave B Architecture Review (`docs/reviews/wave-b.md`): Reviewer certification PASS.
- **Milestone Outcome**: 529 unit tests passing, 91.22% test coverage.

### Wave C: Empirical Benchmarks, Dashboard SPA, & Final Audit
- **Commit**: Current
- **Scope**:
  - Executed evaluation experiments E1 through E7 via `scripts/run_experiments.py`.
  - Exported benchmark reports, run manifest, and visualization plots to `reports/`.
  - Built dark cyber-terminal single-page application dashboard in `parallax/dashboard/app.py`.
  - Added unit test suite for dashboard in `tests/test_dashboard.py`.
  - Authored comprehensive Phase 2 technical report (`docs/REPORT.md`) with 10 Viva Defense Q&As.
  - Authored final compliance and security audit (`docs/AUDIT_FINAL.md`).
- **Milestone Outcome**: 533 unit tests passing, 91.19% test coverage, CI gate 100% clean.

---

## 2. Empirical Verification of Design Claims

| Claim | Objective | Experimental Result | Validation Status |
|:---|:---|:---|:---:|
| **Claim C1** | Heterogeneous detectors fail on different techniques; disagreement is informative. | **50.0% Disagreement Rate** observed in Experiment E7. Rules and embeddings exhibit orthogonal blindspots. | **CONFIRMED** |
| **Claim C2** | Parallel fan-out cuts end-to-end latency vs sequential execution. | **3.89x Speedup** (`p50: 16.12 ms` parallel fan-out vs `62.66 ms` sequential baseline in Experiment E5). | **CONFIRMED** |
| **Claim C3** | PARALLAX reduces attack success rate (ASR) against victim LLMs in a sandbox. | **85.7% $\to$ 0.0% ASR** in Experiment E6. 100% of attack episodes blocked before tool execution or canary leakage. | **CONFIRMED** |

---

## 3. Engineering Metrics Summary

- **Total Source Files**: 68 Python modules across `parallax/`.
- **Total Test Count**: 533 hermetic tests passing.
- **Total Code Coverage**: **91.19%** (exceeding 80% DoD gate).
- **Static Analysis**: Ruff Linter (0 errors), Ruff Formatter (0 violations), Mypy Strict (0 issues), Bandit Security Audit (0 high/medium issues).
- **Maximum Tracked File Size**: 239.8 KB (well within 1 MB limit).
- **Secret Leaks**: 0 detected.
- **Contract Drift**: 0 modified lines against baseline commit `35a04ab`.
