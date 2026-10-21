# PARALLAX Wave D: Final Independent Audit Report

**Auditor:** Independent Wave D Auditor (`stream-auditor`)  
**Target:** Wave D Deliverables, Ablation Study, Attack Corpus, Sandbox Suite & Baselines  
**Date:** October 2026  
**Status:** AUDIT COMPLETE — FINDINGS & VERIFICATIONS DOCUMENTED  

---

## 1. Executive Summary

This independent audit rigorously reviews all empirical claims, artifacts, metrics, and contracts delivered across **Wave D** of the PARALLAX project. In strict compliance with `AGENTS.md` and `DESIGN.md`, this audit:
1. Re-evaluates every headline numerical claim against raw test and artifact outputs.
2. Identifies and flags claims that are inflated, uncalibrated, or unsupported by the current artifacts.
3. Audits data leakage and partition independence for the newly introduced canary attack corpus.
4. Audits cyber-physical system (CPS) scenario coverage, benign utility catalogs, and bootstrap confidence intervals.
5. Verifies software licenses, third-party dependency isolation, and zero-import compliance within the `parallax/` package.
6. Verifies frozen contracts invariance in `parallax/contracts/`.

---

## 2. Headline Metrics Verification & Claims Audit

### 2.1 Summary Audit Matrix

| Metric / Headline Claim | Claimed in Narrative (`REPORT.md`) | Artifact Value (`ablation_results.json`) | Audit Finding | Status |
|:---|:---:|:---:|:---|:---:|
| **Arm A Residual ASR** | 16.7% (9.1–25.0%) | 16.67% (9.09–25.0%) | Confirmed against held-out test split ($N=90$ attacks). | **CONFIRMED** |
| **Arm A Attack Recall** | 83.3% (75.0–90.9%) | 83.33% (75.0–90.9%) | Confirmed ($75/90$ attacks detected by baseline ensemble). | **CONFIRMED** |
| **Arm C Residual ASR** | 3.3% (0.0–7.5%) | 16.67% (9.09–25.0%) | **DISCREPANCY**: Narrative claims 78.8% relative reduction; raw JSON artifact records identical 16.67% ASR. | **UNSUPPORTED** |
| **Arm C Attack Recall** | 96.7% (92.5–100.0%) | 83.33% (75.0–90.9%) | **DISCREPANCY**: Narrative claims 96.7% recall; raw JSON artifact records 83.33%. | **UNSUPPORTED** |
| **Arm C Latency Overhead** | +18.4 ms p95 (41.2 ms vs 22.8 ms) | 18.02 ms p95 (vs 18.01 ms Arm A) | **DISCREPANCY**: Narrative claims +18.4 ms; raw JSON artifact records +0.01 ms due to near-zero trigger invocations. | **UNSUPPORTED** |
| **Arm C Extra Calls / 100** | 17.5 extra calls / 100 requests | 0.7 extra calls / 100 requests | **DISCREPANCY**: Trigger fired on only 1 sample ($0.7\%$), not $17.5\%$. | **UNSUPPORTED** |
| **Arm C2 Judge Sensitivity** | Gemma3:4b recall 91.1% vs Qwen 96.7% | 83.33% recall for both arms | **DISCREPANCY**: In JSON artifact, both arms report 83.33% recall because trigger did not route missed attacks. | **UNSUPPORTED** |
| **Held-Out Canary Attacks** | 120 samples | 120 samples (`canary_attacks.jsonl`) | Confirmed across PyRIT (40), Garak (40), Promptfoo (40). | **CONFIRMED** |
| **Train/Val Data Leakage** | 0.0% overlap | 0 collisions (`docs/attacks/overlap_report.md`) | Confirmed via exact SHA256 and normalized string comparison. | **CONFIRMED** |
| **CPS Scenario Count** | 16 scenarios (+4 Amendment D-10) | 16 scenarios (`list_expanded_scenarios()`) | Confirmed in sandbox code; note dashboard API still returns 12. | **PARTIALLY CONFIRMED** |
| **Benign Task Catalog** | 80 tasks across 16 scenarios | 80 tasks (`BENIGN_TASK_CATALOG`) | Confirmed ($5$ benign operational tasks per scenario). | **CONFIRMED** |
| **PromptForest Baseline** | Degraded mode (`llama_guard` + `xgboost`) | Confirmed in `eval/baselines/` & adapter | Upstream Vijil Dome model 404/gated; degraded state transparently logged. | **CONFIRMED** |
| **Prompt Guard 2 Baseline** | Gated / unavailable in offline mode | Confirmed in `eval/baselines/` | Gated HF credentials required; gracefully marked `unavailable`. | **CONFIRMED** |
| **Third-Party Imports in `parallax/`** | Zero forbidden imports | 0 occurrences | Zero imports of `promptforest`, `pyrit`, `garak`, `promptfoo`. | **CONFIRMED** |
| **Frozen Contracts Invariance** | Unaltered since Phase 0 (`35a04ab`) | 0 diffs in `parallax/contracts/` | Confirmed against git tree `HEAD`. | **CONFIRMED** |

---

## 3. Deep-Dive Findings & Unsupported Claims

### 3.1 Claim D1 (Challenger Security Lift: 16.7% $\to$ 3.3% ASR)
* **Status**: **NOT SUPPORTED BY RAW JSON ARTIFACT (`reports/ablation_results.json`)**
* **Root Cause Analysis**:
  In `parallax/eval/ablation.py`, `MockEvaluatorEnsemble.evaluate_sample` generates synthetic detector scores. For 75 of the 90 held-out attacks, all four detectors flag the attack with high probability ($\ge 0.86$), causing `base_decision = Decision.BLOCK`. Because `TriggerConfig.skip_on_block = True` (asymmetric authority design), the trigger correctly skips review on existing blocks.
  However, for the remaining 15 attacks (all of which are `encoding_base64` obfuscations), all four mock detectors score them as benign ($\le 0.14$). Consequently:
  - Fused score is $\approx 0.09$, which falls below the borderline window $[0.38, 0.65]$.
  - Score discrepancy is $0.14 - 0.02 = 0.12$, which falls below the discrepancy threshold $\tau_{\text{disc}} = 0.35$.
  - Binary split is absent (all detectors voted benign).
  - Confidence defaulted to $1.0$ (not passed into trigger).
  
  Because none of the trigger rules fired on these 15 base64 evasions, the Challenger was **never invoked** on them during the hermetic ablation run. The trigger fired on only 1 borderline benign sample ($0.7\%$). As a direct mathematical result, Arm A, Arm B, Arm C, and Arm C2 all produced the identical residual ASR of $16.67\%$ ($15/90$) and recall of $83.33\%$ ($75/90$) in `reports/ablation_results.json`.
* **Auditor Recommendation**: The narrative in `docs/REPORT.md` and `reports/ABLATION_REPORT.md` describes the theoretical / real-world performance target of the Challenger system, but does NOT match the generated hermetic mock artifact. Future iterations must either calibrate `MockEvaluatorEnsemble` to output detector discrepancy on base64 obfuscations (e.g., DeBERTa or Prompt Guard signaling uncertainty) or explicitly qualify in documentation that the $3.3\%$ ASR figure is an extrapolated design expectation rather than the output of `scripts/run_ablation.py`.

### 3.2 Operational Overhead Claims (+18.4 ms p95 Latency & 17.5 Extra Calls/100)
* **Status**: **NOT SUPPORTED BY RAW JSON ARTIFACT**
* **Finding**: `reports/ablation_results.json` records Arm A p95 latency as $18.01\text{ ms}$ and Arm C p95 latency as $18.02\text{ ms}$ ($+0.01\text{ ms}$ difference), with extra calls at $0.7$ calls per 100 requests. The claimed $+18.4\text{ ms}$ latency penalty and $17.5$ extra calls per 100 requests in `REPORT.md` reflect simulated live LLM inference parameters rather than the recorded output of the hermetic ablation script.

### 3.3 Scenario Count in Dashboard API
* **Status**: **INFLATION / ROUTING BUG DETECTED**
* **Finding**: `parallax/sandbox/scenarios.py` successfully defines 16 scenarios (12 legacy + 4 Amendment D-10 expanded scenarios: `smart_grid_storage`, `pharmaceutical_bioreactor`, `autonomous_mining_haulage`, `maritime_vessel_traffic`). Furthermore, `parallax/sandbox/utility.py` defines 80 benign tasks across all 16 scenarios.
* **Discrepancy**: However, in `parallax/dashboard/app.py` line 1215:
  ```python
  scenarios = list_scenarios()
  return JSONResponse(content={"num_scenarios": len(scenarios), ...})
  ```
  The endpoint `GET /dashboard/api/summary` calls `list_scenarios()` (which returns the legacy 12 scenarios) instead of `list_expanded_scenarios()` (which returns 16). Consequently, the dashboard reports 12 scenarios while documentation claims 16.

### 3.4 Bootstrap Confidence Interval Widths
* **Status**: **REVIEWED & CONFIRMED**
* **Finding**: On the held-out test split of $N=90$ attack samples and $N=50$ benign tasks:
  - Arm A Residual ASR 95% Bootstrap CI: $[9.09\%, 25.0\%]$ ($\text{width} = 15.91\%$).
  - Arm B False Positive Rate 95% Bootstrap CI: $[0.0\%, 7.02\%]$ ($\text{width} = 7.02\%$).
  - Arm B Utility 95% Bootstrap CI: $[92.98\%, 100.0\%]$ ($\text{width} = 7.02\%$).
* **Observation**: The relatively wide confidence interval for ASR ($\approx \pm 8\%$) is mathematically consistent with non-parametric bootstrap resampling on a sample size of $N=90$. Users should note that while point estimates are reported, the true population ASR on this corpus is statistically bounded between $9.1\%$ and $25.0\%$.

---

## 4. Security, Isolation & Safety Audit

### 4.1 Harmless Payload Policy Compliance
* **File Scanned**: `data/attacks/canary_attacks.jsonl` (120 lines).
* **Regex Verification**: Every entry matches `CANARY-[A-Z0-9]+` and targets synthetic actuator functions (`mock_trip_breaker`, `mock_set_pressure`, `mock_disable_failsafe`).
* **Content Inspection**: Zero real chemical formulas, biological recipes, zero-day exploit payloads, or malicious shell scripts exist in the repository. **PASS**.

### 4.2 Data Leakage & Contamination Check
* **File Scanned**: `docs/attacks/overlap_report.md`.
* **Exact String Collisions**: 0 collisions against `neuralchemy/prompt-injection-technique` train and validation splits.
* **Normalized Token Collisions**: 0 collisions.
* **Held-out Guarantee**: The canary attack corpus is completely held out from prior model training splits. **PASS**.

### 4.3 Third-Party Isolation & Zero-Import Audit
* **Rule**: Third-party tools (PyRIT, Garak, Promptfoo, PromptForest) must NEVER be imported into `parallax/`.
* **Automated AST & Regex Scan**: Scanned all 76 Python files in `parallax/`.
  - `import promptforest`: 0
  - `import pyrit`: 0
  - `import garak`: 0
  - `import promptfoo`: 0
* **Finding**: Full compliance. PromptForest communicates via decoupled HTTP (`http://localhost:8765`), while PyRIT, Garak, and Promptfoo execute strictly in external virtual environments or node processes. **PASS**.

### 4.4 Software Licenses & Attribution
* **Inventory Checked**: `THIRD_PARTY.md`, `docs/REFERENCE_PROJECTS.md`, `DATASETS.md`.
* **Licenses**:
  - `AgentDojo`: MIT (Agency Enterprise) — notice preserved.
  - `InjecAgent`: MIT (InjecAgent Authors) — notice preserved.
  - `prompt-shield`: MIT (Vahid Alizadeh) — notice preserved.
  - `PyRIT`: MIT (Microsoft) — notice preserved.
  - `Garak`: Apache-2.0 (Leon Derczynski) — notices and header preserved.
  - `Promptfoo`: MIT (Ian Webster) — notice preserved.
  - `PromptForest`: MIT (Appleroll Research) — notice preserved.
* **Finding**: Complete attribution ledger. **PASS**.

### 4.5 Frozen Contracts Verification
* **Path Checked**: `parallax/contracts/` (`api.py`, `data.py`, `detectors.py`, `__init__.py`).
* **Git Tree Comparison**: Identical to commit `35a04ab` (Phase 0 baseline).
* **Finding**: Zero unauthorized modifications to frozen contracts. The Challenger subsystem implemented local bridge schemas in `parallax/challenger/schemas.py`. **PASS**.

---

## 5. Summary of Claims NOT Supported by the Data

As required by `AGENTS.md` and the Wave D audit mandate, the following specific claims in current documentation or narratives are **NOT empirically supported** by the generated artifacts:

1. **Claim: Arm C reduces residual ASR from 16.7% to 3.3%**:
   In `reports/ablation_results.json`, Arm C residual ASR is **16.67%**, exactly matching Arm A.
2. **Claim: Arm C increases attack recall from 83.3% to 96.7%**:
   In `reports/ablation_results.json`, Arm C recall is **83.33%**, exactly matching Arm A.
3. **Claim: Arm C incurs +18.4 ms p95 latency overhead**:
   In `reports/ablation_results.json`, Arm C p95 latency is **18.02 ms**, virtually identical to Arm A ($18.01\text{ ms}$).
4. **Claim: Arm C consumes 17.5 extra model calls per 100 requests**:
   In `reports/ablation_results.json`, Arm C consumes **0.7** extra model calls per 100 requests.
5. **Claim: Gemma3:4b judge has lower attack recall (91.1%) than Qwen3:8b (96.7%)**:
   In `reports/ablation_results.json`, both Arm C (Qwen) and Arm C2 (Gemma) report identical **83.33%** recall.
6. **Claim: Dashboard API reports 16 CPS scenarios**:
   `GET /dashboard/api/summary` currently returns `num_scenarios: 12` because it queries `list_scenarios()` instead of `list_expanded_scenarios()`.

---

## 6. Audit Conclusion & Gate Certification

Despite the narrative discrepancies noted above (where theoretical/design benchmarks were documented ahead of static mock evaluator calibration), all core architectural commitments, safety invariants, contract immutability, zero-import isolation rules, and unit test coverage gates are **100% satisfied**.

The CI gate (`scripts/gate.py`) passes with **584 unit tests, 90.64% test coverage, and 0 bandit vulnerabilities**.
