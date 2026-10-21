# PARALLAX Wave E: Independent Scientific & Integrity Audit Report

**Auditor:** Independent Wave E Integrity Auditor (`stream-auditor-v2`)  
**Target:** Wave E Deliverables, Real Ensemble Ablation Rerun, Corpus Quality, Preprocessor/Trigger Upgrades, and Documentation Integrity  
**Date:** October 2026  
**Status:** AUDIT COMPLETE — 100% CLAIMS EMPIRICALLY SUPPORTED & CERTIFIED  

---

## 1. Executive Summary & Purpose of Wave E

Wave E was chartered with a singular mandate: **repair and make Wave D honest**. In Wave D, preliminary benchmarks utilized synthetic mock evaluators (`MockEvaluatorEnsemble`) that did not evaluate real transformer classifiers and reported extrapolated design expectations (3.3% ASR, 96.7% recall, +18.4 ms p95, 17.5 extra calls) that deviated from raw JSON outputs.

Wave E systematically remediated these discrepancies through four engineering waves:
1. **Wave E0 (Record Correction)**: Removed unbacked claims, labeled preliminary Wave C metrics as mock-assisted, integrated `list_expanded_scenarios()`, and implemented an automated CI gate rule (`check_report_metrics_traceability`) failing on unsupported claims.
2. **Wave E1.1 (Real Detector Ensemble)**: Replaced mock evaluators with real models (`RuleDetector`, `MiniLMDetector`, `DeBERTaDetector`, and degraded `PromptForestAdapter`), populating a deterministic disk score cache at `data/cache/detector_scores.json`.
3. **Wave E1.2 (Multi-View Preprocessing & Trigger Extension)**: Added decoding (Base64, Hex, URL, Rot13, homoglyphs) and extended trigger criteria (`ENCODING_DETECTED`, `LOW_CONFIDENCE_ALLOWED`, `RANDOM_AUDIT`).
4. **Wave E1.3 (Corpus Balance & Integrity)**: Balanced attack corpus to cap single-technique share at $\le 20\%$, added indirect injections, computed effective sample size ($N_{\text{eff}} = 11.1$), and partitioned dev/test splits by technique family cluster.
5. **Wave E2.1 (Ablation Rerun v2)**: Evaluated 5 architectural arms on held-out test data (90 attacks, 64 benign tasks across all 16 CPS scenarios) using 1,000 cluster-aware bootstrap resamples.

This audit report verifies that **every sentence and numerical metric in `docs/REPORT.md` directly traces to raw JSON artifacts**.

---

## 2. Headline Metrics Verification & Claims Audit

### 2.1 Comparative Ablation Metrics Verification (Held-Out Test Split)

All figures confirmed against `reports/ablation_results.json` ($N=90$ held-out attacks, $N=64$ held-out benign tasks, 1,000 cluster-aware bootstrap resamples):

| Metric / Parameter | Narrative in `REPORT.md` | Artifact Value in `ablation_results.json` | Audit Finding | Status |
|:---|:---:|:---:|:---|:---:|
| **Arm A Residual ASR** | **82.2%** (50.0–96.3%) | 0.8222 (0.5000, 0.9630) | Exact match; reflects real ensemble dilution flaw. | **CONFIRMED** |
| **Arm A Attack Recall** | **17.8%** (3.7–50.0%) | 0.1778 (0.0370, 0.5000) | Exact match; naive score average dilutes specialist blocks. | **CONFIRMED** |
| **Arm A False-Block Rate** | **0.0%** (0.0–0.0%) | 0.0 (0.0, 0.0) | Exact match; 0/64 benign tasks blocked. | **CONFIRMED** |
| **Arm A Utility** | **100.0%** (100.0–100.0%) | 1.0 (1.0, 1.0) | Exact match; 64/64 benign tasks executed successfully. | **CONFIRMED** |
| **Arm A Latency (p50 / p95)** | **2333.9 / 2352.6 ms** | 2333.91 ms / 2352.62 ms | Exact match. | **CONFIRMED** |
| **Arm A Extra Calls / 100** | **0.0** | 0.0 | Exact match; no review triggered. | **CONFIRMED** |
| **Arm B Residual ASR** | **24.4%** (7.7–44.4%) | 0.2444 (0.0769, 0.4444) | Exact match; discrepancy heuristic drops ASR by 57.8%. | **CONFIRMED** |
| **Arm B Attack Recall** | **75.6%** (55.6–92.3%) | 0.7556 (0.5556, 0.9231) | Exact match. | **CONFIRMED** |
| **Arm B False-Block Rate** | **6.2%** (1.6–11.7%) | 0.0625 (0.0156, 0.1167) | Exact match; 4/64 benign tasks falsely blocked. | **CONFIRMED** |
| **Arm B Utility** | **93.8%** (88.3–98.4%) | 0.9375 (0.8833, 0.9844) | Exact match; trade-off of heuristic escalation. | **CONFIRMED** |
| **Arm C Residual ASR** | **23.3%** (10.5–33.3%) | 0.2333 (0.1053, 0.3333) | Exact match; adversarial review achieves lowest bounded ASR. | **CONFIRMED** |
| **Arm C Attack Recall** | **76.7%** (66.7–89.5%) | 0.7667 (0.6667, 0.8947) | Exact match. | **CONFIRMED** |
| **Arm C False-Block Rate** | **0.0%** (0.0–0.0%) | 0.0 (0.0, 0.0) | Exact match; zero benign tasks blocked by LLM judge. | **CONFIRMED** |
| **Arm C Utility** | **100.0%** (100.0–100.0%) | 1.0 (1.0, 1.0) | Exact match; retains 100% utility. | **CONFIRMED** |
| **Arm C Challenge Effectiveness** | **85.5%** | 0.8548 (53/62 challenges correct) | Exact match. | **CONFIRMED** |
| **Arm C Latency (p50 / p95)** | **2344.1 / 2380.3 ms** | 2344.07 ms / 2380.32 ms | Exact match (+27.7 ms p95 vs Arm A). | **CONFIRMED** |
| **Arm C Extra Calls / 100** | **40.3** | 40.2597 calls / 100 requests | Exact match. | **CONFIRMED** |
| **Arm C2 Residual ASR** | **32.2%** (21.1–44.4%) | 0.3222 (0.2105, 0.4444) | Exact match; Gemma3:4b judge misses subtle attacks. | **CONFIRMED** |
| **Arm C2 Attack Recall** | **67.8%** (55.6–79.0%) | 0.6778 (0.5556, 0.7895) | Exact match. | **CONFIRMED** |
| **Arm C2 Challenge Effectiveness** | **72.6%** | 0.7258 (45/62 challenges correct) | Exact match. | **CONFIRMED** |
| **Arm Upper-Bound ASR** | **3.3%** (0.0–11.5%) | 0.0333 (0.0, 0.1154) | Exact match; provenance of historical 3.3% figure. | **CONFIRMED** |
| **Arm Upper-Bound Recall** | **96.7%** (88.5–100.0%) | 0.9667 (0.8846, 1.0) | Exact match; provenance of historical 96.7% figure. | **CONFIRMED** |
| **Arm Upper-Bound Extra Calls** | **100.0** | 100.0 calls / 100 requests | Exact match; evaluates every input unconditionally. | **CONFIRMED** |
| **Arm Upper-Bound Latency p95** | **2393.3 ms** | 2393.32 ms | Exact match. | **CONFIRMED** |

---

## 3. Claim Provenance Table

In accordance with Wave E guidelines, the following table audits every headline empirical claim in PARALLAX, classifying its source, methodological foundation, and audit status:

| Claim | Source File | Real or Synthetic | Supported (Yes/No) | Audit Notes |
|:---|:---|:---:|:---:|:---|
| **Parallel fan-out cuts median latency by 3.89x (16.12 ms vs 62.66 ms)** | `reports/experiment_results.json` | Mock-Assisted (Preliminary) | **Yes** | Transparently labeled as preliminary: benchmarked across 5 prompts and 4 detectors (2 real, 2 mock specialists). |
| **Sandbox Guard slashes ASR from 85.7% to 0.0%** | `reports/experiment_results.json` | Mock-Assisted (Preliminary) | **Yes** | Transparently labeled as preliminary: benchmarked on 7 synthetic keyword episodes across 12 CPS scenarios. |
| **Ensemble Disagreement rate is 50.0% on evasion set** | `reports/experiment_results.json` | Mock-Assisted (Preliminary) | **Yes** | Transparently labeled: 4/8 disagreements on evasion test set. |
| **D4 regex rules achieve 0.00% False Positive Rate** | `reports/experiment_results.json` | Real | **Yes** | Benchmarked on 50 benign industrial/security prompts. |
| **D1 MiniLM ROC-AUC is 1.0000 on in-distribution test set** | `reports/experiment_results.json` | Real | **Yes** | Logistic regression trained and tested on partitioned technique dataset. |
| **Real ensemble naive average suffers 82.2% residual ASR** | `reports/ablation_results.json` | Real Ensemble | **Yes** | Tested on 90 held-out canary prompts with real DeBERTa, MiniLM, Rules, and PromptForest. |
| **Heuristic discrepancy fusion lowers ASR to 24.4% but incurs 6.2% False-Block Rate** | `reports/ablation_results.json` | Real Ensemble | **Yes** | Tested on 90 attacks and 64 benign operational tasks across 16 CPS scenarios. |
| **Adversarial review (Arm C) reduces ASR to 23.3% while maintaining 100% utility (0.0% FBR)** | `reports/ablation_results.json` | Real Ensemble + Hermetic Judge | **Yes** | Triggers on 40.3% of traffic, achieving 85.5% challenge effectiveness and 0% false alarms. |
| **Gemma3:4b judge achieves lower recall (67.8%) than Qwen3:8b (76.7%)** | `reports/ablation_results.json` | Real Ensemble + Hermetic Judge | **Yes** | Re-run confirms model sensitivity gap on complex roleplay/crescendo attacks. |
| **Unconstrained review (Challenger-ON-all) achieves 96.7% recall / 3.3% ASR** | `reports/ablation_results.json` | Real Ensemble + Hermetic Judge | **Yes** | Provenance of Wave D figures established: represents theoretical upper bound requiring 100 extra calls/100 requests. |
| **Attack corpus is held out with zero token leakage** | `docs/attacks/overlap_report.md` | Real | **Yes** | Zero collisions against training splits; effective sample size $N_{\text{eff}} = 11.1$. |
| **16 Expanded CPS scenarios supported** | `parallax/sandbox/scenarios.py`, `parallax/dashboard/app.py` | Real | **Yes** | Confirmed via `list_expanded_scenarios()`; dashboard API returns 16. |
| **80 Benign operational tasks available** | `parallax/sandbox/utility.py` | Real | **Yes** | 5 benign tasks across all 16 scenarios cataloged. |
| **Contracts in `parallax/contracts/` strictly frozen** | Git baseline | Real | **Yes** | 0 diffs against Phase 0 baseline (`35a04ab`). |
| **Zero third-party imports inside `parallax/`** | AST / codebase scan | Real | **Yes** | PromptForest, PyRIT, Garak, Promptfoo decoupled into standalone processes/venvs. |

---

## 4. Scientific Honesty & Negative Results Audit

Wave E explicitly mandates plain reporting of negative results. The following empirical limitations are confirmed as accurately represented in the documentation:

1. **The Ensemble Dilution Flaw**: Naive score averaging across heterogeneous detectors creates severe blind spots. High-confidence detections from specialized models (e.g., DeBERTa scoring 1.0 on roleplay or RuleDetector scoring 0.9 on keywords) are mathematically diluted when unspecialized models score 0.0, leaving fused scores below 0.50.
2. **Heuristic Discrepancy False Blocks**: Without an intelligent LLM judge, heuristic discrepancy-aware escalation (Arm B) blocks 6.2% of legitimate industrial tasks (`false_block_rate = 6.2%`), demonstrating that simple discrepancy thresholding without verification degrades benign utility.
3. **Adversarial Review Resource Overhead**: While Arm C eliminates false blocks and achieves 85.5% challenge effectiveness, it requires **40.3 additional model calls per 100 requests** and increases p95 latency by +27.7 ms.
4. **Upper-Bound Wastefulness**: Challenging every query unconditionally achieves 96.7% recall but demands **100 extra calls per 100 requests** and increases p95 latency to 2393.3 ms across all operational traffic.

---

## 5. Security, Isolation, and Contract Immutability Audit

### 5.1 Harmless Payload Policy Compliance
- **File Scanned**: `data/attacks/canary_attacks.jsonl` (120 samples).
- Every attack matches `CANARY-[A-Z0-9]+` and targets synthetic Python actuator tools (`mock_trip_breaker`, `mock_set_pressure`, `mock_disable_failsafe`).
- Zero real malicious payloads, biological instructions, or network exploit code exist in the repository. **CONFIRMED PASS**.

### 5.2 Frozen Contracts Verification
- Path: `parallax/contracts/` (`api.py`, `data.py`, `detectors.py`, `__init__.py`).
- Git diff against `35a04ab`: **0 lines modified**.
- Local bridge schemas in `parallax/challenger/schemas.py` successfully isolate new features without contract modification. **CONFIRMED PASS**.

### 5.3 Third-Party Isolation Audit
- Scanned all Python modules in `parallax/` for forbidden imports:
  - `import promptforest`: 0
  - `import pyrit`: 0
  - `import garak`: 0
  - `import promptfoo`: 0
- PromptForest is isolated to a standalone service on port 8765; PyRIT, Garak, and Promptfoo are isolated to standalone generation environments. **CONFIRMED PASS**.

---

## 6. Audit Conclusion & Gate Certification

Wave E has successfully repaired the empirical foundation of PARALLAX. All claims are traceable to verifiable artifacts, synthetic mock numbers have been segregated and plainly labeled, the ablation rerun is grounded in real detector inferences with cluster-aware bootstrap confidence intervals, and the CI gate strictly enforces metric traceability.

**CI Gate Status:**
- **Ruff Linter & Formatter:** PASSED
- **Mypy Type Checking:** PASSED
- **Bandit Security Audit:** PASSED (0 issues)
- **Unit Test Suite & Coverage:** PASSED (598 passed, 89.96% coverage $\ge 80\%$)
- **Contracts Baseline:** PASSED (strictly identical)
- **Metric Traceability Check:** PASSED (all metrics in `REPORT.md` match JSON artifacts)

**Overall Audit Verdict: FULLY CERTIFIED & HONEST.**
