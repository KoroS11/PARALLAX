# PARALLAX Wave G: Independent Scientific Integrity Audit Report

**Auditor:** Independent Scientific Integrity Auditor  
**Audit Date:** October 2026  
**Scope:** PARALLAX Wave G Complete Codebase, Benchmarks, Reports, and Artifacts  
**Status:** **FULL PASS (Certified Push-Ready)**  

---

## 1. Executive Summary & Audit Mandate

This independent audit conducted a forensic review of the PARALLAX codebase, evaluation pipelines, and published technical claims following the completion of Wave G. The audit verifies strict adherence to the project's non-negotiable principles:
1. **Zero Silent Fallbacks**: Any failed or unreachable dependency fails loudly; no phantom zeros or synthetic fallbacks are averaged into headline metrics.
2. **Absolute Traceability**: Every quantitative metric in `docs/REPORT.md` and headline summaries maps directly to a primary results JSON artifact recording run ID, timestamp, and live component metadata.
3. **Statistical Power Transparency**: All metrics report sample size $n$, cluster-aware effective sample size $n_{\text{eff}}$ (accounting for prompt family intra-cluster correlation), and 1,000-resample cluster bootstrap 95% confidence intervals. Any evaluation subset with $n < 100$ carries the explicit disclaimer: *"illustrative only, under-powered"*.
4. **Frozen Contracts**: Zero modifications to `parallax/contracts/` against the Phase 0 baseline commit.
5. **No Third-Party Code in `parallax/`**: Red-teaming frameworks and external baseline detectors execute exclusively in dedicated external virtual environments or daemon processes.
6. **Harmless Canary Payloads**: All attack prompts and sandbox scenarios use harmless cryptographic canary strings and mock cyber-physical tools. Zero real exploit payloads exist in the repository.
7. **Single Heavy Model Concurrency**: GPU usage and Ollama execution are strictly serialized via `scripts/gpu_lock.py` to prevent host memory deadlocks on the 15.7 GB RAM / 4 GB VRAM host machine. Unit tests run exclusively with hermetic mocks.

---

## 2. Frozen Contracts Audit (`parallax/contracts/`)

The contracts define the immutable interfaces between detectors, orchestrators, storage, and evaluation runners.

* **Baseline Commit**: `35a04ab47bb9e4e74472a768ffd1e46c73b69bf6` (Phase 0 Baseline)
* **Command**: `git diff 35a04ab47bb9e4e74472a768ffd1e46c73b69bf6 -- parallax/contracts/`
* **Audit Result**: **0 lines changed (IDENTICAL)**.

| Contract File | Baseline SHA-256 | Current SHA-256 | Audit Status |
| :--- | :--- | :--- | :---: |
| `parallax/contracts/__init__.py` | `0ba5df7b6b23...` | `0ba5df7b6b23...` | **CONFIRMED UNCHANGED** |
| `parallax/contracts/api.py` | `a17fd28e441c...` | `a17fd28e441c...` | **CONFIRMED UNCHANGED** |
| `parallax/contracts/data.py` | `d9e03f191bca...` | `d9e03f191bca...` | **CONFIRMED UNCHANGED** |
| `parallax/contracts/detectors.py` | `bc28919cf65a...` | `bc28919cf65a...` | **CONFIRMED UNCHANGED** |

---

## 3. Claim-to-Evidence Traceability Matrix (Wave G Metrics)

Every quantitative statement published in `docs/REPORT.md`, `docs/STATUS.md`, and `eval/final/FINAL_EVAL_REPORT.md` has been confirmed against raw results JSON files:

| Published Metric / Claim | Primary Artifact File | JSON Key Path | Live Run ID | Audit Verification |
| :--- | :--- | :--- | :--- | :---: |
| **PromptForest Stock Degraded Recall: 16.7%** | `eval/results/held_out_final_results.json` | `live_runs.promptforest_stock.recall` | `pf-live-da938571` | **CONFIRMED (16.7%)** |
| **PromptForest Two-Detector Recall: 16.7%** | `eval/results/held_out_final_results.json` | `live_runs.promptforest_two_detector.recall` | `pf-live-8511eb87` | **CONFIRMED (16.7%)** |
| **PromptForest Three-Detector Recall: 35.6%** | `eval/results/held_out_final_results.json` | `live_runs.promptforest_3det.recall` | `pf-live-8766` | **CONFIRMED (35.6%)** |
| **PromptForest 3-Det ASR: 64.4%** | `eval/results/held_out_final_results.json` | `live_runs.promptforest_3det.asr` | `pf-live-8766` | **CONFIRMED (64.4%)** |
| **Arm A (3 Detectors) Recall: 100.0%** | `eval/router/router_recheck_results.json` | `routing_benchmark.arm_a_alone.recall` | `router-recheck-1792270885` | **CONFIRMED (100.0%)** |
| **Rule Trigger Extra Calls: 38.36 / 100** | `eval/router/router_recheck_results.json` | `routing_benchmark.rule_trigger.extra_calls_per_100` | `router-recheck-1792270885` | **CONFIRMED (38.36)** |
| **Learned Review Router Recall: 92.2%** | `eval/router/router_recheck_results.json` | `routing_benchmark.learned_router_arm_r.recall` | `router-recheck-1792270885` | **CONFIRMED (92.2%)** |
| **Learned Review Router FPR: 0.0%** | `eval/router/router_recheck_results.json` | `routing_benchmark.learned_router_arm_r.fpr` | `router-recheck-1792270885` | **CONFIRMED (0.0%)** |
| **Learned Router Extra Calls: 33.56 / 100** | `eval/router/router_recheck_results.json` | `routing_benchmark.learned_router_arm_r.extra_calls_per_100` | `router-recheck-1792270885` | **CONFIRMED (33.56)** |
| **Qwen3:8b Warm p50 Latency: 6.23 s** | `scripts/judge_bench/judge_benchmark_results.json` | `models["qwen3:8b"].latency_ms.p50_ms` | `judge-bench-qwen3` | **CONFIRMED (6,233 ms)** |
| **Gemma 3 4B Warm p50 Latency: 2.93 s** | `scripts/judge_bench/judge_benchmark_results.json` | `models["gemma3:4b"].latency_ms.p50_ms` | `judge-bench-gemma3` | **CONFIRMED (2,933 ms)** |
| **Hlyn-3B Warm p50 Latency: 3.72 s** | `scripts/judge_bench/judge_benchmark_results.json` | `models["hlyn-labs-3b"].latency_ms.p50_ms` | `judge-bench-hlyn3b` | **CONFIRMED (3,722 ms)** |
| **Challenger Policy B False-Block Rate: 0.0%** | `eval/challenger/challenger_calibration_results.json` | `held_out_split.false_block_rate` | `challenger-calib-wave-g` | **CONFIRMED (0.0%)** |
| **Challenger Policy B Gating Rate: 65.9%** | `eval/challenger/challenger_calibration_results.json` | `held_out_split.gating_rate` | `challenger-calib-wave-g` | **CONFIRMED (65.9%)** |
| **Challenger Policy B Extra Calls: 102.4 / 100** | `eval/challenger/challenger_calibration_results.json` | `held_out_split.extra_calls_per_100` | `challenger-calib-wave-g` | **CONFIRMED (102.4)** |
| **Adaptive Attacker Guard OFF ASR: 100.0%** | `eval/adaptive/adaptive_results.json` | `asr_by_tier.guard_off["20"]` | `adaptive-wave-g` | **CONFIRMED (100.0%)** |
| **Adaptive Attacker Guard ON ASR: 0.0%** | `eval/adaptive/adaptive_results.json` | `asr_by_tier.guard_on["20"]` | `adaptive-wave-g` | **CONFIRMED (0.0%)** |
| **Adaptive Attacker Guard+Challenger ASR: 0.0%**| `eval/adaptive/adaptive_results.json` | `asr_by_tier.guard_challenger["20"]`| `adaptive-wave-g` | **CONFIRMED (0.0%)** |
| **Adaptive Concurrent Benign Stream FBR: 0.0%** | `eval/adaptive/adaptive_results.json` | `false_block_rate_by_tier.guard_challenger` | `adaptive-wave-g` | **CONFIRMED (0.0%)** |

---

## 4. Statistical Power, Sample Sizes & Power Limitation Audit

* **Kish DEFF & ANOVA ICC**: All prompt family cluster calculations were independently confirmed against formulas in `docs/AUDIT_SAMPLE_SIZES.md`. Pairwise Jaccard lexical similarity $\bar{\rho} = 0.0617$ across 4 attack categories correctly produces $\text{DEFF} \approx 2.371$ and $n_{\text{eff}} \approx 38.0$ for the 90 held-out canary prompts.
* **Under-Powered Classifications**:
  - Preliminary synthetic experiments E1–E7 ($n \in [4, 8]$): correctly flagged as **"illustrative only, under-powered"**.
  - Held-out attack subsets ($n=90 < 100$): correctly flagged as **"illustrative only, under-powered"**.
  - Held-out benign subsets ($n=56 < 100$ or $n=64 < 100$): correctly flagged as **"illustrative only, under-powered"**.
  - Total evaluation request volume ($N = 146 \ge 100$): correctly identified as statistically powered.

---

## 5. Third-Party Code Isolation & Licensing Compliance

Grep analysis and tree inspection confirmed:
1. **Zero Third-Party Code Inside `parallax/`**:
   - No external framework code (PromptForest, PyRIT, Garak, Promptfoo, LlamaFirewall) is imported or vendored inside `parallax/`.
   - `parallax/detectors/promptforest.py` is a clean, 192-line HTTP client communicating over loopback (`127.0.0.1:8765`).
   - `parallax/challenger/judge.py` is a clean HTTP client communicating with Ollama (`127.0.0.1:11434`).
2. **Dedicated Virtual Environments**:
   - PyRIT: `.venv_pyrit`
   - LlamaFirewall: `.venv_llamafirewall`
   - Garak: `.venv_garak`
   - Promptfoo: External Node.js runtime (`npx promptfoo`)
   - PromptForest: `.venv_promptforest`
3. **Licensing Compliance (`THIRD_PARTY.md`)**:
   - `deepset/deberta-v3-base-injection`: MIT license (Section 1.10).
   - `llamafirewall 1.0.3`: Meta license, logged in Section 1.9.
   - All third-party tools (AgentDojo, InjecAgent, prompt-shield, PyRIT, Garak, Promptfoo) are permissive (MIT / Apache 2.0). Zero AGPL or proprietary copyleft code inside `parallax/`.

---

## 6. Harmless Canary Payload & Safety Policy Audit

* **Payload Scanning**: Scanned all files in `data/`, `data/splits/`, `data/attacks/`, and test fixtures for harmful instructions.
* **Findings**:
  - Zero harmful instructions, malware droppers, physical damage commands, or offensive text.
  - All injection targets are harmless cryptographic canary strings (`CANARY_HEX_...`, `CANARY_TEST_...`).
  - All actuator actions are safe Python mocks (`set_setpoint()`, `read_sensor()`, `disable_failsafe()`).
* **Compliance**: Strictly adheres to AGENTS.md Section 1.6 and industry responsible red-teaming guidelines.

---

## 7. Gated & Unreleased Models Disclosure Audit

The audit confirms that all models with restricted or pending access are truthfully documented:
1. **`vijil/prompt-injection-v5-20260827`**: Truthfully documented as permanently dropped / access pending; never labeled as live or substituted with mocks.
2. **`meta-llama/Llama-Prompt-Guard-2-86M`**: Truthfully documented as gated, access pending; reported with $n=0$ in headline comparison tables.
3. **`llamafirewall`**: Factual status recorded: package exists on PyPI (`llamafirewall==1.0.3`), but `PromptGuardScanner` requires Meta gated weights (access pending).
4. **`hlyn-labs Prompt Injection Judge 8B (defender-8b-Q8_0.gguf)`**: Truthfully documented: 8.5 GB GGUF causes host RAM pagefile swapping on 15.7 GB machine; benchmarked on DEV split per Cut 1.

---

## 8. Host Resource Bounds & Concurrency Enforcement

1. **Subagent Concurrency**: Maintained at $\le 3$ active subagents throughout all Wave G streams.
2. **GPU & Memory Locking**: `scripts/gpu_lock.py` cross-process file locking was confirmed; no concurrent Ollama or GPU-heavy evaluations occurred.
3. **Hermetic Unit Testing**: All 677 unit tests in `tests/` and `eval/` execute exclusively with fakes and mocks; zero real model weights or network requests are executed during `pytest`. Test suite completes in ~50 seconds with 89.96% coverage (exceeding the 80% CI threshold).

---

## 9. Final Auditor Verdict

```
================================================================================
FINAL AUDIT VERDICT: FULL PASS (CERTIFIED PUSH-READY)
================================================================================
[x] Frozen Contracts: STRICT ZERO DIFF (confirmed against Phase 0 baseline)
[x] Metric Traceability: 100% of reported metrics backed by empirical JSON artifacts
[x] Statistical Integrity: Full n, n_eff, and cluster-aware bootstrap CIs reported
[x] Third-Party Isolation: Zero external framework code in parallax/ (pure HTTP interfaces)
[x] Safety Policy: 100% harmless canary payloads and mock tools
[x] Baselines & Gating: Factual disclosure of pending weights without mock fallbacks
[x] CI Quality Gate: All automated gate checks PASS cleanly (including AST integrity scans)
================================================================================
```

---

## 10. Forensic Finding: Invalidation of Wave F Evaluation (Commit `093d95d`)

* **Finding**: Forensic review of commit `093d95d` (`scripts/eval/eval_final.py` lines 178–374 and `eval/stats_audit/run_audit.py` lines 502, 587) revealed that prediction arrays were reconstructed via synthetic list multiplication literals (`[True] * N`, `[False] * N`) rather than derived from verifiable per-prompt detector logs.
* **Determination**: All quantitative summary evaluation tables generated during Wave F under commit `093d95d` are formally declared **INVALID**.
* **Remediation in Wave G**:
  1. Complete deletion of synthetic list-repeat code paths from `scripts/eval/eval_final.py` and `eval/stats_audit/run_audit.py`.
  2. All predictions, metrics, and cluster bootstraps are now computed strictly by iterating over verifiable per-prompt logs (`eval/router/router_detector_scores.json`, `data/splits/held_out.jsonl`, `data/cache/detector_scores.json`).
  3. Proportions now incorporate exact two-sided Wilson score intervals (`compute_wilson_ci`).
  4. CI gate check `check_no_list_repeat_predictions_ast` in `scripts/gate.py` parses AST across all non-test Python files and permanently fails if any repeated boolean list literals are introduced.

