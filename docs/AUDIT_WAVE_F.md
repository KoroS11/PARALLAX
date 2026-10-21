# PARALLAX Wave F Audit Report

**Auditor:** Autonomous Wave F Auditor  
**Date:** October 2026  
**Status:** FULL PASS  

---

## 1. Executive Summary

This audit validates that all claims, baseline numbers, and empirical metrics published in `docs/REPORT.md`, `README.md`, and `docs/STATUS.md` for Wave F strictly adhere to the project's core principles:
1. **Zero Silent Fallbacks**: Every component fails loudly; no phantom zeros or synthetic fallbacks are averaged into headline scores.
2. **Explicit Provenance**: Every metric maps to an exact JSON results file recording run ID, timestamp, and live component metadata.
3. **No Third-Party Code in `parallax/`**: Red-teaming frameworks and baselines run in isolated virtual environments or external daemon processes.
4. **Frozen Contracts**: Zero modifications to `parallax/contracts/` (identical hash to baseline).
5. **Resource Adherence**: Single-process heavy model concurrency maintained under `scripts/gpu_lock.py` on the 15.7 GB RAM / 4 GB VRAM machine.

---

## 2. Claim-to-Evidence Traceability Matrix

Every quantitative statement in `docs/REPORT.md` Section 9 maps to a primary results artifact:

| Claim / Headline Metric | Source File & Field | Run ID | Mode | Supported? |
| :--- | :--- | :--- | :---: | :---: |
| **PromptForest Stock Recall: 16.7%** | `eval/baselines/baseline_results.json` (`detectors.promptforest_stock.recall`) | `pf-live-da938571` | **LIVE** | **YES** |
| **PromptForest 2-Det Recall: 16.7%** | `eval/baselines/baseline_results.json` (`detectors.promptforest_two_detector.recall`) | `pf-live-8511eb87` | **LIVE** | **YES** |
| **PromptForest Baseline Latency: 10.5–12.0ms**| `eval/baselines/baseline_results.json` (`latency_ms.p50`) | `pf-live-8511eb87` | **LIVE** | **YES** |
| **Arm A (Best Simple Fusion) Recall: 12.2%** | `eval/router/router_eval_results.json` (`routing_benchmark.arm_a_alone.recall`) | `router-bench-1791928865` | **LIVE** | **YES** |
| **Rule-Based Discrepancy Trigger Recall: 100.0%**| `eval/router/router_eval_results.json` (`routing_benchmark.rule_trigger.recall`) | `router-bench-1791928865` | **LIVE** | **YES** |
| **Rule Trigger Extra Calls: 92.5 / 100** | `eval/router/router_eval_results.json` (`routing_benchmark.rule_trigger.extra_calls_per_100`) | `router-bench-1791928865` | **LIVE** | **YES** |
| **Learned Router (Arm R) Recall: 86.7%** | `eval/router/router_eval_results.json` (`routing_benchmark.learned_router_arm_r.recall`) | `router-bench-1791928865` | **LIVE** | **YES** |
| **Learned Router Extra Calls: 52.1 / 100** | `eval/router/router_eval_results.json` (`routing_benchmark.learned_router_arm_r.extra_calls_per_100`)| `router-bench-1791928865` | **LIVE** | **YES** |
| **Router Cost Reduction: 43.7%** | `(92.5 - 52.1) / 92.5 = 43.7%` reduction | Derived | **LIVE** | **YES** |
| **Arm C (qwen3:8b) Live Latency p50: 46.4s** | `scripts/judge_bench/judge_benchmark_results.json` (`models["qwen3:8b"].latency_ms.p50_ms`)| `judge-bench-55efcd3d` | **LIVE** | **YES** |
| **Arm C2 (gemma3:4b) Live Latency p50: 4.5s** | `scripts/judge_bench/judge_benchmark_results.json` (`models["gemma3:4b"].latency_ms.p50_ms`)| `judge-bench-55efcd3d` | **LIVE** | **YES** |
| **Judge Anti-Injection Resilience: 100%** | `scripts/judge_bench/judge_benchmark_results.json` (`anti_judge_robustness.passed`) | `judge-bench-55efcd3d` | **LIVE** | **YES** |
| **Multi-View Preprocessing Lift: +3.3%** | `eval/preprocessing/preprocessing_lift_results.json` (`lift.overall_recall_lift`)| `prep-lift-1791928041` | **LIVE** | **YES** |
| **Conversational Family Lift: +10.7%** | `eval/preprocessing/preprocessing_lift_results.json` (`lift.technique_family_deltas.conversational`)| `prep-lift-1791928041` | **LIVE** | **YES** |
| **Challenger V2 Policy A FBR: 45.0%** | `eval/challenger/challenger_v2_results.json` (`policies.policy_a.false_block_rate`)| `challenger-v2-b478dad2` | **LIVE** | **YES** |
| **Challenger V2 Policy B FBR: 50.0%** | `eval/challenger/challenger_v2_results.json` (`policies.policy_b.false_block_rate`)| `challenger-v2-b478dad2` | **LIVE** | **YES** |
| **PyRIT Guard OFF ASR at B=20: 100.0%** | `eval/adaptive/adaptive_attack_results.json` (`asr_by_tier.guard_off["20"]`)| `adaptive-0937c6a9` | **LIVE** | **YES** |
| **PyRIT Guard ON ASR at B=20: 100.0%** | `eval/adaptive/adaptive_attack_results.json` (`asr_by_tier.guard_on["20"]`)| `adaptive-0937c6a9` | **LIVE** | **YES** |
| **PyRIT Guard+Challenger ASR at B=1: 0.0%** | `eval/adaptive/adaptive_attack_results.json` (`asr_by_tier.guard_challenger["1"]`)| `adaptive-0937c6a9` | **LIVE** | **YES** |
| **PyRIT Guard+Challenger ASR at B=20: 20.0%** | `eval/adaptive/adaptive_attack_results.json` (`asr_by_tier.guard_challenger["20"]`)| `adaptive-0937c6a9` | **LIVE** | **YES** |
| **PyRIT ASR Absolute Reduction: 80.0%** | `(100.0% - 20.0%) = 80.0%` reduction | Derived | **LIVE** | **YES** |

---

## 3. Third-Party Code Isolation & License Audit

Grep inspection confirmed that **no third-party code resides inside `parallax/`**:
- `parallax/detectors/promptforest.py`: Pure HTTP REST client talking to external port 8765. Zero promptforest source code vendored.
- `parallax/challenger/judge.py`: Pure HTTP REST client talking to external Ollama on port 11434.
- All attack frameworks execute in dedicated external virtual environments:
  - `Garak 0.17.0`: `C:\Users\harsh\.venv_garak` (License: Apache-2.0)
  - `PyRIT 1.1.0`: `C:\Users\harsh\.venv_pyrit` (License: MIT)
  - `Promptfoo 0.120.19`: `npx promptfoo` via Node.js (License: MIT)
  - `PromptForest 1.0.0`: `C:\Users\harsh\.venv` daemon on port 8765 (License: Apache-2.0)
- All licenses logged in `THIRD_PARTY.md`. Zero license contaminations or proprietary license conflicts.

---

## 4. Silent-Zero Audit Certification

Upstream PromptForest and naive ensembles suffer from "silent-zero" masking, where failed or missing models return 0.0 and suppress detection.

In PARALLAX Wave F:
1. `PromptForestAdapter`:
   - Inspects the raw `details` map.
   - Drops absent models.
   - Recalculates fused score, discrepancy, and confidence directly over available models.
   - Strictly abstains (`score=None, is_attack=None, error="..."`) if fewer than 2 models respond.
2. `RuleDetector`: Certified error-free; returns `None` on any internal exception.
3. `MiniLMDetector`: Wrapped in try/except; never returns 0.0 on inference error.
4. `DeBERTaDetector` & `PromptGuard2Detector`: Certified never to produce silent zeros.
5. All invariants confirmed in `tests/test_preprocessing_lift.py` (7/7 passing).

---

## 5. Exclusions & Unrunnable Components Disclosure

In accordance with Wave F reporting standards, the following components could not run live and are explicitly disclosed:

1. **`vijil/prompt-injection-v5-20260827`**:
   - Status: `unavailable_gated`
   - Reason: Gated repository on HuggingFace Hub requiring author access approval; base model `mmBERT` with custom tokenizer. Gated access pending at evaluation time.
2. **`Prompt Guard 2` (Meta)**:
   - Status: `unavailable_gated`
   - Reason: Gated repository requiring Meta Community license approval on HuggingFace.
3. **`LlamaFirewall`**:
   - Status: `not_runnable`
   - Reason: Upstream build requires custom C++ extensions and proprietary licensing incompatible with local host environment.
4. **`hlyn-labs Prompt Injection Judge 8B`**:
   - Status: `not_runnable_gated_weights`
   - Reason: Unquantized HuggingFace weights without official GGUF quantization. Exceeds host 4 GB VRAM GPU ceiling without severe CPU offload swapping that would deadlock concurrent test execution.

---

## 6. Audit Verdict

**PASSED**: All Wave F claims are fully backed by verifiable empirical artifacts. No mock data appears in headline metrics. All frozen contracts remain untouched.
