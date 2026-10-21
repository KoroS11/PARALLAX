# PARALLAX Preprocessing & Multi-View Evaluation Report

**Run ID:** `prep-lift-1791928202`  
**Timestamp:** `2026-10-13T21:50:02Z`  
**Corpus:** 120 attacks, 80 benign operational samples  

## 1. Executive Summary

- **Overall Recall Lift:** `+3.3%` (7.5% raw -> 10.8% multi-view).
- **Benign False Alarm Rate (FPR):** `0.0%` (zero degradation on benign prompts).
- **`encoding_present` Feature:** Detects `65.6%` of obfuscation attacks with `0.0%` benign false positive rate.
- **Silent-Zero Defense Audit:** `PASSED` (all detectors explicitly return `score=None` and fail loudly on error).

## 2. Detection Lift per Technique Family

| Technique Family | N Samples | Raw Recall | Multi-View Recall | Recall Lift |
| :--- | :---: | :---: | :---: | :---: |
| `conversational` | 28 | 10.7% | 21.4% | **+10.7%** |
| `direct` | 38 | 5.3% | 7.9% | **+2.6%** |
| `indirect` | 22 | 0.0% | 0.0% | **0.0%** |
| `obfuscation` | 32 | 12.5% | 12.5% | **0.0%** |

## 3. Dataset Split Breakdown (Dev vs Held-Out)

| Split | N Samples | Raw Recall | Multi-View Recall | Lift |
| :--- | :---: | :---: | :---: | :---: |
| `dev` | 30 | 3.3% | 6.7% | **+3.3%** |
| `held_out` | 90 | 8.9% | 12.2% | **+3.3%** |

## 4. Silent-Zero Audit Certification

| Component / Scenario | Error Recorded | Score | Attack Verdict | Status |
| :--- | :--- | :---: | :---: | :---: |
| `RuleDetector_empty` | `Empty or invalid input prompt` | `None` | `None` | **PASSED (No Silent 0.0)** |
| `MiniLMDetector_empty` | `Empty or invalid input prompt` | `None` | `None` | **PASSED (No Silent 0.0)** |
| `PromptForestAdapter_unreachable` | `ConnectTimeout: connection failure` | `None` | `None` | **PASSED (No Silent 0.0)** |
| `DeBERTaDetector_error` | `DeBERTa inference server disconnect...` | `None` | `None` | **PASSED (No Silent 0.0)** |
| `PromptGuard2Detector_error` | `Gated access required / HF token mi...` | `None` | `None` | **PASSED (No Silent 0.0)** |

---
*Generated autonomously by PARALLAX Wave F Orchestrator.*
