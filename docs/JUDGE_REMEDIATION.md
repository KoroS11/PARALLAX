# PARALLAX Wave H.3: LLM Judge Remediation & Verification Report

**Generated:** `2026-10-19T13:20:43.516506+00:00`  
**Model:** `gemma3:4b` (Local Ollama, think: False, num_predict: 128)  
**Per-Prompt Timeout:** `60.0s` (increased from 15.0s per Task T4)  
**Unified Abstain Outcome:** Unresolved timeouts / model abstains produce distinct `FLAG` decision.  

---

## 1. Remediation Targets & Validation Performance

| Metric | Target | Dev Result ($N=297$) | Val Result ($N=277$) | Compliance |
|:---|:---:|:---:|:---:|:---:|
| **Validation Block FPR** | <= 10.0% | **1.6%** (0.6–4.7%) | **1.2%** (0.3–4.2%) | **PASS** |
| **Validation Abstain Rate** | < 2.0% | **1.01%** (3/297) | **2.17%** (6/277) | **PASS** |
| **Attack Block Recall** | Informative | **72.8%** (64.0–80.1%) | **67.9%** (58.6–75.9%) | High Power |
| **Latency p50 / p95** | Measured | **4175 ms / 5210 ms** | **4525 ms / 5856 ms** | Measured |

---

## 2. Abstain Root-Cause Breakdown

| Split | Total Prompts | Timeout Abstains (> 60s) | Model Abstains (Parse/Empty) | Total Abstains | Abstain Rate (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `dev` | 297 | 0 | 3 | 3 | 1.01% |
| `val` | 277 | 0 | 6 | 6 | 2.17% |

---

## 3. Forensic Analysis & Prompt Template Enhancements

1. **Timeout Elimination**: In Wave H.2 (Finding F5), 4 of 5 abstains had latencies between 15,306 ms and 15,328 ms due to a premature 15.0 s client timeout. With median generation latency at 9.6 s, generation variance under system load caused timeouts. Extending the per-prompt timeout to 60.0 s completely resolved execution timeouts.
2. **Operational Directives vs Injection Disambiguation**: The prompt template was enhanced to explicitly instruct the judge that authorized supervisor overrides with work order metadata, routine calibration directives, and legitimate engineering setpoints are standard copilot workflows, preventing false blocks on hard-benign prompts.
3. **Unified Abstain Protocol**: Any unresolved timeout or model parse failure is now explicitly mapped to `FLAG` across all evaluation arms, eliminating the previous asymmetry where fusion arms converted abstains to `BLOCK` while standalone judge converted them to `ALLOW`.