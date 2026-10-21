# PARALLAX Challenger Calibration Report (Wave G)

**Run ID:** `challenger-calib-411712dc`  
**Judge Model:** `gemma3:4b` (Live Ollama, GPU Locked)  
**Self-Consistency:** N=3 with strict abstain on divergence  
**Timestamp:** `2026-10-17T21:33:37.598509+00:00`  

## 1. Root Cause Analysis
In Wave F, a 45–50% False-Block Rate (FBR) was observed due to two structural issues:
1. **Unconstrained Judge False Positives:** The adversarial reviewer escalated `ALLOW -> BLOCK` whenever the judge returned a false-positive on allowed prompts.
2. **Missing Downgrade Path:** `skip_on_block` was set to `True` by default in `TriggerConfig`, preventing initial `BLOCK` decisions (caused by oversensitive classifiers on tricky benign tasks) from ever being reviewed or downgraded.

## 2. Dev Split Calibration (Sweep on dev.jsonl + Benign Catalog)
*Calibration performed strictly on the DEV split; held-out was not touched.*

| Operating Point | Threshold Config | Dev Attack Recall | Dev FBR | Dev FPR | Downgrades | Gating Rate | Extra Calls / 100 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Point 1 (Target <= 5% FBR)** | tau_conf=0.65, block_ceil=0.85 | `63.3%` | `0.0%` | `17.6%` | `0` | `67.8%` | `96.5` |
| **Point 2 (Target <= 10% FBR)** | tau_conf=0.80, block_ceil=0.88 | `63.3%` | `0.0%` | `17.6%` | `0` | `67.8%` | `96.5` |

## 3. Policy Comparison on DEV Split (Policy A vs Policy B)

| Metric | Policy A (Conservative) | Policy B (Calibrated) | Delta |
| :--- | :---: | :---: | :---: |
| **Attack Recall** | `60.0%` | `63.3%` | `+0.0%` |
| **False-Block Rate (FBR)** | `17.6%` | `0.0%` | `-17.6%` |
| **False Positive Rate (FPR)** | `17.6%` | `17.6%` | `-0.0%` |
| **Downgraded BLOCK -> FLAG** | `0` | `0` | `+0` |
| **Extra Calls / 100 Reqs** | `91.3` | `96.5` | `0.0` |

## 4. Single Final Evaluation on Held-Out Split (held_out.jsonl)
*Evaluated ONCE at the end with Point 1 calibrated thresholds.*

- **Held-Out Attack Recall:** `47.8%` (90 canary attacks)
- **Benign Operational FBR:** `0.0%` (80 benign operational tasks)
- **Benign Operational FPR:** `16.2%`
- **Downgraded Blocks:** `0`
- **Gating Rate:** `65.9%`
- **Extra Calls / 100 Reqs:** `102.4`

---
*Generated autonomously by PARALLAX Wave G Challenger Calibration Specialist.*
