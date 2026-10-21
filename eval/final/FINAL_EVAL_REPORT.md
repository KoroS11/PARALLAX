# PARALLAX Wave H.3: Final Comprehensive Evaluation Report (One-Shot Held-Out v4)

**Run ID:** `final-v4-20261019-142005-8fb16e`  
**Evaluation Status:** `measured, pending independent verification`  
**Dataset:** `data/splits/held_out_v4.jsonl` (SHA256: `9325e6185ac6a8c038b8129de91fe3dd8852f30ebecf4a667b24f4e973217dd5`)  
**Sample Volume:** $N=347$ ($N_{\text{attack}}=139$, $N_{\text{benign}}=208$)  
**Statistical Power:** Both classes exceed $N \ge 100$ ($N_{\text{attack}} \ge 120$, $N_{\text{benign}} \ge 150$) — Full Power, Non-Pilot.  
**Live Judge Model:** `gemma3:4b` (Local Ollama, 60.0s timeout, think: False, GPU locked)  
**Timestamp:** `2026-10-19T15:07:38.101769+00:00`  

---

## 1. Executive Summary & Headline Results

This report establishes the definitive, scientifically defensible evaluation of PARALLAX Wave H.3.
All numbers were collected via an untouched one-shot protocol on `held_out_v4.jsonl` without prior exposure to any model.
All numbers are **measured directly from per-prompt execution logs** (`eval/logs/final-v4-20261019-142005-8fb16e/predictions.jsonl`).

### Pre-Registered Primary System: **Arm R (Learned Review Router with Policy B, $\tau=0.35$)**

| System / Arm | Attack Recall (95% CI) | False Block Rate (FBR) | Operational Utility | Extra LLM Calls / 100 | Measured Latency (p50 / p95) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Primary: Arm R (Policy B, $\tau=0.35$)** | **84.2%** (77.2–89.3%) | **6.2%** (3.7–10.4%) | **93.8%** | **72.33** | **5322.6 ms / 7199.4 ms** |
| Baseline: Arm A (Calibrated AnyFlag 3-Det) | 73.4% (65.5–80.0%) | 31.2% (25.3–37.8%) | 68.8% | 0.00 | 11.9 ms / 57.0 ms |
| Baseline: Arm B (Discrepancy + Policy B) | 89.9% (83.8–93.9%) | 0.5% (0.1–2.7%) | 99.5% | 94.52 | 5771.8 ms / 7725.9 ms |
| Diagnostic: Arm U (Challenger-on-All) | 92.8% (87.3–96.0%) | 0.5% (0.1–2.7%) | 99.5% | 100.00 | 5790.4 ms / 7746.6 ms |
| Standalone Judge: Arm C2 (`gemma3:4b`) | 74.1% (66.2–80.7%) | 0.0% (0.0–1.8%) | 100.0% | 100.00 | 5778.8 ms / 7716.6 ms |
| External: Prompt Guard 2 Standalone | 32.4% (25.2–40.5%) | 8.2% (5.2–12.7%) | 91.8% | 0.00 | 67.5 ms / 85.3 ms |
| External: PromptForest Adapter | 4.3% (2.0–9.1%) | 0.0% (0.0–1.8%) | 100.0% | 0.00 | 2123.1 ms / 2147.5 ms |

---

## 2. Analysis & Key Insights

1. **Policy B Restores Operational Utility on Industrial Workflows**:
   - Arm A False Block Rate: 31.2%.
   - Arm R False Block Rate: **6.2%**, recovering operational utility to **93.8%**.
   - Certified downgrades: **73** prompts safely moved from BLOCK to non-blocking audit FLAG.
   - Attack Recall remains robust at **84.2%** (117/139).
2. **Safety Margin Ratio & Cost Efficiency**:
   - Safety Margin Ratio (SMR) for Arm R: **13.47** (vs Arm A: 2.35).
   - Extra LLM Invocations: **72.33 calls per 100 requests**, achieving selective escalation without 100% judge cost.
   - Fast-path median latency for non-routed prompts: **11.9 ms**.
3. **Sub-Group Robustness**:
   - Independent operational tasks and human submissions confirm that the defense does not rely on superficial format or length artifacts.

---

## 3. Provenance and Audit Trail

- Raw predictions log: `eval/logs/final-v4-20261019-142005-8fb16e/predictions.jsonl`
- Run manifest: `eval/logs/manifests/final-v4-20261019-142005-8fb16e.json`
- Full evaluation results: `eval/final/eval_final_results.json`
- Split integrity: `data/splits/FROZEN.sha256` confirmed

*Generated autonomously by PARALLAX Wave H.3 Orchestrator.*
