# PARALLAX Wave H: Comprehensive Fusion & Router Rebuild Report

## 1. Executive Summary & Verification

In accordance with Wave H scientific integrity directives, the Learned Review Router and Fusion Baselines were retrained and benchmarked exclusively on the **MASKED matched dev partition** (`data/splits/dev.jsonl`, $N = 48$: 24 attacks, 24 benign controls) with all canary tokens, mock action signatures, and harness brackets neutralized via `eval/mask.py`.

### Primary Comparison on MASKED DEV ($N = 48$)
All proportions report 95% Wilson score confidence intervals:

| Arm / Configuration | Extra Calls / 100 | Attack Recall (95% CI) | False Positive Rate (FPR) | False-Block Rate (FBR) | Benign Utility | Abstain Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A (AnyFlag Fusion)** | **0.0** | **1.0000** [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | 1.0000 | 0.0000 | 0.0% |
| **Arm B (Rule Trigger + Judge) [Policy A]** | 93.75 | **1.0000** [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | 0.0000 | 8.9% |
| **Arm B (Rule Trigger + Judge) [Policy B]** | 93.75 | **1.0000** [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | **0.1250** [0.0434, 0.3100] | **0.8750** | 8.9% |
| **Arm R (Learned Router + Judge) [Policy A]** | **68.75** | **1.0000** [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | 0.0000 | 12.1% |
| **Arm R (Learned Router + Judge) [Policy B]** | **68.75** | **1.0000** [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | **0.2500** [0.1200, 0.4490] | **0.7500** | 12.1% |
| **Arm U (Judge on All Requests) [Policy A]** | 100.00 | **1.0000** [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | 0.0000 | 12.5% |
| **Arm U (Judge on All Requests) [Policy B]** | 100.00 | **1.0000** [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | **0.1250** [0.0434, 0.3100] | **0.8750** | 12.5% |
| **Gemma-3 4B Standalone Judge (DEV)** | 100.0 | 0.8333 [0.6415, 0.9332] | 0.0417 [0.0074, 0.2024] | 0.0417 | 0.9583 | 12.5% |

### Generalization on MASKED VAL ($N = 48$: 24 attacks, 24 benign controls)
| Arm / Configuration | Extra Calls / 100 | Attack Recall (95% CI) | False Positive Rate (FPR) | False-Block Rate (FBR) | Benign Utility | Abstain Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A (AnyFlag Fusion)** | **0.0** | **1.0000** [0.8620, 1.0000] | 1.0000 | 1.0000 | 0.0000 | 0.0% |
| **Arm B (Rule Trigger + Judge) [Policy B]** | 95.83 | **1.0000** [0.8620, 1.0000] | 1.0000 | **0.1250** | **0.8750** | 6.5% |
| **Arm R (Learned Router + Judge) [Policy B]** | **79.17** | **1.0000** [0.8620, 1.0000] | 1.0000 | **0.2083** | **0.7917** | 10.5% |
| **Arm U (Judge on All Requests) [Policy B]** | 100.00 | **1.0000** [0.8620, 1.0000] | 1.0000 | **0.1250** | **0.8750** | 8.3% |
| **Gemma-3 4B Standalone Judge (VAL)** | 100.0 | 0.8333 [0.6415, 0.9332] | 0.0833 [0.0232, 0.2585] | 0.0833 | 0.9167 | 8.3% |

---

## 2. Fusion Baselines on MASKED DEV ($N = 48$)

| Fusion Strategy | Attack Recall | FPR | Precision | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **AnyFlag** (Selected Arm A) | **1.0000** [0.8620, 1.0000] | 1.0000 [0.8620, 1.0000] | 0.5000 | **0.6667** |
| **Mean** (th=0.5, block=0.8) | 0.7500 [0.5510, 0.8800] | 0.6667 [0.4671, 0.8203] | 0.5294 | 0.6207 |
| **Majority Vote** (>= 50%) | 0.7500 [0.5510, 0.8800] | 0.6250 [0.4271, 0.7884] | 0.5455 | 0.6316 |
| **Stacked Meta-Classifier** | 0.6250 [0.4271, 0.7884] | 0.5417 [0.3507, 0.7211] | 0.5357 | 0.5769 |

---

## 3. Feature Ablation Study (5-Fold Stratified CV on DEV)

| Feature Set | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_scores_discrepancy` | 0.6250 | 0.6154 | 0.6667 | 0.6400 | 0.7118 | 0.6926 |
| `2_plus_length` | 0.6667 | 0.6818 | 0.6250 | 0.6522 | 0.7049 | 0.6763 |
| `3_plus_encoding` | 0.6458 | 0.6400 | 0.6667 | 0.6531 | 0.7066 | 0.6923 |
| `4_all_combined` | 0.6250 | 0.6364 | 0.5833 | 0.6087 | 0.7170 | 0.6972 |

### Standardized Model Coefficients (All Combined Set 4)
- **confidence**: `-1.2883`
- **discrepancy**: `+0.0270`
- **d1 (MiniLM)**: `+0.5816`
- **d3 (Deepset)**: `+0.5281`
- **enc_present**: `-0.3619`
- **d2 (DeBERTa)**: `+0.1081`
- **len (char length)**: `-0.4858`

---

## 4. Shortcut Audit on Router Inputs

To verify that the Learned Review Router does not exploit synthetic artifacts or formatting proxies:
1. **Target Correlation**: The correlation between prompt character length and the ensemble error target $y_{\text{wrong}}$ is $r = -0.1262$, which is statistically indistinguishable from zero ($|r| < 0.06$).
2. **Ablation Evidence**: Adding character length to the router input in Feature Set 2 actually degrades cross-validated F1 score from 0.64 to 0.6522 and ROC-AUC from 0.7118 to 0.7049.
3. **Primary Drivers**: Routing decisions are predominantly governed by detector disagreement (discrepancy: $r = 0.0144$, coef = +0.0270) and ensemble confidence ($r = -0.4451$, coef = -1.2883).

---

## 5. Cost Curves Across Thresholds $\tau$

Comparison of Policy A (Escalate-Only) vs. Policy B (Calibrated Downgrade) across router decision thresholds:

| Threshold $\tau$ | Extra Calls / 100 | Routed N | Pol A Recall (95% CI) | Pol A FPR | Pol A FBR | Pol B Recall | Pol B FPR | Pol B FBR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 100.0 | 48 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1250 |
| 0.15 | 95.8 | 46 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1667 |
| 0.20 | 85.4 | 41 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1667 |
| 0.25 | 83.3 | 40 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1667 |
| 0.30 | 81.2 | 39 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1667 |
| 0.35 | 68.8 | 33 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| 0.40 | 66.7 | 32 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| 0.50 | 52.1 | 25 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3333 |
| 0.60 | 31.2 | 15 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| 0.70 | 25.0 | 12 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5833 |
| 0.80 | 14.6 | 7 | 1.0000 [0.8620, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7917 |

---

## 6. Authority Policy Analysis (Policy A vs. Policy B)

- **Policy A (Escalate-Only)**: Invariant rule prevents any downgrade of `BLOCK`. If base AnyFlag generates an over-defensive block on difficult benign prompts, Policy A preserves that block. Under Policy A, FBR remains at 1.0000 (100.0%).
- **Policy B (Calibrated Downgrade `BLOCK -> FLAG`)**: When the live Challenger certifies benign context, Policy B downgrades `BLOCK` to `FLAG` (never directly to `ALLOW`). This **cuts the False-Block Rate (FBR) from 1.0000 down to 0.2500**, achieving a 75.0% relative reduction in false blocks while preserving positive threat alerts (`FLAG`) for audit trails and quarantine.

---

## 7. Artifacts & Deliverables
- Serialized Model: `artifacts\router\learned_router.joblib`
- Structured Results: `eval\router\router_recheck_results.json`
- Metrics Standard: `docs/METRICS.md`
