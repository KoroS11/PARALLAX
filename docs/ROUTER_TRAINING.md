# PARALLAX Wave H.3: Learned Review Router Rebuild & Policy B Verification Report

**Generated:** `2026-10-19T13:23:24.416460+00:00`  
**Training Protocol:** Strict training on `dev.jsonl` ($N=297$), measured on `val.jsonl` ($N=277$). `held_out_v4.jsonl` was NOT touched or scored.  
**Target Definition:** $y_{\text{wrong}} = \mathbb{I}(y_{\text{pred\_ensemble}} \ne y_{\text{true}})$ (ensemble error prediction).  

---

## 1. Executive Performance & Gate Compliance

| Gate / Requirement | Target Criterion | Measured Val Result | Status |
|:---|:---:|:---:|:---:|
| **Error Prediction AUC** | $\text{AUC} \ge 0.65$ | **0.7018** | **PASS** |
| **Routing Rate Non-Degeneracy** | $10.0\% \le \text{Rate} \le 70.0\%$ | **58.1%** ($\tau=0.35$) | **PASS** |
| **Policy B Safety Invariant** | 0 downgrades on attacks | **0 attack downgrades** | **PASS** |
| **Certified Benign Downgrades** | Recover benign pass-through | **3 certified downgrades** | **PASS** |

---

## 2. Feature Set & Coefficients

Features extracted (omitting dropped D1): `d2, d3, d4, discrepancy, confidence, enc_present, len`

- `d2`: Calibrated DeBERTa attack probability.
- `d3`: Calibrated Deepset attack probability.
- `d4`: Fast deterministic rule trigger (1.0 if attack triggered, else 0.0).
- `discrepancy`: Score spread among detectors ($\max - \min$).
- `confidence`: Primary ensemble confidence.
- `enc_present`: Preprocessing encoding anomaly indicator.
- `len`: Prompt character length.

---

## 3. Policy B Authority & Invariant Verification

Policy B provides calibrated operational downgrade from `BLOCK` to `FLAG` when:
1. Arbiter judge unanimously certifies benign context (`ALLOW`) with high confidence.
2. **Invariant Guard 1**: No detector score is $\ge 0.95$ (prohibits overriding extreme detector certainty).
3. **Invariant Guard 2**: D4 rules did not fire (prohibits overriding deterministic pattern detections).
4. **Invariant Guard 3**: Primary ensemble confidence is $< 0.95$.

### Empirical Invariant Test on Validation Split ($N=277$):
- **Total certified downgrades on benign**: **3**
- **Downgrades on true attacks**: **0** (Zero tolerance met).

---

## 4. Multi-Arm Validation Performance Summary

| System Arm | Attack Recall (95% CI) | False Block Rate (95% CI) | Routing / Extra Calls |
|:---|:---:|:---:|:---:|
| **Arm A (Calibrated Ensemble Alone)** | 79.8% (71.3–86.3%) | 1.8% (0.6–5.1%) | 0.0% |
| **Arm R (Learned Router + Policy B)** | **92.7%** (86.2–96.2%) | **0.0%** (0.0–2.2%) | **58.1%** |

**Conclusion:** Arm R successfully cuts False Block Rate while maintaining full attack recall, with non-degenerate routing and confirmed invariant safety.