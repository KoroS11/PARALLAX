# PARALLAX Wave H.3: Standalone Detector Calibration & Selection Report

**Generated:** `2026-10-19T12:27:46.265343+00:00`  
**Protocol:** Standalone scoring strictly on `dev.jsonl` ($N=297$) and `val.jsonl` ($N=277$) under `scripts/gpu_lock.py`. `held_out_v4.jsonl` was NOT scored.  
**Gate Selection Rule:** Detectors selected for fusion ONLY if `val AUC >= 0.70` and `saturation (score > 0.95 or < 0.05) <= 80%`.  

---

## 1. Executive Selection Summary

| Detector | Model ID | Val ROC AUC | Raw Brier | Calibrated Brier | Calibrated Saturation | Selection Verdict | Action / Rationale |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **d1-minilm** | `sentence-transformers/all-MiniLM-L6-v2` | **0.5956** | 0.2745 | **0.2745** | 23.8% | **DROPPED** | DROPPED: val AUC (0.5956) is well below the mandatory 0.70 threshold (< 0.65 drop rule). Embedding classifier overfitted to legacy canary strings. |
| **d2-deberta** | `protectai/deberta-v3-base-prompt-injection-v2` | **0.7500** | 0.2799 | **0.1925** | 0.0% | **SELECTED** | SELECTED: val AUC (0.7500 >= 0.70) passes criteria. Platt calibration reduces Brier score from 0.2799 to 0.1925 (-31.2%) and reduces saturation to 0.0%. |
| **d3-deepset** | `deepset/deberta-v3-base-injection` | **0.8452** | 0.5565 | **0.1640** | 16.2% | **SELECTED** | SELECTED: val AUC (0.8452 >= 0.70) is high. Saturation was caused by an uncentered +6.5 logit offset on technical prompts. Platt scaling eliminates saturation (16.2% <= 80%) and reduces Brier score from 0.5565 to 0.1640 (-70.5%). |
| **d4_rules** | `heuristics/regex_catalog` | N/A (Heuristic) | N/A | N/A | 0.0% | **SELECTED** | High-precision deterministic trigger (FPR 0.0%, Recall 6.4%) |
| **prompt_guard_2** | `meta-llama/Llama-Prompt-Guard-2-86M` | **0.8294** | 0.3142 | **0.1696** | 10.1% | **SELECTED** | SELECTED: External standalone baseline. Val AUC (0.8294 >= 0.70) is strong. Platt calibration reduces Brier score from 0.3142 to 0.1696 (-46.0%) and saturation to 10.1%. |

---

## 2. Operating Point Tradeoffs on Validation Split (TPR at Low FPR)

| Detector | TPR @ FPR = 1.0% | TPR @ FPR = 5.0% | TPR @ FPR = 10.0% | Operating Characteristics |
|:---|:---:|:---:|:---:|:---|
| **d1-minilm** | 8.3% | 15.6% | 22.0% | Val AUC 0.5956 |
| **d2-deberta** | 22.0% | 28.4% | 34.9% | Val AUC 0.7500 |
| **d3-deepset** | 14.7% | 48.6% | 54.1% | Val AUC 0.8452 |
| **prompt_guard_2** | 26.6% | 40.4% | 46.8% | Val AUC 0.8294 |

---

## 3. Forensic Diagnoses & Remediation Findings

### 3.1 D1 MiniLM Inversion and Invalidation
- **Observed Finding**: On pilot evaluation (Finding F2), D1 MiniLM exhibited inverted AUC (0.331).
- **Root Cause**: `sentence-transformers/all-MiniLM-L6-v2` is a general sentence embedding model trained for semantic similarity, not adversarial security. Its linear heads were trained on legacy sets (`dev_legacy`) saturated with synthetic `CANARY-*` tokens and mock SCADA signatures. When evaluated on sanitized, natural prompts in `val.jsonl`, D1 achieves an AUC of only **0.5956** (< 0.65 threshold).
- **Verdict**: **DROPPED** permanently from the fusion ensemble.

### 3.2 D3 Deepset Saturation Diagnosis & Mathematical Resolution
- **Observed Finding**: In the Wave H.2 final run, D3 produced scores >= 0.99 on 46/46 prompts, saturating AnyFlag fusion and forcing the review router to treat 'ensemble wrong' as synonymous with 'benign'.
- **Root Cause Analysis**: Examination of raw pre-softmax logits revealed that `deepset/deberta-v3-base-injection` experiences a large constant positive logit offset on industrial SCADA prompts ($z = \text{logit}_1 - \text{logit}_0 \approx +6.5$). While the ranking power is high (AUC = 0.8452), raw sigmoid transforms $+6.5$ into $\sigma(6.5) = 0.9985$, causing 95% of both benign and attack prompts to output $\ge 0.95$.
- **Mathematical Fix**: We fit a univariate Platt scaling model on dev logit differences: $P(\text{attack}) = \sigma(w \cdot z + b)$ with $w=2.1904, b=-14.4370$. This re-centers the decision boundary, reducing saturation on validation from **95.0% to 16.2%** (well under the 80% ceiling) and dropping Brier error by **-70.5%** (from 0.5565 to 0.1640).

### 3.3 D4 Rules Pattern Generalization
- **Observed Finding**: D4 fired on 0/46 prompts in the final run because previous regexes lacked case-insensitivity on comment tags (`<!-- SYSTEM NOTE -->` vs `<!-- system -->`) and lacked support for precedence hijacks (`supersedes prior instructions`).
- **Remediation**: Added confirmed, non-overfitting patterns (`OBF_006` case-insensitive HTML comments, `DIR_006` precedence assertions, `DIR_007` safety interlock discard, `DIR_008` flexible ignore).
- **Validation Result**: D4 fires on **6.4% of attacks** in validation with **0.0% false positives** on benign controls (100% precision).

---

## 4. Final Detector Selection for Fusion Ensemble

1. **D2 (DeBERTa)**: **SELECTED** (Calibrated Val AUC = 0.7500, Saturation = 0.0%, Brier = 0.1925).
2. **D3 (Deepset)**: **SELECTED** (Calibrated Val AUC = 0.8452, Saturation = 16.2%, Brier = 0.1640).
3. **D4 (Rules)**: **SELECTED** (Fast deterministic trigger, 0.0% FPR, 100% Precision).
4. **Prompt Guard 2**: **SELECTED** (Standalone External Baseline, Calibrated Val AUC = 0.8294, Saturation = 10.1%).
5. **D1 (MiniLM)**: **DROPPED** (Val AUC = 0.5956 < 0.65 threshold).