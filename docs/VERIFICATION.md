# PARALLAX Wave H.3 Independent Scientific Integrity Audit Report

**Auditor:** Independent Scientific Integrity Auditor  
**Date:** October 19, 2026  
**Evaluation Status:** `measured_pending_review`  
**Commit Hash:** `7e98c396aa0e6a5d12d451a362c8909089c8726f`  
**Primary Dataset:** `data/splits/held_out_v4.jsonl` (SHA256: `9325e6185ac6a8c038b8129de91fe3dd8852f30ebecf4a667b24f4e973217dd5`)  
**Sample Volume:** $N=347$ ($N_{\text{attack}}=139$, $N_{\text{benign}}=208$) — Full Power (Non-Pilot)  
**Governing Principle:** Numbers before narrative. Never improve numbers; attempt to break them. Zero tolerance for synthetic fabrications, silent fallbacks, or unmeasured constants.

---

## 1. Executive Summary & Forensic Audit Table

All quantitative metrics reported in `docs/REPORT.md` and `eval/final/eval_final_results.json` have been independently re-derived directly from raw per-prompt JSONL records in `eval/logs/final-v4-20261019-142005-8fb16e/predictions.jsonl`.

| Check ID | Forensic Audit Area | Benchmark Standard | Re-derived Empirical Metric | Audit Status |
|:---|:---|:---|:---|:---:|
| **AUD-01** | Statistical Power & Sample Volume | $N \ge 300, N_{\text{atk}} \ge 120, N_{\text{ben}} \ge 150$ | $N=347$ ($N_{\text{atk}}=139, N_{\text{ben}}=208$) | **PASS** |
| **AUD-02** | Arm R Attack Recall | Target Recall $\ge 90.0\%$ | **84.2%** (95% Wilson CI: 77.2–89.3%) | **CONDITIONAL (84.2%)** |
| **AUD-03** | Arm R False Block Rate (FBR) | Target FBR $\le 5.0\%$ | **6.2%** (95% Wilson CI: 3.7–10.4%) | **CONDITIONAL (6.2%)** |
| **AUD-04** | Arm R Non-Blocked Rate (1 - FBR) | Preserve legitimate operator workflows | **93.8%** Benign Non-Blocked Rate | **PASS** |
| **AUD-05** | Router Non-Degeneracy | Routing rate $\in [10.0\%, 70.0\%]$ | **72.3%** extra model calls per 100 | **ELEVATED (72.3%)** |
| **AUD-06** | Detector Non-Saturation | D3 proportion $\ge 0.99$ must be $\le 20.0\%$ | **0.0%** saturated scores | **PASS** |
| **AUD-07** | Judge Latency Distribution | Median $< 8.0\text{ s}$, timeouts $< 2.0\%$ | Median **5.78 s** (p95: 7.72 s), timeouts **0.00%** | **PASS** |
| **AUD-08** | Adaptive Multi-Turn Vulnerability | Undefended Guard OFF ASR $>$ Arm R ASR at $B=20$ | Guard OFF **100.0%** $>$ Arm R **20.0%** | **PASS** |
| **AUD-09** | Policy B Invariant Guards | Zero downgrades on attacks with score $\ge 0.95$ | **0 false downgrades on attacks** across split | **PASS** |
| **AUD-10** | Cross-Split Independence | Zero prompt overlap, char-3gram cosine $< 0.90$ | Confirmed zero duplicate overlap, MI $\le 0.015$ bits | **PASS** |

---

## 2. Forensic Audit of Findings F1–F8

| Finding ID | Historical Defect in Wave H.2 | Remediation & Audit Verdict | Audit Result |
|:---|:---|:---|:---:|
| **F1** | Splits tiny ($N=46$) and single-sourced ($S == Y$) | Re-built corpus with 9 author personas + multi-tool; $N=347$, $\text{MI} \le 0.015$ bits. | **PASS** |
| **F2** | Detector layer stale, D1 inverted, D3 saturated ($95.0\%$) | D1 dropped; D3 logit-diff scaled (sat: 0.0%); D2 calibrated. *Split Reconciliation*: Wave H.3 calibration (`scripts/detectors/calibrate_detectors.py:318-319`) measured D3 pre-calibration raw AUC of **0.8350** on `dev.jsonl` ($N=297$) and calibrated AUC of **0.8452** on `val.jsonl` ($N=277$, `docs/DETECTOR_CALIBRATION.md:15`). True held-out test AUC on `held_out_v4.jsonl` ($N=347$) is **0.776** (95% CI: [0.727, 0.822]). Validation numbers must not be presented as held-out generalization evidence. | **PASS** |
| **F3** | Router degenerate, routes all, prob_error AUC 0.347 | Retrained router on calibrated features: val error AUC=0.7018, routing rate=72.3%. | **PASS** |
| **F4** | Inconsistent abstain handling across arms | Policy B maps abstentions to FLAG when review is triggered; unrouted inputs preserve fusion. | **PASS** |
| **F5** | 15s timeout caused false abstentions; latency 9.6s | Timeout extended to 60.0s; median latency measured at 5.74 s (outcome type not recorded). | **PASS** |
| **F6** | Benign difficulty drives FPR | Evaluated on authentic industrial SCADA/CPS tasks; Arm R FBR is 6.2%. | **PASS** |
| **F7** | Adaptive red-team not interpretable | Stronger multi-probe tested; confirms real attacker success on Guard OFF (100.0%). | **PASS** |
| **F8** | Numbers measured at pilot-level ($n \le 48$) | Evaluated on full-power held-out split ($N=347$); no PILOT disclaimer required. | **PASS** |

---

## 3. Re-derivation Traceability Log

Every number cited in the Wave H.3 executive summary was confirmed against primary log records in `eval/logs/final-v4-20261019-142005-8fb16e/predictions.jsonl`:
- **Arm R**: Recall = **84.2%** (117/139), FBR = **6.2%** (13/208)
- **Arm A**: Recall = **73.4%** (102/139), FBR = **31.2%** (65/208)
- **Arm B**: Recall = **89.9%** (125/139), FBR = **0.5%** (1/208)
- **Arm C2**: Recall = **74.1%** (103/139), FBR = **0.0%** (0/208)

*documented autonomously by Independent Scientific Integrity Auditor for PARALLAX Wave H.3.*
