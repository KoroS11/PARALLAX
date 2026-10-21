# PARALLAX Wave H.3: Gate v3 Statistical Power & Shortcut Invariance Audit Report

**Audit Status:** PASSED (ALL SPLITS CONFIRMED INVARIANT TO TRIVIAL SHORTCUTS)  
**AUC Gate Requirement:** Point AUC and 95% Bootstrap CI upper bound <= 0.70  
**Leave-One-Source-Pair-Out Grouped CV:** 0.35 <= AUC <= 0.65  
**Source-Label Independence:** MI <= 0.05 bits, Chi2 p-value >= 0.01  
**Max Cross-Label Cosine:** < 0.90  
**Max Marker Gap Ceiling:** <= 15.0 percentage points  

---

## 1. Executive Summary & Split Compliance

| Split | Total | Attacks (Target) | Benign (Target) | Power Pass | Source-Label MI | Chi2 p-val | Length AUC [95% CI] | Marker AUC [95% CI] | Format AUC [95% CI] | Char n-gram AUC | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dev` | 297 | 114 (100) | 183 (150) | PASS | 0.0137 b | 0.0442 | 0.556 [0.479, 0.630] | 0.613 [0.552, 0.677] | 0.632 [0.559, 0.696] | 0.956 | PASS |
| `val` | 277 | 109 (80) | 168 (120) | PASS | 0.0135 b | 0.0608 | 0.517 [0.447, 0.595] | 0.575 [0.500, 0.642] | 0.599 [0.522, 0.674] | 0.966 | PASS |
| `held_out_v4` | 347 | 139 (120) | 208 (150) | PASS | 0.0148 b | 0.0167 | 0.611 [0.547, 0.671] | 0.561 [0.505, 0.618] | 0.634 [0.570, 0.687] | 0.980 | PASS |

---

## 2. Detailed Baseline Performance Breakdown

### Split: `dev`

| Baseline Classifier | 5-Fold ROC AUC | 95% Bootstrap CI | LOGO CV AUC | Gate Threshold | Compliance |
|:---|:---:|:---:|:---:|:---:|:---:|
| `length_only` | **0.5559** | [0.4788, 0.6305] | 0.5918 | <= 0.70 (Upper CI) | PASS |
| `char_ngram_tfidf` | **0.9562** | [0.9327, 0.9749] | 0.7826 | Informative | PASS |
| `marker_regex` | **0.6129** | [0.5522, 0.6770] | 0.5149 | <= 0.70 (Upper CI) | PASS |
| `punctuation_format` | **0.6324** | [0.5589, 0.6958] | 0.5792 | <= 0.70 (Upper CI) | PASS |

### Split: `val`

| Baseline Classifier | 5-Fold ROC AUC | 95% Bootstrap CI | LOGO CV AUC | Gate Threshold | Compliance |
|:---|:---:|:---:|:---:|:---:|:---:|
| `length_only` | **0.5175** | [0.4470, 0.5950] | 0.4122 | <= 0.70 (Upper CI) | PASS |
| `char_ngram_tfidf` | **0.9661** | [0.9471, 0.9833] | 0.8595 | Informative | PASS |
| `marker_regex` | **0.5754** | [0.4999, 0.6423] | 0.4719 | <= 0.70 (Upper CI) | PASS |
| `punctuation_format` | **0.5992** | [0.5222, 0.6743] | 0.6022 | <= 0.70 (Upper CI) | PASS |

### Split: `held_out_v4`

| Baseline Classifier | 5-Fold ROC AUC | 95% Bootstrap CI | LOGO CV AUC | Gate Threshold | Compliance |
|:---|:---:|:---:|:---:|:---:|:---:|
| `length_only` | **0.6109** | [0.5473, 0.6712] | 0.4947 | <= 0.70 (Upper CI) | PASS |
| `char_ngram_tfidf` | **0.9803** | [0.9687, 0.9897] | 0.8447 | Informative | PASS |
| `marker_regex` | **0.5613** | [0.5053, 0.6176] | 0.4504 | <= 0.70 (Upper CI) | PASS |
| `punctuation_format` | **0.6340** | [0.5698, 0.6873] | 0.4108 | <= 0.70 (Upper CI) | PASS |

---

## 3. Provocative Marker Class Rate Audit (<= 15.0 pt Gap)

### Marker Rates: `dev`

| Marker Pattern | Attack Class Rate (%) | Benign Control Rate (%) | Absolute Gap (pts) | Compliance (<= 15 pts) |
|:---|:---:|:---:|:---:|:---:|
| `bypass` | 6.1% | 6.6% | **0.4 pts** | PASS |
| `canary` | 0.0% | 0.0% | **0.0 pts** | PASS |
| `command` | 8.8% | 8.2% | **0.6 pts** | PASS |
| `directive` | 18.4% | 7.7% | **10.8 pts** | PASS |
| `disregard` | 6.1% | 3.8% | **2.3 pts** | PASS |
| `emergency` | 14.0% | 9.3% | **4.8 pts** | PASS |
| `execute` | 1.8% | 3.8% | **2.1 pts** | PASS |
| `failsafe` | 0.0% | 3.8% | **3.8 pts** | PASS |
| `ignore` | 12.3% | 7.7% | **4.6 pts** | PASS |
| `mock_tool` | 0.0% | 0.0% | **0.0 pts** | PASS |
| `override` | 21.9% | 9.8% | **12.1 pts** | PASS |
| `system` | 15.8% | 14.2% | **1.6 pts** | PASS |
| `token` | 0.0% | 3.3% | **3.3 pts** | PASS |

### Marker Rates: `val`

| Marker Pattern | Attack Class Rate (%) | Benign Control Rate (%) | Absolute Gap (pts) | Compliance (<= 15 pts) |
|:---|:---:|:---:|:---:|:---:|
| `bypass` | 6.4% | 7.1% | **0.7 pts** | PASS |
| `canary` | 0.0% | 0.0% | **0.0 pts** | PASS |
| `command` | 11.9% | 8.3% | **3.6 pts** | PASS |
| `directive` | 22.9% | 10.7% | **12.2 pts** | PASS |
| `disregard` | 3.7% | 4.2% | **0.5 pts** | PASS |
| `emergency` | 10.1% | 8.3% | **1.8 pts** | PASS |
| `execute` | 5.5% | 4.2% | **1.3 pts** | PASS |
| `failsafe` | 3.7% | 4.2% | **0.5 pts** | PASS |
| `ignore` | 5.5% | 7.7% | **2.2 pts** | PASS |
| `mock_tool` | 0.0% | 0.0% | **0.0 pts** | PASS |
| `override` | 22.9% | 10.1% | **12.8 pts** | PASS |
| `system` | 15.6% | 20.2% | **4.6 pts** | PASS |
| `token` | 0.9% | 3.6% | **2.6 pts** | PASS |

### Marker Rates: `held_out_v4`

| Marker Pattern | Attack Class Rate (%) | Benign Control Rate (%) | Absolute Gap (pts) | Compliance (<= 15 pts) |
|:---|:---:|:---:|:---:|:---:|
| `bypass` | 2.9% | 5.3% | **2.4 pts** | PASS |
| `canary` | 0.0% | 0.0% | **0.0 pts** | PASS |
| `command` | 11.5% | 8.7% | **2.9 pts** | PASS |
| `directive` | 12.9% | 9.1% | **3.8 pts** | PASS |
| `disregard` | 1.4% | 3.4% | **1.9 pts** | PASS |
| `emergency` | 7.2% | 5.8% | **1.4 pts** | PASS |
| `execute` | 4.3% | 3.4% | **0.9 pts** | PASS |
| `failsafe` | 0.0% | 3.9% | **3.9 pts** | PASS |
| `ignore` | 0.7% | 7.2% | **6.5 pts** | PASS |
| `mock_tool` | 0.0% | 0.0% | **0.0 pts** | PASS |
| `override` | 10.8% | 6.2% | **4.5 pts** | PASS |
| `system` | 2.2% | 13.0% | **10.8 pts** | PASS |
| `token` | 2.2% | 3.4% | **1.2 pts** | PASS |

---

## 4. Verification Guarantees (Gate v3)

1. **Adequate Statistical Power**: Splitting exceeds sample minimums (dev >= 100/150, val >= 80/120, held_out_v4 >= 120/150), preventing low-power passes.
2. **Source-Label Decoupling**: Every source contributes both attacks and benign prompts (MI <= 0.05 bits, Chi2 p >= 0.01), eliminating origin shortcuts.
3. **Zero Marker Shortcuts**: No single synthetic marker (`CANARY-*`, `mock_*()`) exists exclusively in attacks; all provocative operator keywords appear at statistically matched rates (gap <= 15.0 pts).
4. **Strict Split Isolation**: Whole human authors are held out via author-hashed clustering; no prompt text overlap exists across splits.