# PARALLAX Closeout Numbers Sheet (One-Page Fact Source)

> **PURPOSE**: Canonical, script-derived reference numbers for teammates drafting the final report and presentation slides.
> **GOVERNING RULE**: Every number below is read directly from `numbers.json` or `results/final-v4/predictions.jsonl`. Zero hard-coded literals.
> **GIT COMMIT**: `2f0605073db035919fd53fcec5c6fe21d40e34b7`

---

## 1. Primary System: Arm R (`held_out_v4`, N=347: 139 attacks, 208 benign)

| Metric | Exact Value | 95% Confidence Interval | Source Key / File |
|---|:---:|:---:|---|
| **Loose Recall (FLAG+BLOCK)** | **84.17%** | [77.2%, 89.3%] | `arm_r.loose.recall` (`numbers.json`) |
| **Strict Recall (BLOCK only)** | **58.27%** | [50.0%, 66.1%] | `arm_r.strict.recall` (`numbers.json`) |
| **Benign Pass-Through (ALLOW)** | **68.75%** | [62.2%, 74.7%] | `arm_r.benign.pass_through` (`numbers.json`) |
| **Benign FLAG Load** | **25.00%** | [19.6%, 31.3%] | `arm_r.benign.flag_load` (`numbers.json`) |
| **Benign False Block Rate (FBR)** | **6.25%** | [3.7%, 10.4%] | `arm_r.benign.block_rate` (`numbers.json`) |
| **Judge Invocations per 100** | **72.33** | *Deterministic routing* | `arm_r.extra_calls_per_100` (`numbers.json`) |
| **Latency p50** | **5322.6 ms** | — | `arm_r.latency_p50_ms` (`numbers.json`) |
| **Latency p95** | **7199.4 ms** | — | `arm_r.latency_p95_ms` (`numbers.json`) |

---

## 2. Descriptive Comparison Arms on `held_out_v4` (Post-Hoc)

| Arm | Loose Recall | Strict Recall | Benign Pass-Through | Benign False Block | Calls / 100 | Dominance Status |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Arm A (AnyFlag Fusion)** | 73.38% | 73.38% | 68.75% | 31.25% | 0.00 | Non-dominated |
| **Arm B (Discrepancy Trigger)** | 89.93% | 52.52% | 68.75% | 0.48% | 94.52 | Non-dominated |
| **Arm U (Review Upper Bound)** | 92.81% | 52.52% | 68.75% | 0.48% | 100.00 | Non-dominated |
| **Arm C2 (Judge Standalone)** | 74.10% | 66.91% | 100.00% | 0.00% | 100.00 | Non-dominated |
| **Prompt Guard 2 (86M)** | 32.37% | 32.37% | 91.83% | 8.17% | 0.00 | Non-dominated |
| **PromptForest (Tree)** | 4.32% | 4.32% | 100.00% | 0.00% | 0.00 | Non-dominated |

---

## 3. Router Diagnostics & Random-Routing Benchmark

- **Router AUC vs Ground Truth Label**: **0.4350** [0.3703, 0.4991] (CI excludes 0.500; reflects asymmetric routing rate: 58.99% on attacks vs 81.25% on benign)
- **Router AUC vs AnyFlag Fusion Error**: **0.5296** [0.4579, 0.6006] (CI includes 0.500; confirms **no demonstrated skill** at error prediction)
- **Random Routing Empirical Simulation** ($k=251$ calls, 2,000 draws, seed 0):
  - Loose Recall Mean: **87.48%** [84.17%, 90.65%]
  - Benign Block Mean: **9.05%** [6.25%, 12.02%]
  - Benign Flag Mean: **22.20%** [19.23%, 25.00%]
- **Conclusion**: Arm R's loose recall (84.17%) does not surpass the random routing mean (87.48%), but achieves a lower benign block rate (6.25% vs 9.05%).

---

## 4. Judge Diagnostics & Abstain Audit

- **Total Judge Decisions Evaluated**: 347 (226 ALLOW, 111 BLOCK, 10 ABSTAIN)
- **Abstain Asymmetry**: 10 on attacks (10/139 = 7.19%) vs 0 on benign (0/208 = 0.00%); Fisher exact $p = 8.71 \times 10^{-5}$
- **Abstain Latency Impact**: Non-abstain mean = 5,322 ms; Abstain mean = 7,422 ms (+39.5% latency overhead)
- **Policy B Downgrade Effect**: In 7 of the 10 abstain rows, judge abstain turned a fusion BLOCK into a final FLAG

---

## 5. Dataset Split Sizes

- `dev`: 297 prompts (149 attacks, 148 benign)
- `val`: 277 prompts (138 attacks, 139 benign)
- `held_out_v4`: 347 prompts (139 attacks, 208 benign)
- `external`: 621 prompts (Gandalf: 112 attacks, NotInject: 309 benign, Dolly: 200 benign)

---

## 6. External Benchmark Sets & NotInject Subgroup Addendum (Post-Hoc)

| Dataset / Partition | Category / Subset | Sample Size (n) | Primary Arm R Metric | 95% Wilson Interval | Pre-registered Scope |
|---|---|:---:|:---:|:---:|---|
| **Lakera/gandalf** | Attack prompts (similarity >= 0.825) | 112 | **100.00%** Loose Recall | [96.7%, 100.0%] | One narrow attack family (ceiling effect) |
| **Lakera/gandalf** | Attack prompts (strict recall) | 112 | **83.04%** Strict Recall | [75.0%, 88.9%] | BLOCK only decisions |
| **leolee99/NotInject** | All Benign (Trigger Words) | 309 | **44.34%** Pass-Through | [38.9%, 49.9%] | Hard trigger-word benign |
| **leolee99/NotInject** | English Queries (Post-hoc Addendum) | 225 | **55.56%** Pass-Through | [49.0%, 61.9%] | Baseline trigger resistance (FBR: 17.78%) |
| **leolee99/NotInject** | Multilingual (Post-hoc Addendum) | 84 | **14.29%** Pass-Through | [8.4%, 23.3%] | Multilingual trigger shift (FBR: 26.19%) |
| **leolee99/NotInject** | 1 Trigger Word | 101 | **68.32%** Pass-Through | [58.7%, 76.6%] | Monotonic trigger degradation (FBR: 7.92%) |
| **leolee99/NotInject** | 2 Trigger Words | 101 | **41.58%** Pass-Through | [32.5%, 51.3%] | Monotonic trigger degradation (FBR: 23.76%) |
| **leolee99/NotInject** | 3 Trigger Words | 107 | **24.30%** Pass-Through | [17.2%, 33.2%] | Monotonic trigger degradation (FBR: 28.04%) |
| **databricks-dolly-15k** | General Instruction Benign | 200 | **85.50%** Pass-Through | [80.0%, 89.8%] | Stratified sample across 8 tasks (FBR: 1.00%) |

