# Primary Arm Check & Router Efficacy Audit (K3 & K12c)

## 1. Verbatim Evaluation Protocol & Primary Arm Determination

### 1.1 Complete Audit of `docs/EVAL_PROTOCOL_v2.md`
A full audit of `docs/EVAL_PROTOCOL_v2.md` (lines 1–116) was conducted to identify any stopping rules, call budget ceilings, or routed-fraction constraints:

- **Primary System Pre-Registration (Section 3.1, Line 66)**:
  > `### 3.1 Primary System: Arm R (Learned Review Router with Policy B)`  
  > `- Base Ensemble: AnyFlag fusion of Calibrated D2, Calibrated D3, and D4 Rules.`  
  > `- Routing Decision: Escalates to Challenger V2 if: \hat{P}(\text{ensemble wrong}) \ge \tau \quad (\tau = 0.35)`  
  > `  where features are extracted strictly from [d_2, d_3, d_4, \text{discrepancy}, \text{confidence}, \text{enc\_present}, \text{len}].`  
  > `- Authority Policy B (Certified Downgrade with Invariant Guards): ...`
- **Component Detector Invalidation & Selection (Section 2.1, Lines 34–56)**:
  Specifies rules for dropping individual component detectors (e.g. D1 MiniLM dropped for validation AUC $< 0.65$).
- **Absence of Stopping or Budget Rules**:
  **Zero stop rules, call budgets, or routed-fraction criteria exist anywhere in `docs/EVAL_PROTOCOL_v2.md`.**

### 1.2 Primary Arm Conclusion
Because `docs/EVAL_PROTOCOL_v2.md` pre-registers Arm R as the primary system and contains no stop rule, call budget, or performance threshold, **Arm R is retained as the primary system under evaluation**. Prior speculative dual readings ("Reading B") are deleted.

---

## 2. Post-Hoc Pareto Analysis Across Defense Arms

All metrics are compiled directly from `results/final-v4/predictions.jsonl` ($N=347$: 139 attacks, 208 benign) and cross-checked against `numbers.json`.

> **Reconciliation Note on Previous K12e/K12f Pareto Drafts**:  
> In prior draft iterations, a temporary table reported 120 attacks and an identical 123 benign ALLOW / 85 benign non-ALLOW across arms R, B, and C2. The exact file/commit origin of those specific values was not found in repo commit history (origin not established; likely an ephemeral draft script). Furthermore, the earlier K12e report incorrectly claimed that Arm C2's 123/85 split was computed from `predictions.jsonl`, whereas independent audit confirms Arm C2 produces 208 ALLOW, 0 FLAG, 0 BLOCK on benign inputs ($100.0\%$ pass-through). All values below are computed dynamically from `results/final-v4/predictions.jsonl` with strict assertions.

**Dominance Definition (computed by `scripts/eval/compute_pareto.py`)**:
An arm X dominates an arm Y (X > Y) iff:
1. Recall_loose(X) >= Recall_loose(Y)
2. Benign_FLAG+BLOCK(X) <= Benign_FLAG+BLOCK(Y)
3. Calls(X) <= Calls(Y)
AND strictly better on at least one of (1), (2), or (3). Under this definition, all evaluated arms are non-dominated trade-offs across the 3 operational objectives. Note that "non-dominated" is a property of the Pareto frontier (no single arm is universally worse than another), not a claim that any arm is superior. For example, Arm R achieves 22.19 fewer judge calls per 100 requests than Arm B (72.33 vs 94.52), but catches fewer attacks (117 vs 125) and has higher benign BLOCK (13 vs 1).

| Arm Identifier | Arm Description | Attack Decisions (ALLOW / FLAG / BLOCK) | Recall (Loose: FLAG/BLOCK) | Recall (Strict: BLOCK only) | Benign Decisions (ALLOW / FLAG / BLOCK) | Benign Pass-Through (ALLOW only) | Benign FLAG Load | Benign BLOCK Rate | Extra Calls / 100 | Post-Hoc Pareto Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Arm A** | AnyFlag Calibrated Fusion (D2+D3+D4) | 37 / 0 / 102 | **73.38%** [65.5%, 80.0%] | 73.38% [65.5%, 80.0%] | 143 / 0 / 65 | **68.75%** [62.2%, 74.7%] | 0.00% [0.0%, 1.8%] | 31.25% [25.3%, 37.8%] | **0.00** | Non-dominated (Pareto optimal) |
| **Arm R** | Learned Review Router + Policy B (tau=0.35) | 22 / 36 / 81 | **84.17%** [77.2%, 89.3%] | 58.27% [50.0%, 66.1%] | 143 / 52 / 13 | **68.75%** [62.2%, 74.7%] | 25.00% [19.6%, 31.3%] | 6.25% [3.7%, 10.4%] | 72.33 | Non-dominated (Pareto optimal) |
| **Arm B** | Discrepancy Trigger + Policy B | 14 / 52 / 73 | **89.93%** [83.8%, 93.9%] | 52.52% [44.3%, 60.6%] | 143 / 64 / 1 | **68.75%** [62.2%, 74.7%] | 30.77% [24.9%, 37.3%] | 0.48% [0.1%, 2.7%] | 94.52 | Non-dominated (Pareto optimal) |
| **Arm U** | Full Review Upper Bound (Policy B) | 10 / 56 / 73 | **92.81%** [87.3%, 96.0%] | 52.52% [44.3%, 60.6%] | 143 / 64 / 1 | **68.75%** [62.2%, 74.7%] | 30.77% [24.9%, 37.3%] | 0.48% [0.1%, 2.7%] | **100.00** | Non-dominated (Pareto optimal) |
| **Arm C2** | Standalone Gemma-3 4B Judge | 36 / 10 / 93 | **74.10%** [66.2%, 80.7%] | 66.91% [58.7%, 74.2%] | 208 / 0 / 0 | **100.00%** [98.2%, 100.0%] | 0.00% [0.0%, 1.8%] | 0.00% [0.0%, 1.8%] | **100.00** | Non-dominated (Pareto optimal) |
| **Prompt Guard 2** | Standalone Calibrated Baseline | 94 / 0 / 45 | **32.37%** [25.2%, 40.5%] | 32.37% [25.2%, 40.5%] | 191 / 0 / 17 | **91.83%** [87.3%, 94.8%] | 0.00% [0.0%, 1.8%] | 8.17% [5.2%, 12.7%] | **0.00** | Non-dominated (Pareto optimal) |
| **PromptForest** | External Tree Baseline | 133 / 0 / 6 | **4.32%** [2.0%, 9.1%] | 4.32% [2.0%, 9.1%] | 208 / 0 / 0 | **100.00%** [98.2%, 100.0%] | 0.00% [0.0%, 1.8%] | 0.00% [0.0%, 1.8%] | **0.00** | Non-dominated (Pareto optimal) |

---

## 3. Router Analysis: Skill Evaluation & Routing Asymmetry

### 3.1 ROC-AUC Evaluation
1. **AUC vs. Primary Fusion Error (Training Target)**:
   - Point ROC-AUC: **0.5296** (95% Bootstrap CI: **[0.4579, 0.6006]**).
   - **Assessment**: The confidence interval spans 0.5000, indicating **no demonstrated skill** (consistent with chance performance on the held-out distribution).
2. **AUC vs. Ground Truth Label**:
   - Point ROC-AUC: **0.4350** (95% Bootstrap CI: **[0.3703, 0.4991]**).
   - **Assessment**: Because the upper bound of the 95% CI is $0.4991 < 0.5000$, this reflects an operational **routing-rate asymmetry** across classes rather than an effective error predictor.

### 3.2 Routing-Rate Class Asymmetry
- **Benign Operational Prompts Routed**: 165 / 208 = **79.33%**
- **Attack Prompts Routed**: 86 / 139 = **61.87%**
- **Total Invocations**: 251 / 347 = **72.33%** (72.33 extra calls / 100 requests)
- **Operational Effect**: The router invoked the reviewer on benign traffic at a rate 17.46 percentage points higher than on malicious traffic.

---

## 4. Random Routing Comparison at Matched Budget

A 2,000-draw Monte Carlo simulation evaluated uniform random routing at the identical reviewer budget ($k=251$ reviewer calls per 347 requests, seed 0).

### 4.1 Comparative Performance Table

| Metric | Arm R (Empirical) | Random Routing (Equal Budget Mean) | Random Routing 95% Interval | Description |
|---|:---:|:---:|:---:|---|
| **Attack Recall (Loose)** | **84.17%** [77.2%, 89.3%] | **87.48%** | **[84.17%, 90.65%]** | Empirical 2.5%–97.5% quantile of the 2,000 draw means. |
| **Benign BLOCK Rate** | **6.25%** [3.7%, 10.4%] | **9.05%** | **[6.25%, 12.02%]** | Empirical 2.5%–97.5% quantile of 2,000 draws. |
| **Benign FLAG Load** | **25.00%** [19.6%, 31.3%] | **22.20%** | **[19.23%, 25.00%]** | Empirical 2.5%–97.5% quantile of 2,000 draws. |
| **Reviewer Invocations** | **72.33 / 100** | **72.33 / 100** | [72.33, 72.33] | Budget identical by construction ($k=251$). |

### 4.2 Three-Metric Conclusion
A balanced evaluation across all three operational metrics demonstrates trade-offs:
1. **Attack Detection**: Uniform random routing captures more attacks than Arm R (+3.31 percentage points loose recall, 87.48% vs 84.17%), because uniform sampling allocates 72.33% of calls to attacks, whereas Arm R only routed 61.87% of attacks.
2. **Human Audit Overhead**: Uniform random routing lowers the benign FLAG review load by 2.80 percentage points (22.20% vs 25.00%).
3. **Legitimate Availability**: Arm R preserves a lower hard false block rate by 2.80 percentage points (6.25% vs 9.05%), because its higher routing on benign inputs (79.33%) allowed Policy B to downgrade more benign blocks to non-blocking flags.
