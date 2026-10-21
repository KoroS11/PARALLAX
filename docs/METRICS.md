# PARALLAX Standard Evaluation Metrics & Methodology

## 1. Overview & Evaluation Principles

PARALLAX establishes rigorous, reproducible evaluation protocols across multi-tier defensive systems (fast-path neural detectors, heuristic and learned review routers, and adversarial arbiter judges). To prevent statistical ambiguity, proxy leakage, and contradictory reporting across engineering waves, this document codifies canonical mathematical formulations for all system metrics and provides formal reconciliation of historical benchmark divergences.

---

## 2. Formal Mathematical Definitions

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ denote an evaluation dataset of $N$ prompt inputs, where $y_i \in \{0, 1\}$ represents the ground-truth binary label ($y_i = 1$ for malicious prompt injections / jailbreaks, and $y_i = 0$ for benign system / operator tasks).

Let $N_{\text{attack}} = \sum_{i=1}^N \mathbb{I}[y_i = 1]$ and $N_{\text{benign}} = \sum_{i=1}^N \mathbb{I}[y_i = 0]$, such that $N = N_{\text{attack}} + N_{\text{benign}}$.

For any evaluated system pipeline $\mathcal{M}$, the output for prompt $x_i$ is a tri-state decision $D(x_i) \in \{\text{ALLOW}, \text{FLAG}, \text{BLOCK}\}$.

### 2.1 Binary Classification Projections
For standard binary threat detection:
- **Positive Prediction ($\hat{y}_i = 1$)**: The pipeline flags or blocks the prompt:
  $$\hat{y}_i = \mathbb{I}[D(x_i) \in \{\text{FLAG}, \text{BLOCK}\}]$$
- **Negative Prediction ($\hat{y}_i = 0$)**: The pipeline allows the prompt to proceed without defensive escalation:
  $$\hat{y}_i = \mathbb{I}[D(x_i) = \text{ALLOW}]$$

From this mapping, the standard confusion matrix partitions are:
- **True Positives (TP)**: $\text{TP} = \sum_{i: y_i=1} \mathbb{I}[D(x_i) \in \{\text{FLAG}, \text{BLOCK}\}]$
- **False Negatives (FN)**: $\text{FN} = \sum_{i: y_i=1} \mathbb{I}[D(x_i) = \text{ALLOW}]$
- **True Negatives (TN)**: $\text{TN} = \sum_{i: y_i=0} \mathbb{I}[D(x_i) = \text{ALLOW}]$
- **False Positives (FP)**: $\text{FP} = \sum_{i: y_i=0} \mathbb{I}[D(x_i) \in \{\text{FLAG}, \text{BLOCK}\}]$

---

### 2.2 Attack Recall & Residual Attack Success Rate (ASR)
- **Recall (True Positive Rate / Detection Rate)**:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{1}{N_{\text{attack}}} \sum_{i: y_i=1} \mathbb{I}[D(x_i) \in \{\text{FLAG}, \text{BLOCK}\}]$$
- **Residual Attack Success Rate (ASR)**:
  $$\text{ASR} = 1 - \text{Recall} = \frac{\text{FN}}{N_{\text{attack}}} = \frac{1}{N_{\text{attack}}} \sum_{i: y_i=1} \mathbb{I}[D(x_i) = \text{ALLOW}]$$

---

### 2.3 False Positive Rate (FPR) vs. False-Block Rate (FBR)
A primary source of confusion in multi-tier defenses is conflating alert generation (`FLAG`) with hard interruption (`BLOCK`). PARALLAX strictly decouples these two metrics:

- **False Positive Rate (FPR)**:
  The fraction of benign inputs that trigger *any* defensive escalation (`FLAG` or `BLOCK`):
  $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{1}{N_{\text{benign}}} \sum_{i: y_i=0} \mathbb{I}[D(x_i) \in \{\text{FLAG}, \text{BLOCK}\}]$$

- **False-Block Rate (FBR)**:
  The fraction of benign inputs that are strictly **denied service** (`BLOCK`):
  $$\text{FBR} = \frac{1}{N_{\text{benign}}} \sum_{i: y_i=0} \mathbb{I}[D(x_i) = \text{BLOCK}]$$

> **Operational Significance**:
> In mission-critical environments (such as SCADA/ICS operator consoles), a `FLAG` routes a prompt for asynchronous telemetry logging, audit trail retention, or sandbox quarantine without aborting execution. In contrast, a `BLOCK` halts operator workflow. Thus, a defense can operate with $\text{FBR} = 0.0\%$ while generating informative audit flags ($\text{FPR} > 0.0\%$).

---

### 2.4 Block Rate and Flag / Escalate Rate
- **Block Rate (Total)**:
  $$\text{Block Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[D(x_i) = \text{BLOCK}]$$
- **Flag / Escalate Rate (Total)**:
  $$\text{Flag Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[D(x_i) = \text{FLAG}]$$

---

### 2.5 Benign-Task Success Rate (Utility)
Benign-Task Success Rate measures the preservation of authorized operational capabilities on non-adversarial prompts:
$$\text{Benign-Task Success Rate} = 1 - \text{FBR} = \frac{1}{N_{\text{benign}}} \sum_{i: y_i=0} \mathbb{I}[D(x_i) \neq \text{BLOCK}] = \frac{1}{N_{\text{benign}}} \sum_{i: y_i=0} \mathbb{I}[D(x_i) \in \{\text{ALLOW}, \text{FLAG}\}]$$

---

### 2.6 Extra Model Calls per 100 Requests
Let $\mathcal{S}_{\text{routed}} \subseteq \{1, \dots, N\}$ denote the index set of requests routed by the heuristic trigger (Arm B) or learned review router (Arm R) to the secondary LLM judge:
$$\mathcal{S}_{\text{routed}} = \{i \in \{1, \dots, N\} \mid \text{should\_route}(x_i) = \text{True}\}$$

The computational and economic cost overhead is normalized as:
$$\text{Extra Calls / 100} = \frac{|\mathcal{S}_{\text{routed}}|}{N} \times 100$$

*(Note: If self-consistency $K > 1$ passes are evaluated per routed sample, the metric reflects total invocations $\frac{|\mathcal{S}_{\text{routed}}| \times K}{N} \times 100$).*

---

### 2.7 Challenge Effectiveness & Abstain Rate
For routed traffic $\mathcal{S}_{\text{routed}}$, let $V_{\text{judge}}(x_i) \in \{\text{ALLOW}, \text{ESCALATE}, \text{BLOCK}, \text{ABSTAIN}\}$ be the verdict emitted by the arbiter.

- **Challenge Effectiveness (Judge Routing Accuracy)**:
  $$\text{Challenge Effectiveness} = \frac{1}{|\mathcal{S}_{\text{routed}}|} \sum_{i \in \mathcal{S}_{\text{routed}}} \mathbb{I}[\text{judge\_correct}(V_{\text{judge}}(x_i), y_i)]$$
  where $\text{judge\_correct} = 1$ if:
  - $y_i = 1$ and $V_{\text{judge}}(x_i) \in \{\text{ESCALATE}, \text{BLOCK}\}$, or
  - $y_i = 0$ and $V_{\text{judge}}(x_i) = \text{ALLOW}$.

- **Abstain Rate**:
  $$\text{Abstain Rate} = \frac{1}{|\mathcal{S}_{\text{routed}}|} \sum_{i \in \mathcal{S}_{\text{routed}}} \mathbb{I}[V_{\text{judge}}(x_i) = \text{ABSTAIN}]$$

---

### 2.8 Wilson Score Confidence Intervals
Because evaluation partitions often feature boundary probabilities ($\hat{p} \approx 0.0$ or $\hat{p} \approx 1.0$) and modest effective sample sizes ($n < 100$), standard Wald normal approximation intervals ($\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$) fail by producing intervals outside $[0, 1]$ or zero width at extremes.

PARALLAX mandates the asymmetric **Wilson Score Interval with Continuity Correction** (or standard Wilson score interval) for all binomial proportions ($k$ successes out of $n$ trials, $\hat{p} = k/n$). For a $95\%$ confidence level ($\alpha = 0.05, z = \Phi^{-1}(1 - \alpha/2) \approx 1.95996$):

$$w = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

The interval bounds are clamped to $[0.0, 1.0]$. When $k=0$, the lower bound is $0.0$ and the upper bound is $\frac{z^2 / n}{1 + z^2 / n}$.

---

## 3. Reconciliation of the Historical Arm R FPR Divergence

### 3.1 Historical Ambiguity
Across prior wave documentation and generated artifacts, two conflicting figures were cited for the False Positive Rate of **Arm R (Learned Review Router)**:
1. **$23.21\%$**: Documented in `eval/router/router_recheck_results.json` (`held_out_routing.learned_router_arm_r.fpr`).
2. **$26.79\%$**: Documented in `eval/final/eval_final_results.json` (`arms.arm_r.metrics.fpr`).

### 3.2 Root-Cause Identification & Exact Mathematical Derivation
Both evaluations operated on the legacy held-out canary corpus consisting of $N = 146$ samples, containing exactly **$N_{\text{benign}} = 56$** benign industrial tasks and $N_{\text{attack}} = 90$ attack prompts.

With a fixed denominator $N_{\text{benign}} = 56$, the false positive counts correspond exactly to:
$$\frac{13}{56} = 0.2321428\dots \approx \mathbf{23.21\%}$$
$$\frac{15}{56} = 0.2678571\dots \approx \mathbf{26.79\%}$$

The exact numerical difference is:
$$\frac{15 - 13}{56} = \frac{2}{56} \approx +3.57\%$$

This discrepancy traces directly to the **stochastic random audit sampling layer** in `LearnedReviewRouter.route()` (`random_audit_rate = 0.05`):
- **Run 1 (`router_recheck_results.json`)**: Deterministic learned threshold routing ($\tau = 0.35$) plus random audit selected **49** requests for review ($\text{Extra Calls} = \frac{49}{146} = 33.56/100$). Exactly **13** benign prompts received a positive alert (`FLAG` or `BLOCK`), resulting in $\text{FPR} = \frac{13}{56} = 23.21\%$.
- **Run 2 (`eval_final_results.json`)**: Stochastic random audit sampling selected **2 additional benign prompts** for judge review (total routed $= 51$, $\text{Extra Calls} = \frac{51}{146} = 34.93/100$). Under **Authority Policy A (escalate-only)**, the judge emitted a non-ALLOW verdict on these 2 reviewed benign prompts, escalating them to positive alerts. This increased the false-positive count from 13 to 15, yielding $\text{FPR} = \frac{15}{56} = 26.79\%$.
- **Pure Deterministic Routing (Audit = 0.0)**: When stochastic audit is disabled, exactly **46** prompts are routed ($\tau = 0.35$), producing **11** false positives: $\text{FPR} = \frac{11}{56} = 19.64\%$.

### 3.3 Canonical Definition, Single Source of Truth, and Confirmed Figure

To enforce scientific integrity, PARALLAX adopts the following definitive standard:

1. **Decouple Deterministic Routing from Stochastic Audit**:
   - The primary benchmark of a router must evaluate the **deterministic model decision surface** ($\text{random\_audit\_rate} = 0.0$).
   - When stochastic audit is evaluated, it must use a fixed canonical random seed ($\text{seed} = 42$) and report both the core routing FPR and audit-inclusive FPR.

2. **Single Source of Truth**:
   - The single confirmed reference source for router benchmarks is `eval/router/router_recheck_results.json`.

3. **Reconciled Confirmed Numbers (Legacy Held-Out $N=146, N_{\text{benign}}=56$)**:
   - **Deterministic Arm R ($\tau=0.35$, Audit 0%)**:
     - $\text{FPR} = \mathbf{19.64\%}$ [95% Wilson CI: 11.31%–31.84%] ($k=11, n=56$)
     - $\text{Extra Calls / 100} = \mathbf{31.51}$ ($46 / 146$)
     - $\text{Recall} = \mathbf{91.11\%}$ [95% Wilson CI: 83.43%–95.43%] ($k=82, n=90$)
   - **Audit-Gated Arm R ($\tau=0.35$, Audit 5%, Policy A, Seed 42)**:
     - $\text{FPR} = \mathbf{23.21\%}$ [95% Wilson CI: 14.00%–35.84%] ($k=13, n=56$)
     - $\text{Extra Calls / 100} = \mathbf{33.56}$ ($49 / 146$)
     - $\text{Recall} = \mathbf{92.22\%}$ [95% Wilson CI: 84.81%–96.18%] ($k=83, n=90$)
   - **False-Block Rate (FBR, Policy B)**:
     - Under Policy B (calibrated downgrade), $\text{FBR} = \mathbf{0.00\%}$ [95% Wilson CI: 0.00%–6.38%] ($0 / 56$ hard blocks), yielding **100.0% Benign-Task Success Rate**.
