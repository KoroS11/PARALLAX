# PARALLAX Wave H.2: Pre-Registered Evaluation Protocol & Hypothesis Registry

**Protocol Status:** DRAFT / PRE-REGISTRATION (Corpus Rebuild In Progress)  
**Governing Principle:** Pre-register hypotheses, primary configurations, authority policies, and statistical recipes BEFORE any model scores the final test partition (`held_out_v3.jsonl`).

---

## 1. Corpus Partitioning & Provenance Ledger

### 1.1 Invalidation of Historical Splits
Forensic audit (`check_corpus2.py`) revealed that previous splits (`dev.jsonl`, `val.jsonl`, `held_out_v2.jsonl`) suffered from systematic twin-pair contamination:
- 80%–85% of prompts possessed an opposite-label nearest neighbor with cosine similarity > 0.8, including byte-identical texts across classes.
- One-sided gate checks allowed inverted shortcut signals ($\text{AUC} = 0.13 - 0.38$) to pass.
- In Task H3, `gemma3:4b` scored all 1,467 prompts across the repository, including `held_out_v2.jsonl` (recall 73.11%, FPR 45.60%, 25 abstains).

> [!CAUTION]
> **Diagnostic Demotion**: `data/splits/held_out_v2.jsonl`, `dev.jsonl`, and `val.jsonl` are formally designated **CONTAMINATED / DIAGNOSTIC ONLY**. They are permanently retired from model selection, threshold calibration, and headline reporting.

### 1.2 The Clean Held-Out Partition (`held_out_v3.jsonl`)
- The clean, untouched test split is **`held_out_v3.jsonl`**.
- It is generated entirely in Wave H.2 from independently authored CPS operational tasks and confirmed raw attack logs.
- Clustered strictly by **Source and Template ID** (no template spans splits).
- Cryptographically frozen in `data/splits/FROZEN.sha256`.
- **Zero-Touch Invariant**: `held_out_v3.jsonl` will **never** be accessed, read, scored, or evaluated by any detector, judge, or router model until Task H7 execution.

---

## 2. Pre-Registered Primary Configuration (Locked via Task R3)

Prior to evaluating `held_out_v3.jsonl`, exactly ONE primary configuration is selected and locked based strictly on empirical tuning across rebuilt `dev.jsonl` ($N=48$) and `val.jsonl` ($N=48$):

### 2.1 Locked Primary Configuration
- **Primary System**: **Arm R** (Learned Review Router + live Judge `gemma3:4b`).
- **Primary Authority Policy**: **Policy B** (Calibrated Downgrade: `BLOCK -> FLAG` when reviewer certifies benign operational context).
- **Primary Operating Point ($\tau$)**: **$\tau = 0.35$** (LearnedReviewRouter classification threshold on $y_{\text{wrong}}$).
- **Ensemble Base**: AnyFlag fusion of D1 (MiniLM), D2 (DeBERTa-v3), and D3 (Deepset).
- **Reviewer**: Live local Ollama `gemma3:4b` using strict prompt template hash `2510046f1ada9bb3` and quote evidence validation.
- **Input Text Transformation**: Neutral text masking (`eval/mask.py`) enabled by default (`input_masked=true`).

### 2.2 Empirical Baseline Expectations (from Dev & Val Splits)
- **Attack Recall**: Expected $\ge 95\%$ (empirical dev: 100.0% [95% CI: 0.8620, 1.0000], val: 100.0% [95% CI: 0.8620, 1.0000]).
- **False-Block Rate (FBR)**: Expected $20\% - 25\%$ (empirical dev: 25.00% [6/24], val: 20.83% [5/24]).
  *Note on Honest Reporting*: FBR of 20–25% represents a dramatic 75–80 percentage-point cut relative to unreviewed AnyFlag (100% FBR), while eliminating the pathological 94.6% false-block rate from Wave H.1. It is reported transparently as a bounded operational trade-off, not as 0.0%.
- **Reviewer Call Overhead**: Expected $65 - 80$ extra calls / 100 requests (empirical dev: 68.75 calls/100, val: 79.17 calls/100).

### 2.3 Secondary Systems (All Other Configurations in Final Report)
Every other evaluated configuration is explicitly designated **SECONDARY / COMPARATIVE BASELINE**:
1. **Arm A**: 3 Detectors Alone (AnyFlag fusion, 0 reviewer calls).
2. **Arm B**: Rule-Triggered Challenger (Policy B, threshold 0.35).
3. **Arm U**: Challenger on All Requests (Policy B, 100 reviewer calls/100).
4. **Arm R (Policy A)**: Learned Router with Escalate-Only Policy.
5. **Standalone Judge Arms**: Arm C (`qwen3:8b`), Arm C2 (`gemma3:4b`), Arm C3 (`hlyn-3b`).
6. **External Baselines**: PromptForest (Stock, Two-Detector, Three-Detector), Prompt Guard 2 Standalone, LlamaFirewall PG2.

---

## 3. Evaluation Invariants & Reporting Rules

1. **Masked Text Consumption**: All detectors, judges, and routers evaluate masked text (`eval.mask.mask_prompt_text`) by default (`input_masked=true`). Unmasked execution is reserved strictly for ablation diagnostics.
2. **Numbers Before Narrative**: Every reported metric must trace directly to a confirmed JSON results log backed by a per-run manifest (`eval/logs/manifests/<run_id>.json`).
3. **Transparent Operational Costs**: Attack Success Rate (ASR) must always be accompanied by the concurrent False-Block Rate (FBR) and extra model call rate.
4. **Small-$n$ Statistical Guard**: Any sub-corpus or technique family with $n < 20$ is strictly annotated as *illustrative only, under-powered* with wide confidence intervals.
5. **No Selective Reporting**: All results on `held_out_v3.jsonl` will be reported exactly as logged, regardless of whether they confirm or contradict prior hypotheses.
