# Evolution from Pilot Evaluation to Scientific Closeout

> **DOCUMENT**: Architectural and Methodological Shifts Since Pilot Evaluation  
> **SCOPE**: Summary of design, protocol, governance, and evaluation upgrades implemented between Wave A (Pilot) and Wave K (Scientific Closeout).

---

## 1. Primary Methodological Transformations

| Dimension | Pilot Formulation (Wave A/B) | Scientific Closeout Formulation (Wave K) | Empirical / Methodological Rationale |
|:---|:---|:---|:---|
| **Primary System Arm** | Multiple competing arms; post-hoc best-metric selection across variants. | **Single pre-registered primary arm: Arm R** (Learned Router + Challenger V2 Judge + Policy B). | Prevents multi-testing selection bias. Baselines (Arm A, Arm B, Arm U, Arm C2, Prompt Guard 2) are formally designated as post-hoc descriptive comparisons. |
| **Judge Abstention Handling** | Silent fallback or unrecorded retry loops; unquantified timeout behavior. | **Unified Abstain Protocol**: explicit logging of `abstain`, `timeout`, and `parse_fail` states. | Eliminates hidden failure modes. On `held_out_v4`, the judge abstained on 10 attack prompts and 0 benign prompts (Fisher exact $p = 8.71 \times 10^{-5}$), incurring 39.5% higher latency. Under Policy B, an abstain prevents any block downgrade. |
| **Evaluation Scope** | In-distribution synthetic splits only (`held_out_v4`, $N=347$). | **Independent external benchmark suite** ($N=621$ across 3 families) evaluated as separate families and never pooled. | Tests performance on external corpora while preventing misleading aggregate figures: Gandalf ($n=112$, direct instructional overrides), NotInject ($n=309$, benign text with injection triggers), and Dolly ($n=200$, general instruction following). |
| **Data Governance & Integrity** | Ad-hoc split generation; undocumented deduplication. | **Cryptographic provenance locks**: SHA-256 manifests (`FROZEN.sha256`, `FROZEN_EXTERNAL.sha256`), text normalization, and leak exclusion. | Discovered and excluded 30 exact-match rows from NotInject (8.85% overlap with dev split). Disqualified deepset benchmark due to explicit pretraining overlap documented in the model card. |
| **Reproducibility Guarantee** | Untracked dynamic execution states. | **Deterministic policy replay**: 347/347 exact match check (`policy_replay_check.py`) and single-process hardware locks (`scripts/gpu_lock.py`). | Asserts identical arm decisions given logged scores and prevents GPU memory contention or deadlocks on constrained hardware (15.7 GB RAM, 4 GB VRAM). |

---

## 2. Key Architectural Upgrades

### 2.1 Unified Abstain Protocol and Policy B Invariants
In the pilot implementation, ambiguous prompts or LLM judge parsing errors lacked standardized downstream resolution. Under the Wave K protocol:
- **Authority Policy B**: Preliminary fusion `BLOCK` verdicts can only be downgraded to `FLAG` when the LLM judge explicitly outputs `ALLOW` with high confidence.
- **Fail-Safe Abstain Containment**: If the judge returns `abstain`, times out, or encounters a parse error, Policy B treats the judge verdict as non-authoritative and retains the underlying fusion verdict. A preliminary `BLOCK` is never downgraded on an abstain.

### 2.2 Single Pre-Registered Primary Arm (Arm R)
Rather than presenting opportunistic metrics across varying threshold combinations, Wave K established Arm R as the single primary pre-registered evaluation target:
- Operates a learned logistic regression error predictor ($\tau = 0.35$).
- Routes 72.33 calls per 100 requests to `gemma3:4b` on held-out evaluation.
- Delivers 84.17% loose recall (95% CI [77.20%, 89.31%]) with 68.75% benign pass-through and a 6.25% hard false-block rate.

### 2.3 Strict Separation of External Benchmarks
To address the risk of benchmark gaming and Simpson's paradox:
- External evaluations are presented strictly per-family (Gandalf vs NotInject vs Dolly) with 95% Wilson score confidence intervals.
- Pooling attack recall and benign false-alarm rates into a single composite score is explicitly barred under the protocol.

### 2.4 Cryptographic Audit and Execution Provenance
Every artifact is anchored to cryptographic digests:
- Protocol lock commit `7e98c39` (19:02 IST) preceded final evaluation run start (19:50 IST).
- Dataset files are bound by SHA-256 checksums in `FROZEN.sha256` and `FROZEN_EXTERNAL.sha256`.
- Figure sidecars log source data digests and assert exact equality against source JSON records.
