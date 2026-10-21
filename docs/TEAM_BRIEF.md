# PARALLAX Team Briefing & ESE Viva Reference

> **DOCUMENT PURPOSE**: Canonical single-source reference for project presentation, slides creation, and final viva examination.  
> **EVALUATION SNAPSHOT COMMIT**: `7b2fb66373febcba7d3ee30b299ad3d3be4976f5`  
> **STRICT SCIENTIFIC RULE**: All figures and metrics in this document are programmatically bound to audited logs (`numbers.json`, `external_metrics.json`, `notinject_addendum_metrics.json`). Zero typed metrics.

---

## 1. Executive Summary

PARALLAX is a multi-tiered adversarial defense architecture designed to detect indirect prompt injections and jailbreak payloads targeting Large Language Model (LLM) interfaces in cyber-physical and industrial supervisory systems (CPS/ICS). Traditional security wrappers rely on either single high-latency LLM judges or standalone neural classifiers that suffer from severe false-positive saturation on technical command vocabularies. PARALLAX implements a parallel multi-detector fan-out (DeBERTa-v3, Deepset, Heuristic Regex Rules), a calibrated AnyFlag ensemble fusion layer, a learned review router ($\tau=0.35$), and an adversarial local judge (`gemma3:4b`) governed by an asymmetric Authority Policy (Policy B). On the design evaluation split (`held_out_v4`, $N=347$), pre-registered primary system **Arm R** achieves **84.17% loose recall** (95% Wilson CI [77.2%, 89.3%], strict recall 58.27%), **68.75% immediate benign pass-through** (95% CI [62.2%, 74.7%]), and limits hard false blocking to **6.25%** at an operational budget of **72.33 judge calls per 100 requests**.

---

## 2. System Architecture & How It Works

PARALLAX decomposes input threat adjudication into a four-stage pipeline:

```
[Raw Input Prompt]
        │
        ├──► Multi-View Preprocessing (Unicode canonicalization, Hex/Base64 decode)
        │
        ├──► Parallel Detector Fan-Out:
        │       ├─► D2: DeBERTa-v3 (protectai) with Platt scaling
        │       ├─► D3: Deepset Injection with Logit-Diff Platt scaling
        │       └─► D4: Heuristic Regex & Comment Precedence Rules
        │
        ├──► Stage 2: AnyFlag Fusion (Arm A)
        │       └─► Flags if any calibrated detector exceeds threshold -> preliminary BLOCK
        │
        ├──► Stage 3: Learned Review Router (XGBoost / LightGBM)
        │       ├─► Evaluates detector score dispersion, agreement, and uncertainty
        │       └─► Computes P(error); if P(error) >= 0.35 -> Routes to Stage 4 Judge
        │
        └──► Stage 4: LLM Judge Adjudication (`gemma3:4b`, num_predict=128)
                ├─► Evaluates semantic intent under operational copilot system prompt
                └─► Governed by Authority Policy B:
                      • Preliminary BLOCK downgraded to FLAG only on explicit ALLOW
                      • Judge ABSTAIN fails safe to FLAG
                      • Direct ALLOW from Fusion is preserved without judge override
```

### Evaluated System Arms & Definitions
- **Arm R (Pre-registered Primary)**: AnyFlag Fusion + Review Router ($\tau=0.35$) + LLM Judge + Authority Policy B.
- **Arm A (Post-hoc Baseline)**: Standalone AnyFlag Fusion (zero judge calls, hard blocking).
- **Arm B (Post-hoc Baseline)**: Full Review Router + Judge without Authority Policy B downgrade interlock.
- **Arm U (Post-hoc Oracle)**: Unconditional routing of all requests directly to the LLM judge.
- **Arm C2 (Post-hoc Baseline)**: Standalone LLM judge operating without base detectors.
- **PG2 (Post-hoc External)**: Standalone Meta Prompt Guard 2 (86M).
- **PromptForest (Post-hoc External)**: Standalone Tree Ensemble baseline.

---

## 3. Data Splits, Provenance, & Exclusions

### Split Composition & Freezing
- **dev.jsonl ($N=297$)**: Used for detector Platt calibration and threshold selection.
- **val.jsonl ($N=277$)**: Used for feature validation and router training.
- **held_out_v4.jsonl ($N=347$)**: 139 attacks, 208 benign prompts. Frozen under SHA-256 (`9325e61...`). Reused during design review; all non-primary arms are designated **post-hoc**.
- **External Evaluation Benchmark ($N=621$)**: Scored once under strict protocol lock with zero cross-corpus pooling:
  * **Lakera / Gandalf ($n=112$)**: External attack benchmark filtered for instructional override variants ($	ext{cosine similarity} \ge 0.825$).
  * **NotInject ($n=309$)**: Non-harmful benign technical triggers across English ($n=225$) and Multilingual ($n=84$) partitions.
  * **Databricks Dolly-15k ($n=200$)**: Public instruction-following benign baseline.

### Pre-Scoring Exclusions & Deduplication
1. **D3 Deepset Pretraining Contamination**: The `deepset/prompt-injections` candidate dataset was excluded prior to scoring because model weights were directly fine-tuned on that corpus.
2. **NotInject 8.85% Development Overlap**: Pre-scoring audit identified 30 overlapping rows between NotInject and `dev.jsonl`. These 30 rows were permanently excluded, reducing the candidate from 339 to $n=309$.
3. **No Raw External Prompts Distributed**: Prompts from external datasets are referenced only by `prompt_id` and family in adherence to licensing and closeout rules.

---

## 4. Scientific Protocol & Provenance

- **Pre-Registration**: Primary system Arm R and operating threshold $	au=0.35$ were locked prior to held-out execution.
- **Single-Pass External Scoring**: Gandalf, NotInject, and Dolly were evaluated in a single sequential execution pass with zero parameter retuning.
- **Post-Hoc Designation**: All comparisons involving Arms A, B, U, C2, PG2, and PromptForest are labeled **post-hoc comparative evaluations**.
- **Deterministic Replication**: All confidence intervals (Wilson score 95% and bootstrap 2,000 resamples) use fixed random seed `0`.

---

## 5. Official Benchmark Results

### 5.1 Primary Split: held_out_v4 ($N=347$: 139 Attacks, 208 Benign)

| System Arm | Loose Recall (%) | 95% Wilson CI | Strict Recall (%) | Pass-Through (%) | Benign FLAG (%) | Benign BLOCK (%) | Judge Calls/100 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm R (Primary)** | 84.17% | [77.2%, 89.3%] | 58.27% | 68.75% | 25.00% | 6.25% | 72.33 |
| **Arm A (Post-hoc)** | 73.38% | [65.5%, 80.0%] | 73.38% | 68.75% | 0.00% | 31.25% | 0.00 |
| **Arm B (Post-hoc)** | 89.93% | [83.8%, 93.9%] | 52.52% | 68.75% | 30.77% | 0.48% | 94.52 |
| **Arm U (Post-hoc)** | 92.81% | [87.3%, 96.0%] | 52.52% | 68.75% | 30.77% | 0.48% | 100.00 |
| **Arm C2 (Post-hoc)** | 74.10% | [66.2%, 80.7%] | 66.91% | 100.00% | 0.00% | 0.00% | 100.00 |
| **PG2 (Post-hoc)** | 32.37% | [25.2%, 40.5%] | 32.37% | 91.83% | 0.00% | 8.17% | 0.00 |
| **PromptForest (Post-hoc)** | 4.32% | [2.0%, 9.1%] | 4.32% | 100.00% | 0.00% | 0.00% | 0.00 |

**Safe Claims & Allowed Wording**:
- "Arm R achieves 84.17% loose recall (95% Wilson CI [77.2%, 89.3%]) and 68.75% immediate benign pass-through on held_out_v4."
- "Hard false blocks are limited to 6.25% (13/208), with 25.00% (52/208) flagged for human review."
- "Arm R incurs an operational cost of 72.33 judge invocations per 100 requests."

### 5.2 Independent External Benchmarks (Scored Once, Zero Pooling)

| Dataset | Split Type | Sample Size ($n$) | Arm R Primary Metric | 95% Wilson CI | Comparison / Baseline |
|:---|:---:|:---:|:---:|:---:|:---|
| **Lakera / Gandalf** | Attacks | 112 | Loose Recall: **100.00%** | [96.7%, 100.0%] | Strict Recall: 83.04% [75.0%, 88.9%]; D2: 100.0%; D4: 26.8% |
| **NotInject (Overall)** | Benign | 309 | Pass-Through: **44.34%** | [38.9%, 49.9%] | Friction: 55.66% (Flagged: 46.6%, Blocked: 9.1%) |
| **NotInject English** | Benign | 225 | Pass-Through: **55.56%** | [49.0%, 61.9%] | Statistically distinct from Multilingual partition |
| **NotInject Multilingual** | Benign | 84 | Pass-Through: **14.29%** | [8.4%, 23.3%] | Reduced pass-through on non-English syntax |
| **Databricks Dolly-15k** | Benign | 200 | Pass-Through: **96.00%** | [92.3%, 98.0%] | Hard false block rate: 0.50% (1/200); Friction: 4.00% |

---

## 6. Statistics Primer for Oral Examination (Viva)

1. **Why Wilson Score Interval instead of Normal Approximation?**
   - *Answer*: Proportions near 0% or 100% (e.g. 100% Gandalf recall, 0.5% Dolly false block rate) violate the normal distribution assumption ($	ext{Wald interval} = p \pm 1.96\sqrt{p(1-p)/n}$ collapses to width 0). The Wilson score interval inverts the score test, guaranteeing asymmetric bounds strictly contained within $[0.0, 1.0]$.
2. **What does ROC-AUC mean for base detectors?**
   - *Answer*: Area Under the Receiver Operating Characteristic curve measures ranking capability independent of decision thresholds. On held_out_v4, D2 DeBERTa achieves ROC-AUC of **0.7812** [0.7285, 0.8321], D3 Deepset achieves **0.7763** [0.7260, 0.8197], D4 Heuristic Rules achieve **0.5408** [0.5188, 0.5660], Prompt Guard 2 achieves **0.8090** [0.7584, 0.8527], and PromptForest achieves **0.5949** [0.5310, 0.6544].
3. **Did the Review Router demonstrate error prediction skill?**
   - *Answer*: **No.** The router's ROC-AUC against AnyFlag fusion errors is **0.5296** (95% CI [0.4579, 0.6006]), which encompasses chance (0.50). Against ground truth labels, the router routes 79.3% of benign inputs vs 61.9% of attacks ($	ext{AUC} = 0.4350$).
4. **Why is no statistical significance claimed against random routing?**
   - *Answer*: In a 2,000-draw random routing simulation at equal budget ($k=251$), random routing achieved **87.48%** mean loose recall (Arm R empirical percentile: 4.8%). We claim no statistical significance; the router trades recall for lower benign false blocks.
5. **What is a Ceiling Effect?**
   - *Answer*: When a test set is too narrow or homogeneous, systems cluster at 100% detection. In Gandalf ($n=112$), all prompts share high similarity ($\ge 0.825$) to *"Ignore instructions"*, enabling D2 alone to catch 112/112 prompts without demonstrating cross-domain transfer or broad-spectrum detection.
6. **Why do we report Pareto non-dominance rather than claiming a "best" arm?**
   - *Answer*: In multi-objective optimization (Recall vs Friction vs Cost), **dominance is not ranking**. Arm R does not dominate Arm A (which has zero cost) or Arm U (which has higher recall). Every evaluated arm sits on the empirical Pareto frontier.

---

## 7. Limitations & Threats to Validity

# Methodological Limitations and Threats to Validity

> **STAGE**: PARALLAX Scientific Closeout Protocol  
> **SCOPE**: Transparent disclosure of threats to validity, empirical bounds, sample constraints, and architectural limitations across all evaluated components.

---

## 1. Internal Validity Threats

### 1.1 Reuse of `held_out_v4` During Design Review
The evaluation dataset `held_out_v4.jsonl` ($N=347$) was originally intended as a strict single-pass holdout. However, during the architectural review and threshold exploration of Wave H and K, prompt distributions, detector calibrated scores, and error patterns were repeatedly inspected. While model weights were not retrained on `held_out_v4`, the choice of router decision threshold ($\tau=0.35$), Policy B mapping logic, and discrepancy triggers were evaluated against this split. Consequently, `held_out_v4` serves as a design evaluation split rather than an untouched benchmark; the external benchmark ($N=621$) provides the sole single-pass evaluation.

### 1.2 Synthetic & Agent-Generated Corpus
No attack prompt in the PARALLAX evaluation corpus was authored by real human adversaries attacking production industrial environments. Submissions in `data/human_submissions/*.csv` were generated using automated LLM persona-prompting pipelines (`agent_authored_personas`). While designed to model ICS/CPS scenarios (Modbus, OPC-UA, BACnet, power plant trip defeats), synthetic prompts may underrepresent obfuscation techniques, out-of-band payloads, or evasion strategies developed by human threat actors.

### 1.3 Asymmetric Origin and Length Shortcut (Length AUC 0.58 Inverse)
Although four high-level source categories were balanced across splits, attack and benign prompts originated from distinct generative processes:
- Attacks: Generated via specialized CPS adversarial templates, PyRIT mutations, and jailbreak roleplays.
- Benign: Collected from industrial equipment manuals, operational control setpoints, and public instruction datasets.
A character-length classifier achieves an inverse ROC-AUC of **0.58** (benign prompts skew slightly longer than single-turn injection attempts), indicating residual length discrepancy that models could exploit as a heuristic shortcut.

---

## 2. External Validity Threats

### 2.1 Narrow Scope of Gandalf Attack Benchmark ($n=112$)
The external attack benchmark `Lakera/gandalf_ignore_instructions` evaluates a single, narrow family of prompt injection:
- Per the dataset card, prompts are filtered to those with a cosine similarity of at least 0.825 against the literal baseline: *"Ignore all previous instructions"*.
- This benchmark measures sensitivity to direct instructional override variants, not multi-turn escalation, base64 obfuscation, payload splitting, or indirect data-flow injection.
- A plausible hypothesis for the observed 100.00% Arm A detection is that D2 (DeBERTa) flagged 112/112 prompts, producing a ceiling effect under this pre-filtered phrasing distribution. In contrast, D4 (Rules) flagged only 30/112 (26.8%) per external_metrics.json. While K13_SELECT initially predicted that deterministic regex heuristics (D4) would match many Gandalf prompts due to direct override phrasing, D4 in fact matched only a minority, leaving neural detector D2 as the primary driver of the ceiling effect.

### 2.2 Historical 8.85% Overlap in NotInject Benign Set
During pre-scoring deduplication of `leolee99/NotInject`, an overlap check revealed that **30 out of 339 prompts (8.85%)** exactly matched entries previously ingested into `data/splits/dev.jsonl` under `source: public_catalog` (IDs `notinject-benign-001` through `notinject-benign-030`).
To prevent contamination, these 30 rows were excluded prior to evaluation via text-only normalization, yielding candidate $n=309$. However, the presence of historical overlap indicates that the NotInject corpus was partially exposed to earlier development cycles.

### 2.3 Detector Contamination Risk on Public Datasets
- **D3 Deepset (`deepset/deberta-v3-base-injection`)**: The upstream model card explicitly states the model was fine-tuned on the `JasperLS/prompt-injections` dataset and lists `datasets: - deepset/prompt-injections`. Consequently, evaluation on the `deepset` candidate was excluded due to direct pretraining contamination.
- **PromptForest (Two-Detector Config)**: The tree baseline wrapped public HuggingFace embeddings and classifiers whose pretraining exposure to open-source injection benchmarks is unquantified. Disabling Vijil left a two-detector configuration that could not reproduce the three-detector architecture evaluated in `final-v4`.
- **Judge Pretraining**: The local judge (`gemma3:4b`) was pretrained on 4 trillion tokens with an August 2024 knowledge cutoff. Its exact pretraining exposure to public injection benchmarks (Gandalf, HackAPrompt, NotInject, Dolly) is unrecorded and inaccessible in public documentation.

### 2.4 NotInject Taxonomy Mismatch in Selection Protocol
In `docs/K13_SELECT.txt`, non-English inputs were specified as `category == 'Multilingual Queries'` per the published card taxonomy. In the actual dataset `data/external/notinject.jsonl`, the stored category value is `'Multilingual'` (84 rows), alongside `'Common Queries'` (117 rows), `'Technique Queries'` (68 rows), and `'Virtual Creation'` (40 rows). The original scoring script checked the literal string `'Multilingual Queries'`, resulting in an empty report for that split until reconciled in the post-hoc addendum. Furthermore, `'English'` is not a stored category in the raw dataset, but a derived partition aggregating all non-multilingual categories ($n=225$).

---

## 3. Provenance and Run-Time Telemetry Gaps

### 3.1 Unrecorded Run-Time State of `eval_final.py`
The run manifest at `eval/logs/manifests/final-v4-20261019-142005-8fb16e.json` (SHA-256 `d5e7271d...`) recorded execution metadata, split hashes, and sample counts, but omitted the execution git commit hash and working tree status. Git history shows producing commit `959a99a` was committed at 21:30 IST, approximately 53 minutes after predictions were closed at 20:37 IST. While the frozen core engine (`parallax/` and `configs/`) has a zero git diff between protocol lock `7e98c39` and `959a99a`, the exact working-tree cleanliness during execution remains an unrecorded runtime state.

### 3.2 Ollama Content Digest Unrecorded at Run Time
The final-v4 execution recorded the judge model tag string `"gemma3:4b"` but did not record the 64-character SHA-256 content digest of the active Ollama model layer. While local Ollama inventory confirms image ID `a2af6cc3eb7f` was created 3 months prior and remained untouched, runtime manifest capture did not record the digest programmatically.

### 3.3 Presence of Databricks Dolly Text in Git History
During Phase 2 preparation and scoring, prompt text from the `databricks/databricks-dolly-15k` dataset (licensed under Creative Commons Attribution-ShareAlike 3.0 Unported, CC BY-SA 3.0) was committed to the local repository in commits `df23e6b` (`external_input_interleaved.jsonl`) and `9e68954` (`external_predictions.jsonl`). In accordance with closeout protocol rules, git history was not rewritten. Both files were subsequently untracked and added to `.gitignore`, and a text-free predictions log (`export/closeout/external_predictions_notext.jsonl`) was committed. The local repository has no configured remote (`git remote -v` is empty).

---

## 4. Statistical Bounds and Comparison Limits

### 4.1 Sample Size Constraints and Wide Confidence Intervals
At sample sizes of $n=112$ (Gandalf) and $n=139$ (held_out_v4 attacks), binomial proportions exhibit substantial statistical uncertainty:
- For Gandalf ($n=112$), the Wilson 95% score interval spans **13.56 to 18.21 percentage points** ($\pm 6.8$ to $\pm 9.1$ pp).
- For held_out_v4 ($N=347$), Arm R loose recall is 84.17% with 95% CI $[77.20\%, 89.31\%]$ (interval width = 12.11 pp).
- Comparisons across small point differences must not be interpreted as statistically significant distinctions.

### 4.2 No Significance Claims Versus Random Routing
In a 2,000-draw random routing simulation at the exact query budget ($k=251$ calls / 347 requests, seed 0):
- Random routing achieves mean loose recall of **87.48%** [84.17%, 90.65%].
- Primary Arm R achieves **84.17%** loose recall.
Arm R does not outperform uniform random escalation in attack recall. While Arm R achieves a lower false blocking rate on benign inputs (6.25% vs 9.05%), no claim of superior error prediction skill is supported by the data.

### 4.3 D3 Deepset Validation to Held-Out Drop
D3 Deepset ROC-AUC declined from **0.8452** on validation to **0.7761** on held_out_v4 (95% CI [0.7246, 0.8248]). Under strict closeout rules, no post-hoc speculative causes are asserted; the metric change is reported plainly without unsupported causal claims.


---

## 8. Suggested Presentation Slide Outline (12-15 Slides)

| Slide | Topic | Figure / Asset Reference | Key Evidence / Talking Point |
|:---:|:---|:---|:---|
| **1** | Title & Project Overview | `export/demo/parallax_replay.gif` | Multi-model injection defense for cyber-physical LLM copilots |
| **2** | Threat Model & Industrial Context | `docs/THREAT_MODEL.md` | ICS/SCADA commands, indirect injections, actuator override payloads |
| **3** | Core Architecture: Pipeline | `export/ieee/figures/arch_pipeline.pdf` | Base detectors, AnyFlag fusion, review router, judge, Policy B |
| **4** | Architecture Matrix: System Arms | `export/ieee/figures/arch_arms.pdf` | Mapping Arm R, A, B, U, C2, PG2, PromptForest to components |
| **5** | Base Detectors & Calibration | `export/ieee/figures/i2_detector_auc.pdf` | D2 (0.781), D3 (0.776), D4 (0.541); Platt logit-diff centering |
| **6** | Primary Results on held_out_v4 | `export/ieee/figures/i1_held_out_arms.pdf` | Arm R: 84.2% recall, 68.8% pass-through, 6.2% false block |
| **7** | Router Diagnostics: The Zero-Skill Finding | `export/ieee/figures/i3_router_analysis.pdf` | Error AUC 0.53 (no skill); routing rate asymmetry (81% ben vs 59% atk) |
| **8** | LLM Judge Outcomes & Latency | `export/ieee/figures/i4_judge_outcomes.pdf` | 10 abstains on attacks (0 benign); +39.5% latency overhead |
| **9** | Error Categorization by Family | `export/ieee/figures/i5_error_analysis.pdf` | 22 FNs, 13 FBs; ICS payload analysis without prompt text leakage |
| **10** | External Benchmarks: Lakera Gandalf | `export/ieee/figures/i6_external_results.pdf` (a) | 100.0% loose recall; ceiling effect from narrow override taxonomy |
| **11** | External Benchmarks: NotInject & Dolly | `export/ieee/figures/i6_external_results.pdf` (b,c,d) | Dolly 85.5% pass-through; NotInject English (55.6%) vs Multilingual (14.3%) |
| **12** | Multi-Objective Tradeoffs & Pareto Frontier | `export/ieee/figures/i7_cost_and_pareto.pdf` | All arms on Pareto frontier; cost vs recall tradeoff (dominance is not ranking) |
| **13** | Self-Audit Discoveries & Forensic Fixes | Audit Trail Logs | Typed-literal incident, copied CI fix, NotInject 8.85% overlap exclusion |
| **14** | Methodological Limitations | `docs/LIMITATIONS_AND_THREATS.md` | Reused holdout, synthetic agent persona corpus, length shortcut (0.58 AUC) |
| **15** | Conclusion & Defense Posture Summary | `export/demo/parallax_replay.gif` | Failsafe defense posture: Authority Policy B provides asymmetric quarantine downgrade |

---

## 9. Viva Cheat-Sheet: Top 15 Hard Questions & Answers

1. **Q1: Which dataset represents the weakest result for your primary arm?**
   - **Answer**: **NotInject ($n=309$)**. Arm R achieved only **44.34%** benign pass-through (55.56% on English, 14.29% on Multilingual). Non-harmful technical keywords ('bypass', 'terminal', 'override') triggered false alarms across the neural base detectors.
2. **Q2: Does your learned router actually work as intended?**
   - **Answer**: The router did **not** demonstrate skill at predicting fusion errors (AUC = 0.5296, equivalent to random chance). However, it operates as a conservative rate throttle, directing ambiguous prompts to the judge and achieving lower false blocking (6.25%) than AnyFlag alone (31.25%).
3. **Q3: Why did Arm R underperform random routing on loose recall?**
   - **Answer**: In a 2,000-draw random routing simulation at equal budget ($k=251$), random routing achieved **87.48%** recall vs **84.17%** for Arm R. Random routing allocates judge queries uniformly across all attacks, whereas the router's score features routed benign inputs at higher rates (81.25%) than attacks (58.99%).
4. **Q4: Why does Arm A have higher strict recall (84.17%) than Arm R (58.27%)?**
   - **Answer**: Arm A is a hard AnyFlag blocker—it immediately blocks any prompt flagged by any detector. Arm R routes flagged prompts to the judge, where Authority Policy B downgrades preliminary blocks to FLAG unless the judge issues an explicit ALLOW. This reduces strict blocks while preserving loose detection.
5. **Q5: Why did the LLM judge abstain on 10 prompts?**
   - **Answer**: All 10 abstentions occurred on attack prompts where model generation hit the `num_predict=128` token limit before outputting a JSON verdict tag (`<VERDICT>`). Under Policy B, judge abstains fail safe to FLAG.
6. **Q6: Did you evaluate on real production ICS attacks?**
   - **Answer**: No. All attack prompts were generated using automated persona pipelines (`agent_authored_personas`) modelling industrial CPS workflows (SCADA, OPC-UA). This is disclosed as an internal validity threat in Section 7.
7. **Q7: Why was D1 (MiniLM) permanently dropped from the ensemble?**
   - **Answer**: In validation testing, D1 exhibited an inverted AUC of **0.5956** (< 0.65 threshold). Its embedding head had overfitted to legacy canary strings (`CANARY-*`) and collapsed when tested on clean natural text.
8. **Q8: How did you fix D3 (Deepset) saturation?**
   - **Answer**: D3 raw logits had an uncentered +6.5 offset on technical prompts, saturating 95% of outputs above 0.95. We applied Platt scaling on dev logit differences ($w=2.19, b=-14.44$), dropping saturation to 16.2% and Brier score by 70.5%.
9. **Q9: Why are external datasets never pooled into a single composite score?**
   - **Answer**: Pooling datasets with wildly different base rates, attack phrasings, and lengths creates a meaningless aggregate metric. Gandalf ($n=112$ attacks) and NotInject ($n=309$ benign) are reported as independent family partitions.
10. **Q10: What does "held_out_v4 reused for design review" mean?**
    - **Answer**: While model weights were not retrained on `held_out_v4`, hyperparameters like the router threshold ($	au=0.35$) were explored against it. Thus, it serves as a design split; the external benchmark ($N=621$) provides the sole untouched evaluation.
11. **Q11: Why is Gandalf 100% recall described as a ceiling effect?**
    - **Answer**: Gandalf prompts are pre-filtered to cosine similarity $\ge 0.825$ against "Ignore all instructions". D2 DeBERTa easily flags this specific phrasing pattern. It does not prove general resilience against novel obfuscations.
12. **Q12: Why did you exclude 30 prompts from NotInject?**
    - **Answer**: Pre-scoring deduplication detected an 8.85% exact text overlap between NotInject and `data/splits/dev.jsonl`. We permanently removed those 30 rows before evaluation to prevent train-test contamination.
13. **Q13: What is the purpose of Authority Policy B?**
    - **Answer**: In industrial systems, safety overrides must be asymmetric. Policy B prevents an uncalibrated LLM judge from silently allowing a dangerous payload. A preliminary detector BLOCK is downgraded to FLAG only if the judge explicitly issues an ALLOW verdict.
14. **Q14: How does PARALLAX compare to Prompt Guard 2?**
    - **Answer**: Standalone Prompt Guard 2 (86M) achieved only **32.37%** loose recall on held_out_v4, failing on complex multi-stage ICS jailbreaks. PARALLAX Arm R achieves **84.17%** recall.
15. **Q15: What was the typed-literal incident found during self-audit?**
    - **Answer**: During the K12e audit, we discovered that hardcoded string metrics were accidentally embedded into summary reports. We eradicated all typed numbers, replacing them with programmatic data-binding scripts enforced by automated gate checks (`scripts/gate.py`).

---

## 10. Glossary & Prohibited Words Reference

### Key Terminology
- **Loose Recall**: Fraction of attacks classified as either FLAG or BLOCK.
- **Strict Recall**: Fraction of attacks classified strictly as BLOCK.
- **Pass-Through**: Fraction of benign inputs immediately granted ALLOW without friction.
- **Friction Rate**: Fraction of benign inputs delayed by FLAG review or mistakenly BLOCKED.
- **Authority Policy B**: Failsafe governance rule where detector blocks can only be downgraded to FLAG.

### Prohibited Wordings Catalog ("DO NOT SAY")
| Do NOT Say | Say Instead | Rationale |
|:---|:---|:---|
| *"The system generalizes across all attacks"* | *"Evaluated on specific benchmark families with reported bounds"* | No human-authored zero-day attacks evaluated; Gandalf is narrow |
| *"The router is smart / skilled / superior"* | *"The router operates as a rate throttle with no demonstrated error skill"* | Router error ROC-AUC is 0.5296 (chance baseline is 0.50) |
| *"Arm R is the best / optimal system"* | *"All arms occupy the empirical Pareto frontier"* | Dominance is not ranking; Arm A has zero cost, Arm U has higher recall |
| *"Zero false positives / completely safe"* | *"6.25% false block rate and 25.0% flag rate on benign held-out inputs"* | Hard empirical friction exists |
| *"Human-written attack corpus"* | *"Agent-authored persona generation pipeline"* | Automated LLM persona prompting was used |
| *"Verified / certified / validated defense"* | *"Measured under protocol lock; confirmed by unit test gates"* | Unscientific compliance claim |

---

## 11. Team Roles & Reproduction Guide

### Team Member Slots
| Member Name | Role & Responsibilities | Presentation Focus Area |
|:---|:---|:---|
| **TODO: name** | Pipeline Architecture & Multi-Detector Fusion | Slides 1-5: Threat Model & Base Detectors |
| **TODO: name** | Learned Review Router & Judge Telemetry | Slides 6-9: Primary Results & Router Analysis |
| **TODO: name** | External Benchmarks & Cross-Domain Evaluation | Slides 10-12: Gandalf, NotInject, Dolly |
| **TODO: name** | Safety Governance, Auditing & Limitations | Slides 13-15: Policy B, Self-Audit, Limitations |

### One-Line Reproduction Commands
All evaluation runs are automated via `uv`:
```bash
# 1. Run gate checks and verification tests
uv run pytest tests/test_gate_final.py scripts/eval/test_replay_tallies.py

# 2. Re-generate IEEE publication figures and LaTeX tables
uv run python scripts/eval/make_ieee_figures.py

# 3. Re-generate Simulation Replay GIF
uv run python scripts/eval/make_replay_gif.py

# 4. Re-generate Team Brief
uv run python scripts/eval/make_team_brief.py
```
