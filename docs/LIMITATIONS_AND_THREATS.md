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
