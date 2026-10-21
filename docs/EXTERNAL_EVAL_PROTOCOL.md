# External Generalization Benchmark Protocol (Pre-Registered Protocol)

> **STAGE**: K13 External Evaluation  
> **STATUS**: **FORMALLY PRE-REGISTERED & AUTHORIZED BY HUMAN** (`docs/K13_SELECT.txt`)  
> **AUTHORIZATION**: B1 (Runner Preparation & Freeze) and B2 (Dataset Download, Preprocessing, Hash Freeze, Length Audit) ONLY. Scoring is **NOT GRANTED**.  
> **GOVERNING POLICY**: This protocol is committed BEFORE downloading any dataset and BEFORE any scoring execution.

---

## 1. Candidate Dataset Pre-Registration & Extraction Rules

Human-selection decisions incorporated verbatim from `docs/K13_SELECT.txt`:

### 1.1 Attack Dataset
- **Selected Dataset: `Lakera/gandalf_ignore_instructions`**
  - **Pinned HF Revision**: `04737b65e90a6794ec227012e4a255a7def6344b`
  - **Split**: `test` ($n=112$ expected before deduplication)
  - **Selected $n$**: `112` expected rows before deduplication.
  - **Extraction Rule**: Extract `text` column directly without modification. (Columns documented in card: `text`, `similarity`).
  - **Exclusion Rule**: None for held-out `test` split.
  - **Scope Limitation & Expected Ceiling Effect**:
    - Card specifies: *"Filter to those prompts that have similarity of at least 0.825."* against *"Ignore all previous instructions"*. This benchmark **measures one narrow attack family, not general injection recall**.
    - Because Gandalf prompts are direct "ignore instructions" variants, **the regex detector D4 is expected to match many of them**. Per-detector recall (including D4) must be reported alongside Arm R to quantify any ceiling effect.
  - **License**: MIT License.

### 1.2 Benign Datasets (Evaluated as Separate Families; Never Pooled)
- **Benign Family 1 (Hard Trigger-Word Benign): `leolee99/NotInject`**
  - **Pinned HF Revision**: `847ae76cf8fea5ed325429e569ae8cfef022d2e0`
  - **License**: MIT License.
  - **Redistribution Policy**: Permitted under MIT license terms.
  - **Split Structure**: All three documented subsets included (`NotInject_one`, `NotInject_two`, `NotInject_three`), expected total $n=339$ before deduplication and filtering. Card documents subsets by trigger-word count (1, 2, or 3 trigger words); no split named `test` exists.
  - **Selected $n$**: `339` (all 3 subsets combined before deduplication).
  - **Extraction Rule**: Extract `prompt` column. (Columns documented in card: `prompt`, `word_list`, `category`).
  - **Multilingual Rows Decision**: Retained and reported separately. Non-English rows are identified by `category == 'Multilingual Queries'` per card taxonomy.
  - **Filtering**: No filtering based on detector scores, router uncertainty, or model predictions.
- **Benign Family 2 (General Instruction Benign): `databricks/databricks-dolly-15k`**
  - **Pinned HF Revision**: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`
  - **License**: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0). Attribution: Databricks, Inc.
  - **Redistribution Policy**: Row IDs and hashes only; **no raw prompt text committed to the repository**.
  - **Split**: Single default split ($n=15,011$).
  - **Prompt Fields Decision**: `instruction + context` (if `context` is empty, `instruction` alone; otherwise `instruction + "\n\n" + context`).
  - **Character Cap**: 2,000 characters evaluated on combined prompt, applied before sampling. Truncation is not used; prompts exceeding 2,000 characters are excluded prior to sampling.
  - **Sample Decision**: Stratified sample of $n=200$ across documented categories, random seed `0`. Cap and sampling are deterministic and completely independent of detector scores or evaluation outcomes.

### 1.3 Excluded and Deferred Datasets
- **`deepset/prompt-injections`**: Excluded; documented overlap with D3 training data. D3's model card cites training on `JasperLS/prompt-injections` and lists `deepset/prompt-injections` in metadata (exact repository identity between JasperLS and deepset not established from card texts).
- **`walledai/XSTest`**: Dropped; documented overlap in d2's model card (`natolambert/xstest-v2-copy` cited in training data).
- **`truthfulqa/truthful_qa`**: Dropped; no distinct evidence for this evaluation (broad factual QA rather than adversarial injection or false-refusal boundary test). Not excluded for contamination.
- **`hackaprompt/hackaprompt-dataset`**: Deferred; gated access requiring human click-through and unspecified split sizes; not included in this evaluation.

---

## 2. Pinned Baseline Commit & Provenance Locks

All behavior-affecting components are frozen and pinned by exact cryptographic hashes:

1. **Evaluated Codebase Commit & Tree Hashes**:
   - Evaluated Codebase Commit: `959a99a`
   - `parallax/` Git Tree Hash: `cb250bfad8b6320094ec6deadfc42cbf31c61360` (Directory SHA-256: `6f2bc5f9ab903e43cdec2e2d4f4e2d2dcf17a369a461460cd6f9444a7a5ed0f7`)
   - `configs/` Git Tree Hash: `c3502d47cd91c2f8542d7729ef3b8a80c5e01c5f` (Directory SHA-256: `72d9adb58a501d27ba21459e14a8e4bfad3106979d0f0aa49f98612ebb6a65c1`)
   - Evaluated Runner (`scripts/eval/eval_final.py`): SHA-256 `68b31ea540455dbd795f54aa62391ce4f8f41530e3bb41c6f7b1d1f0eaee91b9` (same file content as at 959a99a; run-time state unrecorded).
   - Protocol Status: Pinned to evaluated code (`959a99a` plus tree hashes above).
2. **Configuration & Dependency Locks**:
   - Judge Prompt (`parallax/detectors/prompts/judge_prompt.py`) SHA-256: `c64f565306e6e86da8d8baef9f04c9dc23664f7b79159091077d984db17ebb2c`
   - Dependency Lock (`uv.lock`) SHA-256: `28023c2a4be4bc607be20be5082a8f88af3c4976a181d46ae4344b847faad505`
   - `final-v4` Manifest SHA-256: `d5e7271d7b8df7aeea3ad965c4d9004d289b33d11d6dce873d46cac821a273bf`

3. **Scoring Runner Specification & Logging Changes**:
   - **Runner Script**: `scripts/eval/eval_external.py` (SHA-256: `992ba83a0536ddb3d5efbdc4350bfed6d47dd87d51a03287a7e2a4d7fc0a240c`)
   - **Approved Logging-Only Changes (to be implemented in B1 before scoring)**:
     - Write a dedicated scoring script `scripts/eval/eval_external.py`.
     - Script must import evaluated detectors from `parallax/` and the exact learned router / Policy B parameters ($\tau = 0.35$).
     - Script must wrap model inference inside `scripts/gpu_lock.py`.
     - Script must output an append-only JSONL log containing row-level timestamps, raw detector scores, router uncertainty, judge verdicts, and arm decisions.
     - Its SHA-256 must be computed by script and committed in B1 prior to any scoring run.
   - **Metrics Script**: `scripts/eval/external_metrics.py`
     - SHA-256: `404287d95138c34976130cdd5d6a88801920258dd3da540f089771daff25f077`
     - Provenance: Reproduces all 37 overlapping keys in `numbers.json` byte-for-byte on `final-v4` replay.

---

## 3. Pre-Registered Primary Arm & Post-Hoc Arms

- **Pre-Registered Primary Arm**: **Arm R** (Learned Review Router with Policy B, $\tau = 0.35$).
  - All primary claims, trade-offs, and performance metrics are pre-registered on Arm R.
- **Descriptive Arms (Labelled Post-Hoc)**:
  - Arm A (Real Ensemble Only, no judge)
  - Arm B (Static Heuristic Router with Policy B)
  - Arm U (Static Heuristic Router with Uncertainty Routing)
  - Arm C2 (Judge-First Standalone)
  - All arms other than Arm R are purely descriptive and will be reported in a table explicitly labelled **"Post-Hoc Descriptive Arms"**.
- **Excluded Baseline Arm**:
  - **PromptForest (Port 8765)**: In K13 Preflight 2, PromptForest with the 2-detector config (`configs/pf_two_detectors.yaml`) could not be matched because the upstream `promptforest` CLI daemon binary is not installed on PATH or in the environment (`Get-Command promptforest` failed). Per protocol stop rule, PromptForest is excluded from the descriptive comparison arms.

---

## 4. Execution Rules, Crash Recovery & Time Budget

1. **One-Shot Execution Under GPU Lock**:
   - Scored strictly in a single pass under `scripts/gpu_lock.py` to prevent multi-process interference.
2. **Crash-Resume Rule**:
   - If a run crashes, execution resumes from the append-only predictions log.
   - Rows already scored and logged are **never re-scored**.
   - The incomplete row is re-scored once and explicitly flagged as `resumed: true` in the log.
3. **Phase B2 Pre-Scoring Length Audit**:
   - In Phase B2, **before any scoring begins**, run a text-only length audit: compute length-only AUC (attack vs. benign) on the frozen external dataset.
   - State in results that attack and benign prompts originate from different sources, quantifying the extent to which length alone discriminates between the two distributions.
4. **Time Budget Reconciliation & Formula**:
   - Reconciled Timing Sources:
     - The **2,831.3 s** sum of logged component latencies ($\sum [T_{\text{arm\_a}} + T_{\text{judge}} + T_{\text{pg2}} + T_{\text{pf}}] = 2,831.31\text{ s}$, average $8.16\text{ s/row}$) is consistent with the **2,853 s** wall-clock span between run start (19:50:05 IST) and manifest created_at (20:37:38 IST) under sequential execution.
     - The **1,485.8 s** total ($\sum T_{\text{arm\_r}} = 1,485.83\text{ s}$, average $4.28\text{ s/row}$) is Arm R's attributed latency.
     - Per-row rates and runtime ranges are estimates:
       - **Gandalf** ($n=112$): $479\text{ s} - 914\text{ s}$ ($\mathbf{8.0\text{ min} - 15.2\text{ min}}$)
       - **NotInject** ($n=339$): $1,451\text{ s} - 2,766\text{ s}$ ($\mathbf{24.2\text{ min} - 46.1\text{ min}}$)
       - **Dolly Sample** ($n=200$): $856\text{ s} - 1,632\text{ s}$ ($\mathbf{14.3\text{ min} - 27.2\text{ min}}$)

---

## 5. Statistical Interpretation & Reporting Rules

1. **95% Confidence Intervals**:
   - All evaluation metrics (Loose Recall, Strict Recall, Pass-Through, False Block Rate, Judge Call Rate) must be reported with 95% bootstrap or Wilson score confidence intervals.
2. **No Binary Pass/Fail & No "Generalizes" Claims**:
   - Binary claims of success or failure are prohibited.
   - The word "generalizes" is prohibited.
3. **Statistical Uncertainty at $n=112$**:
   - For an attack sample of $n=112$, the Wilson 95% score interval width is mathematically derived:
     - At $p = 0.8417$ (baseline recall): 95% CI is $[76.02\%, 89.59\%]$ (interval width = **13.56 percentage points**).
     - At $p = 0.5000$: 95% CI is $[40.89\%, 59.11\%]$ (interval width = **18.21 percentage points**).
   - This wide interval ($\pm 6.8$ to $\pm 9.1$ percentage points) reflects substantial sample uncertainty; reports must explicitly disclose this limitation.
4. **Benign Isolation**:
   - Benign families are **never pooled**. Each benign candidate must be reported in a dedicated table column with its own distinct False Block Rate.

---

## 6. Formal Protocol Amendments

### 6.1 Amendment 1: NotInject `public_catalog` Overlap Stop & Human Authorization (2026-10-20)
- **Trigger**: During Phase B2 automated pre-scoring deduplication, the pre-registered stop rule fired:
  - 30 out of 339 rows (8.85%) in `leolee99/NotInject` matched prompts previously ingested into `data/splits/dev.jsonl` under `source: public_catalog` (IDs `notinject-benign-001` through `notinject-benign-030`).
  - Because 8.85% exceeds the pre-registered 5% overlap threshold, execution halted immediately without scoring.
- **Human Decision (Harsh Jain, 2026-10-20)**:
  - Exclude the 30 exact-match rows (`notinject-benign-001` through `notinject-benign-030`) from the external evaluation dataset.
  - The exclusion is strictly **text-only (exact normalized match)**, completely blind to detector scores, model predictions, or router uncertainty.
  - Evaluate `leolee99/NotInject` at candidate $n = 309$.
  - Disclose and report the 8.85% historical overlap in all evaluation artifacts and tables.
- **Original Stop Rule**: The original pre-registered stop rule remains in place and unedited as part of the permanent historical record.

### 6.2 Amendment 2: PromptForest Baseline Exclusion & Script Pinning (2026-10-21)
- **PromptForest Evaluation**: Upstream `promptforest` 1.0.0 was located at `C:\Users\harsh\.venv\Scripts\promptforest.exe`. Config `C:\Users\harsh\pf_two_detectors.yaml` (44 bytes, SHA-256 `a717f128c84f4128eafafb8591187766c2bbc93b87e66b9cb0976eb86f6847ef`) was tested on port 8765. Per protocol stop rule, PromptForest is excluded from descriptive comparison arms.
  - **Exact Reason**: final-v4 used the adapter's default three_detector mode (eval_final.py:428; promptforest.py:91); the two-detector daemon cannot reproduce it.
- **Metrics Script**: `scripts/eval/external_metrics.py` pinned at SHA-256 `404287d95138c34976130cdd5d6a88801920258dd3da540f089771daff25f077`.
- **Runner Script**: `scripts/eval/eval_external.py` pinned at SHA-256 `992ba83a0536ddb3d5efbdc4350bfed6d47dd87d51a03287a7e2a4d7fc0a240c`.
