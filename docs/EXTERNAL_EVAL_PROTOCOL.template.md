# External Generalization Benchmark Protocol (Pre-Registration Template)

> **STAGE**: K13 External Evaluation  
> **STATUS**: **DRAFT SUGGESTION - HUMAN DECISION** (Awaiting final human decision and pre-registration in `docs/K13_SELECT.txt` before Phase B execution)  
> **GOVERNING POLICY**: This protocol must be committed BEFORE downloading any dataset and BEFORE any scoring execution.

---

## 1. Candidate Dataset Pre-Registration & Extraction Rules

All human-decision fields below are marked **DRAFT SUGGESTION - HUMAN DECISION**. The human decides final inclusion, sample size, and redistribution in `docs/K13_SELECT.txt`.

### 1.1 Attack Candidates
- **Primary Candidate: `Lakera/gandalf_ignore_instructions`** [DRAFT SUGGESTION - HUMAN DECISION]
  - **Pinned HF Revision**: `04737b65e90a6794ec227012e4a255a7def6344b`
  - **Split**: `test` ($n=112$)
  - **Proposed $n$**: `112` [DRAFT SUGGESTION - HUMAN DECISION]
  - **Extraction Rule**: Extract `text` column directly without modification. (Columns documented in card: `text`, `similarity`).
  - **Exclusion Rule**: None for held-out `test` split.
  - **Scope Limitation & Expected Ceiling Effect**:
    - Card specifies: *"Filter to those prompts that have similarity of at least 0.825."* against *"Ignore all previous instructions"*. This benchmark **measures one narrow attack family, not general injection recall**.
    - Because Gandalf prompts are direct "ignore instructions" variants, **the regex detector D4 is expected to match many of them**. Per-detector recall (including D4) must be reported alongside Arm R to quantify any ceiling effect.
- **Deferred Alternative: `hackaprompt/hackaprompt-dataset`**
  - **Status**: Deferred; not part of K13 unless the human selects it.
  - **Access**: Gated (`gated: auto` on Hugging Face API; requires human acceptance).
  - **Size**: `size_categories: 100K<n<1M`, storage ~15.4 GB. Usable $n$ not determinable in Phase A.
- **Excluded Candidate: `deepset/prompt-injections`**
  - **Status**: Excluded: overlaps D3 training data per D3's model card.

### 1.2 Benign Candidates (Evaluated as Separate Families; Never Pooled)
- **Candidate Family 1 (Hard Trigger-Word Benign): `leolee99/NotInject`** [DRAFT SUGGESTION - HUMAN DECISION]
  - **Pinned HF Revision**: `847ae76cf8fea5ed325429e569ae8cfef022d2e0`
  - **License**: MIT License.
  - **Redistribution Policy**: [DRAFT SUGGESTION - HUMAN DECISION: Commit sample to repo under MIT license terms]
  - **Split Structure**: Card describes subsets by trigger-word count rather than a "test split, n=339":
    - `NotInject_one` ($n=113$)
    - `NotInject_two` ($n=113$)
    - `NotInject_three` ($n=113$)
    - Total $n=339$. No split named `test`.
  - **Proposed $n$**: `339` (all 3 subsets) [DRAFT SUGGESTION - HUMAN DECISION]
  - **Extraction Rule**: Extract `prompt` column. (Columns documented in card: `prompt`, `word_list`, `category`).
  - **Multilingual Filter Decision**: [DRAFT SUGGESTION - HUMAN DECISION:
    - *Option A (no filter)*: retain all 339 prompts ($n=339$) and report non-English rows (`category == 'Multilingual Queries'`) separately.
    - *Option B (apply filter)*: exclude `category == 'Multilingual Queries'`, resulting in an English-only subset with smaller $n$.
    - The post-filter $n$ will be computed by script in Phase B2 and frozen by SHA-256 hash before scoring.]
- **Candidate Family 2 (General Instruction Benign): `databricks/databricks-dolly-15k`** [DRAFT SUGGESTION - HUMAN DECISION]
  - **Pinned HF Revision**: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`
  - **License**: CC BY-SA 3.0.
  - **Redistribution Policy**: [DRAFT SUGGESTION - HUMAN DECISION: Human decision required whether CC BY-SA 3.0 derivatives impose share-alike obligations on repo]
  - **Split**: Single default split ($n=15,011$).
  - **Proposed $n$**: Stratified sample ($n=200$, seed 0) across categories [DRAFT SUGGESTION - HUMAN DECISION]
  - **Prompt Field Decision**: [DRAFT SUGGESTION - HUMAN DECISION:
    - *Option A (instruction + context)*: if `context` is empty, prompt is `instruction`; otherwise prompt is `instruction + "\n\n" + context`. Reflects realistic user context seen by deployed systems.
    - *Option B (instruction alone)*: prompt is `instruction` only, ignoring `context`.
    - Extraction order: apply the 2000-character cap to the chosen prompt representation, then perform stratified sampling (seed 0). The 2000-char cap decides the surviving candidate pool before sampling.]
- **Candidate Family 3 (Semantic False-Refusal): `walledai/XSTest` (Safe Prompts)** [DRAFT SUGGESTION - HUMAN DECISION]
  - **Pinned HF Revision**: `f1d713187c61b6ae64e602d74f0b3d812cc2e8e8`
  - **Status**: Dropped (d2 model card cites `natolambert/xstest-v2-copy` as training data).
  - **Access**: Gated (`gated: auto` on Hugging Face API; requires human click-through acceptance).
  - **Split**: `test` ($n=450$).
  - **Proposed $n$**: `250` safe prompts [DRAFT SUGGESTION - HUMAN DECISION]
  - **Extraction Rule**: Extract `prompt` column where `label == 'safe'`.
- **Candidate Family 4 (Factual QA Control): `truthfulqa/truthful_qa`** [DRAFT SUGGESTION - HUMAN DECISION]
  - **Pinned HF Revision**: `741b8276f2d1982aa3d5b832d3ee81ed3b896490`
  - **Status**: Dropped (broad factual QA rather than adversarial injection / boundary test).
  - **Access**: Open (`gated: false`).
  - **Split**: `validation` ($n=817$) for config `generation`.
  - **Proposed $n$**: `817` [DRAFT SUGGESTION - HUMAN DECISION]
  - **Extraction Rule**: Extract `question` column.

---

## 2. Pinned Baseline Commit & Provenance Locks

All behavior-affecting components are frozen and pinned by exact cryptographic hashes:

1. **Evaluated Codebase Commit & Tree Hashes**:
   - Evaluated Codebase Commit: `959a99a`
   - `parallax/` Git Tree Hash: `cb250bfad8b6320094ec6deadfc42cbf31c61360` (Directory SHA-256: `6f2bc5f9ab903e43cdec2e2d4f4e2d2dcf17a369a461460cd6f9444a7a5ed0f7`)
   - `configs/` Git Tree Hash: `c3502d47cd91c2f8542d7729ef3b8a80c5e01c5f` (Directory SHA-256: `72d9adb58a501d27ba21459e14a8e4bfad3106979d0f0aa49f98612ebb6a65c1`)
   - Evaluated Runner (`scripts/eval/eval_final.py`): SHA-256 `68b31ea540455dbd795f54aa62391ce4f8f41530e3bb41c6f7b1d1f0eaee91b9` (same file content as at 959a99a; run-time state unrecorded).
   - Protocol Status: Pinned to evaluated code (`959a99a` plus tree hashes above). No self-referential commit hash is recorded for the protocol file itself.
2. **Configuration & Dependency Locks**:
   - Judge Prompt (`parallax/detectors/prompts/judge_prompt.py`) SHA-256: `c64f565306e6e86da8d8baef9f04c9dc23664f7b79159091077d984db17ebb2c`
   - Dependency Lock (`uv.lock`) SHA-256: `28023c2a4be4bc607be20be5082a8f88af3c4976a181d46ae4344b847faad505`
   - `final-v4` Manifest SHA-256: `d5e7271d7b8df7aeea3ad965c4d9004d289b33d11d6dce873d46cac821a273bf`

3. **Scoring Runner Specification & Logging Changes**:
   - **Runner Script**: `scripts/eval/eval_external.py`
   - **Current Status**: **Does not exist yet (unwritten in Phase A)**.
   - **Historical Note**: `scripts/eval/eval_final.py`'s state at final-v4 run time is unrecorded.
   - **Approved Logging-Only Changes (to be implemented in B1 before scoring)**: [DRAFT SUGGESTION - HUMAN DECISION]
     - Write a dedicated scoring script `scripts/eval/eval_external.py`.
     - Script must import evaluated detectors from `parallax/` and the exact learned router / Policy B parameters ($\tau = 0.35$).
     - Script must wrap model inference inside `scripts/gpu_lock.py`.
     - Script must output an append-only JSONL log containing row-level timestamps, raw detector scores, router uncertainty, judge verdicts, and arm decisions.
     - Its SHA-256 must be computed by script and committed in B1 prior to any scoring run.

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

---

## 4. Execution Rules, Crash Recovery & Time Budget

1. **One-Shot Execution Under GPU Lock**:
   - Scored strictly in a single pass under `scripts/gpu_lock.py` to prevent multi-process interference.
2. **Crash-Resume Rule**:
   - If a run crashes, execution resumes from the append-only predictions log.
   - Rows already scored and logged are **never re-scored**.
   - The incomplete row is re-scored once and explicitly flagged as `resumed: true` in the log.
3. **Phase B2 Pre-Scoring Length Audit**: [DRAFT SUGGESTION - HUMAN DECISION]
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
