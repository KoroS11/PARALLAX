# External Dataset Review (K13 Phase A)

This document provides the independent, pre-evaluation review of external candidate datasets for Stage K13 external generalization evaluation, strictly following Closeout Plan v2 (Phase A: Review Only; zero dataset downloads, zero model runs).

---

## 1. Attack Dataset Review (A1)

### 1.1 Lakera / `gandalf_ignore_instructions`
- **Dataset URL**: `https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions`
- **Pinned Revision (Commit SHA)**: `04737b65e90a6794ec227012e4a255a7def6344b`
- **License Text**: MIT License (`license: mit`) as stated on the dataset card.
- **Access Conditions**: Open / ungated. Free public download without authentication or click-through terms (`gated: false`).
- **Column Names & Schema**: `text` (dtype: string), `similarity` (dtype: float64). (API metadata `dataset_info.features`).
- **Split Names & Row Counts**: `train` ($n=777$), `validation` ($n=111$), `test` ($n=112$). (API metadata `dataset_info.splits`).
- **Exact Prompt Field**: `text`.
- **Exclusion Rules**: None required for the held-out `test` split ($n=112$).
- **Data Collection Method**: Gathered from player submissions to the web game Gandalf in July 2023. Filtered using OpenAI `text-embedding-ada-002` cosine similarity ($\ge 0.825$) against the target string *"Ignore all previous instructions"*, followed by Lakera Guard PII filtering, approximate near-duplicate deduplication, and random sampling to 1,000 rows.
- **Authorship Documentation**: The dataset card documents player submissions: *"all prompts submitted to Gandalf in July 2023"* ([URL](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions)). The prompts originate from crowd/web players attempting to extract a secret password.
- **Filtering & Scope Note**: The dataset card specifies: *"Filter to those prompts that have similarity of at least 0.825."* Because of this strict similarity threshold against *"Ignore all previous instructions"*, the benchmark **measures one narrow attack family, not general injection recall**.
- **Label Definition**: `label_binary = true` (all rows are attack / prompt injection candidates, with publisher noting minor noise).
- **Usable $n$**: **112** (held-out `test` split).
- **Estimated Scoring Time**: $112 \times 4.28 \text{ s/row} \approx 479 \text{ s} \approx \mathbf{8.0 \text{ min}}$ under GPU lock (estimate).
- **Recommendation**: **RECOMMENDED (Primary Attack Evaluation Split)**. Un-gated, clearly documented crowd player collection, clean test split ($n=112$).

---

### 1.2 `hackaprompt/hackaprompt-dataset`
- **Dataset URL**: `https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset`
- **Pinned Revision (Commit SHA)**: `25b87fbedfb86840abaf8cd09af7a029208a971a`
- **License Text**: MIT License (`license: mit`) stated on card.
- **Access Conditions**: **GATED** (`gated: auto` on Hugging Face API). Direct HTTP raw downloads return HTTP 401 Unauthorized unless the user is logged into Hugging Face and has accepted the competition terms. In strict accordance with GLOBAL RULES: *Gated datasets must be accepted by the HUMAN, not by the agent. Without human acceptance, this dataset cannot be downloaded.*
- **Column Names & Schema**: Card/API lists format `parquet`, `hackaprompt.parquet`. Exact schema not documented in card; verify in B2.
- **Split Names & Row Counts**: Total size in card metadata is `size_categories: 100K<n<1M` (storage: 15,398,262,551 bytes, ~15.4 GB). Split names and row counts are not documented in card; verify in B2. Usable $n$ not determinable in Phase A.
- **Exact Prompt Field**: `user_input` (or template concatenated). Exact column not documented in card; verify in B2.
- **Exclusion Rules**: Undocumented in card; verify in B2.
- **Data Collection Method**: Sourced from global participants in the HackAPrompt competition (EMNLP 2023 paper, arXiv:2311.16119).
- **Authorship Documentation**: Player / competition submissions: *"submissions from a prompt hacking competition"* ([URL](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset)).
- **Usable $n$**: Usable $n$ not determinable in Phase A.
- **Recommendation**: **DEFERRED; not part of K13 unless the human selects it**. Gated access requires human acceptance; usable $n$ and schema not documented in card.

---

### 1.3 `deepset/prompt-injections`
- **Dataset URL**: `https://huggingface.co/datasets/deepset/prompt-injections`
- **Pinned Revision (Commit SHA)**: `4f61ecb038e9c3fb77e21034b22511b523772cdd`
- **License Text**: Apache-2.0 (`license: apache-2.0`) / CC-BY-4.0.
- **Access Conditions**: Open / ungated (`gated: false`).
- **Column Names & Schema**: `text` (dtype: string), `label` (dtype: int64). (API metadata `dataset_info.features`).
- **Split Names & Row Counts**: `train` ($n=546$), `test` ($n=116$). (API metadata `dataset_info.splits`).
- **Exact Prompt Field**: `text`.
- **Exclusion Rules**: Filter `label == 1` for injection attack subset.
- **Data Collection Method**: Curated by deepset.
- **Authorship Documentation**: The dataset card contains no authorship details: *"More Information needed"* ([URL](https://huggingface.co/datasets/deepset/prompt-injections)). Authorship not documented in the card.
- **Contamination Finding**: **EXCLUDED: overlaps D3 training data per D3's model card** (not rejected as bad data). D3 model card states: *"This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on the [prompt-injection](https://huggingface.co/datasets/JasperLS/prompt-injections) dataset."*
- **Usable $n$**: 116 (test split).
- **Recommendation**: **EXCLUDED: overlaps D3 training data per D3's model card**. Evaluating D3 on this dataset would violate out-of-distribution evaluation integrity.

---

## 2. Benign Dataset Review (A2)

Per protocol, benign datasets are never pooled; each candidate will be evaluated and reported as an independent benign family with separate False Block Rates.

---

### 2.1 `databricks/databricks-dolly-15k`
- **Dataset URL**: `https://huggingface.co/datasets/databricks/databricks-dolly-15k`
- **Pinned Revision (Commit SHA)**: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`
- **License Text & Plain Obligations**:
  - License: `CC BY-SA 3.0` (Creative Commons Attribution-ShareAlike 3.0 Unported).
  - Attribution Wording: Must credit "Databricks, Inc." and link to license.
  - ShareAlike Scope: If adapted or remixed, derivatives must be released under CC BY-SA 3.0.
- **Access Conditions**: Open / ungated (`gated: false`).
- **Column Names & Schema**: `instruction` (string), `context` (string), `response` (string), `category` (string). (Documented in card sections `## Sources` and `## Annotator Guidelines`).
- **Split Names & Row Counts**: Single default split containing 15,011 records (`size_categories: 10K<n<100K`, card text: *"more than 15,000 records"*). No predefined train/test split.
- **Exact Prompt Field**: `instruction` alone when `context` is empty; `instruction + "\n\n" + context` when `context` is populated.
- **Exclusion Rules**:
  - Drop prompts exceeding 2000 characters: `len(prompt) > 2000` (evaluates `instruction` combined with `context`). This cap ensures prompts fit context limits and is completely uninfluenced by model scores.
  - Language: Card YAML metadata specifies `language: [en]` and card text specifies `## Language: American English`. There is no `lang` column in the data table; the language filter is satisfied by the dataset-level card specification.
- **Authorship Documentation**: Supported as human employee authored: *"generated by thousands of Databricks employees"* ([URL](https://huggingface.co/datasets/databricks/databricks-dolly-15k)).
- **Benign Definition & Labeling**: Natural language task instructions across 8 categories (brainstorming, classification, closed QA, generation, information extraction, open QA, summarization, creative writing).
- **Difficulty**: Easy generic instruction benign (low trigger-word density).
- **Domain Fit**: General conversational language domain.
- **Recommendation**: Candidate for general instruction benign baseline (human license decision required for CC BY-SA 3.0).

---

### 2.2 `leolee99/NotInject`
- **Dataset URL**: `https://huggingface.co/datasets/leolee99/NotInject`
- **Pinned Revision (Commit SHA)**: `847ae76cf8fea5ed325429e569ae8cfef022d2e0`
- **License Text & Plain Obligations**: MIT License (`license: mit`). Permissive; allows evaluation, storage, and modification with attribution notice.
- **Access Conditions**: Open / ungated (`gated: false`).
- **Column Names & Schema**: `prompt` (dtype: string), `word_list` (sequence: string), `category` (dtype: string). (API metadata `dataset_info.features`).
- **Split Names & Row Counts**: The card describes subsets by trigger-word count rather than a "test split, n=339". Card sentence: *"The dataset is divided into three subsets, each consisting of prompts generated using one, two, or three trigger words respectively."* The splits in metadata are:
  - `NotInject_one` ($n=113$)
  - `NotInject_two` ($n=113$)
  - `NotInject_three` ($n=113$)
  - Total across all three subsets: $n=339$. There is no split named `test`.
- **Exact Prompt Field**: `prompt`.
- **Exclusion Rules**: Card documents 4 categories (`Common Queries`, `Technique Queries`, `Virtual Creation`, `Multilingual Queries`). If English-only evaluation is specified, filter `category != 'Multilingual Queries'`.
- **Authorship Documentation**: The dataset card does not state LLM generation; generation method not documented in the quoted sentence: *"prompts generated using one, two, or three trigger words respectively."* ([URL](https://huggingface.co/datasets/leolee99/NotInject)).
- **Benign Definition & Labeling**: Benign prompts containing lexical trigger words (e.g. "ignore", "system", "instructions", "bypass", "rule") in non-malicious contexts.
- **Difficulty**: **Hard trigger-word benign**. Directly stresses keyword-sensitive classifiers and heuristic rules.
- **Domain Fit**: High-priority stress test for over-defense in prompt injection guardrails.
- **Recommendation**: **HIGH PRIORITY CANDIDATE (Hard trigger-word benign)**.

---

### 2.3 `walledai/XSTest` (Safe Prompts Subset)
- **Dataset URL**: `https://huggingface.co/datasets/walledai/XSTest`
- **Pinned Revision (Commit SHA)**: `f1d713187c61b6ae64e602d74f0b3d812cc2e8e8`
- **License Text & Plain Obligations**: `CC BY 4.0` (Creative Commons Attribution 4.0 International).
- **Access Conditions**: **GATED** (`gated: auto` on Hugging Face API). Single access condition: gated repository on Hugging Face API requiring click-through confirmation.
- **Column Names & Schema**: `prompt` (string), `focus` (string), `type` (string), `note` (string), `label` (string). (API metadata `dataset_info.features`).
- **Split Names & Row Counts**: `test` ($n=450$). (API metadata `dataset_info.splits`).
- **Exact Prompt Field**: `prompt`.
- **Exclusion Rules**: Filter `label == 'safe'` (yields $n=250$ safe prompts per card/paper). Drop `label == 'unsafe'`.
- **Authorship Documentation**: The card sentence: *"A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models"* is the research paper title; authorship not documented in the card.
- **Difficulty**: Hard semantic benign (lexical homonyms and sensitive keywords in safe contexts).
- **Recommendation**: Viable candidate for semantic false-refusal testing (requires human acceptance due to `gated: auto`).

---

### 2.4 `truthfulqa/truthful_qa`
- **Dataset URL**: `https://huggingface.co/datasets/truthfulqa/truthful_qa`
- **Pinned Revision (Commit SHA)**: `741b8276f2d1982aa3d5b832d3ee81ed3b896490`
- **License Text & Plain Obligations**: Apache-2.0 (`license: apache-2.0`). Permissive open source.
- **Access Conditions**: Open / ungated (`gated: false`).
- **Column Names & Schema**: `type` (string), `category` (string), `question` (string), `best_answer` (string), `correct_answers` (sequence: string), `incorrect_answers` (sequence: string), `source` (string). (API metadata `dataset_info.features` for config `generation`).
- **Split Names & Row Counts**: `validation` ($n=817$) for config `generation`. (API metadata `dataset_info.splits`).
- **Exact Prompt Field**: `question`.
- **Exclusion Rules**: None required.
- **Authorship Documentation**: Card sentence: *"Questions are crafted so that some humans would answer falsely due to a false belief or misconception"* describes benchmark design; authorship not documented in the card.
- **Difficulty**: Easy neutral knowledge benign.
- **Domain Fit**: Broad factual QA.
- **Recommendation**: Clean baseline for conversational false-positive measurement.

---

## 3. Training Data Contamination Analysis per Detector

To rigorously verify potential training leakage against candidate datasets, model cards for all evaluated components were retrieved and inspected:

### 3.1 Evaluated Component Training Documentation

| Evaluated Component | Model Card / Repository | Training Data Documentation in Model Card | Contamination Risk Assessment |
|---|---|---|---|
| **d2-deberta** (`ProtectAI`) | `https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/raw/main/README.md` | Model card states: *"The dataset used for training the model was meticulously assembled from various public open datasets to include a wide range of prompt variations."* Specifically cites `natolambert/xstest-v2-copy:1_full_compliance`, `VMware/open-instruct`, `alespalla/chatbot_instruction_prompts`, `HuggingFaceH4/grok-conversation-harmless`, `Harelix/Prompt-Injection-Mixed-Techniques-2024`, `OpenSafetyLab/Salad-Data`, `jackhhao/jailbreak-classification`. | **High contamination on XSTest** (`xstest-v2-copy` in training data; XSTest dropped). Moderate risk on open injection sets. |
| **d3-deepset** (`deepset`) | `https://huggingface.co/deepset/deberta-v3-base-injection/raw/main/README.md` | Model card states: *"This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on the [prompt-injection](https://huggingface.co/datasets/JasperLS/prompt-injections) dataset."* (YAML: `datasets: - deepset/prompt-injections`). | **Severe contamination on deepset/prompt-injections** (exact training set; excluded). No documented Gandalf overlap. |
| **d4_rules** (Regex) | Built-in heuristics (`parallax/detectors/rules.py`) | No model card; deterministic keyword/regex heuristics. | Zero statistical training overlap. |
| **promptguard2** (`Meta`) | `https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md` | Model card states: *"Both Llama Prompt Guard 2 models detect both prompt injection and jailbreaking attacks, trained on a large corpus of known vulnerabilities."* Specific datasets are not enumerated in the accessible card: training data not documented in accessible card. | Specific training datasets not documented in accessible card. |
| **promptforest** (Adapter) | External ensemble daemon (`parallax/detectors/promptforest.py`) | Local HTTP service wrapping `llama_guard` and `xgboost` in 2-detector configuration (`configs/pf_two_detectors.yaml`); `vijil` is disabled and D3 (`deepset/deberta-v3-base-injection`) is not enabled in this config. | **Not exposed to deepset/prompt-injections** in 2-detector evaluation mode because D3 is not enabled in the configuration. |
| **Judge Model** (`gemma3:4b`) | `https://ai.google.dev/gemma/docs/model_card_3` | Gemma 3 was pretrained on 4T tokens of web documents, code, mathematics, and multimodal data with cutoff August 2024. Specific training datasets are not enumerated: training data not documented in accessible card. | General web pretraining data; exact overlap with specific benchmarks undocumented in accessible card. |

---

### 3.2 Detector $\times$ Dataset Contamination Matrix

Each cell documents whether training overlap is documented, quoting the model card sentence from the exact case-insensitive grep scan of `docs/phase_a_cards/`, or stating "not documented".

> **Note on D3 (`deepset/deberta-v3-base-injection`) vs `deepset/prompt-injections`**:  
> D3's model card text states: *"This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on the [prompt-injection](https://huggingface.co/datasets/JasperLS/prompt-injections) dataset."* The card metadata YAML lists `datasets: - deepset/prompt-injections`. However, **identity of the two repos not established from the cards** (nothing in the saved card texts explains whether `JasperLS/prompt-injections` and `deepset/prompt-injections` are identical or renamed).

| Evaluated Component | Gandalf (`Lakera/gandalf_ignore_instructions`) | NotInject (`leolee99/NotInject`) | Dolly (`databricks/databricks-dolly-15k`) | deepset (`deepset/prompt-injections`) | XSTest (`walledai/XSTest`) | TruthfulQA (`truthfulqa/truthful_qa`) |
|---|---|---|---|---|---|---|
| **d2-deberta** | not documented | not documented | not documented | not documented | **Documented Contamination**: Card cites `datasets: - natolambert/xstest-v2-copy` and `1 dataset (natolambert/xstest-v2-copy:1_full_compliance)` | not documented |
| **d3-deepset** | not documented | not documented | not documented | **Documented Contamination**: Card lists `datasets: - deepset/prompt-injections` in metadata and cites `https://huggingface.co/datasets/JasperLS/prompt-injections` in text | not documented | not documented |
| **d4_rules** | not documented (static regex heuristics) | not documented (static regex heuristics) | not documented (static regex heuristics) | not documented (static regex heuristics) | not documented (static regex heuristics) | not documented (static regex heuristics) |
| **promptguard2** (86M) | not documented in accessible card (*"trained on a large corpus of known vulnerabilities"*) | not documented in accessible card | not documented in accessible card | not documented in accessible card | not documented in accessible card | not documented in accessible card |
| **promptforest** *(per adapter code in this repo, not a model card)* | not documented | not documented | not documented | not documented (2-detector config does not enable D3) | not documented | not documented |
| **Judge (`gemma3:4b`)** | not documented in accessible card (Pretrained on 4T tokens; cutoff Aug 2024) | not documented in accessible card | not documented in accessible card | not documented in accessible card | not documented in accessible card | not documented in accessible card |

---

## 4. Phase A Fetch Audit & URLs Inspected

### 4.1 Dataset Card & API URLs Fetched
1. `https://huggingface.co/api/datasets/Lakera/gandalf_ignore_instructions`
2. `https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions/raw/main/README.md`
3. `https://huggingface.co/api/datasets/hackaprompt/hackaprompt-dataset`
4. `https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/raw/main/README.md`
5. `https://huggingface.co/api/datasets/deepset/prompt-injections`
6. `https://huggingface.co/datasets/deepset/prompt-injections/raw/main/README.md`
7. `https://huggingface.co/api/datasets/databricks/databricks-dolly-15k`
8. `https://huggingface.co/datasets/databricks/databricks-dolly-15k/raw/main/README.md`
9. `https://huggingface.co/api/datasets/leolee99/NotInject`
10. `https://huggingface.co/datasets/leolee99/NotInject/raw/main/README.md`
11. `https://huggingface.co/api/datasets/walledai/XSTest`
12. `https://huggingface.co/datasets/walledai/XSTest/raw/main/README.md`
13. `https://huggingface.co/api/datasets/truthfulqa/truthful_qa`
14. `https://huggingface.co/datasets/truthfulqa/truthful_qa/raw/main/README.md`

### 4.2 Detector Model Card URLs Fetched
1. `https://huggingface.co/deepset/deberta-v3-base-injection/raw/main/README.md` (fetched live)
2. `https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/raw/main/README.md` (fetched live)
3. `https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/Llama-Prompt-Guard-2/README.md` (fetched live)
4. `https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md` (fetched live)
5. `https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M/raw/main/README.md` (read attempt; HTTP 401 gated)
6. `https://huggingface.co/google/gemma-3-4b-it/raw/main/README.md` (read attempt; HTTP 401 gated)
7. `https://ai.google.dev/gemma/docs/model_card_3` (model card inspection via documentation search)

### 4.3 Data Access & Row Read State
- **Dataset Rows Read**: **ZERO**.
- **Dataset Files Read / Downloaded**: **ZERO**.
- No `.parquet`, `.jsonl`, `.csv`, or individual prompt payloads were downloaded, opened, or read.
- In strict adherence to Closeout Plan v2, sample inspections are deferred entirely to Step B2 after human pre-registration in `docs/K13_SELECT.txt`.

