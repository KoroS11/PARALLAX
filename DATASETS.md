# DATASETS: Dataset Provenance, Licenses, Gating Ledger, and Split Distributions

All datasets used in PARALLAX are documented here with their exact source paths, licenses, gating access requirements, split distributions, and architectural role within the system.

Per `AGENTS.md` and `DESIGN.md`:
* **Harmless Payloads Only:** All benchmark samples and sandbox triggers use benign canary tokens, mock commands, or non-destructive probes. No real malware, credentials, or dangerous exploits are stored or evaluated.
* **Normalized Schema:** All loaders emit uniform `DatasetSample` objects (`text, label_multiclass, label_binary, source, split, metadata`).
* **Honest Evaluation (In-Distribution vs OOD):** `neuralchemy` splits are in-distribution only for detector D1 (MiniLM+LR) and the validation fusion stacker. Pretrained binary models D2 (`deberta-v3`) and D3 (`Prompt Guard 2`) were pretrained externally. Independent benchmarks (`Simsonsun`, `NotInject`, and `Mindgard`) serve as fair cross-detector out-of-distribution (OOD) tests.

---

## 1. Dataset Inventory & Provenance

| Dataset Name | HuggingFace / Source Repository | License | Gated Access? | Primary Role in PARALLAX |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt Injection Technique** | `neuralchemy/prompt-injection-technique` | Apache-2.0 | No (Open) | Train D1 MiniLM+LR classifier, holdout validation partition for fusion stacker, and in-distribution multi-class evaluation (E1). |
| **Prompt Injection Dataset** | `neuralchemy/Prompt-injection-dataset` | Apache-2.0 | No (Open) | Binary classification baseline cross-checks and calibration. |
| **Mindgard Evaded Samples** | `Mindgard/evaded-prompt-injection-and-jailbreak-samples` | Research / Gated | Yes (HF approval required) | OOD robustness benchmark (E2) evaluating evasion techniques (character perturbation, AML optimization) by `attack_name`. |
| **Simsonsun Jailbreaks** | `Simsonsun/JailbreakPrompts` | Open / MIT | No (Open) | Independent jailbreak benchmark dataset (E2) covering in-the-wild prompt attacks. |
| **NotInject** | `leolee99/NotInject` | Apache-2.0 | No (Open) | False Positive Rate (FPR) and over-defense evaluation (E3) using benign prompts containing sensitive trigger words. |
| **CPS-Themed Attacks** | Local red-team generators (`parallax/redteam/`) | CC0 / MIT (Project-created) | No | Original CPS-specific industrial automation attacks for sandbox ASR evaluation (E6). |

---

## 2. Partition Counts and Class Distributions

### Summary of Dataset Partitions

| Dataset | Train Split | Validation Split | Test Split | Total Samples | Multiclass Classes | Binary Positive Ratio |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `neuralchemy/prompt-injection-technique` | 25,648 | 3,298 | 3,211 | **32,157** | 8 canonical classes | ~87.5% attack / 12.5% benign (`none`) |
| `neuralchemy/Prompt-injection-dataset` | 4,391 | 941 | 942 | **6,274** (core) | Binary + category metadata | ~50% attack / 50% benign |
| `Simsonsun/JailbreakPrompts` | — | — | 2,426 (`Dataset_1` & `Dataset_2`) | **2,426** | `jailbreak` | 100% attack |
| `leolee99/NotInject` | — | — | 339 (`NotInject_one/two/three`) | **339** | `none` (benign controls) | 0% attack (100% benign) |
| `Mindgard/evaded-prompt-injection-and-jailbreak-samples` | — | — | Gated (Evaluated on access) | Variable | Evasion techniques by `attack_name` | 100% attack |

### Canonical Multi-Class Technique Breakdown (`neuralchemy/prompt-injection-technique`)
Detector D1 and multi-class benchmarks utilize the 8 canonical classes defined in `DESIGN.md` Section 3:
1. `none` (ID 0) — Benign control prompts (`label_binary = False`)
2. `keyword_override` (ID 1) — Direct instruction overrides and ignore prefixes
3. `persona_play` (ID 2) — Roleplay and character-adoption attacks
4. `encoding` (ID 3) — Base64, hex, rot13, and cipher obfuscation
5. `payload_splitting` (ID 4) — Token fragmentation across sentences or variables
6. `context_overflow` (ID 5) — Padding with voluminous text to exhaust context windows
7. `few_shot_poisoning` (ID 6) — Demonstrations nudging model toward refusal overrides
8. `multilingual` (ID 7) — Cross-lingual translation bypasses

---

## 3. Gating & Fallback Instructions

### Mindgard Dataset (`Mindgard/evaded-prompt-injection-and-jailbreak-samples`)
1. Create a Hugging Face account and request access on the repository landing page: `https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples`.
2. Authenticate locally via `huggingface-cli login` or export `HF_TOKEN=<token>`.
3. **Graceful Degradation:** The PARALLAX loader `load_mindgard_samples(allow_missing=True)` catches missing tokens or access denial without raising unhandled exceptions. In environments without access, the pipeline logs a warning and proceeds with `Simsonsun/JailbreakPrompts` as the primary OOD benchmark.

---

## 4. Data Hygiene, Deduping & Contamination Safeguards

* **Split Integrity:** The fusion stacker meta-classifier is trained strictly on the holdout validation split from `neuralchemy/prompt-injection-technique`. No detector is trained on this split.
* **Exact Overlap Verification:** `parallax.data.splits.check_exact_overlaps` searches for exact string collisions between train and evaluation partitions.
* **Near-Duplicate Detection:** `parallax.data.splits.MinHashIndexer` and `check_near_duplicates` compute n-gram Jaccard similarities to flag potential paraphrase leakage.
* **Automated Audit Script:** Run `uv run python -m scripts.data.generate_overlap_report` to generate an audit report at `docs/eval/overlap_report.md`.
