# Rebuilding Excluded External Evaluation Datasets

To respect upstream redistribution policies and maintain zero-leakage repository hygiene, raw prompt text for external benchmark datasets is excluded from the release archive.

This document describes how to re-download the upstream sources, apply the identical frozen filtering and deduplication rules, and check the reconstructed files against `data/external/FROZEN_EXTERNAL.sha256`.

---

## 1. Upstream Datasets & Pinned Revisions

1. **Lakera / Gandalf (`gandalf.jsonl`, n=112 attacks)**:
   - Source: `Lakera/gandalf_ignore_instructions` (split: `test`)
   - Pinned HuggingFace Revision: `04737b65e90a6794ec227012e4a255a7def6344b`
   - License: MIT License (Lakera AI)

2. **NotInject (`notinject.jsonl`, n=309 benign)**:
   - Source: `leolee99/NotInject` (subsets: `NotInject_one`, `NotInject_two`, `NotInject_three`)
   - Pinned HuggingFace Revision: `847ae76cf8fea5ed325429e569ae8cfef022d2e0`
   - License: Apache-2.0 (`leolee99`)
   - Deduplication: 30 exact matches (8.85% development overlap with `data/splits/dev.jsonl`) are excluded, producing $n=309$.

3. **Databricks Dolly-15k (`dolly_sample.jsonl`, n=200 benign)**:
   - Source: `databricks/databricks-dolly-15k` (split: `train`)
   - Pinned HuggingFace Revision: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`
   - License: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)
   - Sampling: Stratified sample of $n=200$ across 8 task categories ($\le 2,000$ characters, seed 0). Committed as row indices, categories, and prompt SHA-256 hashes (`data/external/dolly_sample.jsonl`).

---

## 2. Automated Reconstruction Command

Run the committed freeze script using `uv`:

```bash
uv run python scripts/eval/freeze_external_data.py
```

This script performs the pinned downloads, applies the 30-row deduplication filter, generates the stratified Dolly sample, and writes:
- `data/external/gandalf.jsonl`
- `data/external/notinject.jsonl`
- `data/external/dolly_sample.jsonl`

---

## 3. Cryptographic Hash Checking

Check the generated files against `data/external/FROZEN_EXTERNAL.sha256`:

```bash
# On Linux / macOS:
sha256sum -c data/external/FROZEN_EXTERNAL.sha256

# On Windows (PowerShell):
Get-FileHash data/external/gandalf.jsonl, data/external/notinject.jsonl, data/external/dolly_sample.jsonl -Algorithm SHA256
```

Expected SHA-256 Hashes:
- `gandalf.jsonl`: `ebaf8567f6bf31295e1afff37c9ba1a2a119f6de8d93dfda5ae1e10521866229`
- `notinject.jsonl`: `dda92615712eac164dc18ef54e7a3aa5bf2eca29bbd2aae6634ed4e51b5ac90e`
- `dolly_sample.jsonl`: `b89f71eaff49b66362267e54be79d427dee030648bbd6e85cffc8af399f0a59c`
