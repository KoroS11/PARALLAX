# Evaluation Provenance: final-v4

## 1. Run Identification
- **Run ID**: `final-v4-20261019-142005-8fb16e`
- **Run Start Time (UTC)**: `2026-10-19 14:20:05 UTC` (19:50:05 IST)
- **Log File**: `eval/logs/final-v4-20261019-142005-8fb16e/predictions.jsonl`
- **Copied Results File**: `results/final-v4/predictions.jsonl`
- **Predictions SHA-256**: `74e0975eccda410597a865e3b0b5581e7732eca04406c4ca032cfccc29e1a444`
- **Row Count**: 347 (139 attacks, 208 benign)

## 2. Timestamps and Commit Sequence
- **Protocol-Lock Commit**: `7e98c396aa0e6a5d12d451a362c8909089c8726f` (`7e98c39`)
- **Protocol-Lock Commit Time**: `2026-10-19 19:02:52 +0530` (19:02 IST)
- **Run Initiation Time**: `2026-10-19 19:50:05 IST` (14:20:05 UTC)
- **Log File mtime**: `2026-10-19 20:37:38 IST` (20:37 IST)
- **Producing Commit**: `959a99a69c1a17266f52f689c59f3242747a057a` (`959a99a`) committed at `2026-10-19 21:30:07 +0530` (21:30 IST)
- **Temporal Invariant**: The protocol-lock commit (19:02 IST) strictly preceded run start (19:50 IST), and log completion (20:37 IST) preceded the producing commit (21:30 IST).

## 3. Split Integrity and Hash Verification
- **Target Dataset Split**: `data/splits/held_out_v4.jsonl`
- **`data/splits/FROZEN.sha256` recorded hash**: `9325e6185ac6a8c038b8129de91fe3dd8852f30ebecf4a667b24f4e973217dd5`
- **`data/splits/held_out_v4.jsonl` computed hash**: `9325e6185ac6a8c038b8129de91fe3dd8852f30ebecf4a667b24f4e973217dd5`
- **Verification Result**: Hashes match byte-for-byte; FROZEN integrity intact.

## 4. Run Manifest Audit
A manifest file exists at `eval/logs/manifests/final-v4-20261019-142005-8fb16e.json`.

- **Manifest File SHA-256**: `d5e7271d7b8df7aeea3ad965c4d9004d289b33d11d6dce873d46cac821a273bf`
- **Recorded Fields**:
  - `run_id`: `"final-v4-20261019-142005-8fb16e"`
  - `created_at`: `"2026-10-19T15:07:38.124050+00:00"` (ISO 8601 UTC)
  - `eval_type`: `"held_out_final_one_shot"`
  - `split_name`: `"held_out_v4"`
  - `split_sha256`: `"9325e6185ac6a8c038b8129de91fe3dd8852f30ebecf4a667b24f4e973217dd5"`
  - `sample_count`: `347` (`attack_count`: 139, `benign_count`: 208)
  - `model_digest`: `"gemma3:4b"`
  - `primary_arm`: `"Arm R (Policy B, tau=0.35)"`
  - `files`: paths to predictions log and results json.

- **What the Manifest Does NOT Contain**:
  - Git commit hash at execution time (e.g. `HEAD` commit ID).
  - Git working-tree status or dirty flag.
  - Complete Ollama manifest content digest or exact model weights SHA-256 (only records model string `"gemma3:4b"`).
  - Complete configuration dump, detector threshold tables, calibration parameters (only mentions `tau=0.35` inside the arm label string).
  - Environment details: Python version, PyTorch/CUDA runtime versions, `uv.lock` hash, OS information.
  - Random seed values used for prompt shuffling or inference execution.
  - GPU lock or resource concurrency telemetry.

## 5. Run-Time Code State & Commit Hunk Classification

### 5.1 Temporal Sequence vs. Commit History
- **19:02:52 IST**: Protocol-lock commit `7e98c39` created.
- **19:50:05 IST**: Final evaluation run initiated (`final-v4-20261019-142005-8fb16e`).
- **20:37:16 IST**: Predictions log file modified/closed.
- **20:37:38 IST**: Manifest file created (`created_at: 2026-10-19T15:07:38Z`).
- **21:30:07 IST**: Producing commit `959a99a` committed.

### 5.2 Git Diff between `7e98c39` and `959a99a`
Running `git diff 7e98c39 959a99a --stat -- parallax configs scripts/eval`:
- `parallax/`: **0 files changed, 0 insertions, 0 deletions** (strictly frozen core engine).
- `configs/`: **0 files changed, 0 insertions, 0 deletions** (strictly frozen configurations).
- `scripts/eval/eval_final.py`: **780 insertions, 28 deletions** (behavior-bearing runner script).
  - *Hunks 1–4*: Imports, logging setup, and CLI argument parsing.
  - *Hunks 5–8*: Implementation of `run_live_one_shot_evaluation()` (the execution loop that queries Ollama, executes detector inference, and runs Policy B arbitration). In `7e98c39`, this function was a stub containing `pass`.
  - *Hunks 9–12*: Offline metric re-derivation from log and markdown report writer.
- `scripts/eval/run_adaptive_attack.py`: **472 insertions, 260 deletions** (behavior-bearing adaptive attack runner).
  - *Hunks 1–6*: Multi-probe PyRIT attack loop replacing earlier single-probe placeholder.

### 5.3 Provenance Evidence & Run-Time Identity Verdict
1. **Unchanged Core Contracts**: The underlying detector wrappers, fusion contracts, orchestrator logic, and policy rules in `parallax/` were completely identical between `7e98c39` and `959a99a`.
2. **Behavior-Affecting Code in Runner**: The runner script `scripts/eval/eval_final.py` (which implements arm composition and Policy B execution) was a stub containing `pass` at the 19:02:52 IST protocol lock (`7e98c39`). It was completed in the working tree between 19:02 IST and 19:50 IST, but its exact working-tree state at the 19:50 IST runtime execution remains unrecorded in the manifest.
3. **Model Digest Limitation**: In the execution manifest (`final-v4-20261019-142005-8fb16e.json`), the field `model_digest` is recorded only as a string model tag (`"gemma3:4b"`), rather than a cryptographic weights digest, model layer manifest, or container image hash.
4. **Ollama Runtime Environment**: Local model state at the time of evaluation audit (`ollama list`):
   - Model: `gemma3:4b`
   - Ollama Image ID: `a2af6cc3eb7f`
   - Model Size: `3.3 GB`
   - Ollama Modified Timestamp: `3 months ago`
5. **Absence of Working-Tree Telemetry**: Because the execution manifest does not record a git commit hash or git dirty flag, it **cannot be cryptographically proven** that the code running in memory at 19:50 IST was byte-for-byte identical to commit `959a99a`.
6. **Defensible Claim**: We do NOT claim *"the evaluated system was cryptographically proven as commit 959a99a at runtime"*. We state: *"The evaluated benchmark corresponds to the implementation committed at 959a99a, with core libraries frozen since 7e98c39, while exact runtime working-tree provenance remains unrecorded in the manifest."*
