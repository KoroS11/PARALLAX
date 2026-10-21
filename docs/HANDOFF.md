# PARALLAX Wave G: Teammate & Examiner Handoff Guide

**Document:** Engineering & Reproduction Handoff Manual  
**Repository:** PARALLAX (Parallel Adversarial-Prompt Recognition and Layered Aggregation for LLM eXploits)  
**Release:** Wave G Final Release (v1.2.0)  
**Host Hardware Target:** Windows 11 / Linux / macOS (Tested on AMD Ryzen CPU, 15.7 GB RAM, 4 GB VRAM RTX 3050 GPU)  
**Quality Gate Status:** ALL 6 GATES PASSING (677 unit tests, 89.96% coverage, frozen contracts diff=0)  

---

## 1. Executive Summary & Repository Organization

PARALLAX protects cyber-physical systems (CPS), SCADA operations, and LLM agent tool dispatching from prompt injection and adversarial jailbreaks. It pairs an asynchronous, multi-model parallel fast-path (sub-20ms) with a learned review router and an evidence-grounded adversarial challenger judge.

### Core Architecture Directories
* `parallax/contracts/`: **FROZEN CONTRACTS** (`Detector`, `DetectorResult`, `DatasetSample`, API schemas). Immutable interfaces.
* `parallax/detectors/`: Pure detector wrappers (D1 MiniLM embeddings, D2 DeBERTa, D4 Rule/Regex heuristics, PromptForest HTTP adapter, Ollama LLM Judge).
* `parallax/orchestrator/`: Asynchronous parallel fan-out, circuit breakers, and stacked logistic regression fusion.
* `parallax/router/`: `LearnedReviewRouter` predicting ensemble misclassification from multi-detector signals.
* `parallax/challenger/`: Evidence-grounded Challenger V2 (Role 1 Prosecutor, Role 2 Arbiter, calibrated Policy B downgrade).
* `parallax/preprocess/`: Multi-view proactive decoders (Unicode NFKC, Base64, Hexadecimal, Leetspeak, ROT13).
* `parallax/sandbox/`: 12-scenario CPS cyber-physical test harness with cryptographic canary tracking and mock tools.
* `parallax/dashboard/`: Self-contained, air-gapped terminal dashboard SPA.
* `data/splits/`: Canonical benchmark splits (`dev.jsonl` with 54 samples, `held_out.jsonl` with 146 samples).

---

## 2. Quickstart & Verification (Single Command)

### Step 1: Environment Synchronization
Clone the repository and install all locked dependencies using `uv`:
```bash
uv sync
```

### Step 2: Run CI Quality Gate
Verify that all 6 quality checks pass cleanly (unit test suite, 80%+ coverage, type checks, linting, formatting, security audit, frozen contracts check, and metric traceability):
```bash
uv run python scripts/gate.py
```

### Step 3: Run Environment Preflight
Inspect local services, ports, available RAM/VRAM, and Ollama models:
```bash
uv run python scripts/preflight.py
```

### Step 4: Automated Tool Setup (Optional for Framework Red-Teaming)
To set up all external virtual environments (`.venv_pyrit`, `.venv_llamafirewall`, `.venv_garak`) and pull Ollama models:
* **Windows (PowerShell)**:
  ```powershell
  .\scripts\setup_external.ps1
  ```
* **Linux / macOS (Bash)**:
  ```bash
  bash scripts/setup_external.sh
  ```

---

## 3. How to Reproduce Every Benchmark Row in `docs/REPORT.md`

All quantitative claims published in `docs/REPORT.md` and `docs/STATUS.md` are 100% reproducible and backed by raw JSON results artifacts.

### 3.1 Consolidated Final Multi-Arm Benchmark (Table 1 & Primary Headline Table)
Generates `eval/results/held_out_final_results.json` and `eval/final/FINAL_EVAL_REPORT.md` across all arms with 1,000 cluster-aware bootstrap confidence intervals:
```bash
uv run python scripts/eval/eval_final.py
```

### 3.2 Preliminary Synthetic Experiments (Table 4.1, Experiments E1–E7)
Benchmarks in-distribution classification, OOD evasion, rules over-defense, fusion strategies, latency speedup, and CPS sandbox ASR:
```bash
uv run python scripts/run_experiments.py
```
*Note: Evaluated across $n \in [4, 8]$ samples; designated as "illustrative only, under-powered".*

### 3.3 Comparative Architecture Ablation (Table 8.2 & Table 8.3)
Evaluates Arm A (Ensemble Only), Arm B (Discrepancy Fusion), Arm C (Qwen3:8b), Arm C2 (Gemma3:4b), and Arm U (Upper Bound) on held-out canary attacks ($N=90$) and benign tasks ($N=64$):
```bash
uv run python scripts/run_ablation.py
```

### 3.4 PromptForest Baseline & Third Detector Service (Table 9.2)
1. Launch PromptForest 2-detector daemon on port 8765:
   ```bash
   promptforest serve --port 8765 --config configs/pf_two_detectors.yaml
   ```
2. Launch Deepset DeBERTa third detector daemon on port 8766:
   ```bash
   uv run python scripts/detector_services/third_detector_service.py
   ```
3. Execute live baseline evaluation:
   ```bash
   uv run python eval/baselines/runner.py
   ```
*Output: `eval/baselines/baseline_results.json` certifying 16.7% recall for Stock/Two-Detector and 35.6% recall for Three-Detector.*

### 3.5 Learned Review Router Evaluation (Table 9.3)
Trains and benchmarks `LearnedReviewRouter` on 3-detector feature representations ($d_1, d_2, d_3$, discrepancy, confidence, enc_present, len):
```bash
uv run python eval/router/recheck_router.py
```
*Output: `eval/router/router_recheck_results.json` showing 92.2% recall at 33.56 extra calls per 100 requests.*

### 3.6 Challenger V2 Authority Calibration (Table 9.4)
Calibrates uncertainty gating and Policy B downgrade (`BLOCK -> FLAG`):
```bash
uv run python eval/challenger/calibrate_challenger.py
```
*Output: `eval/challenger/challenger_calibration_results.json` proving 0.0% False-Block Rate on 80 benign industrial tasks.*

### 3.7 PyRIT Adaptive Red-Team Attacker (Table 9.5)
Runs multi-turn adaptive mutations across budgets $B \in [1, 5, 10, 20]$ against Guard OFF, Guard ON, and Guard+Challenger:
```bash
uv run python scripts/eval/run_adaptive_attack.py
```
*Output: `eval/adaptive/adaptive_results.json` and `eval/adaptive/ADAPTIVE_ATTACK_REPORT.md`.*

### 3.8 Ollama LLM Judge Latency Benchmarking (Table 9.6)
Measures live warm p50 and p95 latencies and certifies anti-judge resilience:
```bash
uv run python scripts/judge_bench/run_judge_benchmark.py
```
*Output: `scripts/judge_bench/judge_benchmark_results.json` documenting Qwen3:8b warm p50 of 6.23s (-86.6% reduction).*

### 3.9 Statistical Sample Size & $n_{\text{eff}}$ Power Audit
Calculates Kish DEFF, ANOVA ICC, and 1,000 cluster-aware bootstrap CIs:
```bash
uv run python eval/stats_audit/run_audit.py
```
*Output: `docs/AUDIT_SAMPLE_SIZES.md` and `eval/stats_audit/sample_size_audit.json`.*

---

## 4. Hardware Constraints & Concurrency Rules

1. **Host Specifications**: AMD Ryzen CPU, 15.68 GB physical RAM, 4.0 GB VRAM NVIDIA RTX 3050 Laptop GPU.
2. **GPU Lock Context Manager (`scripts/gpu_lock.py`)**:
   - Because loading multiple transformer models concurrently triggers CUDA OOM errors, `scripts/gpu_lock.py` enforces a strict file-lock mutex.
   - Any live script invoking Ollama or PyTorch on the GPU must acquire this lock:
     ```python
     from scripts.gpu_lock import gpu_lock

     with gpu_lock():
         run_live_inference()
     ```
3. **Hermetic Unit Tests**:
   - All 677 unit tests in `tests/` use mocks and fakes. Never load live model weights or make external network calls during `pytest`.
4. **Hlyn-8b Judge Limitation**:
   - `defender-8b-Q8_0.gguf` is an 8.5 GB model. Running it on a 15.7 GB RAM host with 4 GB VRAM causes Windows pagefile disk swapping (38.4s p50). It is responsibly capped at DEV split evaluation under Cut 1.

---

## 5. Third-Party Licenses & Pending Gated Models

All external software is tracked in `THIRD_PARTY.md`. Zero proprietary or non-permissive code exists inside `parallax/`.

### Models Requiring Gated Access
* **`meta-llama/Llama-Prompt-Guard-2-86M`**:
  - Gated repository requiring Meta Community license approval on Hugging Face.
  - Needed for: LlamaFirewall `PromptGuardScanner` and standalone `PromptGuard2Detector`.
  - Status: **APPROVED**. Evaluated live (`pg2-live-988c7dce`: Recall 17.8%, FPR 0.0%, p50 62.3ms).
* **`vijil/prompt-injection-v5-20260827`**:
  - Gated repository requiring author approval. Gating was never granted; model was permanently dropped and replaced with `deepset/deberta-v3-base-injection`.
* **`LlamaFirewall 1.0.3`**:
  - Available on PyPI (`llamafirewall==1.0.3`) in `.venv_llamafirewall`.
  - Status: Evaluated live (`llamafw-pg2-75f0fb74`: Recall 20.0%, FPR 0.0%, p50 286.1ms) using the approved Meta Prompt Guard 2 weights via `eval/baselines/llamafirewall/runner.py --include-promptguard`.

---

## 6. Examiner & Peer Reviewer FAQ

For detailed technical defenses addressing the toughest criticisms without spin, consult [`docs/VIVA_QA.md`](./VIVA_QA.md):
* **Q1**: Why was Vijil deleted and why was PromptForest stock recall only 16.7%?
* **Q2**: Why was Qwen3:8b warm latency 46.4s in Wave F, and how was it reduced to 6.2s in Wave G?
* **Q3**: Why did Challenger V2 have a 45–50% False-Block Rate in Wave F, and how was it reduced to 0.0%?
* **Q4**: Why is LlamaFirewall marked as access pending?
* **Q5**: Why do sample sizes carry the disclaimer "illustrative only, under-powered"?
* **Q6**: Why use harmless cryptographic canaries instead of real weaponized exploits?
* **Q7**: Why use a Learned Review Router instead of a heuristic rule trigger?

---

## 7. CI Quality Gate Specification (`scripts/gate.py`)

`scripts/gate.py` runs all automated checks required for local merge and push certification:
1. **Ruff Linter**: `uv run ruff check .`
2. **Ruff Formatter**: `uv run ruff format --check .`
3. **Mypy Type Checking**: `uv run mypy parallax`
4. **Pytest & Coverage**: `uv run pytest -m "not integration and not gpu" --cov=parallax --cov-fail-under=80` (enforces $\ge 80\%$ test coverage; current: **89.96%**).
5. **Bandit Security Audit**: `uv run bandit -r parallax -ll` (zero high/medium security flaws).
6. **Frozen Contracts Integrity**: `git diff 35a04ab47bb9e4e74472a768ffd1e46c73b69bf6 -- parallax/contracts/` (must be 0 diffs).
7. **File Size & Secret Scanner**: Verifies all tracked files $\le 1$ MB and scans for API key leaks.
8. **Metric Traceability**: Verifies every quantitative number quoted in `docs/REPORT.md` matches raw JSON artifacts.

To run the complete gate:
```bash
uv run python scripts/gate.py
```
