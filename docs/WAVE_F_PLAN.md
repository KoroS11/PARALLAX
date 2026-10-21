# PARALLAX Wave F Plan: Live Evaluation, Real Attack Tools & Learned Review Routing

**Target:** Wave F Implementation, Empirical Validation, and Scientific Audit  
**Status:** Preflight Passed — Wave F1 Active  
**Preflight Certification:** `scripts/preflight.py` PASSED (PromptForest 8765 details confirmed, Ollama models active, HF token confirmed)  

---

## 1. Research Question & Hypotheses

### Primary Question
> *Does adversarial review (a Red-Team Challenger plus an LLM Judge) and learned review routing improve an ensemble's security without unacceptable latency or false positives under real, live multi-model detector inferences and adaptive attacks?*

### Formal Hypotheses
- **H1 (Ensemble Dilution & Learned Routing)**: Naive score averaging across real detectors dilutes specialist detections (resulting in $>80\%$ ASR). A Learned Review Router trained on dev-split discrepancy, confidence, and encoding features can target review to only the edge cases ($<45\%$ traffic), outperforming rule-based discrepancy triggers in recall while keeping false-block rate at 0.0%.
- **H2 (Live Judge Discrimination & Anti-Injection)**: A live local LLM Judge (`qwen3:8b`) evaluates under thinking-suppressed, strict JSON schemas and maintains robustness against prompt-injections targeting the judge itself.
- **H3 (Adaptive Attack Resilience)**: Under query-budgeted multi-turn attacks from an adaptive PyRIT adversary, PARALLAX with adversarial review maintains significantly lower ASR across turn budgets (1, 3, 5, 10) compared to unguarded or static ensemble baselines.

---

## 2. Agent Roster & File Ownership Table

| Agent | Owned Directories / Files | Responsibilities |
|:---|:---|:---|
| **preflight-agent** | `scripts/preflight.py`, `docs/WAVE_F_PLAN.md` | Preflight verification of live services, models, tokens, and hardware headroom. |
| **pf-adapter** | `parallax/detectors/promptforest.py`, `tests/test_promptforest.py` | Live PromptForest client reading only `details`, dropping missing detectors, calculating discrepancy/confidence, truncation/padding evasion tests. |
| **live-judge** | `parallax/challenger/judge.py`, `scripts/judge_bench/` | Live Ollama judge benchmarking (`qwen3:8b`, `gemma3:4b`), cold/warm latency measurement, anti-judge injection tests. |
| **attack-tools** | `scripts/attacks/`, `data/attacks/`, `docs/TOOLS_STATUS.md` | Execute PyRIT, Garak, Promptfoo in dedicated environments; store raw outputs in `data/attacks/raw/<tool>/`; balance corpus ($N_{\text{eff}}$). |
| **baselines** | `eval/baselines/` | External baselines: PromptForest (stock-degraded vs two-detector), standalone Prompt Guard 2. |
| **fusion-router** | `parallax/fusion/`, `parallax/router/`, `tests/test_router.py` | Fusion comparison; Learned Review Router trained on dev split; cost curve (recall vs extra calls). |
| **preprocessing** | `parallax/preprocess/`, `tests/test_preprocess.py` | Multi-view decoders (Base64, Hex, URL, Rot13, homoglyphs); expose `encoding_present`; report feature effect per technique. |
| **challenger-v2** | `parallax/challenger/` (except `judge.py`), `tests/test_challenger*.py` | Evidence grounding (span citation), $N=3$ self-consistency, authority policy ablation. |
| **adaptive-attack** | `parallax/sandbox/adaptive/`, `scripts/adaptive/` | Multi-turn PyRIT adaptive red-teamer vs PARALLAX sandbox across query budgets. |
| **eval-final** | `parallax/eval/`, `reports/` | Multi-arm ablation (Arms A, B, R, C, C2, C3, U) with cluster-aware bootstrap CIs on held-out test split. |
| **dashboard-docs** | `parallax/dashboard/`, `README.md`, `docs/REPORT.md` | Interactive cost curves, router comparison, adaptive attack curves, updated report with viva Q&A. |
| **auditor** | `docs/AUDIT_WAVE_F.md` (read-only elsewhere) | Independent final audit of all metrics, manifests, licenses, and contract immutability. |

---

## 3. Confirmed Preflight Facts & Environment Constants

- **PromptForest Service (`http://127.0.0.1:8765`)**:
  - Running via `C:\Users\harsh\.venv\Scripts\promptforest.exe serve --port 8765 --config C:\Users\harsh\pf_two_detectors.yaml`.
  - Endpoint `POST /analyze` returns `details` containing `{"llama_guard", "xgboost"}`.
  - `vijil` is disabled in `pf_two_detectors.yaml` (0.0 phantom vote prevented).
  - Preflight validates `details` keys match expected config.
- **Ollama Service (`http://127.0.0.1:11434`)**:
  - Service operational; local models present: `qwen3:8b`, `gemma3:4b`, `qwen3.5:9b`, `mistral:latest`.
  - Zero models loaded in VRAM initially (`ollama ps` clean).
- **HuggingFace Access**:
  - Token present at `~/.cache/huggingface/token`.
  - `meta-llama/Llama-Prompt-Guard-2-86M`: Access granted.
  - `vijil/prompt-injection-v5-20260827`: Status is `gated: manual` (manual approval required). Unusable for live weights until granted; labeled `"Vijil v5 substituted"`.
  - `hlyn-labs/prompt-injection-judge-8b`: Available on Hugging Face (Q8_0 GGUF is ~8.5 GB). Requires CPU offloading due to 4 GB VRAM limit.
- **Hardware Resources**:
  - Total RAM: 15.68 GB.
  - Total VRAM: 4.0 GB (3757 MB free).
  - Concurrency limit: Maximum 3 subagents concurrent.
  - GPU locking: Strictly 1 heavy model at a time wrapped in `scripts/gpu_lock.py`.
- **Contracts Invariant**:
  - `parallax/contracts/` strictly frozen (0 diffs).

---

## 4. Wave Execution Schedule

- **Wave F0**: Preflight checks (`scripts/preflight.py`), subagent manifests in `.agents/`, `docs/WAVE_F_PLAN.md`.
- **Wave F1** (Parallel, max 3):
  1. `pf-adapter`
  2. `live-judge`
  3. `attack-tools`
  *Merge F1 to main after CI gate passes.*
- **Wave F2** (Parallel, max 3):
  4. `baselines`
  5. `fusion-router`
  6. `preprocessing`
  *Merge F2 to main after CI gate passes.*
- **Wave F3** (Parallel, then final eval):
  7. `challenger-v2`
  8. `adaptive-attack`
  *Merge 7 & 8, then spawn:*
  9. `eval-final`
  *Merge F3 to main after CI gate passes.*
- **Wave F4**:
  10. `dashboard-docs`
  *Merge 10, then spawn:*
  11. `auditor` (`docs/AUDIT_WAVE_F.md`)
  *Merge F4 to main after CI gate passes.*
- **Final**: Run `scripts/gate.py`, verify 100% metrics traceability, format AGENTS.md 5-part report and claim table.
