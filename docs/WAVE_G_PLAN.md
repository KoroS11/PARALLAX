# PARALLAX Wave G Plan: Weakness Remediation, Third Detector, and GitHub Ship Preparation

**Target:** Wave G Implementation, Empirical Calibration, and Open Source Preparation  
**Status:** Preflight Passed — Wave G0 Complete, Ready for Wave G1  
**Preflight Certification:** `scripts/preflight.py` PASSED (PromptForest 8765 confirmed, Ollama models confirmed, HF token confirmed, Port 8766 free)  

---

## 1. Research Objectives & Targeted Weakness Remediation

Wave G directly addresses the five weakest empirical findings of Wave F:

1. **Challenger V2 False-Block Rate (45–50%)**:
   - *Problem*: High false-alarm rate on industrial and conversational benign tasks renders the secondary review layer unacceptable for real deployment.
   - *Wave G Solution*: Ablate review authority on DEV (router-gated review, calibrated downgrade `BLOCK -> FLAG`, and confidence threshold tuning) targeting $\le 5\%$ and $\le 10\%$ FBR.
2. **Qwen3:8b Warm Latency (46.4s)**:
   - *Problem*: Latency spike caused by unconstrained generation context and thinking tokens.
   - *Wave G Solution*: Explicitly suppress thinking tags in the Ollama request options, clamp `num_ctx: 1024` and `num_predict: 128`, bringing warm p50 to $<5$s.
3. **PromptForest Low Baseline Recall (16.7%)**:
   - *Problem*: Vijil was permanently lost upstream, crippling PromptForest to a 2-detector ensemble.
   - *Wave G Solution*: Permanently drop Vijil; evaluate 4 ungated DeBERTa candidates on DEV; deploy top candidate on local port 8766 to restore full 3-detector ensemble ($N_{\text{available}}=3$).
4. **Metrics with Unstated Sample Sizes**:
   - *Problem*: Several headline statements in prior reports lacked explicit sample sizes.
   - *Wave G Solution*: Audit every numerical claim, recording $N$, effective sample size $N_{\text{eff}}$, and cluster-aware bootstrap 95% CIs in results JSON; flag any metric with $N < 100$ as "illustrative only".
5. **Review Router Cost Efficiency**:
   - *Problem*: Wave F router required 52.1 extra calls / 100 requests for 86.7% recall.
   - *Wave G Solution*: Re-tune the Learned Review Router on 3-detector feature representations.

---

## 2. Agent Roster & File Ownership Table

| Agent / Stream | Owned Directories & Files | Core Responsibilities |
| :--- | :--- | :--- |
| **preflight-agent** | `scripts/preflight.py`, `docs/WAVE_G_PLAN.md`, `docs/OVERNIGHT_LOG.md` | Host validation, preflight checks, ledger tracking. |
| **detector-sub** | `parallax/detectors/`, `scripts/detector_services/`, `THIRD_PARTY.md`, `tests/test_promptforest.py` | Evaluate 4 ungated detectors on DEV; deploy best candidate as HTTP service on `127.0.0.1:8766`; wire into adapter for $N=3$; add 512-token truncation/padding evasion tests; audit silent-zero paths. |
| **llamafirewall** | `eval/baselines/llamafirewall/`, `THIRD_PARTY.md` | Set up `.venv_llamafirewall` with `llamafirewall==1.0.3`; build baseline runner; mark Prompt Guard 2 as `gated, access pending` with single-command replay instructions. |
| **judge-c3** | `parallax/challenger/judge.py`, `scripts/judge_bench/`, `tests/test_judge.py` | Fix `qwen3:8b` latency ($<5$s target); evaluate Arm C3 (`hlyn` 8B Q8_0) and Arm C3b (`hlyn` 3B); test anti-judge injection resilience; enforce GPU locks. |
| **challenger-fix** | `parallax/challenger/` (except `judge.py`), `eval/challenger/`, `tests/test_challenger_v2.py` | **PRIORITY 1**: Calibrate review policies on DEV to achieve $\le 5\%$ and $\le 10\%$ FBR; plot recall vs FBR and extra calls; touch held-out set once at end. |
| **router-recheck** | `parallax/router/`, `parallax/fusion/`, `eval/router/`, `tests/test_learned_router.py` | Re-train Learned Review Router and fusion baselines on 3-detector signals; maintain random audit sample of allowed traffic. |
| **adaptive-rerun** | `parallax/sandbox/adaptive/`, `scripts/adaptive/`, `eval/adaptive/` | Re-run PyRIT adaptive attacker across budgets $B \in [1, 5, 10, 20]$ against Guard Off, Guard On, Guard+Challenger; report ASR and false-block cost. |
| **sample-size-audit**| `eval/stats/` | Audit every claim in `REPORT.md`; record $N$, $N_{\text{eff}}$, and cluster-aware bootstrap CIs; flag $N<100$ metrics as illustrative only. |
| **eval-final** | `parallax/eval/`, `reports/`, `eval/final/` | Consolidated held-out benchmark across all arms with 95% bootstrap CIs. |
| **dashboard-docs** | `parallax/dashboard/`, `README.md`, `docs/REPORT.md`, `docs/STATUS.md`, `docs/VIVA_QA.md` | Terminal dashboard API, report update, comprehensive Viva defense Q&A. |
| **auditor** | `docs/AUDIT.md` (read-only elsewhere) | Independent verification of 100% claim-to-evidence mappings, zero silent fallbacks, license checks, and frozen contracts. |
| **github-ship** | `.gitignore`, `configs/`, `scripts/setup_external.*`, `docs/HANDOFF.md` | Clean commit, secret scan, config centralization, multi-platform setup scripts, handoff doc, clean-clone verification. |

---

## 3. Overnight Safe Cut Order

If execution time or quota runs short, degrade tasks strictly in this order:
1. **Cut 1**: `hlyn 8B` heavy offload run (retain `hlyn 3B` and latency-fixed `qwen3:8b`).
2. **Cut 2**: `LlamaFirewall` extended scan rows (retain offline baseline and command stub).
3. **Cut 3**: `adaptive-rerun` turn depth (evaluate budgets $B \in [1, 5, 10]$ instead of 20).
4. **Cut 4**: Length of `docs/VIVA_QA.md`.

**NEVER CUT (MANDATORY)**:
- `detector-sub` (Third detector integration)
- `challenger-fix` (Challenger FBR repair)
- `sample-size-audit` ($N$, $N_{\text{eff}}$, and bootstrap CIs)
- `auditor` (`docs/AUDIT.md`)
- `Wave G4` (GitHub ship preparation and handoff)

---

## 4. Hardware & Operating Rules

1. **Host Specifications**: Windows 11, 15.7 GB RAM, NVIDIA RTX 3050 (4 GB VRAM).
2. **Concurrency**: Maximum 3 subagents running concurrently.
3. **GPU Serialization**: Exactly **ONE** heavy model process (Ollama or PyTorch transformer) at any time, wrapped inside `scripts/gpu_lock.py`.
4. **Zero Network / Real Models in Tests**: All pytest unit tests must use hermetic mocks/fakes.
5. **Frozen Contracts**: `parallax/contracts/` must strictly maintain 0 diffs against Phase 0 commit (`35a04ab47bb9e4e74472a768ffd1e46c73b69bf6`).
6. **No Third-Party Code in `parallax/`**: External tools and frameworks execute in dedicated external virtual environments (`.venv_garak`, `.venv_pyrit`, `.venv_llamafirewall`, `.venv_detector3`).
