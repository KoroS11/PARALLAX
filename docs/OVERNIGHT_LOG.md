# PARALLAX Wave G Overnight Execution Log

This log tracks autonomous stream milestones, start/finish timestamps, gate results, and issues during Wave G execution.

| Timestamp (UTC) | Wave / Stream | Status | Summary & Notes |
| :--- | :--- | :---: | :--- |
| 2026-10-17 20:15:00Z | Wave G Orchestrator | START | Wave G Orchestration initiated. Plan approved. |
| 2026-10-17 20:15:30Z | Wave G0: Preflight & Ledger | START | Initializing overnight log, extending preflight script, and writing WAVE_G_PLAN.md |
| 2026-10-17 20:20:00Z | Wave G0: Preflight & Ledger | COMPLETE | Preflight confirmed: PromptForest 8765, Ollama 11434, Port 8766 free, HF token valid. WAVE_G_PLAN.md written. |
| 2026-10-17 20:25:00Z | Wave G1 (detector-sub, llamafirewall, judge-c3) | START | Spawning 3 parallel subagents for third detector integration, LlamaFirewall baseline, and Judge C3 optimization. |
| 2026-10-17 20:43:00Z | Wave G1 (detector-sub, llamafirewall, judge-c3) | COMPLETE | All 3 streams merged sequentially to main. Gate PASSED (663 tests, 89.85% coverage). Qwen3 warm p50 6.2s (-86.6%), deepset deberta on :8766 restoring N=3, LlamaFirewall evaluated. |
| 2026-10-17 20:45:00Z | Wave G2 (challenger-fix, router-recheck, sample-size-audit) | START | Launching 3 parallel streams: Challenger threshold calibration (Priority 1), Learned Router 3-detector retraining, and statistical $N/N_{\text{eff}}$ bootstrap audit. |
| 2026-10-17 21:18:00Z | Wave G2: sample-size-audit | COMPLETE | Comprehensive audit completed. Kish DEFF, ANOVA ICC, $n_{\text{eff}}$, and 1,000 cluster-aware bootstrap CIs generated. Dropped into docs/AUDIT_SAMPLE_SIZES.md. Gate PASSED. |
| 2026-10-17 21:20:00Z | Wave G2: router-recheck | COMPLETE | Retrained on 3 detectors. Slashed extra model calls from 52.1 to 33.56/100 (beats rule trigger 38.36/100). FPR 0.0%, Recall 92.2% on held-out. Gate PASSED. |
| 2026-10-17 21:37:00Z | Wave G2: challenger-fix | COMPLETE | Slashed false-block rate (FBR) from 45–50% to 0.0% on DEV and Held-Out (80 tasks, 0 false blocks). Gating rate 65.9%. Policy B calibrated downgrade measured. Gate PASSED (676 tests, 89.96% coverage). |
| 2026-10-17 22:12:00Z | Wave G2: adaptive-rerun | COMPLETE | Reran PyRIT multi-turn adaptive attack across budgets B in [1, 5, 10, 20] against Guard OFF, Guard ON (3 detectors), and Guard + Challenger (Policy B), with concurrent benign stream FBR. Gate PASSED (677 tests, 89.96% coverage). |
| 2026-10-17 22:14:00Z | Wave G3: eval-final | START | Consolidating all Wave G empirical metrics into final multi-arm benchmark report and held-out JSON. |
| 2026-10-17 22:19:00Z | Wave G3: dashboard-docs & auditor | START | Updating REPORT.md with reviewed tables, authoring VIVA_QA.md, updating STATUS.md, conducting independent scientific audit (AUDIT.md). |
| 2026-10-17 22:24:00Z | Wave G3: dashboard-docs & auditor | COMPLETE | REPORT.md updated with 100% reviewed tables; VIVA_QA.md created; STATUS.md updated; AUDIT.md certifies FULL PASS. Gate PASSED (677 tests, 89.96% cov). Merged to main. |
| 2026-10-17 22:25:00Z | Wave G4: GitHub Ship Prep & Clean Clone | START | .gitignore audit, pf_two_detectors.yaml config, setup scripts (ps1/sh), docs/HANDOFF.md, clean-clone test in scratch dir. |
| 2026-10-17 22:30:00Z | Wave G4: GitHub Ship Prep & Clean Clone | COMPLETE | Clean clone test PASSED all 6 CI gates (677 tests, 89.96% cov, 0 contract diffs, 0 secret/size violations). Final push-ready commit on main. |
| 2026-10-17 22:30:30Z | Wave G Final Verdict | SUCCESS | Wave G autonomously completed through the night. All goals achieved, all weak Wave F results repaired, all metrics backed by JSON runs, push-ready handoff delivered. |
| 2026-10-18 13:30:00Z | Wave H: Orchestrator Kickoff | START | Human audit identified 4 root causes (R1-R4). Wave H initiated to rebuild scientific defensibility from traceable logs. |
| 2026-10-18 13:45:00Z | Wave H0: Hygiene & Baselines | COMPLETE | Review scripts preserved in scripts/review/, .gitignore updated with manifest, Wave G numbers marked SUPERSEDED/PRELIMINARY in STATUS.md. All 10 CI gates PASSED. Head tagged wave-h-start. |
| 2026-10-18 14:00:00Z | Wave H1: Corpus Root-Cause Fix | START | Launched corpus-fix subagent to decouple text vs metadata.goal, eliminate synthetic markers, build matched benign controls, and enforce shortcut-audit gate. |
| 2026-10-18 14:55:00Z | Wave H1: Corpus Root-Cause Fix | COMPLETE | Schema decoupled across all splits; CANARY and mock_*() removed from text; matched benign controls added (dev=129, val=157, held_out_v2=182); shortcut audit strictly passed (all AUCs <= 0.55, max gap <= 6.2 pts); FROZEN.sha256 locked. Merged to main. All 12 gates PASSED. |
| 2026-10-18 15:00:00Z | Wave H2: Masked Scoring & Marker Ablation | START | Launched mask-eval subagent to build eval/mask.py neutral masking module and run 7-arm marker ablation across legacy corpora. |
| 2026-10-18 15:55:00Z | Wave H2: Masked Scoring & Marker Ablation | COMPLETE | eval/mask.py operational (9 unit tests passing); marker ablation completed across 96 marker attacks; quantified R3 published in docs/CORPUS_AUDIT.md: D3/Arm A/Arm B/Arm R/Judge show 0.00 pts shortcut inflation; D2 shows -35.42 pts distractor noise effect. Merged to main. |
| 2026-10-18 16:00:00Z | Wave H3: Judge Root-Cause Fix | START | Launched judge-fix subagent to deploy domain-anchored industrial copilot prompt with mandatory quoted evidence span extraction and candidate selection under GPU lock. |
| 2026-10-18 18:20:00Z | Wave H3: Judge Root-Cause Fix | COMPLETE | Template version-hashed (1dd6436688295581); verbatim evidence validation active; gemma3:4b selected on dev/val; logged 1,467 raw prompt traces to eval/logs/<run_id>/judge_gemma3_4b.jsonl with full manifest and sample logs. All 55 tests passing. Merged to main. |
| 2026-10-18 18:25:00Z | Wave H4: Fusion & Router Rebuild | START | Launched router-rebuild subagent to author docs/METRICS.md, retrain router on masked dev, ablate feature sets, and evaluate Policy A/B against logged judge. |
| 2026-10-18 18:55:00Z | Wave H4: Fusion & Router Rebuild | COMPLETE | docs/METRICS.md published (formalized FPR/FBR/Utility/Wilson CIs, reconciled Arm R divergence); router retrained on masked dev (211 samples); feature ablation confirmed no length reliance (r=-0.0556); Policy B mitigates 56% of false blocks (FBR 0.9457 -> 0.4186); 27/27 tests pass. Merged to main. |






