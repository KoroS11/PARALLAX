# PARALLAX Teammate Handoff Briefing (Closeout Plan v2)

Welcome to the PARALLAX evaluation package. This document summarizes the repository state, reproducible numbers, and team workflow for writing the final evaluation report.

---

## 1. Evaluated System & Branch State

- **Evaluated System**: Git commit `959a99a` on branch `closeout`.
- **Benchmark Split**: Touch-free held-out split `held_out_v4.jsonl` ($N=347$, 139 attacks, 208 benign).
- **Predictions Log**: `results/final-v4/predictions.jsonl` (SHA-256 `74e0975eccda410597a865e3b0b5581e7732eca04406c4ca032cfccc29e1a444`).
- **All Experimental / Amendment Steps**: Parked cleanly on branch `wip-amendment1-steps` with patch `plans/parked/working_tree_vs_959a99a.patch`.

---

## 2. One-Command Metric Regeneration

Any team member can regenerate all **676** benchmark numbers byte-for-byte directly from the evaluated log with **zero models** and **zero GPU requirements**:
```bash
uv run python scripts/eval/closeout_metrics.py
```
This updates `numbers.json`. Every number in the report skeleton, safe claims guide, and benchmark dashboard originates from this file.

---

## 3. Primary Findings Summary

1. **Defense Arm Trade-offs**:
   - **Arm R (Learned Router + Policy B)**: 84.17% loose recall, 58.27% strict recall, 68.75% benign pass-through, 25.00% benign FLAG load, 6.25% benign BLOCK rate, 72.33 extra calls/100 requests.
   - **Arm A (3-Detector Ensemble)**: 73.38% recall, 68.75% pass-through, 31.25% BLOCK rate, 0 extra calls.
   - **Arm B (Discrepancy Trigger + Policy B)**: 89.93% loose recall, 68.75% pass-through, 0.48% BLOCK rate, 94.52 extra calls/100.
   - **External Baselines**: Prompt Guard 2 achieves 32.37% recall; PromptForest achieves 4.32% recall.

2. **Critical Negative Finding (Router Failure)**:
   - Arm R does **not** beat random routing at equal budget (Arm R: 84.17% vs Random: 87.48%).
   - The router's error prediction feature was inverted on held-out data (ROC-AUC 0.435), routing benign requests at higher rates (79.33%) than attacks (61.87%).

3. **Judge Abstention Signal**:
   - LLM review judge abstained exclusively on attacks (10/139 vs 0/208 benign, $p = 8.71 \times 10^{-5}$), functioning as an implicit attack indicator.

---

## 4. Key Artifacts for Writing

- **Report Skeleton**: `report/REPORT_SKELETON.md` (structured in ESE rubric order with placeholders).
- **Safe Claims Guide**: `docs/SAFE_CLAIMS.md` (what you can say, cannot say, and pilot limits).
- **Visual Figures**: `figures/` (PNG and SVG ready for inclusion in LaTeX / Word / Markdown).
- **Datasheet**: `docs/DATASHEET.md` (complete dataset documentation).
- **Interactive Dashboard**: `docs/BENCHMARK_DASHBOARD.html`.
- **Demo Replay**: `docs/DEMO_REPLAY.md` and `docs/demo_replay.html`.
