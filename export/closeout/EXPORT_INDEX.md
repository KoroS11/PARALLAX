# PARALLAX Closeout Export Index & Presentation Guide

> **STAGE**: PARALLAX Scientific Closeout Protocol  
> **SCOPE**: Comprehensive mapping of all generated figures and tables to report sections and presentation slides, accompanied by a 3-minute technical viva demo script.  
> **PROVENANCE**: Figures were generated at commit 2f06050 (footers show 2f0605073d); prior closeout review commit was 851da11.

---

## 1. Figure & Table Mapping to Report & Slides

| Item ID | Title / Content | Split / Scope | Artifact Files (PNG / SVG / CSV) | Target Report Section | Suggested Slide Title | Key Takeaway / Note |
|:---|:---|:---:|:---|:---|:---|:---|
| **F1** | Pipeline Architecture & Operating Thresholds | Design / System | `f1_pipeline_architecture.png`, `tables/f1_pipeline_thresholds.csv` | Section 3: Architecture | *Slide 3: Tiered Heterogeneous Architecture* | Shows detector fan-out, fusion threshold, learned router, and Policy B downgrade path. Non-primary arms marked post-hoc. |
| **F2** | Evaluation Data Composition & Provenance | dev, val, held_out_v4, external | `f2_data_composition.png`, `tables/f2_dataset_splits.csv` | Section 4: Data Governance | *Slide 4: Dataset Anatomy & Leakage Controls* | Documents sample sizes ($N=347$ held-out, $N=621$ external) and source provenance. |
| **F3** | Individual Detector AUC Trajectory | dev, val, held_out_v4 | `f3_detector_auc.png`, `tables/f3_detector_auc.csv` | Section 5.1: Detector Baselines | *Slide 5: Base Detector Discrimination* | D2, D3, D4, PG2 AUC with bootstrap CIs; shows val-to-held-out change for D3 without commentary. |
| **F4** | Multi-Arm Performance Comparison | held_out_v4 ($N=347$) | `f4_held_out_arms.png`, `tables/f4_held_out_arms.csv` | Section 5.2: Comparative Systems | *Slide 6: System Arm Comparison (Held-Out)* | Loose recall (Arm R: 84.17%), strict recall, pass-through, and block rate with 95% Wilson intervals. Arm R marked primary. |
| **F5** | Empirical Pareto Frontier (Recall vs Friction) | held_out_v4 ($N=347$) | `f5_pareto_view.png`, `tables/f5_pareto_frontier.csv` | Section 5.3: Trade-off Frontier | *Slide 7: Pareto Trade-offs & Operating Space* | Marker size = judge calls per 100 requests. Caption: "Dominance is not ranking; all evaluated arms lie on the empirical frontier." |
| **F6** | Decision Composition Breakdown | held_out_v4 ($N=347$) | `f6_stacked_composition.png`, `tables/f6_arm_composition.csv` | Section 5.4: Action Profiles | *Slide 8: ALLOW / FLAG / BLOCK Profiles* | Stacked decision distribution on attack vs benign prompts side by side for each arm. |
| **F7** | Learned Router Operating Analysis | held_out_v4 ($N=347$) | `f7_router_analysis.png`, `tables/f7_router_analysis.csv` | Section 6: Routing Analysis | *Slide 9: Router Evaluation: No Demonstrated Skill* | ROC-AUC of error prediction (0.5296) and 2,000-draw random routing distribution ($k=251$). |
| **F8** | LLM Judge Outcomes & Latency Distribution | held_out_v4 ($N=347$) | `f8_judge_outcomes.png`, `tables/f8_judge_outcomes.csv` | Section 7: Judge Behavior | *Slide 10: Judge Telemetry & Abstention Overhead* | Breakdown of judge outcomes; reveals 39.5% latency overhead on abstaining queries. |
| **F9** | Detailed Case Table of 10 Judge Abstains | held_out_v4 ($N=347$) | `f9_abstain_cases.png`, `tables/f9_abstain_cases.csv` | Section 7.2: Edge Cases | *Slide 11: Fail-Safe Handling of Edge Cases* | Per-arm decisions for 10 attack prompts with judge abstention; demonstrates Policy B containment. |
| **F10** | Adjudication Budget vs Detection Recall | held_out_v4 ($N=347$) | `f10_cost_vs_recall.png`, `tables/f10_cost_recall.csv` | Section 5.5: Computational Cost | *Slide 12: Cost-Recall Trade-offs* | Plot of extra judge calls per 100 vs loose recall across all arms. |
| **F11** | Category-Level Error Breakdown | held_out_v4 ($N=347$) | `f11_error_breakdown.png`, `tables/f11_error_breakdown.csv` | Section 8: Error Taxonomy | *Slide 13: Error Distribution by Source Category* | Aggregated counts of false negatives and false blocks per category (no raw attack text). |
| **F12** | Complete Multi-Class Confusion Matrices | held_out_v4 ($N=347$) | `f12_confusion_matrices.png`, `tables/f12_confusion_matrices.csv` | Section 8.2: Confusion Profiling | *Slide 14: Action Confusion: Arm R vs Arm A* | 2x3 confusion matrices (Attack/Benign vs ALLOW/FLAG/BLOCK) for Arm R and Arm A. |
| **F13** | Contamination Risk Heatmap | System / Datasets | `f13_contamination_matrix.png`, `tables/f13_contamination_matrix.csv` | Section 4.2: Data Contamination | *Slide 15: Cross-Corpus Contamination Analysis* | 3-state heatmap ("documented overlap", "not documented in the card", "no model card"). |
| **F14** | Protocol Lock & Execution Provenance Timeline | Engineering Timeline | `f14_provenance_timeline.png`, `tables/f14_provenance_timeline.csv` | Section 9: Execution Integrity | *Slide 16: Cryptographic Provenance Timeline* | Visual timeline linking commit hashes, protocol freeze (19:02 IST), and run completion. |
| **F15** | Methodological Defect & Audit Ledger | Closeout Audit | `f15_audit_trail.png`, `tables/f15_audit_trail.csv` | Section 10: Methodological Rigor | *Slide 17: Transparent Methodological Remediation* | Records detection and fixes for typed literals, duplicate CIs, and data overlap. |
| **X1** | Gandalf External Attack Family Benchmark | Gandalf ($n=112$) | `x1_gandalf_metrics.png`, `tables/x1_gandalf_metrics.csv` | Section 11.1: Gandalf Results | *Slide 18: External Benchmark: Gandalf Family* | Loose and strict recall for Arm R and individual detectors on instructional overrides. |
| **X2** | External Benign Dataset Decision Composition | NotInject ($n=309$), Dolly ($n=200$) | `x2_benign_composition.png`, `tables/x2_benign_composition.csv` | Section 11.2: Benign Benchmarks | *Slide 19: External Friction: NotInject & Dolly* | ALLOW / FLAG / BLOCK composition across arms, reported separately and never pooled. |
| **X3** | NotInject Subgroup & Trigger Analysis | NotInject ($n=309$) | `x3_notinject_subsets.png`, `tables/x3_notinject_subsets.csv` | Section 11.3: Trigger Sensitivity | *Slide 20: NotInject Multilingual & Trigger Splits* | False alarm rates across multilingual prompts and trigger-word density subsets. |
| **X4** | Held-Out vs External Performance Profiling | held_out_v4 vs External | `x4_held_out_vs_external.png`, `tables/x4_held_out_vs_external.csv` | Section 11.4: Comparative Generalization | *Slide 21: In-Distribution vs External Arms* | Distinct panels comparing Arm R metrics across datasets without connecting lines or pooling. |
| **X5** | External Adjudication Outcomes & Latency | External ($N=621$) | `x5_external_judge_outcomes.png`, `tables/x5_external_judge_outcomes.csv` | Section 11.5: External Telemetry | *Slide 22: External Adjudication Latency & Outcomes* | Outcome proportions (ok, abstain, parse_fail) and latency on external candidate sets. |

---

## 2. Three-Minute Technical Viva Demonstration Script

### Timing Breakdown
- **0:00 - 0:45 (45s)**: Architecture, Tiering, and the Pre-Registered Primary System (Arm R).
- **0:45 - 1:45 (60s)**: Empirical Results on Held-Out Data, Router Skill, and Judge Abstention.
- **1:45 - 2:30 (45s)**: External Benchmarking, Contamination Control, and Provenance.
- **2:30 - 3:00 (30s)**: Live Dashboard Replay Demonstration.

---

### Minute 1: System Concept & Tiered Architecture (0:00 – 0:45)
> "Good morning. I'll present PARALLAX, an asynchronous multi-detector defense architecture designed for high-risk language model interactions.
> 
> As shown in **Figure F1**, PARALLAX uses a tiered, multi-model fan-out. Fast base detectors (D2, D3, D4) evaluate the prompt in parallel. Their calibrated probabilities feed a logistic fusion meta-classifier.
> 
> To balance security against inference cost, we pre-registered **Arm R** as our primary system. Arm R introduces a learned review router that estimates ensemble error probability. If error likelihood exceeds threshold tau = 0.35, the prompt escalates to an LLM judge (`gemma3:4b`). Under **Authority Policy B**, preliminary fusion blocks can be downgraded to FLAG for human review, but only when the judge explicitly certifies the input as ALLOW. All non-primary arms are treated strictly as post-hoc comparisons."

---

### Minute 2: Held-Out Performance & Negative Results (0:45 – 1:45)
> "Turning to our held-out evaluation on 347 samples in **Figure F4**, Arm R achieves **84.17% loose recall** (95% CI [77.20%, 89.31%]) while preserving a **68.75% pass-through rate** on benign operations, with a **6.25% hard false-block rate**.
> 
> However, our scientific closeout prioritizes transparent reporting of negative findings:
> 1. **No Demonstrated Router Skill**: As shown in **Figure F7**, the router's error prediction ROC-AUC is **0.5296** (95% CI [0.4579, 0.6006]), which encompasses chance baseline 0.50. Furthermore, in an empirical simulation of 2,000 random routing draws at identical call budgets (k=251), random routing achieved **87.48%** mean recall. The learned router does not outperform random escalation in attack recall.
> 2. **Judge Abstention Overhead**: In **Figure F8 and F9**, we observe that the LLM judge abstained on exactly 10 prompts—all of them adversarial attacks—incurring a 39.5% latency increase. Under Policy B, our fail-safe rule ensures that an abstention never downgrades a block, retaining maximum security."

---

### Minute 3: External Benchmark, Provenance & Live Demo (1:45 – 3:00)
> "To test performance beyond synthetic splits, we evaluated on 621 external samples across three distinct families—Gandalf, NotInject, and Dolly—shown in **Figures X1 and X2**. Rather than pooling them into an unscientific single metric, we report them strictly per-family. 
> 
> Prior to scoring, we discovered an 8.85% overlap between NotInject and our dev split; we excluded those 30 rows under our pre-registered protocol. We also disqualified the deepset benchmark because the model card confirmed pretraining on that exact dataset, as charted in **Figure F13**.
> 
> Finally, looking at `dashboard.html`, we have an entirely self-contained, offline inspection interface. On the **Replay tab**, we can step through held-out rows in real time. We see individual detector outputs, router escalation triggers, judge verdicts, and final arm actions. A text toggle allows hiding prompt content for sanitized presentation.
> 
> All metrics across our documentation, dashboard, and figures are dynamically read from cryptographic source logs anchored to protocol-lock commit `7e98c39`. Figures were generated at commit 23d53ce (footers show 23d53cec7b); prior closeout review commit was 851da11. Thank you."
