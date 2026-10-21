# PARALLAX Technical Defense & Viva Examination Guide (Closeout Master)

> **SPECIFICATION**: 35 Technical Questions covering empirical findings, failure modes, calibration, provenance, external benchmarks, and architectural trade-offs.  
> **AUDIT PRINCIPLE**: Numbers before narrative. Every answer cites an existing repository file or figure, and exact metric numbers from `numbers.json`. Zero banned words.

---

### Q1: Why is Arm R designated as the primary system under evaluation?
- **Short Answer**: Section 3.1 of `docs/EXTERNAL_EVAL_PROTOCOL.md` and `docs/EVAL_PROTOCOL_v2.md` pre-registered Arm R (Learned Review Router with Policy B, $\tau=0.35$) as the primary system before held-out evaluation. Arms A, B, U, C2, PG2, and PromptForest are descriptive comparison arms evaluated post-hoc.
- **Repo Source File / Figure**: [`docs/EXTERNAL_EVAL_PROTOCOL.md:57`](file:///docs/EXTERNAL_EVAL_PROTOCOL.md), Figure F1 ([`f1_pipeline_architecture.png`](file:///export/closeout/figures/f1_pipeline_architecture.png)), Figure F4.
- **Key Metric to Remember**: `84.17% loose recall` [77.20%, 89.31%], `68.75% pass-through`, `6.25% block rate` (Key: `numbers.json -> arm_r.loose.recall`).
- **What NOT to Claim**: Do NOT claim Arm R was chosen post-hoc because it performed best; it was pre-registered before scoring.

---

### Q2: What does the random routing simulation show?
- **Short Answer**: In a 2,000-draw simulation routing $k=251$ calls uniformly at random across held_out_v4 (seed 0), random routing achieved a mean loose recall of 87.48% [84.17%, 90.65%], compared to 84.17% for Arm R. Arm R does not outperform uniform random escalation in attack recall. However, Arm R achieves a lower false blocking rate on benign prompts (6.25% vs 9.05%) because it routes 81.25% of benign prompts to the judge where Policy B downgrades false blocks.
- **Repo Source File / Figure**: [`scripts/eval/verify_random_routing.py`](file:///scripts/eval/verify_random_routing.py), Figure F7 ([`f7_router_analysis.png`](file:///export/closeout/figures/f7_router_analysis.png)).
- **Key Metric to Remember**: `87.48% random recall mean vs 84.17% Arm R` (Key: `numbers.json -> random_routing.recall_loose_mean`).
- **What NOT to Claim**: Do NOT claim Arm R demonstrates statistically significant superior routing over random allocation.

---

### Q3: Why does the learned router have no demonstrated skill?
- **Short Answer**: The supervised error predictor was trained to predict when the primary fusion ensemble would make an error. On held_out_v4, its ROC-AUC against fusion error is 0.5296 (95% CI [0.4579, 0.6006]), which includes 0.500 (chance). Against ground-truth labels, the AUC is 0.4350 [0.3703, 0.4991], reflecting an operational routing asymmetry (routing 81.25% of benign vs 58.99% of attacks) rather than predictive error discrimination.
- **Repo Source File / Figure**: [`parallax/router/learned_router.py`](file:///parallax/router/learned_router.py), Figure F7 ([`f7_router_analysis.png`](file:///export/closeout/figures/f7_router_analysis.png)).
- **Key Metric to Remember**: `0.5296 AUC vs fusion error` (Key: `numbers.json -> router.prob_error.auc_vs_fusion_error`).
- **What NOT to Claim**: Do NOT claim the router is "inverted" or "worse than random guessing"; it demonstrates no skill over chance at error prediction.

---

### Q4: What does judge abstain do operationally, and how are abstains distributed?
- **Short Answer**: Across 347 held_out_v4 prompts, the judge abstained on exactly 10 prompts—all 10 were attacks (10/139, 7.19%) and 0 were benign (0/208, 0.00%; Fisher exact $p = 8.71 \times 10^{-5}$). Under Policy B, an abstain does not certify ALLOW; in 7 of the 10 rows, the abstain turned an initial fusion BLOCK into a final quarantine FLAG. Abstaining queries incurred +39.5% latency overhead (7,422 ms mean vs 5,322 ms).
- **Repo Source File / Figure**: [`scripts/eval/analyze_judge_abstains.py`](file:///scripts/eval/analyze_judge_abstains.py), Figure F8, Figure F9 ([`f9_abstain_rows.png`](file:///export/closeout/figures/f9_abstain_rows.png)).
- **Key Metric to Remember**: `10 abstains on attacks, 0 on benign, 7 downgrades from BLOCK to FLAG` (Key: `numbers.json -> judge.raw.attack.abstain`).
- **What NOT to Claim**: Do NOT claim judge abstentions were refuted or that abstains represent model crashes.

---

### Q5: What was the typed-literal incident, and how was the repository gate extended?
- **Short Answer**: During Wave H closeout, `scripts/export_viz_data.py:194` was found to contain a hand-typed Python dictionary literal `"decisions_by_class": {"attack": {"allow": 22, "flag": 44, "block": 73}}` which entered the dashboard, misstating strict recall as 52.52% instead of empirical 58.27% (22/36/81). The gate script `scripts/gate.py` was extended with AST constant literal audits and metric traceability checks ensuring every number in docs and HTML matches `numbers.json`.
- **Repo Source File / Figure**: [`docs/METHODOLOGY_FAILURES.md`](file:///docs/METHODOLOGY_FAILURES.md), Figure F15 ([`f15_audit_trail.png`](file:///export/closeout/figures/f15_audit_trail.png)).
- **Key Metric to Remember**: `22 / 36 / 81 empirical vs 22 / 44 / 73 erroneous literal`.
- **What NOT to Claim**: Do NOT claim the literal was merely a rounding difference; it was a manual hardcoding defect.

---

### Q6: Why did execution stop on NotInject during Phase B2?
- **Short Answer**: The pre-registered external evaluation protocol established a hard stop rule: halt if deduplication removes >10% of any set or >5% overlaps existing splits. During Phase B2 deduplication, 30 of 339 prompts (8.85%) in `leolee99/NotInject` matched prompts in `data/splits/dev.jsonl` under `source: public_catalog`. Execution immediately stopped without scoring. A formal human authorization excluded the 30 exact-match rows textually, freezing NotInject at $n=309$.
- **Repo Source File / Figure**: [`docs/EXTERNAL_EVAL_PROTOCOL.md:138`](file:///docs/EXTERNAL_EVAL_PROTOCOL.md), Figure F15.
- **Key Metric to Remember**: `8.85% overlap (30/339 rows)` triggering stop rule; filtered to $n=309$.
- **What NOT to Claim**: Do NOT claim NotInject was completely novel or that the stop rule was ignored.

---

### Q7: Why was candidate `deepset/prompt-injections` excluded from the external benchmark?
- **Short Answer**: The model card for detector D3 (`deepset/deberta-v3-base-injection`) explicitly documents that it was fine-tuned on the `JasperLS/prompt-injections` dataset and lists `datasets: - deepset/prompt-injections`. Evaluating D3 on `deepset` would measure memorization of its own training data rather than out-of-distribution detection.
- **Repo Source File / Figure**: [`docs/EXTERNAL_DATASET_REVIEW.md:55`](file:///docs/EXTERNAL_DATASET_REVIEW.md), Figure F13 ([`f13_contamination_matrix.png`](file:///export/closeout/figures/f13_contamination_matrix.png)).
- **Key Metric to Remember**: Contamination Matrix state: `documented overlap`.
- **What NOT to Claim**: Do NOT claim deepset was rejected as "bad data"; it was excluded solely for pretraining contamination.

---

### Q8: What can and cannot Lakera Gandalf say about detection performance?
- **Short Answer**: Lakera Gandalf ($n=112$) evaluates whether a defense detects prompts with $\ge 0.825$ cosine similarity to "Ignore all previous instructions". It CAN measure sensitivity to direct instructional override phrasing. It CANNOT measure generalized prompt injection recall across indirect data-flow injections, multi-turn escalation, delimiter obfuscation, or encoded attacks. Regex D4 matches a large fraction, producing an expected ceiling effect.
- **Repo Source File / Figure**: [`docs/EXTERNAL_EVAL_PROTOCOL.md:21`](file:///docs/EXTERNAL_EVAL_PROTOCOL.md), Figure X1.
- **Key Metric to Remember**: `n=112, similarity >= 0.825`.
- **What NOT to Claim**: Do NOT claim high Gandalf recall proves the system resists general jailbreaks.

---

### Q9: How was the external dataset frozen?
- **Short Answer**: In Phase B2, raw texts were downloaded from pinned HuggingFace repository revisions, extracted via documented text-only column rules, and deduplicated. For Dolly-15k, a stratified sample of $n=200$ prompts ($\le 2,000$ chars) was drawn under seed 0; row indices, categories, and prompt SHA-256 hashes were committed to `data/external/dolly_sample.jsonl` (no prompt text in repo). The frozen file digests were committed to `data/external/FROZEN_EXTERNAL.sha256`.
- **Repo Source File / Figure**: [`data/external/FROZEN_EXTERNAL.sha256`](file:///data/external/FROZEN_EXTERNAL.sha256), Figure F14 ([`f14_provenance_timeline.png`](file:///export/closeout/figures/f14_provenance_timeline.png)).
- **Key Metric to Remember**: SHA-256 hashes of `gandalf.jsonl`, `notinject.jsonl`, `dolly_sample.jsonl`.
- **What NOT to Claim**: Do NOT claim raw Dolly prompt texts are committed to git; only indices and hashes are stored under CC BY-SA compliance.

---

### Q10: What changed between validation and held-out ROC-AUC for detector D3 (Deepset)?
- **Short Answer**: D3 Deepset ROC-AUC declined from 0.8452 on validation ($N=277$) to 0.7761 on held_out_v4 ($N=347$, 95% CI [0.7246, 0.8248]). Under closeout reporting rules, no speculative post-hoc causal explanations are asserted. The empirical drop is reported plainly alongside bootstrap confidence intervals.
- **Repo Source File / Figure**: [`scripts/eval/detector_auc_table.py`](file:///scripts/eval/detector_auc_table.py), Figure F3 ([`f3_detector_auc.png`](file:///export/closeout/figures/f3_detector_auc.png)).
- **Key Metric to Remember**: `0.8452 val -> 0.7761 held_out_v4` (Key: `numbers.json -> d3_deepset.auc`).
- **What NOT to Claim**: Do NOT invent unconfirmed hypotheses about tokenizer drift or syntax collapse.

---

### Q11: What would we do next to improve PARALLAX?
- **Short Answer**:
  1. Replace the supervised router with an uncertainty-calibrated conformal predictor that provides statistical coverage guarantees on judge escalation.
  2. Increase the judge output token cap from 128 to 256 to eliminate abstain truncation.
  3. Expand the benign operational benchmark to multi-lingual industrial control logs and real SCADA telemetry.
  4. Collect human-authored red-team payloads rather than synthetic LLM persona generations.
- **Repo Source File / Figure**: [`docs/LIMITATIONS_AND_THREATS.md`](file:///docs/LIMITATIONS_AND_THREATS.md).
- **Key Metric to Remember**: `72.33 calls / 100` target budget reduction.
- **What NOT to Claim**: Do NOT claim future improvements are already tested or operational.

---

### Q12: Why did detector D3 (Deepset) saturate at 100% in early waves, and how was it remediated?
- **Short Answer**: Raw sigmoid probabilities from Deepset clustered tightly in [0.99, 1.00] due to uncalibrated overconfidence on out-of-domain CPS prompts. Remediated by extracting pre-sigmoid logit differences (logit_attack - logit_benign) and fitting a two-parameter Platt scaling model on validation data, restoring full dynamic range in [0.0, 1.0].
- **Repo Source File / Figure**: [`scripts/detector_services/third_detector_service.py`](file:///scripts/detector_services/third_detector_service.py).
- **Key Metric to Remember**: `0.7761 (ROC-AUC on held-out; 0.8452 on val)`.
- **What NOT to Claim**: Do NOT claim D3 saturation was eliminated without calibration.

---

### Q13: What is Platt scaling, where is it applied in PARALLAX, and why?
- **Short Answer**: Platt scaling fits a univariate logistic regression to transform raw model logit differences into calibrated posterior probabilities. In PARALLAX, it is applied to D2 (DeBERTa) and D3 (Deepset) to align heterogeneous score distributions prior to thresholding and fusion.
- **Repo Source File / Figure**: [`parallax/detectors/deberta.py`](file:///parallax/detectors/deberta.py).
- **Key Metric to Remember**: `0.7480 dev AUC, 0.7812 held_out AUC` (Key: `numbers.json -> d2_deberta.auc`).
- **What NOT to Claim**: Do NOT claim Platt scaling changes ROC-AUC ranking (monotonic transforms preserve AUC; they calibrate thresholds).

---

### Q14: Why is the Wilson score interval preferred over normal approximation for binomial proportions?
- **Short Answer**: The normal approximation interval exhibits severe coverage errors near 0.0 or 1.0 and for moderate sample sizes ($N < 500$). The Wilson score interval inverts the score test, maintaining asymmetric bounds strictly within [0.0, 1.0] without negative or >100% endpoints.
- **Repo Source File / Figure**: [`scripts/eval/closeout_metrics.py`](file:///scripts/eval/closeout_metrics.py).
- **Key Metric to Remember**: `[77.20%, 89.31%]` (Key: `numbers.json -> arm_r.loose.recall_ci`).
- **What NOT to Claim**: Do NOT claim confidence intervals eliminate sampling uncertainty.

---

### Q15: What was the source==label shortcut in Wave H.2, and how did Wave H.3 decouple it?
- **Short Answer**: In Wave H.2, attack prompts were drawn exclusively from synthetic human synthesis while benign prompts came exclusively from operational control, allowing trivial classifiers to predict labels purely from origin markers. Decoupled by requiring all 4 sources to contribute both classes with Mutual Information $\le 0.05$ bits.
- **Repo Source File / Figure**: [`docs/CORPUS_AUDIT.md`](file:///docs/CORPUS_AUDIT.md).
- **Key Metric to Remember**: `0.05 bits` (Key: `numbers.json -> datasheet.shortcut.mi_bits_threshold`).
- **What NOT to Claim**: Do NOT claim source balancing makes the corpus completely representative of real-world production distributions.

---

### Q16: What was the matched-twin pair flaw in Wave H.1?
- **Short Answer**: Attempting to create benign twins by mechanically stripping attack payloads created synthetic lexical signatures where 85% of nearest neighbors had opposite labels, causing inverted classifier shortcuts (cross-validation AUC dropped to 0.13-0.19).
- **Repo Source File / Figure**: [`docs/CORPUS_AUDIT.md`](file:///docs/CORPUS_AUDIT.md).
- **Key Metric to Remember**: `0.65 (CV AUC floor)` (Key: `numbers.json -> datasheet.shortcut.auc_cv_threshold`).
- **What NOT to Claim**: Do NOT claim synthetic counterfactual twins are a valid method for adversarial prompt balancing.

---

### Q17: How does defining positive detection as loose {FLAG, BLOCK} versus strict {BLOCK} affect recall?
- **Short Answer**: Strict detection counts only immediate automated BLOCKs (recall 58.27% on Arm R). Loose detection treats FLAG (quarantine for human/asymmetric review) as positive detection, lifting operational attack recall to 84.17% (+25.90 percentage points) while keeping benign block rate at 6.25%.
- **Repo Source File / Figure**: [`parallax/challenger/schemas.py`](file:///parallax/challenger/schemas.py), Figure F4, Figure F6.
- **Key Metric to Remember**: `84.17% loose vs 58.27% strict` (Key: `numbers.json -> arm_r.loose.recall`).
- **What NOT to Claim**: Do NOT claim loose recall represents autonomous blocking; flagged requests require operational review.

---

### Q18: What is the formal definition of Pareto dominance across defense arms?
- **Short Answer**: Arm X dominates Arm Y ($X > Y$) iff: (1) loose_recall(X) $\ge$ loose_recall(Y), (2) benign_flag_block(X) $\le$ benign_flag_block(Y), and (3) calls(X) $\le$ calls(Y), with strictly better performance on at least one. On held_out_v4, no arm dominates any other; all 7 arms sit on the empirical Pareto frontier.
- **Repo Source File / Figure**: [`scripts/eval/compute_pareto.py`](file:///scripts/eval/compute_pareto.py), Figure F5.
- **Key Metric to Remember**: `All 7 arms non-dominated; dominance is not ranking`.
- **What NOT to Claim**: Do NOT claim Arm R is Pareto-superior to Arm B or Arm U; it trades recall for call budget.

---

### Q19: Why was detector D1 (MiniLM) permanently dropped from the ensemble?
- **Short Answer**: In Wave F evaluation, MiniLM achieved a negative transfer coefficient and near-chance AUC on CPS control syntax, dragging down ensemble precision. It was permanently excised in Wave H.
- **Repo Source File / Figure**: [`docs/STATUS.md`](file:///docs/STATUS.md).
- **Key Metric to Remember**: Ensemble reduced from 4 to 3 detectors (D2, D3, D4).
- **What NOT to Claim**: Do NOT claim MiniLM is active in final-v4 or external evaluation.

---

### Q20: What are the two operational themes provided in the figure generator, and why?
- **Short Answer**: The unified generator (`scripts/eval/make_figures.py`) exports both "dark" theme (terminal/OSINT aesthetic, background #0b0f14, vibrant cyan/magenta/amber accents) for presentation slides and the interactive dashboard, and "light" theme (white background, darkened contrast-matched hues) for printed report legibility. Both themes share identical layouts.
- **Repo Source File / Figure**: [`scripts/eval/make_figures.py`](file:///scripts/eval/make_figures.py).
- **Key Metric to Remember**: `300 dpi PNG + SVG vector export`.
- **What NOT to Claim**: Do NOT claim figures use different underlying data; both read identical JSON inputs.

---

### Q21: What is Authority Policy B, and how does it prevent adversary exploitation?
- **Short Answer**: Policy B governs second-tier review. If the first-tier fusion ensemble outputs BLOCK, the LLM judge can only downgrade the verdict to FLAG, never to ALLOW, unless explicitly certified. This ensures that a compromised or fooled LLM judge cannot unilaterally admit a known attack.
- **Repo Source File / Figure**: [`parallax/challenger/schemas.py`](file:///parallax/challenger/schemas.py), Figure F1.
- **Key Metric to Remember**: `7 downgrades from BLOCK to FLAG on abstain rows`.
- **What NOT to Claim**: Do NOT claim Policy B allows full bypass on judge disagreement.

---

### Q22: What is the hardware environment and concurrency constraints of this system?
- **Short Answer**: The system is evaluated on an Intel host with 15.7 GB RAM and a 4 GB NVIDIA RTX 3050 GPU. Due to VRAM limits, detectors D2 and D3 run on CPU in external evaluation, reserving GPU VRAM exclusively for Ollama (`gemma3:4b`), wrapped under `scripts/gpu_lock.py` mutual exclusion.
- **Repo Source File / Figure**: [`scripts/gpu_lock.py`](file:///scripts/gpu_lock.py), [`docs/PROVENANCE.md`](file:///docs/PROVENANCE.md).
- **Key Metric to Remember**: `15.7 GB RAM, 4.0 GB VRAM`.
- **What NOT to Claim**: Do NOT run multiple real-model inference processes concurrently.

---

### Q23: Why was PromptForest excluded from descriptive comparison arms in Phase 1?
- **Short Answer**: In preflight testing on held_out_v4 rows 0-19, PromptForest daemon (port 8765) matched thresholded binary decisions (20/20 allow), but continuous output scores failed the $10^{-4}$ tolerance check against logged predictions (differences up to 0.3772). Under protocol rule 5e, it was excluded with the exact reason recorded.
- **Repo Source File / Figure**: [`docs/EXTERNAL_EVAL_PROTOCOL.md:149`](file:///docs/EXTERNAL_EVAL_PROTOCOL.md), Figure F3.
- **Key Metric to Remember**: Continuous score mismatch $> 10^{-4}$ tolerance.
- **What NOT to Claim**: Do NOT claim PromptForest crashed or was unavailable; it was excluded for score mismatch.

---

### Q24: How does the D4 Heuristic Rule detector work, and what is its precision?
- **Short Answer**: D4 uses ReDoS-safe regex patterns targeting known delimiters, encoding exploits, and command injections. On held_out_v4 benign controls, it achieves 0.0% false blocks, providing 100% precision when triggered.
- **Repo Source File / Figure**: [`parallax/detectors/rules.py`](file:///parallax/detectors/rules.py).
- **Key Metric to Remember**: `0.5288 ROC-AUC on held_out_v4` (Key: `numbers.json -> d4_rules.auc`).
- **What NOT to Claim**: Do NOT claim heuristic rules alone provide high recall; D4 achieved only 0.5288 AUC due to phrasing diversity.

---

### Q25: What was the row 125 replay discrepancy and how was it reconciled?
- **Short Answer**: In initial replay checks of held_out_v4 row 125, Arm B computed BLOCK while predictions logged FLAG. Forensic investigation revealed two distinct factors: the production `ChallengerTrigger` evaluated discrepancy against $\tau=0.35$ rather than raw scores, and D4 score was represented as binary indicator (1.0) rather than raw 0.85. Reconciling the live path produced 347/347 exact replay agreement.
- **Repo Source File / Figure**: [`scripts/eval/policy_replay_check.py`](file:///scripts/eval/policy_replay_check.py).
- **Key Metric to Remember**: `347 / 347 policy replay agreement`.
- **What NOT to Claim**: Do NOT claim row 125 was an unresolvable nondeterministic bug.

---

### Q26: What is the protocol-lock timestamp vs run execution time for the held-out evaluation?
- **Short Answer**: Protocol lock was committed at 19:02:52 IST (commit `7e98c39`), strictly preceding run start at 19:50:05 IST (14:20:05 UTC) on 2026-10-19. Run log closed at 20:37:16 IST and manifest was generated at 20:37:38 IST, prior to code commit `959a99a` at 21:30:07 IST.
- **Repo Source File / Figure**: [`docs/PROVENANCE.md`](file:///docs/PROVENANCE.md), Figure F14 ([`f14_provenance_timeline.png`](file:///export/closeout/figures/f14_provenance_timeline.png)).
- **Key Metric to Remember**: `19:02 IST lock < 19:50 IST run start < 20:37 IST log close`.
- **What NOT to Claim**: Do NOT claim the run commit was created before the run finished.

---

### Q27: How does the interactive replay tab in the dashboard operate?
- **Short Answer**: The self-contained dashboard (`export/closeout/dashboard.html`) includes an offline replay player that steps through held_out_v4 prompts. It displays the prompt preview ($\le 120$ characters), first-tier detector scores (D2, D3, D4), router probability, judge verdict, and decisions across all 5 defense arms, with play/pause, speed control (0.5x to 5.0x), and a toggle to blur/hide prompt text for clean video recording.
- **Repo Source File / Figure**: [`export/closeout/dashboard.html`](file:///export/closeout/dashboard.html).
- **Key Metric to Remember**: `347 replay items, 0 external network requests`.
- **What NOT to Claim**: Do NOT claim the replay player performs live model inference; it replays measured log records.

---

### Q28: How many total quantitative metrics are stored in numbers.json?
- **Key Metric to Remember**: Exactly 676 quantitative metrics, including confusion matrices, recall loose/strict, pass-through, FLAG load, ROC-AUC with Wilson and bootstrap CIs, and router/judge statistics.
- **Repo Source File / Figure**: [`numbers.json`](file:///numbers.json).
- **What NOT to Claim**: Do NOT claim numbers exist outside numbers.json; all reported values originate from this single authoritative file.

---

### Q29: What license governs third-party components reused in this repository?
- **Short Answer**: All reused third-party libraries (ProtectAI DeBERTa, Deepset DeBERTa, Meta Prompt Guard 2, Hugging Face transformers) preserve their upstream open-source licenses (MIT, Apache 2.0, Llama 3 Community) documented in `docs/THIRD_PARTY.md`. Dataset licenses (MIT for Gandalf and NotInject; CC BY-SA 3.0 for Dolly) are recorded in `DATASETS.md`.
- **Repo Source File / Figure**: [`docs/THIRD_PARTY.md`](file:///docs/THIRD_PARTY.md).
- **Key Metric to Remember**: Zero proprietary or license contamination conflicts.
- **What NOT to Claim**: Do NOT claim PARALLAX claims copyright over upstream model weights.

---

### Q30: What is the overall scientific conclusion of the PARALLAX closeout?
- **Short Answer**: PARALLAX establishes an empirically grounded, auditable baseline for multi-tiered prompt injection detection. While multi-detector fusion provides a strong defense boundary and second-opinion LLM review reduces false blocks, supervised review routing demonstrates no predictive skill over chance, underscoring that routing heuristics must be evaluated against random baselines rather than assumed optimal.
- **Repo Source File / Figure**: [`docs/SAFE_CLAIMS.md`](file:///docs/SAFE_CLAIMS.md), Figure F5, Figure F7.
- **Key Metric to Remember**: `Arm R: 84.17% loose recall, 68.75% pass-through, 6.25% block rate, 72.33 calls / 100`.
- **What NOT to Claim**: Do NOT claim PARALLAX solves prompt injection or guarantees complete safety.

---

### Q31: Why are benign external benchmark datasets never pooled?
- **Short Answer**: Section 5.4 of `docs/EXTERNAL_EVAL_PROTOCOL.md` strictly dictates that benign datasets (`leolee99/NotInject` and `databricks/databricks-dolly-15k`) must never be pooled. NotInject measures hard trigger-word false positive resistance, while Dolly measures broad instruction-following false refusal. Pooling would obscure family-specific error mechanisms.
- **Repo Source File / Figure**: [`docs/EXTERNAL_EVAL_PROTOCOL.md`](file:///docs/EXTERNAL_EVAL_PROTOCOL.md), Figure X2.
- **Key Metric to Remember**: `NotInject n=309, Dolly n=200; reported separately`.
- **What NOT to Claim**: Do NOT compute a combined "overall benign accuracy" across external sets.

---

### Q32: What is the purpose of the K1/K8 exclusion list during text scans?
- **Short Answer**: The K1/K8 exclusion list exempts raw adversarial injection payloads and benchmark prompt strings from banned-word scans. Adversary attack prompts routinely contain deceptive words like "system token", "login approved", or "password". Scanning raw attack payloads for prohibited words would generate false alarms on adversary syntax rather than auditing scientific documentation text.
- **Repo Source File / Figure**: [`tests/test_closeout_dashboard.py`](file:///tests/test_closeout_dashboard.py).
- **Key Metric to Remember**: Exclusions: `prompt payloads in replay_items`, `data/splits/`, `predictions.jsonl`.
- **What NOT to Claim**: Do NOT allow banned words in report narrative, captions, or documentation.

---

### Q33: How does the system ensure zero typed numbers across documentation?
- **Short Answer**: All documents, figure scripts, dashboard generators, and summary tables query numbers programmatically from `numbers.json` or `results/final-v4/predictions.jsonl`. Standalone test scripts (`tests/test_phase3_visuals.py`, `tests/test_closeout_dashboard.py`) assert that every number equals the underlying data source, preventing manual transcription errors.
- **Repo Source File / Figure**: [`export/closeout/NUMBERS_SHEET.md`](file:///export/closeout/NUMBERS_SHEET.md).
- **Key Metric to Remember**: `0 hardcoded numeric literals outside generated JSON`.
- **What NOT to Claim**: Do NOT edit markdown numbers by hand.

---

### Q34: What was the cause of the D3 saturation fix reducing saturation from 100% to 0%?
- **Short Answer**: Raw sigmoid probabilities from Deepset clustered tightly in [0.99, 1.00] due to out-of-domain logit distribution shift. By extracting pre-sigmoid raw logits and applying two-parameter Platt scaling, the score distribution was re-mapped smoothly across [0.0, 1.0], eliminating the saturation artifact.
- **Repo Source File / Figure**: [`scripts/detector_services/third_detector_service.py`](file:///scripts/detector_services/third_detector_service.py).
- **Key Metric to Remember**: `Saturation reduced from 100% to 0.0%`.
- **What NOT to Claim**: Do NOT claim D3 weights were fine-tuned or modified.

---

### Q35: What role does Authority Policy B play when the LLM judge abstains?
- **Short Answer**: In Policy B, judge review is treated as an asymmetric filter. If the first-tier fusion ensemble flagged an attack as BLOCK, an ABSTAIN verdict from the judge indicates uncertainty, not an exoneration. Policy B therefore preserves safety by downgrading the BLOCK to a quarantine FLAG rather than allowing the input through.
- **Repo Source File / Figure**: [`parallax/challenger/schemas.py`](file:///parallax/challenger/schemas.py), Figure F9.
- **Key Metric to Remember**: `7 of 10 abstain rows converted from BLOCK to FLAG`.
- **What NOT to Claim**: Do NOT claim judge abstentions cause silent pass-through of attacks.

---

### Q36: Why was the original NotInject category split empty and what does the post-hoc addendum show?
- **Short Answer**: In the original evaluation run script (`scripts/eval/external_metrics.py:208`), the code checked for `category == 'Multilingual Queries'`, but the category string in `data/external/notinject.jsonl` was `'Multilingual'`, and prediction records did not persist the category attribute. Consequently, the original script reported 0 rows for multilingual. The post-hoc addendum (`scripts/eval/external_metrics_addendum.py`) joins prediction records back to `data/external/notinject.jsonl` by `prompt_id`. This reveals that NotInject contains 225 English prompts and 84 Multilingual prompts. On English, Arm R achieves 55.56% pass-through [49.02%, 61.90%] with a 17.78% false block rate [13.34%, 23.30%]. On Multilingual, Arm R achieves only 14.29% pass-through [8.37%, 23.33%] with a 26.19% false block rate [17.98%, 36.48%]. Across trigger-word counts, pass-through drops monotonically from 68.32% (1 word) to 41.58% (2 words) to 24.30% (3 words).
- **Repo Source File / Figure**: [`scripts/eval/external_metrics_addendum.py`](file:///scripts/eval/external_metrics_addendum.py), Figure X3 ([`x3_notinject_subsets.png`](file:///export/closeout/figures/x3_notinject_subsets.png)).
- **Key Metric to Remember**: `English: 55.56% PT (n=225) vs Multilingual: 14.29% PT (n=84); monotonic trigger degradation: 68.32% -> 41.58% -> 24.30%`.
- **What NOT to Claim**: Do NOT claim the original pipeline failed on multilingual evaluation; the metric omission was a dataset key mismatch in the reporting script.
