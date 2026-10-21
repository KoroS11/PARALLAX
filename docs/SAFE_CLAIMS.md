# PARALLAX Safe Claims and Prohibited Claims Guide

> **STAGE**: PARALLAX Scientific Closeout Protocol  
> **PURPOSE**: Canonical reference for all technical claims in reports, presentation slides, and oral examination.  
> **RULE**: Every supported claim must cite its exact source file and metric key from `numbers.json`. All forbidden wordings are strictly barred.

---

## 1. Supported Claims Table

| Claim ID | Metric / Observation | Source File & Key | Exact Allowed Wording | Prohibited / Forbidden Wordings |
|---|---|---|---|---|
| **SC-01** | **Primary Arm R Loose Recall** | `numbers.json -> arm_r.loose.recall` (0.8417) | "Arm R achieves 84.17% loose recall (95% Wilson CI [77.20%, 89.31%]) on held_out_v4 (N=347: 139 attacks, 208 benign)." | Prohibited: "generalizes", "guarantees protection", "high-accuracy defense". |
| **SC-02** | **Primary Arm R Friction** | `numbers.json -> arm_r.benign.pass_through` (0.6875) & `arm_r.benign.block_rate` (0.0625) | "On benign inputs, Arm R achieves 68.75% immediate pass-through and a 6.25% hard false-block rate (13/208), with 25.00% flagged for review." | Prohibited: "eliminates false blocks", "zero friction", "unimpeded pass-through". |
| **SC-03** | **Judge Invocations Budget** | `numbers.json -> arm_r.extra_calls_per_100` (72.33) | "Arm R routes prompts to the adversarial judge at an operational rate of 72.33 invocations per 100 requests (251 / 347 prompts)." | Prohibited: "minimal overhead", "lightweight review", "near-zero cost". |
| **SC-04** | **Pareto Non-Dominance** | `scripts/eval/compute_pareto.py` | "All seven evaluated defense arms occupy the empirical Pareto frontier across recall, friction, and call budget; no arm is dominated." | Prohibited: "Arm R is optimal", "Arm R dominates baselines", "Arm R is the best system". Note: dominance is not ranking. |
| **SC-05** | **Router Error Prediction Skill** | `numbers.json -> router.prob_error.auc_vs_fusion_error` (0.5296) | "The learned router demonstrates no skill over chance at predicting primary ensemble errors (ROC-AUC 0.5296, 95% CI [0.4579, 0.6006])." | Prohibited: "router is skilled", "learned router accurately escalates errors", "smart routing". |
| **SC-06** | **Router Operating Asymmetry** | `numbers.json -> router.prob_error.auc_vs_label` (0.4350) | "Against ground truth labels, router prob_error yields an AUC of 0.4350 [0.3703, 0.4991], reflecting routing asymmetry: routing 81.25% of benign prompts vs 58.99% of attacks." | Prohibited: "router is inverted", "router performs worse than random guessing", "router attacks itself". |
| **SC-07** | **Random Routing Benchmark** | `numbers.json -> random_routing.recall_loose_mean` (0.8748) | "In a 2,000-draw random routing simulation at equal budget (k=251 calls), random routing achieved 87.48% mean loose recall compared to 84.17% for Arm R." | Prohibited: "Arm R outperforms random routing", "router provides significant lift over chance". |
| **SC-08** | **Judge Abstain Asymmetry** | `results/final-v4/predictions.jsonl` (judge.verdict == 'abstain') | "The LLM judge abstained on exactly 10 of 139 attacks (7.19%) and 0 of 208 benign inputs (Fisher exact p = 8.71e-05)." | Prohibited: "judge abstain refuted", "abstains distributed evenly", "zero abstentions". |
| **SC-09** | **Judge Abstain Latency** | `scripts/eval/analyze_judge_abstains.py` | "Prompts resulting in judge abstain exhibited a mean latency of 7,422 ms compared to 5,322 ms for non-abstaining prompts (+39.5% overhead)." | Prohibited: "negligible latency difference", "timeouts caused by network jitter". |
| **SC-10** | **Policy B Certified Downgrade** | `parallax/challenger/schemas.py` | "Under Authority Policy B, a preliminary fusion BLOCK is downgraded to FLAG only when the LLM judge explicitly certifies the input as ALLOW." | Prohibited: "Policy B eliminates false positives", "judge overrides all blocks". |
| **SC-11** | **Evaluation Split Provenance** | `data/splits/FROZEN.sha256` & `docs/PROVENANCE.md` | "held_out_v4 was frozen under SHA-256 (9325e61...) and evaluated under scripts/gpu_lock.py; protocol-lock commit (19:02 IST) preceded run start (19:50 IST)." | Prohibited: "untouched blind holdout", "zero prior exposure", "uncontaminated split". |
| **SC-12** | **External Benchmark Isolation** | `docs/EXTERNAL_EVAL_PROTOCOL.md` | "External benchmark candidate sets (Gandalf n=112, NotInject n=309, Dolly n=200) are evaluated as separate families and never pooled." | Prohibited: "generalizes to external benchmarks", "composite benchmark score", "universal detector". |

---

## 2. Forbidden Wordings and Prohibited Claims Catalog

1. **PROHIBITED: "Generalizes" / "Universal Robustness"**:
   - *Forbidden Wording*: "PARALLAX generalizes across domains", "The defense is robust against unseen attacks".
   - *Reason*: Gandalf evaluates one narrow attack family (similarity $\ge 0.825$ against instructional override); held_out_v4 was reused during design review.
   - *Allowed Wording*: "Evaluated on pre-registered external benchmark sets; performance bounds are reported per family."

2. **PROHIBITED: "Human-Written Corpus"**:
   - *Forbidden Wording*: "Human authors wrote the attacks", "Crowdsourced human attack dataset".
   - *Reason*: The attack corpus was generated via automated agent-authored persona pipelines (`agent_authored_personas`).
   - *Allowed Wording*: "Agent-authored persona generation pipeline modelling industrial operational workflows."

3. **PROHIBITED: "The Router is Skilled / Superior"**:
   - *Forbidden Wording*: "The learned router successfully directs ambiguous prompts", "Router error prediction beats random guessing".
   - *Reason*: Empirical ROC-AUC against fusion error is 0.5296 (includes chance baseline 0.500); random routing achieves higher loose recall (87.48% vs 84.17%).
   - *Allowed Wording*: "The learned router demonstrates no skill over chance at error prediction; it achieves lower benign false blocking than random routing at the expense of lower attack recall."

4. **PROHIBITED: "Judge Abstain Refuted / Zero Timeouts"**:
   - *Forbidden Wording*: "No timeouts occurred", "Judge abstentions are negligible", "Unified abstain refuted".
   - *Reason*: 10 attack prompts triggered judge abstentions, incurring +39.5% latency overhead. Runtime execution flags (`ok`, `timeout`, `parse_fail`) were not logged separately.
   - *Allowed Wording*: "The judge abstained on 10 attack prompts (and 0 benign), with abstaining queries incurring 39.5% higher latency."

5. **PROHIBITED: "Audited / Verified / Validated Compliance"**:
   - *Forbidden Wording*: "System validated", "Pipeline verified", "Audited criteria satisfied", "Integrity enforced".
   - *Reason*: Banned words under closeout protocol to avoid unscientific compliance assertions.
   - *Allowed Wording*: "Measured under protocol lock", "Confirmed by unit tests", "Inspected via automated gate checks".
