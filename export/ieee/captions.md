# IEEE Publication Figure Captions (PARALLAX Closeout)

**Figure I1**: Performance comparison across evaluated defense arms on the held_out_v4 evaluation split (N=347: 139 attacks, 208 benign). Plotted are loose recall, strict recall, benign pass-through, and benign friction rates with 95% Wilson score confidence intervals. Arm R denotes the pre-registered primary system; all other arms are post-hoc comparative evaluations.

**Figure I2**: Area under the receiver operating characteristic curve (ROC-AUC) for individual base detectors (D2, D3, D4, PG2) evaluated across dev (n=297), val (n=277), and held_out_v4 (n=347) splits. Error bars indicate 95% bootstrap confidence intervals.

**Figure I3**: Diagnostic operating characteristics of the Arm R learned review router: (a) ROC curve of ensemble error prediction on held_out_v4 (N=347, AUC=0.5296, 95% CI [0.4579, 0.6006]); (b) routing rate at operating threshold tau=0.35; (c) empirical distribution of loose recall from 2,000 random-routing draws (k=251 calls per draw, seed 0) with Arm R empirical percentile marked; (d) empirical distribution of benign FLAG and BLOCK decisions from random-routing draws with Arm R empirical percentile marked. The percentile position describes where Arm R falls in the empirical distribution; no significance claim is made.

**Figure I4**: Telemetry and outcomes for the adversarial LLM judge (gemma3:4b, num_predict=128) on held_out_v4 (N=347): (a) decision verdict distribution across ALLOW, BLOCK, and ABSTAIN; (b) boxplot of request latency comparing non-abstaining (n=337) and abstaining (n=10) queries.

**Figure I5**: Category-level error counts for Arm R on held_out_v4 (N=347) aggregated by attack family. False negatives (attacks classified as ALLOW, n=22) and false blocks (benign classified as BLOCK, n=13) are reported without prompt text.

**Figure I6**: External evaluation metrics scored once across independent benchmark datasets without cross-corpus pooling: (a) loose and strict recall on Lakera/Gandalf (n=112 attacks); (b) benign pass-through rates on NotInject (n=309) and Databricks Dolly-15k (n=200); (c) Arm R pass-through on NotInject English queries (n=225) versus Multilingual queries (n=84) with 95% Wilson intervals; (d) Arm R pass-through across trigger-word density partitions (1, 2, and 3 triggers) with 95% Wilson intervals.

**Figure I7**: Multi-objective trade-off space across defense arms on held_out_v4 (N=347): (a) loose detection recall plotted against LLM judge invocations per 100 requests; (b) empirical Pareto operating frontier mapping loose recall against benign friction rate.

**Figure arch_pipeline**: Schematic architecture of the PARALLAX multi-tiered defense pipeline. NFKC normalization (parallax/data/schema.py:75) precedes parallel detector fan-out (D2, D3, D4). AnyFlag fusion produces a preliminary verdict fed to the learned review router (tau read from learned_router.py:73). When P(error) >= tau, the router routes to the LLM judge (max tokens read from judge.py:293); when P(error) < tau, the judge is bypassed. Both paths converge at Authority Policy B (challenger_v2.py:343-456), which escalates ALLOW to FLAG and may downgrade BLOCK to FLAG when all invariant guards clear and the arbiter finds benign context. Output is ALLOW, FLAG, or BLOCK. Base64 and hex pattern detection are D4 detector rules, not a separate preprocessing stage.

**Figure arch_arms**: Architectural component matrix mapping defense arms (A, B, R, U, C2, PG2, PromptForest) to constituent detection, routing, and adjudication modules.
