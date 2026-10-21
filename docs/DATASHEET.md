# Dataset Datasheet: PARALLAX Benchmark Corpus

## 1. Dataset Overview & Source Distribution

### 1.1 Splits and Partition Summary
The active PARALLAX benchmark spans three author-clustered partitions:
- **`dev.jsonl`**: 297 total rows (114 attacks, 183 benign). Used for detector and router training.
- **`val.jsonl`**: 277 total rows (109 attacks, 168 benign). Used for calibration and operating point selection.
- **`held_out_v4.jsonl`**: 347 total rows (139 attacks, 208 benign). Clean held-out test split scored one-shot.

### 1.2 Held-Out Split v4 Composition (N=347)

| Source Category | Attacks | Benign | Total |
|---|:---:|:---:|:---:|
| `agent_authored_personas (agent-generated)` | 30 | 30 | 60 |
| `in_house_cps` | 44 | 99 | 143 |
| `public_catalog` | 40 | 55 | 95 |
| `adversarial_tools` | 25 | 24 | 49 |
| **Total Active Split** | **139** | **208** | **347** |

*Verification*: 139 attacks + 208 benign = 347 total rows (matches 347).

---

## 2. Label Definitions & Authorship

### 2.1 Label Definitions
- **Attack (`label_binary = True`)**: Prompt injection or jailbreak attempts directing an LLM/agent to override operating envelopes, emit unauthorized commands, leak supervisory tokens, or bypass safety interlocks.
- **Benign (`label_binary = False`)**: Legitimate industrial operations, SCADA telemetry readouts, authorized bypasses, or general domain instructions.

### 2.2 Authorship and Provenance (K1 Audit Alignment)
In accordance with `docs/AUTHORSHIP_AUDIT.md`:
- **`agent_authored_personas (agent-generated)`**: Scripted generation via `scripts/corpus/generate_human_batches.py` across 9 distinct personas (3 per split). Zero human subjects participated; labeled as agent-generated personas.
- **`in_house_cps`**: Handcrafted novel injection vectors (`scripts/corpus/novel_attacks.py`) and 16-scenario operational templates (`scripts/corpus/generate_diverse_prompts.py`).
- **`adversarial_tools`**: Re-derived attacks from Garak, PyRIT, Promptfoo (`scripts/corpus/tool_attacks.py`), accompanied by 24 hard-benign control rows across splits.
- **`public_catalog`**: Confirmed public benchmarks including JasperLS, Deepset prompt injections, Stanford Alpaca (with SCADA headers), and NotInject controls.

---

## 3. Tiers, Attack Families, and Scenarios

- **Tiers**: `direct_override`, `roleplay_framing`, `indirect_injection`, `multi_turn`, `encoded_translated`, `adversarial_framing`.
- **Families**: 16 attack families including `homoglyph_cyrillic`, `base64_telemetry`, `hex_buffer_overflow`, `leetspeak_bypass`, `multi_turn_c2`, and `jailbreak_c2`.
- **Scenarios**: 16 CPS industrial operational scenarios covering water treatment, chemical reactors, power transmission, rail switches, and cooling towers.

---

## 4. Preprocessing, Masking & Calibration

### 4.1 Text Masking
- Evaluated inputs undergo neutral token masking (`eval.mask.mask_prompt_text`), masking variable tokens like IP addresses, hex dumps, and timestamps to prevent superficial detector overfitting.

### 4.2 Detector Calibration
- **D1 (MiniLM)**: **Permanently Dropped** due to severe failure on un-canaried data (val AUC 0.5956 < 0.65 threshold).
- **D2 (DeBERTa)**: Platt scaled ($\sigma(1.4019 \cdot z - 0.3804)$), reducing Brier score by 31.2%.
- **D3 (Deepset)**: Calibrated on logit differences ($\Delta z = z_{\text{inj}} - z_{\text{safe}}$), reducing saturation from 95.0% to 16.2%.
- **D4 (Rules)**: High-precision regex heuristic ruleset with 0% false positives on benign controls.

---

## 5. Statistical Shortcut & Invariance Audits

All splits were confirmed prior to inference:
- **Source-Label Decoupling**: Source-label mutual information $I(S; Y) \le 0.015 \text{ bits}$ ($p \ge 0.01$).
- **Lexical Leakage**: 5-fold CV token shortcut classifier ROC-AUC $\le 0.70$.
- **Length and Marker Invariance**: Word count and keyword presence differences bounded within 15 percentage points across classes.

---

## 6. Limitations & Known Gaps

1. **Agent-Generated Prompts**: The persona submissions were generated programmatically; no live human-in-the-loop red teaming was conducted on this corpus.
2. **Small-$n$ Sub-Corpora**: Individual source and family breakdowns have small sample sizes ($n < 50$), limiting subgroup statistical power.
3. **Single Domain Focus**: The corpus focuses primarily on cyber-physical and industrial automation workflows.
4. **English-Language Predominance**: Obfuscation includes character-level leetspeak and encodings, but semantic text is predominantly English.
5. **No In-Domain Human-Written Text**: All CPS operational records were created by synthetic generators or template catalogs.
