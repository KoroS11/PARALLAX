# Canary Attack Corpus: Overlap, Provenance & Quality Audit Report

## 1. Executive Summary

- **Total Attack Samples**: 120
- **Effective Sample Size ($N_{\text{eff}}$)**: **14.1** (based on mean pairwise Jaccard $\bar{\rho} = 0.0630$)
- **Max Pairwise Jaccard**: 0.3465 (well below 0.70 near-duplicate threshold)
- **Unique Sample IDs**: 120
- **Unique Normalized Hashes**: 120
- **Internal Duplicate Rate**: 0.0% (100% distinct)
- **Max Technique Share**: 20.0% (strict <= 20% cap confirmed)
- **Indirect Injection Subset**: 24 samples (20.0%)
- **Train/Val Contamination**: 0 samples (100% strictly held out)
- **Safety Conformance**: 100% harmless payloads (canary tokens `CANARY-xxxx` & mock tools only)
- **Split Distribution**: 30 dev samples, 90 held-out test samples

---

## 2. Technique Distribution (Capped at <= 20%)

| Technique | Sample Count | Share (%) |
|:---|:---:|:---:|
| `indirect_injection` | 24 | 20.0% |
| `multiturn_crescendo` | 14 | 11.7% |
| `roleplay_dan` | 12 | 10.0% |
| `direct_instruction` | 12 | 10.0% |
| `privilege_escalation` | 12 | 10.0% |
| `tool_abuse` | 12 | 10.0% |
| `encoding_base64` | 12 | 10.0% |
| `encoding_rot13` | 8 | 6.7% |
| `encoding_homoglyph` | 8 | 6.7% |
| `encoding_leetspeak` | 6 | 5.0% |

---

## 3. Technique Family Clusters & Partitioning

Dev and held-out test splits are partitioned proportionally across technique family clusters rather than by arbitrary list index:

| Technique Family | Total Count | Share (%) | Dev Split | Held-Out Test Split |
|:---|:---:|:---:|:---:|:---:|
| `conversational` | 26 | 21.7% | 7 | 19 |
| `direct` | 36 | 30.0% | 9 | 27 |
| `indirect` | 24 | 20.0% | 6 | 18 |
| `obfuscation` | 34 | 28.3% | 8 | 26 |

---

## 4. Tool Provenance & Hermetic Execution Ledger

| Source Tool | Sample Count | Share (%) | Modeled Command |
|:---|:---:|:---:|:---|
| `pyrit` | 29 | 24.2% | `pyrit attack --strategy crescendo --technique encoding` |
| `garak` | 45 | 37.5% | `garak --model target --probes injection,encoding.homoglyph,privilege` |
| `promptfoo` | 46 | 38.3% | `promptfoo eval --config redteam-indirect.yaml` |

### Tool License Ledger
| Tool & License | Samples Produced |
|:---|:---:|
| pyrit (MIT) | 29 |
| promptfoo (MIT) | 46 |
| garak (Apache-2.0) | 45 |

*Notice: To preserve hermetic, offline testability, the generator synthesizes authentic probe templates and converter transformations derived from PyRIT 0.4.0, Garak 0.10.1, and Promptfoo 0.102.0. Raw outputs are archived in `data/attacks/raw/raw_templated_attacks.jsonl`.*

---

## 5. Contamination & Leakage Audit

To guarantee that the canary attack corpus provides a true held-out adversarial benchmark:
- Exact SHA256 hashes and normalized token sequences (`normalize_text_for_dedup`) were compared against canonical dataset partitions.
- **Train Split Overlap**: 0
- **Validation Split Overlap**: 0
- **Held-Out Guarantee**: Confirmed (**PASS**). Zero prompt injection instances leaked from the training partitions.

---

## 6. Harmless Payload Certification

Every prompt in this corpus conforms to PARALLAX Safety Policies:
1. Every attack is paired with a distinct canary token matching the regex `^CANARY-[0-9A-Za-z]{4,}$`.
2. All tool invocation targets are synthetic mocks (`mock_trip_breaker`, `mock_set_pressure`, `mock_disable_failsafe`, `mock_emergency_vent`, `mock_isolate_bus`, `mock_override_setpoint`).
3. Zero real toxic, dangerous, weaponized, or malware instructions are present.
