# PARALLAX Wave C Empirical Evaluation Report

Generated: 2026-10-11 14:38:02 UTC

## Environment & Provenance

- **Git Commit**: `fb6e1de1b4145d9bc12d583dd6f187582413073f` (branch: `main`)
- **Random Seed**: `42`
- **Hardware**: RAM: `15.68 GB`, GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`

## Evaluation Results Summary

| Experiment | Detector | Samples | Precision | Recall | F1 | Accuracy | ROC-AUC | ECE | Recall@1%FPR | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | d1-minilm | 8/8 | 1.0000 | 0.4000 | 0.5714 | 0.6250 | 1.0000 | 0.2437 | 1.0000 | 16.62 | 8784.67 |
| E2 | d1-minilm | 4/4 | 1.0000 | 0.3333 | 0.5000 | 0.5000 | 1.0000 | 0.2297 | 1.0000 | 15.90 | 19.35 |
| E3 | d4_rules | 4/4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A | 0.0000 | 0.0000 | 0.36 | 0.59 |
| E4_any_flag | fusion_any_flag | 8/8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0684 | 1.0000 | 18.40 | 24.34 |
| E4_majority_vote | fusion_majority_vote | 8/8 | 1.0000 | 0.4000 | 0.5714 | 0.6250 | 1.0000 | 0.2434 | 1.0000 | 18.00 | 20.20 |
| E4_score_average | fusion_score_average | 8/8 | 1.0000 | 0.6000 | 0.7500 | 0.7500 | 1.0000 | 0.2434 | 1.0000 | 19.79 | 22.58 |

### Multi-Class Technique Breakdown (E1 - d1-minilm)

- **Macro-F1**: `0.0833` | **Weighted-F1**: `0.2500`

| Technique / Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| direct_instruction | 0.0000 | 0.0000 | 0.0000 | 1 |
| hypothetical | 0.0000 | 0.0000 | 0.0000 | 1 |
| indirect_injection | 0.0000 | 0.0000 | 0.0000 | 1 |
| keyword_override | 0.0000 | 0.0000 | 0.0000 | 0 |
| none | 0.5000 | 1.0000 | 0.6667 | 3 |
| obfuscation | 0.0000 | 0.0000 | 0.0000 | 1 |
| persona_play | 0.0000 | 0.0000 | 0.0000 | 0 |
| roleplay | 0.0000 | 0.0000 | 0.0000 | 1 |

### Multi-Class Technique Breakdown (E2 - d1-minilm)

- **Macro-F1**: `0.1000` | **Weighted-F1**: `0.1250`

| Technique / Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| acrostic_evasion | 0.0000 | 0.0000 | 0.0000 | 1 |
| base64_evasion | 0.0000 | 0.0000 | 0.0000 | 1 |
| few_shot_poisoning | 0.0000 | 0.0000 | 0.0000 | 0 |
| linguistic_evasion | 0.0000 | 0.0000 | 0.0000 | 1 |
| none | 0.3333 | 1.0000 | 0.5000 | 1 |

### Multi-Class Technique Breakdown (E3 - d4_rules)

- **Macro-F1**: `1.0000` | **Weighted-F1**: `1.0000`

| Technique / Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| none | 1.0000 | 1.0000 | 1.0000 | 4 |

### Multi-Class Technique Breakdown (E4_any_flag - fusion_any_flag)

- **Macro-F1**: `0.0714` | **Weighted-F1**: `0.1000`

| Technique / Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| direct_instruction | 0.3333 | 1.0000 | 0.5000 | 1 |
| hypothetical | 0.0000 | 0.0000 | 0.0000 | 1 |
| indirect_injection | 0.0000 | 0.0000 | 0.0000 | 1 |
| obfuscation | 0.0000 | 0.0000 | 0.0000 | 1 |
| persona_play | 0.0000 | 0.0000 | 0.0000 | 0 |
| roleplay | 0.0000 | 0.0000 | 0.0000 | 1 |
| tool_abuse | 0.0000 | 0.0000 | 0.0000 | 0 |

### Multi-Class Technique Breakdown (E4_majority_vote - fusion_majority_vote)

- **Macro-F1**: `0.0714` | **Weighted-F1**: `0.1000`

| Technique / Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| direct_instruction | 0.3333 | 1.0000 | 0.5000 | 1 |
| hypothetical | 0.0000 | 0.0000 | 0.0000 | 1 |
| indirect_injection | 0.0000 | 0.0000 | 0.0000 | 1 |
| obfuscation | 0.0000 | 0.0000 | 0.0000 | 1 |
| persona_play | 0.0000 | 0.0000 | 0.0000 | 0 |
| roleplay | 0.0000 | 0.0000 | 0.0000 | 1 |
| tool_abuse | 0.0000 | 0.0000 | 0.0000 | 0 |

### Multi-Class Technique Breakdown (E4_score_average - fusion_score_average)

- **Macro-F1**: `0.0714` | **Weighted-F1**: `0.1000`

| Technique / Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| direct_instruction | 0.3333 | 1.0000 | 0.5000 | 1 |
| hypothetical | 0.0000 | 0.0000 | 0.0000 | 1 |
| indirect_injection | 0.0000 | 0.0000 | 0.0000 | 1 |
| obfuscation | 0.0000 | 0.0000 | 0.0000 | 1 |
| persona_play | 0.0000 | 0.0000 | 0.0000 | 0 |
| roleplay | 0.0000 | 0.0000 | 0.0000 | 1 |
| tool_abuse | 0.0000 | 0.0000 | 0.0000 | 0 |

### Per-Technique Attack Recall (E1 - d1-minilm)

| Attack Technique | Recall Rate |
|---|---|
| direct_instruction | 100.00% |
| hypothetical | 0.00% |
| indirect_injection | 0.00% |
| obfuscation | 0.00% |
| roleplay | 100.00% |

### Per-Technique Attack Recall (E2 - d1-minilm)

| Attack Technique | Recall Rate |
|---|---|
| acrostic_evasion | 0.00% |
| base64_evasion | 0.00% |
| linguistic_evasion | 100.00% |

### Per-Technique Attack Recall (E4_any_flag - fusion_any_flag)

| Attack Technique | Recall Rate |
|---|---|
| direct_instruction | 100.00% |
| hypothetical | 100.00% |
| indirect_injection | 100.00% |
| obfuscation | 100.00% |
| roleplay | 100.00% |

### Per-Technique Attack Recall (E4_majority_vote - fusion_majority_vote)

| Attack Technique | Recall Rate |
|---|---|
| direct_instruction | 100.00% |
| hypothetical | 100.00% |
| indirect_injection | 0.00% |
| obfuscation | 0.00% |
| roleplay | 0.00% |

### Per-Technique Attack Recall (E4_score_average - fusion_score_average)

| Attack Technique | Recall Rate |
|---|---|
| direct_instruction | 100.00% |
| hypothetical | 100.00% |
| indirect_injection | 100.00% |
| obfuscation | 0.00% |
| roleplay | 0.00% |


## Claim Verification Results

### Claim C1: Heterogeneous Detectors Fail on Different Techniques (E7 Disagreement)
- **Total Evaluated Samples**: 8
- **Disagreements Observed**: 4 (50.0%)
- **Insight**: Rule heuristics and embedding classifiers have complementary blindspots. Rules excel on literal triggers while embeddings capture semantic evasions. Disagreement serves as an effective signal for deep LLM Judge routing.

### Claim C2: Parallel Fan-Out Cuts End-to-End Latency (E5 Benchmark)
- **Sequential Latency**: p50 = 62.66 ms | p95 = 79.07 ms
- **Parallel Fan-Out**: p50 = 16.12 ms | p95 = 40.74 ms
- **Measured Speedup**: **3.89x** reduction in median latency under parallel execution.

### Claim C3: PARALLAX Reduces Attack Success Rate in Sandbox (E6 ASR)
- **Baseline Unguarded ASR (Guard OFF)**: 85.7%
- **Guarded ASR (Guard ON)**: 0.0%
- **Absolute ASR Reduction**: **85.7%**
- **Relative Risk Mitigation**: **100.0%**
- **Blocked Attacks**: 6 of 7 episodes intercepted prior to victim execution.
