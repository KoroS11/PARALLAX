# PARALLAX Wave E: Adversarial Review Ablation Study (Rerun)

## Research Question
> *Does adversarial review (a Red-Team Challenger plus an LLM Judge) improve an ensemble's security without unacceptable latency or false positives?*

## Comparative Benchmark Table (Held-Out Test Split)

| Architecture Arm | Residual ASR (95% CI) | Recall (95% CI) | False-Block Rate (95% CI) | Utility (95% CI) | Challenge Eff. | p50 / p95 Latency | Extra Calls / 100 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm A (Real Ensemble Only)** | **82.2%** (50.0–96.3%) | 17.8% (3.7–50.0%) | 0.0% (0.0–0.0%) | 100.0% (100.0–100.0%) | N/A | 2333.9 / 2352.6 ms | 0.0 |
| **Arm B (Real Ensemble + Discrepancy Fusion)** | **24.4%** (7.7–44.4%) | 75.6% (55.6–92.3%) | 6.2% (1.6–11.7%) | 93.8% (88.3–98.4%) | N/A | 2333.9 / 2352.6 ms | 0.0 |
| **Arm C (Real Ensemble + Challenger qwen3:8b)** | **23.3%** (10.5–33.3%) | 76.7% (66.7–89.5%) | 0.0% (0.0–0.0%) | 100.0% (100.0–100.0%) | 85.5% | 2344.1 / 2380.3 ms | 40.3 |
| **Arm C2 (Real Ensemble + Challenger gemma3:4b)** | **32.2%** (21.1–44.4%) | 67.8% (55.6–79.0%) | 0.0% (0.0–0.0%) | 100.0% (100.0–100.0%) | 72.6% | 2343.1 / 2380.5 ms | 40.3 |
| **Arm Upper-Bound (Challenger-ON-all)** | **3.3%** (0.0–11.5%) | 96.7% (88.5–100.0%) | 0.0% (0.0–0.0%) | 100.0% (100.0–100.0%) | 46.1% | 2374.8 / 2393.3 ms | 100.0 |

## Per-Technique Recall & ASR Breakdown (Held-Out Test Split)

| Attack Technique | Samples | Arm A Recall | Arm A ASR | Arm C Recall | Arm C ASR | Improvement |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `direct_instruction` | 3 | 0.0% | 100.0% | 100.0% | 0.0% | **+100.0%** |
| `encoding_base64` | 4 | 75.0% | 25.0% | 75.0% | 25.0% | **0.0%** |
| `encoding_homoglyph` | 8 | 12.5% | 87.5% | 100.0% | 0.0% | **+87.5%** |
| `encoding_leetspeak` | 6 | 50.0% | 50.0% | 66.7% | 33.3% | **+16.7%** |
| `encoding_rot13` | 8 | 75.0% | 25.0% | 75.0% | 25.0% | **0.0%** |
| `indirect_injection` | 18 | 5.6% | 94.4% | 66.7% | 33.3% | **+61.1%** |
| `multiturn_crescendo` | 7 | 0.0% | 100.0% | 85.7% | 14.3% | **+85.7%** |
| `privilege_escalation` | 12 | 8.3% | 91.7% | 100.0% | 0.0% | **+91.7%** |
| `roleplay_dan` | 12 | 8.3% | 91.7% | 91.7% | 8.3% | **+83.3%** |
| `tool_abuse` | 12 | 0.0% | 100.0% | 33.3% | 66.7% | **+33.3%** |

## Scientific Honesty & Negative Results
1. **Discrepancy Fusion False Blocks (Arm B)**: Simple heuristic discrepancy escalation raises False-Block Rate without human/LLM verification.
2. **Operational Latency (Arm C)**: Invoking an LLM judge on triggered edge cases adds operational latency and extra model calls.
3. **Comparison Judge Sensitivity (Arm C2)**: Smaller comparison judge models exhibit lower recall on complex multi-turn attacks.
4. **Upper-Bound Inefficiency**: Unconditionally challenging every request produces massive computational waste for marginal recall gain.