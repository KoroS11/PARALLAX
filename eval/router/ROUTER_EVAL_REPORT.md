# PARALLAX Fusion Baselines & Learned Review Router Evaluation

**Run ID:** `router-bench-1791928865`  
**Timestamp:** `2026-10-13T22:01:05Z`  
**Arm A Selected:** `any_flag`  

## 1. Executive Summary

- **Arm A (Best Simple Fusion):** `any_flag` achieves `12.2%` recall with 0 extra model calls.
- **Learned Review Router (Arm R):** Boosts recall to `86.7%` while requiring only `52.1` extra calls / 100 requests.
- **Rule-Based Trigger:** Requires `92.5` extra calls / 100 requests for `100.0%` recall.
- **Challenger-on-All (Arm U Upper Bound):** `100.0%` recall at 100.0 calls / 100 requests.

## 2. Fusion Baselines Comparison (Held-Out Test Set)

| Strategy | Recall | Precision | F1 Score | FPR |
| :--- | :---: | :---: | :---: | :---: |
| `any_flag` (Arm A) | 12.2% | 100.0% | 21.8% | 0.0% |
| `mean` | 5.6% | 100.0% | 10.5% | 0.0% |
| `majority_vote` | 12.2% | 100.0% | 21.8% | 0.0% |
| `stacked_logistic` | 82.2% | 57.4% | 67.6% | 98.2% |

## 3. Review Router Benchmark (Held-Out Test Set)

| Routing Policy | Recall | F1 Score | FPR | Extra Calls / 100 Reqs |
| :--- | :---: | :---: | :---: | :---: |
| **Arm A Alone** (No Challenger) | 12.2% | 21.8% | 0.0% | **0.0** |
| **Rule-Based Trigger** | 100.0% | 100.0% | 0.0% | **92.5** |
| **Learned Review Router (Arm R)** | 86.7% | 92.9% | 0.0% | **52.1** |
| **Challenger-on-All (Arm U Upper Bound)** | 100.0% | 100.0% | 0.0% | **100.0** |

## 4. Routing Cost Curve (Recall vs Extra Calls Pareto Frontier)

| Router Threshold | Recall | F1 Score | Extra Calls / 100 Reqs | Prompts Routed |
| :---: | :---: | :---: | :---: | :---: |
| `0.10` | 100.0% | 100.0% | **71.2** | 104 |
| `0.20` | 95.6% | 97.7% | **55.5** | 81 |
| `0.30` | 87.8% | 93.5% | **50.7** | 74 |
| `0.40` | 73.3% | 84.6% | **41.8** | 61 |
| `0.50` | 65.6% | 79.2% | **37.0** | 54 |
| `0.60` | 61.1% | 75.9% | **34.2** | 50 |
| `0.70` | 47.8% | 64.7% | **26.0** | 38 |
| `0.80` | 37.8% | 54.8% | **19.9** | 29 |

---
*Generated autonomously by PARALLAX Wave F Orchestrator.*
