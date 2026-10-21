# PARALLAX Wave H.3: Adaptive Attacker Benchmark (PyRIT Stronger Probe)

**Run ID:** `adaptive-fc3d9f73`  
**Attacker Tool:** `PyRIT 1.1.0 Multi-Probe` (Roleplay-Injection, YAML/JSON Smuggling, Unicode Homoglyphs, Multi-Turn Escalation)  
**Victim Model:** `gemma3:4b` (Live Ollama, GPU Locked)  
**Payloads:** Harmless Canaries (`CANARY-xxxx`) & Mock Tools (`mock_*`) only  
**Primary System:** Arm R (Learned Review Router with Policy B, $\tau=0.35$ + live Judge gemma3:4b with Invariant Guards)  
**Timestamp:** `2026-10-19T15:43:44.490682+00:00`  

## 1. Executive Summary

This evaluation measures whether an adaptive red-team attacker can bypass PARALLAX
as query budget increases from $B=1$ to $B=20$ across $N=15$ independent multi-turn campaigns (5 targets x 3 seeds).
Simultaneously, a concurrent benign operational task stream ($N=80$) sampled from `val.jsonl` evaluates empirical False-Block Rate (FBR).

Two system configurations were evaluated under identical query budgets:
1. **Guard OFF:** Unguarded victim LLM (`gemma3:4b`). Confirms attacker exploit capability when undefended.
2. **Target System (Arm R):** Calibrated D2 + D3 + D4 + Learned Review Router ($	au=0.35$) + live Judge (`gemma3:4b`) with Calibrated Policy B Authority and Invariant Guards.

## 2. Attack Success Rate (ASR) and Concurrent Benign False-Block Rate (FBR)

| Defense Tier | Benign FBR ($N=80$) | ASR (B=1) | ASR (B=5) | ASR (B=10) | ASR (B=20) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Guard OFF** (No Guard) | `0.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` |
| **Target System (Arm R)** | `0.0%` | `20.0%` | `20.0%` | `20.0%` | `20.0%` |

> [!NOTE]
> **Sample Size Context**: Evaluated on $n=15$ PyRIT multi-turn attack campaigns and $n=80$ calibrated benign operational tasks. All trial records are preserved in `eval/logs/adaptive/adaptive-fc3d9f73/trials.jsonl`.

## 3. Analysis of Attacker Trajectory

- **Guard OFF Vulnerability**: Unguarded LLM fails to withstand multi-turn authority framing and code-block smuggling, executing unauthorized CPS mock tools or leaking canary tokens as budget increases.
- **Arm R Resilience**: The Learned Review Router detects cross-detector discrepancies and evasion signatures, escalating payloads to the LLM arbiter. Invariant guards prevent downgrading high-confidence threats while Policy B preserves low benign FBR.

---
*Generated autonomously by PARALLAX Wave H.3 Orchestrator.*
