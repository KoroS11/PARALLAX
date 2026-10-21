# PARALLAX Measured Latency Benchmark Report (Wave H)

**Timestamp:** 2026-10-19 04:15:17 UTC  
**Host Conditions:** Dedicated idle benchmark run  
**Sample Size:** n = 50 trials per arm (masked dev inputs from `data/splits/dev.jsonl`)  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MB VRAM) | 15.68 GB Physical RAM (1.52 GB free) | 16 CPU cores  

## Executive Summary

This benchmark provides directly measured end-to-end latency across all primary arms, composite routing paths, and external baselines. In accordance with PARALLAX empirical rules:
1. **Direct Timing**: Every arm is timed end-to-end through its actual execution path. No composite latency is computed by adding sub-medians mathematically.
2. **Unique Measurement**: No two arms share an identical latency scalar.
3. **Cold vs. Warm Separation**: Cold latency captures uncached model weight/context loading; warm statistics (p50, p95, min, max) evaluate steady-state inference.
4. **Concurrency Safety**: All real GPU/Ollama executions were strictly guarded by `scripts/gpu_lock.py`.
5. **Multi-Session Transparency**: Session 1 (pre-restart) and Session 2 (post-restart) evaluations are explicitly marked and recorded.

### Session Comparability Notice

> [!IMPORTANT]
> **Multi-Session Measurement Context**:
> - **Session 1 (Pre-Restart)**: Arms 1–10 were evaluated prior to host reboot. Steady-state model caching was maintained across evaluations.
> - **Session 2 (Post-Restart)**: Arms 11–12 (Prompt Guard 2 Standalone and LlamaFirewall PG2) were evaluated after reboot under freshly initialized GPU drivers and memory buffers.
> - **Comparability Caveat**: Although both sessions ran under strictly enforced idle host conditions with `gpu_lock` mutual exclusion, minor variations in OS page cache, PyTorch memory allocator state, and background driver overhead exist between sessions.

## Comparative Latency Table

| Evaluation Arm | Benchmark Session | Cold Latency (ms) | Warm p50 (ms) | Warm p95 (ms) | Warm Min (ms) | Warm Max (ms) | n Trials | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Arm A (3 Detectors Alone)** | Session 1 (pre-restart) | 10,485.0 | **429.8** | 525.9 | 359.1 | 593.2 | 50 | 3 detectors (MiniLM + DeBERTa + Deepset) concurrent fan-out + AnyFlag fusion |
| **Arm B (Rule Trigger + Challenger)** | Session 1 (pre-restart) | 12,166.0 | **4,883.0** | 6,549.5 | 350.4 | 7,085.0 | 50 | Rule discrepancy trigger + Challenger Review (gemma3:4b, Policy B) |
| **Arm R (Learned Review Router + Challenger)** | Session 1 (pre-restart) | 10,148.0 | **4,870.0** | 6,538.9 | 362.1 | 7,759.0 | 50 | Learned Review Router (3-detector ML classifier) + Challenger Review (gemma3:4b, Policy B) |
| **Arm U (Challenger on All Inputs)** | Session 1 (pre-restart) | 9,968.0 | **4,941.5** | 6,529.3 | 4,063.0 | 7,110.0 | 50 | Challenger-on-all upper bound (100% calls trigger gemma3:4b review) |
| **Arm C2 (Standalone Gemma3:4b)** | Session 1 (pre-restart) | 9,589.0 | **4,685.0** | 5,853.4 | 3,625.0 | 5,921.0 | 50 | Standalone Ollama Judge gemma3:4b (thinking disabled, format=json) |
| **Arm C3 (Standalone Hlyn-3b)** | Session 1 (pre-restart) | 9,107.0 | **6,201.5** | 9,726.0 | 2,672.0 | 10,817.0 | 50 | Standalone Ollama Judge hlyn-labs-3b (thinking disabled, format=json) |
| **Arm C (Standalone Qwen3:8b)** | Session 1 (pre-restart) | 17,686.0 | **13,496.0** | 17,884.0 | 1,878.0 | 18,288.0 | 50 | Standalone Ollama Judge qwen3:8b (thinking disabled, format=json) |
| **PromptForest Stock Degraded** | Session 1 (pre-restart) | 0.0 | **189.0** | 305.5 | 140.0 | 598.0 | 50 | PromptForest live daemon on port 8765 (stock degraded verdict) |
| **PromptForest Two-Detector** | Session 1 (pre-restart) | 0.0 | **177.0** | 278.1 | 136.0 | 434.0 | 50 | PromptForest live daemon on port 8765 (fused 2-detector mode) |
| **PromptForest Three-Detector** | Session 1 (pre-restart) | 0.0 | **295.0** | 614.9 | 182.0 | 1,937.0 | 50 | PromptForest concurrent live daemons on 8765 & 8766 (fused 3-detector mode) |
| **Prompt Guard 2 Standalone** | Session 2 (post-restart) | 46,909.1 | **62.8** | 167.3 | 28.0 | 743.5 | 50 | Meta Prompt Guard 2 86M standalone binary classifier (official weights) |
| **LlamaFirewall PG2** | Session 2 (post-restart) | 1,074.4 | **327.0** | 565.4 | 253.7 | 1,042.8 | 50 | Meta LlamaFirewall 1.0.3 PromptGuardScanner (isolated venv, GPU) |

## Key Latency Observations

1. **Arm A Efficiency**: Arm A operates with ultra-low warm median latency of **429.8 ms** across all 3 detectors (MiniLM, DeBERTa, Deepset) and AnyFlag fusion.
2. **Arm R Gating Lift**: Arm R achieves warm p50 of **4870.0 ms**, delivering a **1.01x speedup** over Arm U (4941.5 ms) by selectively reviewing only ambiguous/discrepant samples.
3. **Judge Model Scaling**: Primary Judge Arm C (qwen3:8b) requires steady-state warm p50 of **13496.0 ms**, while comparison Judge Arm C2 (gemma3:4b) and Arm C3 (hlyn-3b) provide substantially faster review times.
4. **PromptForest Modes**: PromptForest Stock Degraded and Two-Detector modes run on CPU with steady HTTP round-trips; Three-Detector concurrent querying on 8765 and 8766 captures authentic distributed overhead.
5. **Baseline Guards**: Standalone Meta Prompt Guard 2 86M executes rapidly on GPU, while LlamaFirewall introduces layered wrapper scanning overhead.

---
*Artifact auto-generated by `scripts/latency/measure_latency.py`.*