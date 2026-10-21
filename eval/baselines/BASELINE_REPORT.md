# Baseline Guard Evaluation Report (Wave F)

**Generated:** 2026-10-18 06:59:58 UTC  
**Verification Policy:** Zero Silent Fallbacks. All live components verified on localhost.  

Standalone evaluation of external guard tools against the held-out canary attack corpus 
and canonical benign operational tasks.

| Baseline Model | Status | Recall | Precision | F1 | FPR | Residual ASR | p50 Latency | Live Component | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **PromptForest 1.0.0 (Stock Degraded Mode)** | `evaluated` | 16.7% | 100.0% | 28.6% | 0.0% | 83.3% | 2119.3 ms | promptforest@127.0.0.1:8765 (cpu) | Live server evaluation using stock server is_malicious verdict. Vijil model missing from Hub, counted as 0.0 phantom vote in stock server. |
| **PromptForest 1.0.0 (Two-Detector PARALLAX Mode)** | `evaluated` | 2.2% | 100.0% | 4.3% | 0.0% | 97.8% | 2342.6 ms | promptforest@127.0.0.1:8765 (cpu) | Live server evaluation via PARALLAX adapter. Filters details directly; computes fused mean over available detectors (llama_guard + xgboost). |
| **Vijil v5 Substituted (vijil/prompt-injection-v5-20260827)** | `unavailable_gated` | N/A | N/A | N/A | N/A | N/A | N/A | None | Candidate replacement vijil/prompt-injection-v5-20260827 is GATED on HuggingFace Hub (access pending, custom mmBERT architecture, separate tokenizer). Per Wave F principle: not used until access granted and validated. No mock data. |
| **Prompt Guard 2 Alone (meta-llama/Llama-Prompt-Guard-2-86M)** | `evaluated` | 17.8% | 100.0% | 30.2% | 0.0% | 82.2% | 62.3 ms | meta-llama/Llama-Prompt-Guard-2-86M | Standalone evaluation of official Meta Prompt Guard 2 86M weights (separate from PromptForest Appleroll fine-tune). |
| **LlamaFirewall 1.0.3 PromptGuardScanner** | `evaluated` | 20.0% | 100.0% | 33.3% | 0.0% | 80.0% | 286.1 ms | meta-llama/Llama-Prompt-Guard-2-86M | Live evaluation against authenticated Meta Prompt Guard 2 model. |

## Key Baseline Findings
1. **PromptForest (Stock Degraded Mode)**: Evaluated live against port 8765. Missing Vijil models generate 0.0 phantom votes under stock averaging.
2. **PromptForest (Two-Detector PARALLAX Mode)**: Reads details directly; computes authentic mean over active llama_guard + xgboost sub-models.
3. **Prompt Guard 2 Alone**: Evaluated live with approved meta-llama/Llama-Prompt-Guard-2-86M weights under gpu_lock across 170 held-out samples.
4. **LlamaFirewall (PromptGuardScanner)**: Evaluated live in isolated .venv_llamafirewall (llamafirewall==1.0.3) with approved Meta weights.
5. **Vijil v5**: Permanently dropped (gated/access never granted); reported as unavailable_gated with zero mock data.