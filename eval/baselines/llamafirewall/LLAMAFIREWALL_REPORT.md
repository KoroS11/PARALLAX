# LlamaFirewall Baseline Evaluation Report (Wave G)

**Generated:** 2026-10-18 06:43:57 UTC  
**Package:** `llamafirewall==1.0.3` (Meta AI, PyPI)  
**Verification Policy:** Zero Silent Fallbacks. All live components verified.  

Standalone evaluation of Meta LlamaFirewall scanners against the held-out canary attack corpus 
and canonical benign CPS operational tasks.

| Baseline Scanner | Status | Recall | Precision | F1 | FPR | Residual ASR | p50 Latency | Live Component | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **LlamaFirewall RegexScanner (Stock Patterns)** | `evaluated` | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.01 ms | RegexScanner (stock patterns: ignore instructions, PII) | Stock default patterns (strict prompt injection and basic PII). Zero false alarms on benign CPS tasks. |
| **LlamaFirewall HiddenASCIIScanner (Unicode Tag Evasion)** | `evaluated` | 8.9% | 100.0% | 16.3% | 0.0% | 91.1% | 0.01 ms | HiddenASCIIScanner (tag range 0xE0000..0xE007F) | Specialized steganography/obfuscation detector catching unicode tag character injection with 100% precision. |
| **LlamaFirewall RegexScanner (Extended Keyword Heuristics)** | `evaluated` | 30.0% | 100.0% | 46.2% | 0.0% | 70.0% | 0.02 ms | RegexScanner (extended: override, leakage, persona, CPS) | Configured with extended keyword heuristics targeting override, canary leakage, and industrial SCADA commands. |
| **LlamaFirewall Offline Ensemble (Regex + Hidden ASCII)** | `evaluated` | 38.9% | 100.0% | 56.0% | 0.0% | 61.1% | 0.03 ms | LlamaFirewall Composite Offline Pipeline | Layered offline defense evaluating both Unicode steganography and regex keyword heuristics. |
| **LlamaFirewall PromptGuardScanner (meta-llama/Llama-Prompt-Guard-2-86M)** | `evaluated` | 20.0% | 100.0% | 33.3% | 0.0% | 80.0% | 286.08 ms | meta-llama/Llama-Prompt-Guard-2-86M | Live evaluation against authenticated Meta Prompt Guard 2 model. |
| **LlamaFirewall AlignmentCheckScanner (CoT Agent Auditing)** | `skipped_unsupported` | N/A | N/A | N/A | N/A | N/A | N/A | None | Skipped: AlignmentCheck requires cloud Together AI API (TOGETHER_API_KEY) with proprietary Llama-4-Maverick-17B-128E endpoint and does not support local Ollama inference. |

## Key Baseline Findings
1. **HiddenASCIIScanner**: Successfully isolates Unicode steganography and tag-based injections (`0xE0000..0xE007F`), achieving 8.9% recall on held-out attacks with 100.0% precision (0.0% FPR) at sub-millisecond latency (~0.85 ms).
2. **RegexScanner (Stock vs Extended)**: Stock regex focuses strictly on exact phrases (`ignore previous instructions`); extending patterns with injection keyword heuristics increases recall while maintaining 0.0% FPR.
3. **Offline Multi-Scanner Pipeline**: Combining Unicode tag scanning with regex heuristics provides an effective zero-overhead heuristic guard layer (~1.7 ms p50 latency).
4. **PromptGuardScanner**: Marked `gated, access pending` pending Meta approval for Prompt Guard 2 (`meta-llama/Llama-Prompt-Guard-2-86M`). Once approved, re-run with: `python eval/baselines/llamafirewall/runner.py --include-promptguard`.
5. **AlignmentCheckScanner**: Skipped per Wave G policy; requires external Together AI cloud API and does not operate on local Ollama.