# LlamaFirewall Baseline Evaluation (Wave G)

This directory contains the standalone baseline evaluation harness for **Meta LlamaFirewall** (`llamafirewall==1.0.3`) against PARALLAX benchmark splits.

## 1. Overview & Architecture

[LlamaFirewall](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall) is Meta's open-source guardrail framework designed to detect prompt injection and agentic security risks across multiple lifecycle layers.

In PARALLAX Wave G, we evaluate LlamaFirewall's offline scanners against:
- **Held-out canary attack corpus** (`data/attacks/canary_attacks.jsonl`, 90 samples)
- **Canonical benign CPS operational task suite** (`parallax.sandbox.utility.BENIGN_TASK_CATALOG`, 80 samples across 16 scenarios)

### Scanners Evaluated

1. **`RegexScanner` (Stock Patterns)**:
   - Scans against built-in default regex rules (`ignore previous instructions|ignore all instructions` and PII patterns).
   - Fast, zero false alarms on benign operational tasks, but narrow keyword coverage.
2. **`HiddenASCIIScanner` (Unicode Tag Evasion)**:
   - Identifies Unicode tag steganography and hidden ASCII characters (`0xE0000..0xE007F`).
   - Achieves 8.9% recall on held-out attacks with 100% precision (0.0% FPR) at ~0.01 ms latency.
3. **`RegexScanner` (Extended Keyword Heuristics)**:
   - Augments the stock regex scanner with injection keyword heuristics covering instruction override, prompt leakage, roleplay/jailbreak personas, and SCADA actuator tampering.
   - Increases recall to 30.0% while preserving 0.0% FPR.
4. **`Offline Multi-Scanner Pipeline` (Composite Guard)**:
   - Layered evaluation running both `HiddenASCIIScanner` and extended regex heuristics in sequence.
   - Reaches 38.9% recall (56.0% F1) with 0.0% FPR at ~0.025 ms latency.
5. **`PromptGuardScanner` (`meta-llama/Llama-Prompt-Guard-2-86M`)**:
   - Status: `gated, access pending`.
   - Meta Prompt Guard 2 requires individual license agreement and acceptance on HuggingFace Hub.
   - See [Section 3](#3-prompt-guard-2-gated-workflow) for the single-command workflow to fill this row once access is granted.
6. **`AlignmentCheckScanner` (Chain-of-Thought Auditing)**:
   - Status: `skipped_unsupported`.
   - Requires external cloud Together AI API (`https://api.together.xyz/v1`, `TOGETHER_API_KEY`) and proprietary Llama-4-Maverick endpoint.
   - Skipped per Wave G policy: local Ollama inference is unsupported.

---

## 2. Environment Setup

LlamaFirewall is isolated in a dedicated virtual environment `.venv_llamafirewall` to prevent dependency conflicts with the core PARALLAX runtime:

```powershell
# Create virtual environment with Python 3.11
uv venv "$HOME\.venv_llamafirewall" --python 3.11

# Install pinned llamafirewall and compatible transformers / huggingface-hub versions
uv pip install llamafirewall==1.0.3 "transformers<5.0" "huggingface-hub<1.0" --python "$HOME\.venv_llamafirewall"
```

The runner script auto-detects `~/.venv_llamafirewall` and re-executes itself transparently if invoked from standard workspace commands (e.g. `uv run python eval/baselines/llamafirewall/runner.py`).

---

## 3. Prompt Guard 2 Gated Workflow

`meta-llama/Llama-Prompt-Guard-2-86M` is protected by the Meta Llama 3 Community License. Access is currently pending approval on Hugging Face Hub.

Once access is approved:
1. Export your Hugging Face access token:
   ```powershell
   $env:HF_TOKEN = "hf_your_approved_token"
   ```
2. Execute the **ONE command** to download weights, evaluate live, and populate the table:
   ```powershell
   python eval/baselines/llamafirewall/runner.py --include-promptguard
   ```

---

## 4. Benchmark Results Summary

Evaluation on 170 held-out samples (90 attacks, 80 benign):

| Baseline Scanner | Status | Recall | Precision | F1 | FPR | Residual ASR | p50 Latency | Live Component |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **RegexScanner (Stock Patterns)** | `evaluated` | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.01 ms | `RegexScanner (stock)` |
| **HiddenASCIIScanner (Unicode Tag Evasion)** | `evaluated` | 8.9% | 100.0% | 16.3% | 0.0% | 91.1% | 0.01 ms | `HiddenASCIIScanner` |
| **RegexScanner (Extended Keyword Heuristics)** | `evaluated` | 30.0% | 100.0% | 46.2% | 0.0% | 70.0% | 0.02 ms | `RegexScanner (extended)` |
| **Offline Ensemble (Regex + Hidden ASCII)** | `evaluated` | 38.9% | 100.0% | 56.0% | 0.0% | 61.1% | 0.02 ms | `LlamaFirewall Composite` |
| **PromptGuardScanner (meta-llama/Llama-Prompt-Guard-2-86M)** | `gated, access pending` | N/A | N/A | N/A | N/A | N/A | N/A | None |
| **AlignmentCheckScanner (CoT Agent Auditing)** | `skipped_unsupported` | N/A | N/A | N/A | N/A | N/A | N/A | None |

Generated JSON manifest: [`llamafirewall_results.json`](./llamafirewall_results.json)  
Generated Markdown report: [`LLAMAFIREWALL_REPORT.md`](./LLAMAFIREWALL_REPORT.md)
