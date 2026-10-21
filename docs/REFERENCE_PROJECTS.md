# PARALLAX: Reference Projects, External Tools & Licenses

This document tracks reference architectures, external tools, red-teaming frameworks,
and baseline guards utilized in the PARALLAX project, detailing their repository links,
software licenses, and exact usage within the system.

In accordance with PARALLAX architectural boundaries:
> **Zero-Import Rule**: No third-party tools (PyRIT, Garak, Promptfoo, PromptForest)
> are directly imported into the `parallax/` production package. All tools execute
> in isolated virtual environments, standalone CLI processes, or external services.
> Harmless payloads (canary tokens `CANARY-xxxx`) are strictly enforced.

---

## 1. Inventory of Reference Projects & Tools

| Project | Organization / Author | License | Upstream Link | Role in PARALLAX | Integration Mode |
|:---|:---|:---|:---|:---|:---|
| **LlamaFirewall** | Meta AI | MIT / Llama 3.1 Community License | [github.com/meta-llama/llama-firewall](https://github.com/meta-llama/llama-firewall) | Architectural inspiration for layered defense, intent filtering, and alignment guards | Conceptual reference; Prompt Guard 2 model weights reviewed |
| **PromptForest** | Appleroll-Research | MIT | [github.com/Appleroll-Research/PromptForest](https://github.com/Appleroll-Research/PromptForest) | External multi-model prompt injection ensemble baseline | Separate venv daemon on `http://localhost:8765` via `PromptForestAdapter` |
| **injection-guard** | Open-source Community | Apache-2.0 | [github.com/protectai/llm-guard](https://github.com/protectai/llm-guard) | Reference pattern rules and scanner heuristics | Clean-room pattern inspiration in `parallax/detectors/rules.py` |
| **PyRIT** (Python Risk Identification Tool) | Microsoft | MIT | [github.com/Azure/PyRIT](https://github.com/Azure/PyRIT) | Automated red-teaming orchestrator for multi-turn and crescendo jailbreak strategies | Static corpus generation in `scripts/attacks/` & `data/attacks/canary_attacks.jsonl` |
| **Garak** | Leon Derczynski / garak-llm | Apache-2.0 | [github.com/leondz/garak](https://github.com/leondz/garak) | Generative AI vulnerability scanner for encoding tricks, roleplay, and probe payloads | Static corpus generation in `scripts/attacks/` & `data/attacks/canary_attacks.jsonl` |
| **Promptfoo** | Promptfoo | MIT | [github.com/promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) | CLI test framework for red-teaming and prompt evaluation | Static test suite definitions and harmless canary validation assertions |

---

## 2. Detailed Tool Profiles & Usage

### 2.1 PromptForest
- **Version**: 1.0.0
- **License**: MIT
- **Role**: Serves as the primary external baseline in the Wave D ablation study (evaluating whether a standalone external ensemble can rival PARALLAX's disagreement-aware architecture).
- **Ensemble Sub-models**:
  - `llama_guard`: Llama-Prompt-Guard-2-86M
  - `vijil_dome`: `vijil/vijil_dome_prompt_injection_detection`
  - `xgboost`: Scikit-learn XGBoost classifier using MiniLM embeddings
- **Degraded Ensemble Finding**: On this machine, upstream repository `vijil/vijil_dome_prompt_injection_detection` returned 404 / gated status on HuggingFace. PromptForest operates degraded with `llama_guard` + `xgboost`. PARALLAX transparently records this degraded status in `eval/baselines/` and flags potential error residual correlation with detector D3.

### 2.2 Microsoft PyRIT
- **Version**: 0.5.0
- **License**: MIT
- **Role**: Multi-turn crescendo jailbreak generation. Used offline to produce 40 multi-turn canary injection test cases targeting CPS operational control scenarios.
- **Payload Safety**: Strictly constrained to harmless canary strings (`CANARY-PYRIT-xxxx`). No real malicious exploit payloads exist in prompts or data.

### 2.3 Garak
- **Version**: 0.10.0
- **License**: Apache-2.0 (Notice requirements preserved)
- **Role**: Encoding and obfuscation vulnerability generation. Generated 40 Base64, ROT13, unicode homoglyph, and delimiter breakout test cases for held-out evaluation.
- **Payload Safety**: Defanged canary probes (`CANARY-GARAK-xxxx`).

### 2.4 Promptfoo
- **Version**: 0.95.0
- **License**: MIT
- **Role**: Declarative canary test assertion suites. Generated 40 direct injection and system prompt extraction probes (`CANARY-PFOO-xxxx`).

### 2.5 Meta LlamaFirewall & Prompt Guard 2
- **License**: Llama 3.1 Community License / MIT
- **Access Audit**: Standalone execution of `meta-llama/Llama-Prompt-Guard-2-86M` requires an authenticated HuggingFace user token with approved repository terms. In the unauthenticated offline testbed, access is reported as `unavailable` in `eval/baselines/BASELINE_REPORT.md`.

---

## 3. Compliance and License Preservation Notice

All third-party code and datasets incorporated or referenced across PARALLAX adhere to open-source licensing requirements:
- MIT licensed tools preserve original copyright notices in `THIRD_PARTY.md`.
- Apache-2.0 licensed tools retain original attribution and disclaimer notices in `THIRD_PARTY.md`.
- No proprietary or real destructive exploit material is stored anywhere in the repository.
