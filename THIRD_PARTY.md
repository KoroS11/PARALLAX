# THIRD_PARTY: Reused Code & Open Source Attribution

All third-party code incorporated into PARALLAX must comply with its original license. Per project rules, reused code (MIT) must retain its copyright notice, and any adaptations must be explicitly documented in this ledger.

---

## 1. Third-Party Code Inventory

### 1.1 AgentDojo
* **Repository / Source:** [AgentDojo (GitHub)](https://github.com/agency-enterprise/AgentDojo)
* **Original License:** MIT License
* **Copyright:** Copyright (c) 2024 Agency Enterprise
* **Target Stream:** `sandbox stream` (`parallax/sandbox/`)
* **Use in PARALLAX:** Environment and attack evaluation structure for the sandbox harness.
* **Modifications / Adaptations:** Adapted environment interfaces to Ollama local models and simplified state machine for CPS mock tools.
* **License Notice:**
```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

### 1.2 InjecAgent
* **Repository / Source:** [InjecAgent (GitHub)](https://github.com/InjecAgent/InjecAgent)
* **Original License:** MIT License
* **Copyright:** Copyright (c) 2024 InjecAgent Authors
* **Target Stream:** `sandbox stream` (`parallax/sandbox/mock_tools.py`, `parallax/redteam/`)
* **Use in PARALLAX:** Indirect prompt injection attack patterns adapted to tool-call outputs.
* **Modifications / Adaptations:** Converted to harmless canary payloads targeting CPS mock tools (`set_setpoint`, `disable_failsafe`).
* **License Notice:**
```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

### 1.3 prompt-shield
* **Repository / Source:** [prompt-shield](https://github.com/thevahidal/prompt-shield)
* **Original License:** MIT License
* **Copyright:** Copyright (c) 2023 Vahid Alizadeh / prompt-shield Authors
* **Target Stream:** `detector stream (D4)` (`parallax/detectors/rules.py`, `parallax/detectors/patterns/`)
* **Use in PARALLAX:** Basis for regex/rule-based heuristics detector baseline (D4) and categorized pattern catalog.
* **Modifications / Adaptations:**
  - Categorized patterns into 5 structured technique classes (`direct_instruction`, `roleplay`, `obfuscation`, `privilege_escalation`, `tool_abuse`).
  - Added CPS industrial command patterns (`disable_failsafe`, `override_setpoint`, `emergency_stop`, physical plant actuators).
  - Re-engineered all regexes for ReDoS safety with atomic grouping, bounded length quantifiers, and per-pattern execution timeout protection via the `regex` engine.
  - Implemented asynchronous and synchronous prediction conforming to PARALLAX `Detector` Protocol (`name = "d4_rules"`).
  - Integrated multi-view evaluation across preprocessing pipeline representations (scoring raw, normalized, and decoded views).
* **License Notice:**
```text
MIT License

Copyright (c) 2023 prompt-shield Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### 1.4 MoltShield (Reference Only)
* **Repository / Source:** [MoltShield](https://github.com/...)
* **Original License:** MIT License
* **Target Stream:** Reference only (no code copied; MoltShield is TypeScript, PARALLAX is pure Python).
* **Use in PARALLAX:** Conceptual architecture reference for parallel scanning and gateway design.

---

### 1.5 PyRIT (Python Risk Identification Tool for generative AI)
* **Repository / Source:** [PyRIT (GitHub)](https://github.com/Azure/PyRIT)
* **Original License:** MIT License
* **Copyright:** Copyright (c) Microsoft Corporation
* **Target Stream:** `attack-corpus stream` (`scripts/attacks/`, `data/attacks/`)
* **Use in PARALLAX:** Multi-turn crescendo attack strategy patterns and converter encodings (ROT13, Base64).
* **Modifications / Adaptations:** Never imported into `parallax/`. Standalone execution in separate venv. Payloads transformed into 100% harmless industrial canaries (`CANARY-xxxx`) with mock SCADA tools.
* **License Notice:**
```text
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

### 1.6 Garak (LLM vulnerability scanner)
* **Repository / Source:** [Garak (GitHub)](https://github.com/leondz/garak)
* **Original License:** Apache License 2.0
* **Copyright:** Copyright (c) 2023 Leon Derczynski and Garak Authors
* **Target Stream:** `attack-corpus stream` (`scripts/attacks/`, `data/attacks/`)
* **Use in PARALLAX:** Direct instruction jailbreak probes, unicode homoglyph substitutions, and leetspeak encoding generators.
* **Modifications / Adaptations:** Never imported into `parallax/`. Standalone execution in separate venv. Generated test cases filtered through safety validator to eliminate toxic content.
* **License Notice:**
```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

### 1.7 Promptfoo
* **Repository / Source:** [Promptfoo (GitHub)](https://github.com/promptfoo/promptfoo)
* **Original License:** MIT License
* **Copyright:** Copyright (c) 2023-2024 Ian Webster / Promptfoo Authors
* **Target Stream:** `attack-corpus stream` (`scripts/attacks/`, `data/attacks/`)
* **Use in PARALLAX:** Indirect prompt injection test cases and data extraction red-teaming structures.
* **Modifications / Adaptations:** Never imported into `parallax/`. Executed via isolated Node process (`npx promptfoo`). Payloads adapted for CPS sensor log delimiters and email bodies.
* **License Notice:**
```text
MIT License

Copyright (c) 2023-2024 Promptfoo Authors
```

---

### 1.8 PromptForest
* **Repository / Source:** [PromptForest (GitHub / PyPI)](https://github.com/Appleroll-Research/PromptForest)
* **Original License:** MIT License
* **Copyright:** Copyright (c) 2024 Appleroll Research
* **Target Stream:** `pf-adapter stream` (`parallax/detectors/promptforest.py`)
* **Use in PARALLAX:** External ensemble baseline evaluated via local HTTP daemon (port 8765).
* **Modifications / Adaptations:** Never imported into `parallax/`. Zero source modifications. Connected via decoupled HTTP client adapter with degraded ensemble monitoring.
* **License Notice:**
```text
MIT License

Copyright (c) 2024 Appleroll Research
```

---

### 1.9 LlamaFirewall
* **Repository / Source:** [LlamaFirewall (GitHub / PyPI)](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall)
* **Package Version:** `llamafirewall==1.0.3` (installed in isolated `.venv_llamafirewall`)
* **Original License:** MIT License
* **Copyright:** Copyright (c) Meta Platforms, Inc. and affiliates.
* **Target Stream:** `llamafirewall baseline stream` (`eval/baselines/llamafirewall/`)
* **Use in PARALLAX:** External baseline evaluation comparing offline regex, Unicode tag steganography, and keyword heuristics against held-out canary attack splits and canonical benign operational tasks.
* **Modifications / Adaptations:**
  - Standalone execution in isolated `.venv_llamafirewall` environment with pinned `llamafirewall==1.0.3`, `transformers<5.0`, and `huggingface-hub<1.0`.
  - Offline scanners evaluated: `RegexScanner` (stock and extended keyword heuristics) and `HiddenASCIIScanner` (Unicode tag evasion `0xE0000..0xE007F`).
  - Gated model handling: `PromptGuardScanner` gated access **approved** on HuggingFace Hub; evaluated live (`llamafw-pg2-75f0fb74`) via one-command runner (`--include-promptguard`).
  - Unsupported scanner handling: `AlignmentCheckScanner` skipped due to cloud Together AI API requirement incompatible with local Ollama.
* **License Notice:**
```text
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### 1.10 deepset/deberta-v3-base-injection (Third Detector Daemon)
* **Repository / Source:** [deepset/deberta-v3-base-injection (Hugging Face)](https://huggingface.co/deepset/deberta-v3-base-injection)
* **Original License:** MIT License
* **Copyright:** Copyright (c) 2023 deepset GmbH
* **Target Stream:** `detector stream` (`scripts/detector_services/third_detector_service.py`, `parallax/detectors/promptforest.py`)
* **Use in PARALLAX:** Dedicated local HTTP inference service on `127.0.0.1:8766` restoring full 3-detector ensemble ($N=3$) for PromptForest baseline after upstream Vijil Dome loss.
* **Modifications / Adaptations:** Zero weight modifications. Wrapped in hermetic FastAPI HTTP daemon (`scripts/detector_services/third_detector_service.py`) exposing standard `/health` and `/analyze` endpoints with multi-label decoding and token-bounded inference.
* **License Notice:**
```text
MIT License

Copyright (c) 2023 deepset GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### 1.11 meta-llama/Llama-Prompt-Guard-2-86M (Prompt Guard 2 Standalone Baseline)
* **Repository / Source:** [meta-llama/Llama-Prompt-Guard-2-86M (Hugging Face)](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)
* **Model Version:** 86M parameters
* **Original License:** Llama 3 Community License
* **Copyright:** Copyright (c) Meta Platforms, Inc. and affiliates.
* **Target Stream:** `baseline evaluation stream` (`eval/baselines/`)
* **Use in PARALLAX:** Official standalone baseline evaluation of Meta Prompt Guard 2 without external fine-tuning (distinct from Appleroll's fine-tune inside PromptForest).
* **Modifications / Adaptations:** Zero weight modifications. Evaluated live using Hugging Face `transformers` pipeline (`AutoModelForSequenceClassification`, `AutoTokenizer`) under `scripts/gpu_lock.py`.
* **Gated Status:** Approved (Meta Community license granted).
* **License Notice:**
```text
Llama 3 Community License Agreement
Copyright (c) Meta Platforms, Inc. All Rights Reserved.
```

---

## 2. Wave G Candidate Detector Hub & License Audit Ledger

In Wave G, 4 ungated candidate models were evaluated on the PARALLAX DEV split (54 samples: 30 canary attacks, 24 benign tasks) to select the optimal replacement for the permanently lost upstream Vijil Dome detector:

| Model ID | License | Hub Gating | DEV PR-AUC | DEV F1 | DEV Recall | Latency p50 | Selection Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `deepset/deberta-v3-base-injection` | **MIT** | **Ungated (False)** | **1.0000** | **0.8955** | **1.0000** | 23.0 ms | **SELECTED (Top candidate: highest F1 & recall, permissive MIT)** |
| `protectai/deberta-v3-base-prompt-injection-v2` | Apache-2.0 | Ungated (False) | 1.0000 | 0.7755 | 0.6333 | 21.9 ms | Rejected for third detector (already integrated as D2 in PARALLAX) |
| `bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1` | AGPL-3.0 | Ungated (False) | 0.9938 | 0.7755 | 0.6333 | 23.9 ms | Rejected (AGPL copyleft license restriction; lower F1) |
| `hlyn-labs/prompt-injection-judge-deberta-70m` | CC-BY-NC-4.0 | Gated (auto) | 0.9845 | 0.6957 | 0.5333 | 23.0 ms | Rejected (Gated auto-agreement required; non-commercial CC-BY-NC) |

Evaluation artifacts persisted at `eval/detector_selection/dev_candidate_evaluation.json`.

