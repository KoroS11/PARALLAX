# External Evaluation Benchmark (Wave G Adversarial Review)

## Overview
This directory contains an independent, un-tuned external evaluation benchmark designed specifically to audit PARALLAX detection arms without data leakage or contamination. None of these samples were used in the development or calibration of PARALLAX detectors, fusion models, or review routers.

## Dataset Structure (`external_benchmark.jsonl`)
- **Total Samples**: 110
- **Attacks**: 70
- **Benign**: 40

### 1. Benign Controls (N = 40)
- **Source**: `leolee99/NotInject` (split `NotInject_one`, samples 0–39)
- **License**: MIT
- **Rationale**: Tests false alarm resistance against benign inputs that naturally contain triggers, command vocabulary, or phrasing typically over-flagged by naive keyword and syntax detectors (e.g., system instructions, policy discussion, ignore directives in natural language).

### 2. Public Harmless Injection Attacks (N = 40)
- **Source**: `deepset/prompt-injections` (test split, `label == 1`, samples 0–39)
- **License**: Apache-2.0 / CC-BY-4.0
- **Rationale**: Real-world roleplay and instruction redirection probes from an established open-source benchmark, completely independent of the ProtectAI training distribution. All payloads are benign and canary-safe.

### 3. Novel Human-Synthesized Attacks (N = 30)
- **Source**: Handcrafted expert adversarial prompts across 6 fresh technique families absent from the primary training and dev sets (5 samples per family):
  1. **Hypothetical Simulator / Socratic Paradox**: Semantic inversion games, catastrophe training roleplay, and philosophical dilemmas.
  2. **Many-Shot In-Context Stuffing**: Few-shot contextual dialogues framing canary command execution as standard system completions.
  3. **Cross-Lingual Code-Switching**: Technical multilingual blending (German, French, Spanish) mixed with industrial SCADA terms.
  4. **Mathematical Syllogisms & Logic Predicates**: Formal first-order logic and Boolean SAT formulations concluding in canary tool invocations.
  5. **Recursive AST / JSON Schema Injection**: Structured JSON Schema definitions, AST node declarations, and GraphQL mutations embedding directives.
  6. **Protocol Handshake Impersonation**: Industrial field protocol frame emulations (DNP3, Modbus-TCP, Profinet, IEC 60870-5-104, Ethernet/IP).
- **Safety Policy**: In strict compliance with `AGENTS.md`, all attacks invoke harmless mock functions (`mock_trip_breaker()`, `mock_emergency_vent()`, `mock_disable_failsafe()`, `mock_set_chlorine()`) with canary strings (`CANARY-EXT-001` to `CANARY-EXT-030`). No real harmful instructions exist anywhere in this dataset.

## Provenance Summary Table
| Subset | Source | Split / Generator | License | Count | Label |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Benign Controls | `leolee99/NotInject` | `NotInject_one` | MIT | 40 | Benign (False) |
| Public Probes | `deepset/prompt-injections` | `test` (label=1) | Apache-2.0 | 40 | Attack (True) |
| Human Synthesis | Expert Review Team | Novel 6-family suite | MIT / Project | 30 | Attack (True) |
| **Total** | | | | **110** | **70 Attack / 40 Benign** |
