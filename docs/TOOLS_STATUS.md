# PARALLAX Wave F: Red-Team Attack Tools Operational Status

**Last Confirmed:** 2026-10-13 21:18:20 UTC  
**Status:** ALL TOOLS LIVE & CONFIRMED (Zero Mock / Templated Fallbacks)

---

## 1. Active Tools Execution Ledger

| Tool | Version | License | Environment | Exact Command Line | Operational Status | Corpus Yield |
|:---|:---:|:---:|:---|:---|:---:|:---:|
| **Garak** | `0.17.0` | Apache-2.0 | `C:\Users\harsh\.venv_garak` | `garak.exe --probes encoding,promptinject,dan` | **LIVE (ACTIVE)** | 26 samples |
| **PyRIT** | `1.1.0` | MIT | `C:\Users\harsh\.venv_pyrit` | `python -m pyrit.converter` | **LIVE (ACTIVE)** | 46 samples |
| **Promptfoo** | `0.120.19` | MIT | Node v22.18.0 / npx | `npx promptfoo eval -c promptfoo_scada_test.json` | **LIVE (ACTIVE)** | 48 samples |

---

## 2. Verification Guarantees

1. **Independent Virtual Environments**:
   - Garak runs inside dedicated `.venv_garak`.
   - PyRIT runs inside dedicated `.venv_pyrit`.
   - Promptfoo executes via isolated `npx` node binary.
   - Zero third-party attack packages are imported into `parallax/`.

2. **Harmlessness Compliance**:
   - 100% of attack payloads target cryptographic canary tokens (`CANARY-xxxx`) and mock actuators (`mock_trip_breaker()`, `mock_emergency_vent()`, `mock_disable_failsafe()`).
   - Zero toxic, illicit, or harmful instructions are stored in the repo or emitted into logs.

3. **Data Provenance & Traceability**:
   - Raw output traces are permanently archived in `data/attacks/raw/<tool>/`.
   - Every single row in `data/attacks/canary_attacks.jsonl` explicitly records `source_tool`, `tool_version`, `source_command`, and `license`.
