# PARALLAX Visualization & Dashboard Export Schema (v1.0.0)

## 1. Overview

`docs/VIZ_SCHEMA.md` specifies the canonical, stable JSON export schema for PARALLAX evaluation visualizations and dashboard reporting. It is consumed by `eval/export/*.json` (including `eval/export/pilot_viz_data.json` and final benchmark exports).

The schema decouples UI/dashboard development from backend benchmarking runs. It features explicit `status` flags (`pilot`, `preliminary`, `measured`) per table and metric, strict decoupling of timeouts from model abstentions, and before/after corpus audit statistics.

---

## 2. Schema Specification (JSON Schema)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ParallaxVizDataExport",
  "type": "object",
  "required": [
    "schema_version",
    "generated_at",
    "status",
    "metadata",
    "methodology_timeline",
    "corpus_audit",
    "detector_layer",
    "arms",
    "judge_confusion",
    "adaptive_attacks"
  ],
  "properties": {
    "schema_version": { "type": "string", "enum": ["1.0.0"] },
    "generated_at": { "type": "string", "format": "date-time" },
    "status": {
      "type": "string",
      "enum": ["pilot", "preliminary", "measured"],
      "description": "Global status flag for current dataset payload."
    },
    "metadata": {
      "type": "object",
      "required": ["wave", "run_id", "git_commit", "split_evaluated", "sample_size", "power_status"],
      "properties": {
        "wave": { "type": "string" },
        "run_id": { "type": "string" },
        "git_commit": { "type": "string" },
        "split_evaluated": { "type": "string" },
        "sample_size": { "type": "integer" },
        "power_status": { "type": "string", "enum": ["underpowered_pilot", "adequately_powered"] },
        "hardware": {
          "type": "object",
          "properties": {
            "ram_gb": { "type": "number" },
            "vram_gb": { "type": "number" },
            "gpu_device": { "type": "string" }
          }
        }
      }
    },
    "methodology_timeline": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["wave", "status", "summary", "flaw_or_discovery"],
        "properties": {
          "wave": { "type": "string" },
          "status": { "type": "string" },
          "summary": { "type": "string" },
          "flaw_or_discovery": { "type": "string" }
        }
      }
    },
    "corpus_audit": {
      "type": "object",
      "required": ["wave_h2_pilot", "target_wave_h3"],
      "properties": {
        "wave_h2_pilot": {
          "type": "object",
          "required": ["dev_size", "val_size", "held_out_size", "attack_sources", "benign_sources", "source_label_confounding", "gate_power"],
          "properties": {
            "dev_size": { "type": "integer" },
            "val_size": { "type": "integer" },
            "held_out_size": { "type": "integer" },
            "attack_sources": { "type": "array", "items": { "type": "string" } },
            "benign_sources": { "type": "array", "items": { "type": "string" } },
            "source_label_confounding": { "type": "boolean" },
            "gate_power": { "type": "string" },
            "shortcut_aucs": { "type": "object" }
          }
        },
        "target_wave_h3": {
          "type": "object",
          "required": ["dev_min", "val_min", "held_out_min", "sources_per_class_min", "gate_power_rule"],
          "properties": {
            "dev_min": { "type": "integer" },
            "val_min": { "type": "integer" },
            "held_out_min": { "type": "integer" },
            "sources_per_class_min": { "type": "integer" },
            "gate_power_rule": { "type": "string" }
          }
        }
      }
    },
    "detector_layer": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["auc", "attack_median", "benign_median", "benign_saturation_pct", "status", "diagnosis"],
        "properties": {
          "auc": { "type": "number" },
          "auc_ci_95": { "type": "array", "items": { "type": "number" } },
          "attack_median": { "type": "number" },
          "benign_median": { "type": "number" },
          "flag_rate_05_attack": { "type": "number" },
          "flag_rate_05_benign": { "type": "number" },
          "benign_saturation_pct": { "type": "number" },
          "status": { "type": "string", "enum": ["stale_saturated", "stale_inverted", "stale_dormant", "active_calibrated"] },
          "diagnosis": { "type": "string" }
        }
      }
    },
    "arms": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": [
          "display_name", "role", "recall", "recall_ci_95", "fbr", "fbr_ci_95",
          "operational_utility", "extra_calls_per_100", "latency_p50_ms", "latency_p95_ms",
          "status", "decisions_by_class"
        ],
        "properties": {
          "display_name": { "type": "string" },
          "role": { "type": "string" },
          "recall": { "type": "number" },
          "recall_ci_95": { "type": "array", "items": { "type": "number" } },
          "fbr": { "type": "number" },
          "fbr_ci_95": { "type": "array", "items": { "type": "number" } },
          "operational_utility": { "type": "number" },
          "extra_calls_per_100": { "type": "number" },
          "latency_p50_ms": { "type": ["number", "null"] },
          "latency_p95_ms": { "type": ["number", "null"] },
          "status": { "type": "string", "enum": ["pilot", "preliminary", "measured"] },
          "decisions_by_class": {
            "type": "object",
            "required": ["attack", "benign"],
            "properties": {
              "attack": {
                "type": "object",
                "required": ["allow", "flag", "block", "model_abstain", "timeout", "parse_fail", "error"],
                "properties": {
                  "allow": { "type": "integer" },
                  "flag": { "type": "integer" },
                  "block": { "type": "integer" },
                  "model_abstain": { "type": "integer" },
                  "timeout": { "type": "integer" },
                  "parse_fail": { "type": "integer" },
                  "error": { "type": "integer" }
                }
              },
              "benign": {
                "type": "object",
                "required": ["allow", "flag", "block", "model_abstain", "timeout", "parse_fail", "error"],
                "properties": {
                  "allow": { "type": "integer" },
                  "flag": { "type": "integer" },
                  "block": { "type": "integer" },
                  "model_abstain": { "type": "integer" },
                  "timeout": { "type": "integer" },
                  "parse_fail": { "type": "integer" },
                  "error": { "type": "integer" }
                }
              }
            }
          }
        }
      }
    },
    "judge_confusion": {
      "type": "object",
      "required": ["model", "timeout_cap_ms", "splits", "abstain_vs_timeout_latency"],
      "properties": {
        "model": { "type": "string" },
        "timeout_cap_ms": { "type": "number" },
        "splits": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "required": ["attack_block", "attack_allow", "attack_abstain", "attack_timeout", "benign_allow", "benign_flag", "benign_block", "benign_abstain", "benign_timeout"]
          }
        },
        "abstain_vs_timeout_latency": {
          "type": "object",
          "required": ["timeouts", "model_abstains", "non_abstain"],
          "properties": {
            "timeouts": { "type": "object" },
            "model_abstains": { "type": "object" },
            "non_abstain": { "type": "object" }
          }
        }
      }
    },
    "adaptive_attacks": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["asr_by_budget", "concurrent_benign_fbr", "campaigns", "status"],
        "properties": {
          "asr_by_budget": {
            "type": "object",
            "properties": {
              "b1": { "type": "number" },
              "b5": { "type": "number" },
              "b10": { "type": "number" },
              "b20": { "type": "number" }
            }
          },
          "concurrent_benign_fbr": { "type": "number" },
          "campaigns": { "type": "integer" },
          "status": { "type": "string" }
        }
      }
    }
  }
}
```

---

## 3. Field Definitions & Integrity Constraints

1. **`status` Hierarchy**:
   - `pilot`: $N < 100$, illustrative diagnostic run.
   - `preliminary`: Historical metric from contaminated or simulated runs.
   - `measured`: High-power benchmark ($n \ge 120/150$), pre-registered, backed by confirmed per-prompt traces.
2. **Outcome Decoupling**:
   - The UI must render separate counters for `model_abstain` (model emitted abstain syntax), `timeout` (hit client-side cutoff, e.g. $\sim 15.3\text{ s}$), and `parse_fail` (unparseable output).
3. **Decisions by Class**:
   - All defensive evaluations must break down decisions into full distributions across `attack` and `benign` inputs.
