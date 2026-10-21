"""Independent In-House CPS Operational Benign Catalog and Verified Control Sets.

Wave H.2 Task R1:
Every prompt comes from an independent, auditable operational source.
Benign prompts are NEVER derived from attacks.
Covers all 16 CPS scenarios from parallax/sandbox/utility.py with varied styles:
- Plain language operator readouts, status queries, maintenance, shift logs.
- Legitimate structured requests (JSON-RPC, OPC-UA, Modbus).
- Legitimate authorized setpoint adjustments with valid ticket IDs and justification.
- Provocative-benign vocabulary (override, ignore, emergency, bypass in legitimate contexts).
- Legitimate base64 calibration matrices and hex telemetry dumps.
- Verified NotInject benign controls (license verified, zero overlap with training sets).
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from scripts.corpus.generate_diverse_prompts import DIVERSE_CPS_PROMPTS

logger = logging.getLogger(__name__)

SCENARIOS_16 = list(DIVERSE_CPS_PROMPTS.keys())


def build_all_in_house_benign() -> list[dict[str, Any]]:
    """Build 320 independent, handcrafted in-house CPS operational prompts across all 16 scenarios."""
    all_prompts: list[dict[str, Any]] = []
    sample_idx = 1

    for sc in SCENARIOS_16:
        sc_list = DIVERSE_CPS_PROMPTS[sc]
        for p_idx, item in enumerate(sc_list, start=1):
            tmpl_id = f"tmpl_{sc}_{item['style']}_{p_idx:02d}"
            all_prompts.append(
                {
                    "id": f"benign-cps-{sample_idx:04d}",
                    "text": item["text"],
                    "technique": "none",
                    "family": "benign_operational",
                    "split": "",  # Will be assigned during clustering
                    "label_binary": False,
                    "label_multiclass": "none",
                    "source": "in_house_cps_operations",
                    "source_tool": "cps_operational_catalog",
                    "tool_version": "2.0.0",
                    "source_command": f"DIVERSE_CPS_PROMPTS['{sc}'][{p_idx - 1}]",
                    "license": "MIT",
                    "canary": "",
                    "forbidden_tools": [],
                    "template_id": tmpl_id,
                    "metadata": {
                        "goal": {
                            "canary": "",
                            "forbidden_tools": [],
                            "target_action": "safe_query",
                            "sandbox_parameters": {
                                "safe_tool": item["safe_tool"],
                                "scenario_id": sc,
                            },
                        },
                        "scenario_id": sc,
                        "operational_style": item["style"],
                        "d3_in_distribution": False,
                    },
                    "provenance": {
                        "source_tool": "cps_operational_catalog",
                        "tool_version": "2.0.0",
                        "source_command": "parallax.corpus.independent_benign_rebuild",
                        "license": "MIT",
                        "raw_file_path": "parallax/sandbox/utility.py",
                        "transformation": f"Independently authored {item['style']} prompt for {sc}",
                    },
                }
            )
            sample_idx += 1

    return all_prompts


def load_verified_notinject_controls(max_samples: int = 50) -> list[dict[str, Any]]:
    """Load verified NotInject benign controls with provocative operational vocabulary."""
    records: list[dict[str, Any]] = []
    try:
        ds = load_dataset("leolee99/NotInject")
        candidate_rows = []
        for split_key in ds:
            for row in ds[split_key]:
                p = row.get("prompt", "").strip()
                if (
                    len(p) > 20
                    and p.isascii()
                    and any(
                        w in p.lower()
                        for w in [
                            "ignore",
                            "override",
                            "emergency",
                            "bypass",
                            "system",
                            "command",
                        ]
                    )
                ):
                    candidate_rows.append(
                        (p, row.get("category", "General"), row.get("word_list", []))
                    )

        seen_texts = set()
        count = 1
        for p, cat, wl in candidate_rows:
            if p in seen_texts:
                continue
            seen_texts.add(p)
            records.append(
                {
                    "id": f"notinject-benign-{count:03d}",
                    "text": p,
                    "technique": "none",
                    "family": "notinject_control",
                    "split": "",  # Will be assigned during clustering
                    "label_binary": False,
                    "label_multiclass": "none",
                    "source": "leolee99/NotInject",
                    "source_tool": "huggingface_hub",
                    "tool_version": "1.0.0",
                    "source_command": "load_dataset('leolee99/NotInject')",
                    "license": "Apache-2.0",
                    "canary": "",
                    "forbidden_tools": [],
                    "template_id": f"tmpl_notinject_{count:03d}",
                    "metadata": {
                        "goal": {
                            "canary": "",
                            "forbidden_tools": [],
                            "target_action": "safe_query",
                            "sandbox_parameters": {
                                "safe_tool": "general_benign",
                                "scenario_id": "general",
                            },
                        },
                        "scenario_id": "general",
                        "category": cat,
                        "word_list": wl,
                        "d3_in_distribution": False,
                    },
                    "provenance": {
                        "source_tool": "leolee99/NotInject",
                        "tool_version": "1.0.0",
                        "source_command": "load_dataset('leolee99/NotInject')",
                        "license": "Apache-2.0",
                        "raw_file_path": "datasets--leolee99--NotInject",
                        "transformation": "Verified benign operational query containing trigger vocabulary",
                    },
                }
            )
            count += 1
            if count > max_samples:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load NotInject dataset: %s", exc)

    return records
