#!/usr/bin/env python3
"""Prepare deterministic interleaved external evaluation dataset (seed 0)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_interleaved_dataset(seed: int = 0) -> list[dict[str, Any]]:
    gandalf_path = PROJECT_ROOT / "data" / "external" / "gandalf.jsonl"
    notinject_path = PROJECT_ROOT / "data" / "external" / "notinject.jsonl"
    dolly_text_path = PROJECT_ROOT / "data" / "external" / "dolly_text.jsonl"

    if not gandalf_path.exists():
        raise FileNotFoundError(f"Gandalf missing: {gandalf_path}")
    if not notinject_path.exists():
        raise FileNotFoundError(f"NotInject missing: {notinject_path}")
    if not dolly_text_path.exists():
        raise FileNotFoundError(f"Dolly text missing: {dolly_text_path}")

    # 1. Gandalf (n=112 attacks)
    gandalf_rows = []
    for line in gandalf_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gandalf_rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "label_binary": True,
                "family": "gandalf_ignore_instructions",
                "source": "gandalf",
                "similarity": r.get("similarity"),
            }
        )

    # 2. NotInject (n=309 benign)
    notinject_rows = []
    for line in notinject_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        notinject_rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "label_binary": False,
                "family": "notinject",
                "source": "notinject",
                "subset": r.get("subset"),
                "category": r.get("category"),
                "word_list": r.get("word_list"),
            }
        )

    # 3. Dolly (n=200 benign)
    dolly_rows = []
    for line in dolly_text_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        dolly_rows.append(
            {
                "id": f"dolly-{int(r['row_index']):05d}",
                "text": r["text"],
                "label_binary": False,
                "family": r["category"],
                "source": "dolly",
                "category": r["category"],
                "row_index": r["row_index"],
                "prompt_hash": r["prompt_hash"],
            }
        )

    assert len(gandalf_rows) == 112, f"Expected 112 Gandalf, got {len(gandalf_rows)}"
    assert len(notinject_rows) == 309, (
        f"Expected 309 NotInject, got {len(notinject_rows)}"
    )
    assert len(dolly_rows) == 200, f"Expected 200 Dolly, got {len(dolly_rows)}"

    all_rows = gandalf_rows + notinject_rows + dolly_rows
    assert len(all_rows) == 621, f"Expected 621 total, got {len(all_rows)}"

    # Deterministic interleaving using pinned seed 0
    rng = random.Random(seed)
    rng.shuffle(all_rows)

    return all_rows


def main():
    out_dir = PROJECT_ROOT / "export" / "closeout"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "external_input_interleaved.jsonl"

    rows = build_interleaved_dataset(seed=0)
    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    print(f"Successfully generated {len(rows)} interleaved rows to {out_file}")
    attack_count = sum(1 for r in rows if r["label_binary"])
    benign_count = sum(1 for r in rows if not r["label_binary"])
    print(f"Attack rows: {attack_count}, Benign rows: {benign_count}")


if __name__ == "__main__":
    main()
