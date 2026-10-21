"""Review diagnostic script: Benign block analysis & canary/mock-tool leakage.

Analyzes in-house false blocks and computes marker hit rates across attack vs benign.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def main() -> None:
    log_path = Path("eval/logs/latest_judge_gemma3_4b.jsonl")
    if not log_path.exists():
        log_path = Path("eval/logs/sample_judge_gemma3_4b.jsonl")

    if not log_path.exists():
        print("No judge logs found.")
        return

    with open(log_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    seen: set[str] = set()
    uniq: list[dict] = []
    for r in rows:
        pid = r.get("prompt_id") or ""
        if pid not in seen:
            seen.add(pid)
            uniq.append(r)
    print("unique prompts:", len(uniq))

    # benign prompts the judge blocked (in-house)
    bad = [
        r
        for r in uniq
        if r.get("label") == 0
        and str(r.get("parsed_verdict")).lower() == "block"
        and r.get("split") in ("dev", "held_out")
    ]
    print("benign blocked in-house:", len(bad))
    for r in bad[:6]:
        try:
            out = json.loads(r.get("raw_output", "{}"))
            why = str(out.get("reasoning", ""))[:300]
        except (json.JSONDecodeError, KeyError):
            why = str(r.get("raw_output", ""))[:300]
        print(f"- {r.get('prompt_id')} | {why}")

    # canary / mock-tool leak check: does the string appear by label?
    prompts: dict[str, dict] = {}
    for fn in ("data/splits/dev.jsonl", "data/splits/held_out.jsonl"):
        p = Path(fn)
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    prompts[d.get("id") or d.get("prompt_id")] = d

    for pat in (r"CANARY", r"mock_\w+\(", r"override|ignore previous"):
        c = {0: [0, 0], 1: [0, 0]}
        for d in prompts.values():
            y = 1 if d.get("is_attack", d.get("label")) else 0
            txt = d.get("text") or d.get("prompt") or ""
            c[y][0] += bool(re.search(pat, txt, re.IGNORECASE))
            c[y][1] += 1
        print(
            f"{pat} benign hits {c[0][0]}/{c[0][1]} | attack hits {c[1][0]}/{c[1][1]}"
        )


if __name__ == "__main__":
    main()
