"""Review diagnostic script: Judge recall on marker vs marker-free attacks.

Computes judge recall partitioned by whether attacks contain synthetic markers.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path


def main() -> None:
    log_path = Path("eval/logs/latest_judge_gemma3_4b.jsonl")
    if not log_path.exists():
        log_path = Path("eval/logs/sample_judge_gemma3_4b.jsonl")

    rows: dict[str, dict] = {}
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["prompt_id"]] = r

    texts: dict[str, str] = {}
    for fn in (
        "data/splits/dev.jsonl",
        "data/splits/held_out.jsonl",
        "data/external_eval/external_benchmark.jsonl",
    ):
        p = Path(fn)
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    pid = d.get("id") or d.get("prompt_id")
                    if pid and "text" in d:
                        texts[pid] = d["text"]

    c: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    for pid, r in rows.items():
        if r.get("label") != 1 or pid not in texts:
            continue
        marker = bool(re.search(r"CANARY|mock_\w+\(", texts[pid]))
        key = ("marker" if marker else "no-marker", pid.split("-")[0])
        c[key][1] += 1
        c[key][0] += r.get("parsed_verdict") == "block"

    for k in sorted(c):
        print(f"{k} judge blocked {c[k][0]}/{c[k][1]}")

    tot: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for (m, _), (b, n) in c.items():
        tot[m][0] += b
        tot[m][1] += n

    for m, (b, n) in tot.items():
        print(f"{m} recall {b}/{n} = {b / max(1, n):.2f}")


if __name__ == "__main__":
    main()
