"""Review diagnostic script: Marker prevalence across all datasets and sources.

Evaluates CANARY, mock_tool(), [Action:, and override patterns across benchmark sets.
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

    label: dict[str, int] = {}
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    label[r["prompt_id"]] = r.get("label", 0)

    recs: dict[str, dict] = {}
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
                    if pid:
                        recs[pid] = d

    if not recs:
        print("No prompt records found.")
        return

    first = next(iter(recs.values()))
    print("record keys:", list(first.keys()))

    def text(d: dict) -> str:
        for k in ("text", "prompt", "input", "content"):
            if k in d:
                return str(d[k])
        return json.dumps(d)

    pats = {
        "CANARY": r"CANARY",
        "mock_tool()": r"mock_\w+\(",
        "[Action:": r"\[Action:",
        "override/ignore": r"override|ignore (all )?previous",
    }
    tab: dict[str, collections.Counter] = {name: collections.Counter() for name in pats}
    tot = collections.Counter()
    for pid, d in recs.items():
        y = label.get(pid, 1 if d.get("is_attack") else 0)
        tot[y] += 1
        t = text(d)
        for name, p in pats.items():
            if re.search(p, t, re.IGNORECASE):
                tab[name][y] += 1

    print("totals (label 0 = benign, 1 = attack):", dict(tot))
    for name in pats:
        print(
            f"{name:18} benign {tab[name][0]:3d}/{tot[0]}   attack {tab[name][1]:3d}/{tot[1]}"
        )

    # also by id prefix, to see which attack sources carry canaries
    src: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for pid, d in recs.items():
        y = label.get(pid, 1 if d.get("is_attack") else 0)
        if y == 1:
            s = pid.split("-")[0]
            src[s][1] += 1
            src[s][0] += bool(re.search("CANARY", text(d)))
    print("attacks with CANARY by source:", {k: tuple(v) for k, v in src.items()})


if __name__ == "__main__":
    main()
