"""Generate structured per-run manifests in eval/logs/manifests/<run_id>.json."""

from __future__ import annotations

import collections
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "eval" / "logs"
MANIFESTS_DIR = LOGS_DIR / "manifests"
WORKTREE_DIR = Path(
    "C:/Users/harsh/.gemini/antigravity-cli/worktrees/cf3fb819-21cf-4cb6-af84-fd9fb8ccc8b4/subagent-LLM-Judge-Specialist-judge-fix-a8508f6a/eval/logs"
)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def process_run_dir(run_dir: Path) -> dict[str, Any] | None:
    run_id = run_dir.name
    jsonl_files = list(run_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "files": [],
        "total_records": 0,
        "split_counts": {},
        "model_digests": {},
        "template_hashes": {},
    }

    total_splits: collections.Counter[str] = collections.Counter()

    for jf in jsonl_files:
        sha256 = compute_sha256(jf)
        size_bytes = jf.stat().st_size
        records: list[dict[str, Any]] = []
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        split_counts = collections.Counter(r.get("split", "unknown") for r in records)
        total_splits.update(split_counts)

        # Extract model digest and template hash if present
        for r in records[:5]:
            if "model_digest" in r:
                manifest["model_digests"][r.get("model", "unknown")] = r["model_digest"]
            if "template_hash" in r:
                manifest["template_hashes"][r.get("model", "unknown")] = r[
                    "template_hash"
                ]

        manifest["files"].append(
            {
                "filename": jf.name,
                "rel_path": str(
                    jf.relative_to(
                        LOGS_DIR if LOGS_DIR in jf.parents else jf.parent.parent
                    )
                ),
                "size_bytes": size_bytes,
                "sha256": sha256,
                "num_records": len(records),
                "split_counts": dict(split_counts),
            }
        )
        manifest["total_records"] += len(records)

    manifest["split_counts"] = dict(total_splits)
    return manifest


def main() -> None:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # Search local eval/logs and subagent worktree eval/logs
    candidate_dirs: list[Path] = []
    if LOGS_DIR.exists():
        candidate_dirs.extend(
            [
                p
                for p in LOGS_DIR.iterdir()
                if p.is_dir() and p.name.startswith("judge-run-")
            ]
        )
    if WORKTREE_DIR.exists():
        candidate_dirs.extend(
            [
                p
                for p in WORKTREE_DIR.iterdir()
                if p.is_dir() and p.name.startswith("judge-run-")
            ]
        )

    seen_runs: set[str] = set()
    count = 0

    for d in sorted(candidate_dirs):
        if d.name in seen_runs:
            continue
        manifest = process_run_dir(d)
        if manifest:
            out_file = MANIFESTS_DIR / f"{manifest['run_id']}.json"
            out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"Wrote manifest: {out_file.name} ({manifest['total_records']} records across splits: {manifest['split_counts']})"
            )
            seen_runs.add(d.name)
            count += 1

    print(f"\nGenerated {count} per-run manifests in {MANIFESTS_DIR}.")


if __name__ == "__main__":
    main()
