#!/usr/bin/env python3
"""Check all tracked files under export/, docs/, and tests/ for Dolly prompt prefixes."""

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOLLY_PATH = REPO_ROOT / "data" / "external" / "dolly_text.jsonl"


def main():
    dolly_prefixes = []
    with open(DOLLY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                txt = row.get("text", "").strip()
                if len(txt) >= 30:
                    dolly_prefixes.append(txt[:30])
                elif txt:
                    dolly_prefixes.append(txt)

    print(f"Loaded {len(dolly_prefixes)} Dolly prompt prefixes (length >= 30).")

    res = subprocess.run(
        ["git", "ls-files", "export/", "docs/", "tests/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    print(
        f"Scanning {len(tracked_files)} tracked files under export/, docs/, tests/..."
    )

    matches = []
    scanned_text_files = 0
    for fpath_rel in tracked_files:
        fpath = REPO_ROOT / fpath_rel
        if not fpath.exists():
            continue
        # Skip binary image files
        if fpath.suffix.lower() in [".png", ".jpg", ".jpeg", ".zip"]:
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            scanned_text_files += 1
        except (OSError, UnicodeError):
            continue
        for prefix in dolly_prefixes:
            if prefix in content:
                matches.append((fpath_rel, prefix))

    print(f"Scanned {scanned_text_files} text files.")
    print(f"Total Dolly text matches found: {len(matches)}")
    for f, p in matches:
        print(f"  Match in {f}: '{p}'")


if __name__ == "__main__":
    main()
