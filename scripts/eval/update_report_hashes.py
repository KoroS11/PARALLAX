#!/usr/bin/env python3
"""Script to update stale commit hashes in docs/OVERNIGHT_REPORT.md."""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_PATH = REPO_ROOT / "docs" / "OVERNIGHT_REPORT.md"


def get_current_head() -> str:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def main():
    head_commit = get_current_head()
    print(f"Current git HEAD: {head_commit}")

    content = REPORT_PATH.read_text(encoding="utf-8")

    # Replace CURRENT HEAD COMMIT
    new_content, count1 = re.subn(
        r">\s*\*\*CURRENT HEAD COMMIT\*\*:\s*`[0-9a-fA-F]{40}`",
        f"> **CURRENT HEAD COMMIT**: `{head_commit}`",
        content,
    )

    # Replace Pre-commit HEAD
    new_content, count2 = re.subn(
        r"-\s*\*\*Pre-commit HEAD\*\*:\s*`[0-9a-fA-F]{40}`",
        f"- **Pre-commit HEAD**: `{head_commit}`",
        new_content,
    )

    print(
        f"Replaced {count1} CURRENT HEAD COMMIT and {count2} Pre-commit HEAD entries."
    )
    REPORT_PATH.write_text(new_content, encoding="utf-8")
    print(f"Updated {REPORT_PATH}")


if __name__ == "__main__":
    main()
