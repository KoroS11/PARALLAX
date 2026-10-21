#!/usr/bin/env python3
"""Create clean project submission zip using git archive.

Excludes untracked / ignored files automatically according to git tracking rules.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = REPO_ROOT / "export" / "closeout" / "parallax_submission.zip"


def make_submission_zip() -> Path:
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "archive", "--format=zip", "-o", str(OUTPUT_ZIP), "HEAD"]
    print(f"[make_submission_zip] Running: {' '.join(cmd)}")
    res = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        print(f"Error creating zip: {res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)

    file_size = OUTPUT_ZIP.stat().st_size
    sha256 = hashlib.sha256(OUTPUT_ZIP.read_bytes()).hexdigest()
    print(f"[make_submission_zip] Created {OUTPUT_ZIP} ({file_size} bytes)")
    print(f"[make_submission_zip] SHA-256: {sha256}")
    return OUTPUT_ZIP


if __name__ == "__main__":
    make_submission_zip()
