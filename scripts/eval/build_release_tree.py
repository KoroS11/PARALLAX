#!/usr/bin/env python3
"""Build and audit the PARALLAX release tree in an isolated temporary directory.

Performs:
1. `git archive` of current HEAD to a clean temp directory.
2. Applies exclusions from `scripts/eval/release_exclusions.txt`.
3. Runs `scripts/eval/scan_release_tree.py` on the temp tree.
4. Runs `ruff check .` inside the temp tree.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    temp_base = Path(os.environ.get("TEMP", os.environ.get("TMP", "/tmp")))
    release_dir = temp_base / "parallax_release"

    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    head_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
        .decode("utf-8")
        .strip()
    )
    print(f"[build_release] Archiving HEAD commit {head_commit}...")

    tar_path = temp_base / "parallax_release.tar"
    subprocess.check_call(
        ["git", "archive", "--format=tar", "-o", str(tar_path), "HEAD"],
        cwd=str(REPO_ROOT),
    )

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=release_dir)
    if tar_path.exists():
        tar_path.unlink()

    print(f"[build_release] Extracted release tree to {release_dir}")

    # Read exclusions
    exclusions_path = REPO_ROOT / "scripts" / "eval" / "release_exclusions.txt"
    if exclusions_path.exists():
        with open(exclusions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                target = release_dir / line
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                    print(f"  [EXCLUDED] Removed {line} from release tree")

    # Run scan_release_tree.py
    print("\n--- Running Release Safety Scanner ---")
    scanner_path = REPO_ROOT / "scripts" / "eval" / "scan_release_tree.py"
    res = subprocess.run(
        [sys.executable, str(scanner_path), str(release_dir)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if res.returncode != 0:
        print(f"[build_release] Scanner FAILED with code {res.returncode}")
        return res.returncode

    # Run ruff check in the release dir
    print("\n--- Running Ruff in Release Tree ---")
    ruff_res = subprocess.run(["uv", "run", "ruff", "check", "."], cwd=str(release_dir), check=False)
    if ruff_res.returncode != 0:
        print(f"[build_release] Ruff check FAILED with code {ruff_res.returncode}")
        return ruff_res.returncode

    print("\n[build_release] All release tree checks PASSED successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
