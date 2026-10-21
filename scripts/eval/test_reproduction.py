#!/usr/bin/env python3
"""Hermetic reproduction test for K7.

Clones branch 'closeout' to a temporary directory, runs uv sync and closeout_metrics.py,
and verifies that numbers.json matches byte-for-byte with the repository numbers.json.
Also verifies that git grep for secret tokens returns nothing.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_reproduction() -> None:
    print("[K7 TEST] Starting hermetic reproduction verification...")

    # 1. Check for real secret tokens across repo
    print("[K7 TEST] Scanning for secret tokens...")
    token_patterns = [
        r"hf_[a-zA-Z0-9]{34,}",
        r"ghp_[a-zA-Z0-9]{36,}",
        r"xox[baprs]-[a-zA-Z0-9]{10,48}",
        r"sk-[a-zA-Z0-9]{20,}",
    ]
    for pattern in token_patterns:
        proc = subprocess.run(
            ["git", "grep", "-E", pattern],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            print(
                f"[K7 TEST] FAIL: Found potential secret matching {pattern}:\n{proc.stdout}"
            )
            sys.exit(1)
    print("[K7 TEST] Secret token scan passed (0 secrets found).")

    # 2. Clone branch 'closeout' to a temporary directory
    original_numbers_path = REPO_ROOT / "numbers.json"
    if not original_numbers_path.exists():
        print(
            f"[K7 TEST] FAIL: Original numbers.json not found at {original_numbers_path}"
        )
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "parallax_clone"
        print(f"[K7 TEST] Cloning closeout branch to {tmp_path}...")

        clone_proc = subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "closeout",
                "--single-branch",
                str(REPO_ROOT),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone_proc.returncode != 0:
            print(f"[K7 TEST] FAIL: git clone failed:\n{clone_proc.stderr}")
            sys.exit(1)

        # Run closeout_metrics.py in clone
        print(
            "[K7 TEST] Running 'uv run python scripts/eval/closeout_metrics.py' in clone..."
        )
        run_proc = subprocess.run(
            ["uv", "run", "python", "scripts/eval/closeout_metrics.py"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if run_proc.returncode != 0:
            print(
                f"[K7 TEST] FAIL: closeout_metrics.py failed in clone:\n{run_proc.stderr}"
            )
            sys.exit(1)

        cloned_numbers_path = tmp_path / "numbers.json"
        if not cloned_numbers_path.exists():
            print("[K7 TEST] FAIL: numbers.json was not generated in clone.")
            sys.exit(1)

        # Byte-for-byte comparison
        orig_bytes = original_numbers_path.read_bytes()
        clone_bytes = cloned_numbers_path.read_bytes()

        if orig_bytes == clone_bytes:
            print(
                f"[K7 TEST] PASS: numbers.json matches byte-for-byte ({len(orig_bytes)} bytes)."
            )
        else:
            print(
                f"[K7 TEST] FAIL: numbers.json mismatch (orig: {len(orig_bytes)} bytes, clone: {len(clone_bytes)} bytes)."
            )
            sys.exit(1)

    print("[K7 TEST] ALL REPRODUCTION CHECKS PASSED.")


if __name__ == "__main__":
    test_reproduction()
