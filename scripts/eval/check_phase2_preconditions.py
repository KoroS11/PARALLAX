#!/usr/bin/env python3
"""Phase 2 Preconditions Verification Check (scripts/eval/check_phase2_preconditions.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.eval_external import (
    verify_dolly_text_integrity,
    verify_frozen_manifest,
    verify_protocol_digests,
    verify_runner_config,
)
from scripts.eval.policy_replay_check import run_policy_replay_check

BASE_COMMIT = "6d66e5e2199accf689fe981c830a3aacb9268dbf"


def check_all_preconditions() -> bool:
    print("=" * 80)
    print("PHASE 2 PRECONDITIONS VERIFICATION")
    print("=" * 80)

    # 1. K13_SCORE.txt exists
    k13_score = PROJECT_ROOT / "docs" / "K13_SCORE.txt"
    if not k13_score.exists():
        print(f"FAIL: {k13_score} does not exist")
        return False
    print(f"PASS: {k13_score.name} exists ({k13_score.stat().st_size} bytes)")

    # 2. git diff --stat touches only allowed directories
    cmd = ["git", "diff", "--name-only", BASE_COMMIT, "HEAD"]
    res = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True
    )
    changed_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    allowed_prefixes = ("docs/", "scripts/eval/", "tests/", "export/")
    disallowed = [f for f in changed_files if not f.startswith(allowed_prefixes)]
    if disallowed:
        print(f"FAIL: Disallowed files changed since base commit: {disallowed}")
        return False
    print(
        f"PASS: All {len(changed_files)} changed files since base commit are within allowed dirs ({allowed_prefixes})"
    )

    # 3. Working tree clean
    res_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    uncommitted = res_status.stdout.strip()
    if uncommitted:
        print(f"FAIL: Working tree not clean:\n{uncommitted}")
        return False
    print("PASS: Working tree is clean")

    # 4. Protocol digests verify
    digests = verify_protocol_digests()
    print(f"PASS: Runner and metrics hashes match protocol ({digests})")

    # 5. Policy replay check 347/347
    replay_results = run_policy_replay_check()
    all_347 = all(count == 347 for count in replay_results.values())
    if not all_347:
        print(f"FAIL: Policy replay results: {replay_results}")
        return False
    print(f"PASS: 5-arm policy replay succeeded on 347/347 rows ({replay_results})")

    # 6. Config check
    checks = verify_runner_config()
    mismatches = [name for name, _, _, ok in checks if not ok]
    if mismatches:
        print(f"FAIL: Runner config mismatches: {mismatches}")
        return False
    print(
        f"PASS: All {len(checks)} runner configuration parameters match protocol pins"
    )

    # 7. Ollama ID check
    ollama_check = next(c for c in checks if c[0] == "Ollama Model ID")
    if ollama_check[1] != "a2af6cc3eb7f":
        print(f"FAIL: Ollama ID mismatch: {ollama_check[1]} != a2af6cc3eb7f")
        return False
    print(f"PASS: Ollama gemma3:4b ID is {ollama_check[1]}")

    # 8. Frozen manifest verify
    manifest = verify_frozen_manifest()
    print(f"PASS: Frozen external benchmark manifest verified ({len(manifest)} files)")

    # 9. Dolly text integrity
    dolly_rows = verify_dolly_text_integrity()
    print(f"PASS: Dolly prompt text integrity verified ({dolly_rows} rows)")

    print("=" * 80)
    print("ALL PHASE 2 PRECONDITIONS MET SUCCESSFULLY")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = check_all_preconditions()
    sys.exit(0 if success else 1)
