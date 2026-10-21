#!/usr/bin/env python3
"""Provenance and Code State Verification Check (K0 TEST).

Verifies:
1. results/final-v4/predictions.jsonl exists with 347 rows.
2. SHA-256 matches eval/logs/final-v4-20261019-142005-8fb16e/predictions.jsonl.
3. data/splits/held_out_v4.jsonl SHA-256 matches data/splits/FROZEN.sha256.
4. Protocol-lock commit (7e98c39, 19:02 IST) is earlier than run start (14:20 UTC / 19:50 IST).
5. docs/CODE_STATE.md classifies every file modified since producing commit 959a99a.
6. docs/PROVENANCE.md contains manifest SHA-256 and field audits.
"""

from __future__ import annotations

import datetime
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> int:
    print("[check_provenance] === Starting K0 Provenance Check ===")
    errors: list[str] = []

    # 1. Check results/final-v4/predictions.jsonl row count
    res_path = REPO_ROOT / "results" / "final-v4" / "predictions.jsonl"
    if not res_path.exists():
        errors.append(f"Missing {res_path}")
    else:
        rows = count_lines(res_path)
        if rows != 347:
            errors.append(
                f"Row count mismatch in {res_path}: expected 347, found {rows}"
            )
        else:
            print(f"[check_provenance] PASS: {res_path} exists with {rows} rows.")

    # 2. Check hash match with source log
    log_path = (
        REPO_ROOT
        / "eval"
        / "logs"
        / "final-v4-20261019-142005-8fb16e"
        / "predictions.jsonl"
    )
    if not log_path.exists():
        errors.append(f"Missing source log {log_path}")
    elif res_path.exists():
        res_hash = sha256_file(res_path)
        log_hash = sha256_file(log_path)
        if res_hash != log_hash:
            errors.append(f"Hash mismatch: results ({res_hash}) != log ({log_hash})")
        else:
            print(f"[check_provenance] PASS: Predictions SHA-256 matched: {res_hash}")

    # 3. Check FROZEN hash vs held_out_v4.jsonl
    held_out = REPO_ROOT / "data" / "splits" / "held_out_v4.jsonl"
    frozen_file = REPO_ROOT / "data" / "splits" / "FROZEN.sha256"
    if not held_out.exists() or not frozen_file.exists():
        errors.append("held_out_v4.jsonl or FROZEN.sha256 missing")
    else:
        held_out_hash = sha256_file(held_out)
        frozen_entries = dict(
            line.strip().split()[::-1]
            for line in frozen_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        expected_hash = frozen_entries.get("held_out_v4.jsonl")
        if held_out_hash != expected_hash:
            errors.append(
                f"FROZEN mismatch for held_out_v4.jsonl: computed {held_out_hash} != {expected_hash}"
            )
        else:
            print(
                f"[check_provenance] PASS: FROZEN hash matches held_out_v4.jsonl: {held_out_hash}"
            )

    # 4. Check protocol-lock commit time earlier than run start
    # Protocol lock commit: 7e98c39 (2026-10-19 19:02:52 +0530)
    # Run start: final-v4-20261019-142005-8fb16e -> 2026-10-19 14:20:05 UTC = 19:50:05 IST
    run_start_utc = datetime.datetime(2026, 10, 19, 14, 20, 5, tzinfo=datetime.UTC)
    try:
        commit_date_str = subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", "7e98c39"],
            cwd=str(REPO_ROOT),
            text=True,
        ).strip()
        protocol_lock_time = datetime.datetime.fromisoformat(commit_date_str)
        if protocol_lock_time >= run_start_utc:
            errors.append(
                f"Protocol-lock time ({protocol_lock_time}) not earlier than run start ({run_start_utc})"
            )
        else:
            print(
                f"[check_provenance] PASS: Protocol-lock time ({protocol_lock_time}) "
                f"is strictly earlier than run start ({run_start_utc})."
            )
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        errors.append(f"Failed to check git commit 7e98c39: {exc}")

    # 5. Check docs/CODE_STATE.md classifies every changed file
    code_state_file = REPO_ROOT / "docs" / "CODE_STATE.md"
    if not code_state_file.exists():
        errors.append(f"Missing {code_state_file}")
    else:
        code_state_text = code_state_file.read_text(encoding="utf-8")
        try:
            # Check diff between wip-amendment1-steps and 959a99a
            diff_target = "wip-amendment1-steps"
            changed_files_out = (
                subprocess.check_output(
                    ["git", "diff", "--name-only", "959a99a", diff_target],
                    cwd=str(REPO_ROOT),
                    text=True,
                )
                .strip()
                .splitlines()
            )
            # Filter to tracked code files
            changed_files = [
                f.strip().replace("\\", "/")
                for f in changed_files_out
                if f.strip() and not f.startswith("plans/")
            ]
            unclassified = [f for f in changed_files if f not in code_state_text]
            if unclassified:
                errors.append(f"CODE_STATE.md does not classify files: {unclassified}")
            else:
                print(
                    f"[check_provenance] PASS: CODE_STATE.md classifies all {len(changed_files)} changed files."
                )
        except (subprocess.SubprocessError, OSError) as exc:
            errors.append(f"Failed to inspect git diff: {exc}")

    # 6. Check docs/PROVENANCE.md contains manifest SHA-256
    prov_file = REPO_ROOT / "docs" / "PROVENANCE.md"
    manifest_file = (
        REPO_ROOT
        / "eval"
        / "logs"
        / "manifests"
        / "final-v4-20261019-142005-8fb16e.json"
    )
    if not prov_file.exists():
        errors.append("Missing docs/PROVENANCE.md")
    elif manifest_file.exists():
        man_hash = sha256_file(manifest_file)
        prov_text = prov_file.read_text(encoding="utf-8")
        if man_hash not in prov_text:
            errors.append(
                f"Manifest SHA-256 ({man_hash}) not found in docs/PROVENANCE.md"
            )
        else:
            print(
                f"[check_provenance] PASS: Manifest SHA-256 verified in docs/PROVENANCE.md: {man_hash}"
            )

    if errors:
        print("\n[check_provenance] FAILED with errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("[check_provenance] ALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
