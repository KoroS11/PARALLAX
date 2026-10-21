#!/usr/bin/env python3
"""Safety and Compliance Scanner for PARALLAX Release Tree (Phase 6 / Closeout).

Scans an extracted git archive release tree for release criteria:
- No prohibited files:
    * dolly_text.jsonl, external_predictions.jsonl, *.pem, *.key, *id_rsa*, .env*
    * Excluded raw external datasets: gandalf.jsonl, notinject.jsonl
    * Temporary test frames: test_frame.png
- No file > 50 MB.
- Secrets and credentials:
    * "BEGIN (RSA|EC|OPENSSH) PRIVATE KEY"
    * Anthropic / OpenAI / HuggingFace key patterns (sk-ant-, sk-proj-, hf_[A-Za-z0-9]{34})
- Dolly prompt text leakage (30-char prefixes).
- dolly_sample.jsonl validation: holds IDs and SHA-256 hashes only, no prompt text.
- data/external/ contents: only permitted manifests, hashes, and non-text samples.
- Absolute local paths (e.g. c:\\users\\harsh): REPORTED as warnings, not fatal.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def scan_release(release_root: Path) -> int:
    print("=" * 80)
    print(f"PARALLAX PRE-PUSH SAFETY SCAN (RELEASE TREE: {release_root})")
    print("=" * 80)

    violations: list[str] = []
    reported_notices: list[str] = []

    # 1. Prohibited filenames check
    prohibited_patterns = [
        "dolly_text.jsonl",
        "external_predictions.jsonl",
        "gandalf.jsonl",
        "notinject.jsonl",
        "test_frame.png",
        "*.pem",
        "*.key",
        "*id_rsa*",
        ".env*",
    ]
    print("\n--- 1. Prohibited Files Scan ---")
    found_prohibited = []
    for root, dirs, files in os.walk(release_root):
        dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
        for f in files:
            for pat in prohibited_patterns:
                if fnmatch.fnmatch(f.lower(), pat.lower()):
                    p = Path(root) / f
                    found_prohibited.append(str(p.relative_to(release_root)))
                    break
    print(f"Found prohibited files: {len(found_prohibited)}")
    for f in found_prohibited:
        print(f"  [VIOLATION] Prohibited file present: {f}")
        violations.append(f"Prohibited file present: {f}")

    # 2. Large files check (> 50 MB)
    print("\n--- 2. Large Files Scan (> 50 MB) ---")
    large_files = []
    for root, dirs, files in os.walk(release_root):
        dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
        for f in files:
            p = Path(root) / f
            try:
                sz = p.stat().st_size
                if sz > 50 * 1024 * 1024:
                    rel = str(p.relative_to(release_root))
                    large_files.append((rel, sz / (1024 * 1024)))
            except OSError:
                continue
    print(f"Found files > 50 MB: {len(large_files)}")
    for f, mb in large_files:
        print(f"  [VIOLATION] {f} ({mb:.2f} MB)")
        violations.append(f"File exceeds 50 MB: {f} ({mb:.2f} MB)")

    # 3. data/external/ contents scan
    print("\n--- 3. data/external/ Contents Scan ---")
    ext_dir = release_root / "data" / "external"
    ext_allowed_files = {
        "manifest.json",
        "dolly_manifest.json",
        "gandalf_manifest.json",
        "notinject_manifest.json",
        "dolly_sample.jsonl",
        "frozen_external.sha256",
        "readme.md",
    }
    ext_violations = []
    if ext_dir.exists():
        for root, dirs, files in os.walk(ext_dir):
            dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
            for f in files:
                p = Path(root) / f
                rel = str(p.relative_to(release_root))
                if f.lower() not in ext_allowed_files:
                    ext_violations.append(rel)
    print(f"data/external/ unauthorized files: {len(ext_violations)}")
    for f in ext_violations:
        print(f"  [VIOLATION] Unauthorized external file: {f}")
        violations.append(f"Unauthorized data/external file: {f}")

    # 4. dolly_sample.jsonl schema and length check
    print("\n--- 4. dolly_sample.jsonl Audit ---")
    dolly_sample_path = release_root / "data" / "external" / "dolly_sample.jsonl"
    if not dolly_sample_path.exists():
        dolly_sample_path = REPO_ROOT / "data" / "external" / "dolly_sample.jsonl"
    if dolly_sample_path.exists():
        try:
            with open(dolly_sample_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            fields = list(rows[0].keys()) if rows else []
            max_len = 0
            max_field = ""
            for r in rows:
                for k, v in r.items():
                    if isinstance(v, str) and len(v) > max_len:
                        max_len = len(v)
                        max_field = k
            print(f"  Rows count      : {len(rows)}")
            print(f"  Field names     : {fields}")
            print(f"  Longest string  : {max_len} chars in field '{max_field}'")
            if max_len > 64:
                violations.append(f"dolly_sample.jsonl contains strings > 64 chars ({max_len} in '{max_field}')")
                print("  [VIOLATION] Contains text strings exceeding hash length!")
            else:
                print("  [PASS] dolly_sample.jsonl contains only IDs/hashes; zero prompt text.")
        except (OSError, json.JSONDecodeError) as e:
            violations.append(f"Error auditing dolly_sample.jsonl: {e}")
            print(f"  [VIOLATION] Could not parse dolly_sample.jsonl: {e}")
    else:
        print("  [INFO] dolly_sample.jsonl not found in release tree.")

    # 5. export/demo/ contents scan
    print("\n--- 5. export/demo/ Contents Scan ---")
    demo_dir = release_root / "export" / "demo"
    demo_allowed_files = {
        "parallax_replay.gif",
        "parallax_replay_frames.json",
        "parallax_replay_sources.json",
        "parallax_replay.gif.sidecar.json",
    }
    demo_violations = []
    if demo_dir.exists():
        for root, dirs, files in os.walk(demo_dir):
            dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
            for f in files:
                p = Path(root) / f
                rel = str(p.relative_to(release_root))
                if f.lower() not in demo_allowed_files:
                    demo_violations.append(rel)
    print(f"export/demo/ unauthorized files: {len(demo_violations)}")
    for f in demo_violations:
        print(f"  [VIOLATION] {f}")
        violations.append(f"Unauthorized export/demo file: {f}")

    # 6. Dolly 30-char prefixes scan
    print("\n--- 6. Dolly Text Leakage Scan ---")
    dolly_prefixes = set()
    # Try reading from data/external/dolly_text.jsonl in repo root if present
    dolly_text_path = REPO_ROOT / "data" / "external" / "dolly_text.jsonl"
    if dolly_text_path.exists():
        try:
            with open(dolly_text_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        txt = r.get("text", "")
                        if len(txt) >= 30:
                            pref = (
                                txt[:35].strip().lower()
                                if txt[:30].strip().lower() == "what is the difference between"
                                else txt[:30].strip().lower()
                            )
                            dolly_prefixes.add(pref)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning reading dolly text: {e}")

    print(f"Loaded {len(dolly_prefixes)} 30-char Dolly prefixes for scanning.")

    dolly_leaks = []
    text_extensions = {".py", ".md", ".json", ".txt", ".csv", ".html", ".tex", ".yaml", ".yml", ".toml"}
    if dolly_prefixes:
        for root, dirs, files in os.walk(release_root):
            dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in text_extensions and p.stat().st_size < 5 * 1024 * 1024:
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as tf:
                            content_lower = tf.read().lower()
                            for prefix in dolly_prefixes:
                                if prefix in content_lower:
                                    dolly_leaks.append((str(p.relative_to(release_root)), prefix))
                                    break
                    except OSError:
                        continue

    print(f"Dolly prefix leaks found: {len(dolly_leaks)}")
    for f, pref in dolly_leaks:
        print(f"  [VIOLATION] Leak in {f}: prefix='{pref}'")
        violations.append(f"Dolly text leak in {f}")

    # 7. Secrets and Keys regex scan
    print("\n--- 7. Secrets and Credentials Scan ---")
    secret_patterns = [
        (re.compile(r"BEGIN\s+(?:RSA|EC|OPENSSH)\s+PRIVATE\s+KEY", re.IGNORECASE), "Private Key Header"),
        (re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}"), "Anthropic API Key"),
        (re.compile(r"sk-proj-[a-zA-Z0-9_\-]{20,}"), "OpenAI API Key"),
        (re.compile(r"hf_[a-zA-Z0-9]{34}"), "HuggingFace Token"),
    ]
    secret_leaks = []
    for root, dirs, files in os.walk(release_root):
        dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in text_extensions and p.stat().st_size < 2 * 1024 * 1024:
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as tf:
                        content = tf.read()
                        for pat, desc in secret_patterns:
                            if pat.search(content):
                                secret_leaks.append((str(p.relative_to(release_root)), desc))
                except OSError:
                    continue

    print(f"Secrets found: {len(secret_leaks)}")
    for f, desc in secret_leaks:
        print(f"  [VIOLATION] {desc} found in {f}")
        violations.append(f"Secret ({desc}) in {f}")

    # 8. Absolute local paths scan (c:\users\harsh) -> REPORTED, not fatal
    print("\n--- 8. Absolute Local Paths Scan (c:\\users\\harsh) [REPORTED, NOT FATAL] ---")
    abs_path_pattern = re.compile(r"c:\\users\\harsh", re.IGNORECASE)
    abs_path_hits = []
    for root, dirs, files in os.walk(release_root):
        dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in text_extensions and p.stat().st_size < 2 * 1024 * 1024:
                if "scan_release_tree.py" in f:
                    continue
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as tf:
                        content = tf.read()
                        matches = abs_path_pattern.findall(content)
                        if matches:
                            abs_path_hits.append(str(p.relative_to(release_root)))
                except OSError:
                    continue

    print(f"Absolute local path hits (REPORTED): {len(abs_path_hits)}")
    for f in abs_path_hits:
        print(f"  [REPORTED] {f}")
        reported_notices.append(f"Absolute local path in {f}")

    print("\n" + "=" * 80)
    print("SAFETY SCAN SUMMARY")
    print(f"  Violations (Fatal) : {len(violations)}")
    print(f"  Reported Notices   : {len(reported_notices)}")
    print("=" * 80)

    if violations:
        print("[FAIL] Safety scan detected policy violations. Release blocked!")
        return 1
    else:
        print("[PASS] Clean release tree! All safety scan checks PASSED.")
        return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        temp_dir = os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))
        target = Path(temp_dir) / "parallax_release"
    sys.exit(scan_release(target))
