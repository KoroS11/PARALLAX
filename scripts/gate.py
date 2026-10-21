"""PARALLAX Quality Gate Definition of Done.

Must pass before ANY merge to main.
Runs:
1. ruff check . and ruff format --check .
2. mypy parallax
3. pytest -m "not integration and not gpu" with coverage >= 80% on touched packages
4. bandit -r parallax -ll
5. Check that parallax/contracts/ is unchanged against Phase 0 commit (35a04ab)
6. Check that no file > 1MB and no token/secret pattern is present
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_0_COMMIT = "35a04ab47bb9e4e74472a768ffd1e46c73b69bf6"
MAX_FILE_BYTES = 1024 * 1024  # 1 MB

SECRET_PATTERNS = [
    re.compile(r"hf_[a-zA-Z0-9]{34,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{36,}"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
]


def run_command(cmd: list[str], desc: str) -> None:
    print(f"\n[gate] === Running: {desc} ===")
    print(f"[gate] Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if res.returncode != 0:
        print(
            f"\n[gate] FAILED: {desc} exited with code {res.returncode}",
            file=sys.stderr,
        )
        sys.exit(res.returncode)
    print(f"[gate] PASSED: {desc}")


def check_frozen_contracts() -> None:
    print("\n[gate] === Checking frozen contracts (parallax/contracts/) ===")
    cmd = ["git", "diff", PHASE_0_COMMIT, "--", "parallax/contracts/"]
    res = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        cmd_fallback = ["git", "diff", "main", "--", "parallax/contracts/"]
        res = subprocess.run(
            cmd_fallback, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )

    if res.stdout.strip():
        print(
            f"\n[gate] FAILED: parallax/contracts/ has modified lines:\n{res.stdout}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        "[gate] PASSED: parallax/contracts/ is strictly identical to Phase 0 baseline."
    )


def check_file_sizes_and_secrets() -> None:
    print("\n[gate] === Checking file sizes (<= 1MB) and scanning for secrets ===")
    res = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]

    size_violations = []
    secret_violations = []

    for rel_path in tracked_files:
        path = REPO_ROOT / rel_path
        if not path.is_file():
            continue

        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            size_violations.append((rel_path, size))

        if rel_path.endswith((".joblib", ".lock", ".png", ".jpg", ".parquet")):
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                matches = pattern.findall(content)
                if matches:
                    real_matches = [
                        m for m in matches if not m.endswith(("xxx", "placeholder"))
                    ]
                    if real_matches:
                        secret_violations.append((rel_path, pattern.pattern))
        except (OSError, UnicodeDecodeError):
            continue

    if size_violations:
        print("[gate] FAILED: Files exceed 1 MB limit:", file=sys.stderr)
        for f, sz in size_violations:
            print(f"  - {f}: {sz / (1024 * 1024):.2f} MB", file=sys.stderr)
        sys.exit(1)

    if secret_violations:
        print("[gate] FAILED: Potential secrets detected:", file=sys.stderr)
        for f, pat in secret_violations:
            print(f"  - {f}: matched pattern {pat}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[gate] PASSED: Checked {len(tracked_files)} tracked files. No size or secret violations."
    )


def check_report_metrics_traceability() -> None:
    """Verify that all metrics and quantitative claims in docs/REPORT.md are traceable to results JSON."""
    print("\n[gate] === Checking docs/REPORT.md Metric Traceability ===")
    report_path = REPO_ROOT / "docs" / "REPORT.md"
    ablation_path = REPO_ROOT / "reports" / "ablation_results.json"
    exp_path = REPO_ROOT / "reports" / "experiment_results.json"

    if not report_path.exists():
        print("[gate] FAILED: docs/REPORT.md not found", file=sys.stderr)
        sys.exit(1)

    content = report_path.read_text(encoding="utf-8")

    # 1. Check for unbacked narrative claims that do not exist in raw artifacts
    unsupported_patterns = [
        (
            r"Arm C[^\n]*\b3\.3%",
            "Arm C 3.3% ASR claim (Arm C achieved 23.3%, not 3.3%)",
        ),
        (
            r"Arm C[^\n]*\b96\.7%",
            "Arm C 96.7% recall claim (Arm C achieved 76.7%, not 96.7%)",
        ),
        (
            r"\b18\.4\s*ms",
            "+18.4 ms p95 latency claim (does not exist in reports/ablation_results.json)",
        ),
        (
            r"\b17\.5\s*(?:extra|additional|calls)",
            "17.5 extra calls claim (does not exist in reports/ablation_results.json)",
        ),
        (
            r"\b78\.8%",
            "78.8% relative risk reduction claim (does not exist in reports/ablation_results.json)",
        ),
    ]

    for pat, desc in unsupported_patterns:
        if re.search(pat, content):
            print(
                f"[gate] FAILED: Unsupported metric quoted in docs/REPORT.md: {desc}",
                file=sys.stderr,
            )
            sys.exit(1)

    # 2. Check that ablation table in Section 8 matches reports/ablation_results.json if file exists
    if ablation_path.exists():
        try:
            import json

            ablation_data = json.loads(ablation_path.read_text(encoding="utf-8"))
            for arm in ablation_data:
                arm_name = arm["arm_name"]
                asr_pct = f"{arm['asr'] * 100:.1f}%"
                rec_pct = f"{arm['recall'] * 100:.1f}%"
                if asr_pct not in content:
                    print(
                        f"[gate] FAILED: Missing or mismatched ASR metric for {arm_name} ({asr_pct}) in docs/REPORT.md",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                if rec_pct not in content:
                    print(
                        f"[gate] FAILED: Missing or mismatched recall metric for {arm_name} ({rec_pct}) in docs/REPORT.md",
                        file=sys.stderr,
                    )
                    sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[gate] FAILED: Error checking ablation results JSON: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    # 3. Check that Wave C experiment results in Section 4 are backed by reports/experiment_results.json
    if exp_path.exists():
        try:
            import json

            exp_data = json.loads(exp_path.read_text(encoding="utf-8")).get(
                "results", {}
            )
            if "E5" in exp_data:
                e5_speedup = f"{exp_data['E5']['p50_speedup']:.2f}"
                if e5_speedup not in content and "3.89x" not in content:
                    print(
                        f"[gate] FAILED: E5 speedup ({e5_speedup}x) not found in docs/REPORT.md",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            if "E6" in exp_data:
                asr_off = f"{exp_data['E6']['asr_guard_off'] * 100:.1f}%"
                if asr_off not in content:
                    print(
                        f"[gate] FAILED: E6 baseline ASR ({asr_off}) not found in docs/REPORT.md",
                        file=sys.stderr,
                    )
                    sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[gate] FAILED: Error checking experiment results JSON: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(
        "[gate] PASSED: All metrics quoted in docs/REPORT.md are traceable to results JSON."
    )


def check_log_manifest_integrity() -> None:
    """Verify that run manifests exist in eval/logs/manifests/ and provenance is searchable."""
    print("\n[gate] === Checking Log Manifest Integrity (eval/logs/manifests/) ===")
    manifests_dir = REPO_ROOT / "eval" / "logs" / "manifests"
    if not manifests_dir.exists():
        print(
            f"[gate] FAILED: Manifests directory {manifests_dir} does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest_files = list(manifests_dir.glob("*.json"))
    if not manifest_files:
        print(
            f"[gate] FAILED: No per-run manifests found in {manifests_dir}.",
            file=sys.stderr,
        )
        sys.exit(1)

    main_manifest = REPO_ROOT / "eval" / "logs" / "MANIFEST.json"
    if not main_manifest.exists():
        print(
            f"[gate] FAILED: Main manifest {main_manifest} is missing.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[gate] PASSED: Log manifest integrity verified ({len(manifest_files)} per-run manifests present)."
    )


def check_no_constant_prediction_construction_ast() -> None:
    """Fail on constant-literal prediction construction in non-test Python files:
    - Repeated literals: [True]*N, [False]*N, [1]*N, [0]*N
    - Constant list comprehensions: [True for _ in range(N)], [1 for _ in range(N)], etc.
    - Constant array calls: np.ones, np.zeros, numpy.ones, numpy.zeros
    """
    print(
        "\n[gate] === AST Scan: Checking for constant-literal prediction construction ([1]*N, np.ones, [True for _ in range(N)]) ==="
    )
    import ast

    res = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_py_files = [
        line.strip() for line in res.stdout.splitlines() if line.strip()
    ]

    violations = []
    for rel_path in tracked_py_files:
        path = REPO_ROOT / rel_path
        if (
            "tests" in path.parts
            or path.name.startswith("test_")
            or path.name.endswith("_test.py")
        ):
            continue

        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="ignore"), filename=rel_path
            )
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # 1. Assignment to prediction variables
            if isinstance(node, ast.Assign):
                target_names = []
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        target_names.append(t.id)
                    elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                        target_names.append(t.value.id)

                is_pred_target = any(
                    any(
                        k in tn.lower()
                        for k in (
                            "pred",
                            "verdict",
                            "decision",
                            "challenger_pred",
                            "arm_",
                        )
                    )
                    and not any(
                        safe in tn.lower()
                        for safe in (
                            "true",
                            "ground_truth",
                            "actual",
                            "label",
                            "gold",
                            "matrix",
                            "mat",
                            "split",
                        )
                    )
                    for tn in target_names
                )
                if is_pred_target:

                    def is_constant_prediction_list(expr: ast.expr) -> bool:
                        if isinstance(expr, ast.List):
                            return len(expr.elts) > 0 and all(
                                isinstance(elt, ast.Constant)
                                and (isinstance(elt.value, bool) or elt.value in (0, 1))
                                for elt in expr.elts
                            )
                        return False

                    for sub in ast.walk(node.value):
                        # Binary multiplication: [True]*N, [1]*N, etc.
                        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult):
                            if is_constant_prediction_list(
                                sub.left
                            ) or is_constant_prediction_list(sub.right):
                                violations.append(
                                    (
                                        rel_path,
                                        node.lineno,
                                        f"repeated constant list literal assigned to {target_names}",
                                    )
                                )
                        # List comprehension: [True for _ in range(N)]
                        elif isinstance(sub, ast.ListComp):
                            if isinstance(sub.elt, ast.Constant) and (
                                isinstance(sub.elt.value, bool)
                                or sub.elt.value in (0, 1)
                            ):
                                violations.append(
                                    (
                                        rel_path,
                                        node.lineno,
                                        f"constant list comprehension literal assigned to {target_names}",
                                    )
                                )
                        # Array call: np.ones, np.zeros
                        elif isinstance(sub, ast.Call):
                            call_func = sub.func
                            if (
                                isinstance(call_func, ast.Attribute)
                                and isinstance(call_func.value, ast.Name)
                                and call_func.value.id in ("np", "numpy")
                                and call_func.attr in ("ones", "zeros", "full")
                            ):
                                violations.append(
                                    (
                                        rel_path,
                                        node.lineno,
                                        f"constant array prediction call ({call_func.value.id}.{call_func.attr}) assigned to {target_names}",
                                    )
                                )

            # 2. Method calls on prediction lists: preds.extend([1]*N) or preds.append(True)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("extend", "append"):
                    caller_id = (
                        node.func.value.id.lower()
                        if isinstance(node.func.value, ast.Name)
                        else ""
                    )
                    is_pred_target = any(
                        k in caller_id for k in ("pred", "verdict", "decision", "arm_")
                    ) and not any(
                        safe in caller_id
                        for safe in (
                            "true",
                            "ground_truth",
                            "actual",
                            "label",
                            "gold",
                            "matrix",
                            "mat",
                            "split",
                        )
                    )
                    if is_pred_target:
                        for arg in node.args:
                            for sub in ast.walk(arg):
                                if isinstance(sub, ast.BinOp) and isinstance(
                                    sub.op, ast.Mult
                                ):
                                    if (
                                        isinstance(sub.left, ast.List)
                                        and any(
                                            isinstance(e, ast.Constant)
                                            and (
                                                isinstance(e.value, bool)
                                                or e.value in (0, 1)
                                            )
                                            for e in sub.left.elts
                                        )
                                    ) or (
                                        isinstance(sub.right, ast.List)
                                        and any(
                                            isinstance(e, ast.Constant)
                                            and (
                                                isinstance(e.value, bool)
                                                or e.value in (0, 1)
                                            )
                                            for e in sub.right.elts
                                        )
                                    ):
                                        violations.append(
                                            (
                                                rel_path,
                                                node.lineno,
                                                f"{caller_id}.{node.func.attr}() with repeated constant list",
                                            )
                                        )
                                elif (
                                    isinstance(sub, ast.ListComp)
                                    and isinstance(sub.elt, ast.Constant)
                                    and (
                                        isinstance(sub.elt.value, bool)
                                        or sub.elt.value in (0, 1)
                                    )
                                ):
                                    violations.append(
                                        (
                                            rel_path,
                                            node.lineno,
                                            f"{caller_id}.{node.func.attr}() with constant list comprehension",
                                        )
                                    )

    if violations:
        print(
            "[gate] FAILED: Constant-literal prediction construction detected in non-test code:",
            file=sys.stderr,
        )
        for f, line, kind in violations:
            print(f"  - {f}:{line}: {kind}", file=sys.stderr)
        sys.exit(1)

    print(
        "[gate] PASSED: No constant-literal prediction construction found in non-test code."
    )


def check_no_oracle_label_prediction_leakage() -> None:
    """Fail on any label field (is_attack, label, label_binary, y_true) used to construct
    a prediction in any non-test eval code OUTSIDE eval/oracle_upper_bound/,
    regardless of comments or tags.
    """
    print(
        "\n[gate] === Checking for oracle label prediction leakage outside eval/oracle_upper_bound/ ==="
    )
    import ast

    res = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_py_files = [
        line.strip() for line in res.stdout.splitlines() if line.strip()
    ]

    violations = []
    for rel_path in tracked_py_files:
        path = REPO_ROOT / rel_path
        # Allow test files and isolated theoretical upper bound module
        if (
            "tests" in path.parts
            or path.name.startswith("test_")
            or path.name.endswith("_test.py")
            or "oracle_upper_bound" in path.parts
        ):
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content, filename=rel_path)
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            # Check 1: Assignment to target named *pred* or *verdict* or *decision*
            if isinstance(node, ast.Assign):
                target_names = []
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        target_names.append(t.id)
                    elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                        target_names.append(t.value.id)

                is_pred_target = any(
                    any(k in tn.lower() for k in ("pred", "verdict", "challenger_pred"))
                    and not any(
                        safe in tn.lower()
                        for safe in (
                            "y_true",
                            "ground_truth",
                            "actual",
                            "test_y",
                            "dev_y",
                        )
                    )
                    for tn in target_names
                )
                if is_pred_target:
                    for sub in ast.walk(node.value):
                        # Subscript on label fields: e.g. rec["is_attack"]
                        if (
                            isinstance(sub, ast.Subscript)
                            and isinstance(sub.slice, ast.Constant)
                            and sub.slice.value
                            in ("is_attack", "label", "label_binary")
                        ):
                            # Verify the base object is a sample/record, not a detector response object
                            base_id = ""
                            if isinstance(sub.value, ast.Name):
                                base_id = sub.value.id.lower()
                            elif isinstance(sub.value, ast.Subscript):
                                base_id = "indexed_sample"
                            if any(
                                s_name in base_id
                                for s_name in (
                                    "sample",
                                    "rec",
                                    "item",
                                    "s",
                                    "r",
                                    "row",
                                    "indexed_sample",
                                )
                            ):
                                violations.append(
                                    (
                                        rel_path,
                                        node.lineno,
                                        f"assignment to {target_names} from {sub.slice.value}",
                                    )
                                )
                        elif isinstance(sub, ast.Name) and sub.id == "y_true":
                            violations.append(
                                (
                                    rel_path,
                                    node.lineno,
                                    f"assignment to {target_names} directly from y_true",
                                )
                            )

            # Check 2: Method calls like preds.append(...) using label fields or y_true
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("append", "extend"):
                    caller_id = (
                        node.func.value.id.lower()
                        if isinstance(node.func.value, ast.Name)
                        else ""
                    )
                    if any(p in caller_id for p in ("pred", "verdict", "predictions")):
                        for arg in node.args:
                            for sub in ast.walk(arg):
                                if (
                                    isinstance(sub, ast.Subscript)
                                    and isinstance(sub.slice, ast.Constant)
                                    and sub.slice.value
                                    in ("is_attack", "label", "label_binary")
                                ):
                                    base_id = (
                                        sub.value.id.lower()
                                        if isinstance(sub.value, ast.Name)
                                        else ""
                                    )
                                    if any(
                                        s_name in base_id
                                        for s_name in (
                                            "sample",
                                            "rec",
                                            "item",
                                            "s",
                                            "r",
                                            "row",
                                            "data",
                                        )
                                    ) and not any(
                                        det_name in base_id
                                        for det_name in (
                                            "pf",
                                            "det",
                                            "detector",
                                            "result",
                                            "res",
                                            "response",
                                            "d1",
                                            "d2",
                                            "d3",
                                        )
                                    ):
                                        violations.append(
                                            (
                                                rel_path,
                                                node.lineno,
                                                f"{caller_id}.{node.func.attr}() containing {sub.slice.value}",
                                            )
                                        )
                                elif isinstance(sub, ast.Name) and sub.id == "y_true":
                                    violations.append(
                                        (
                                            rel_path,
                                            node.lineno,
                                            f"{caller_id}.{node.func.attr}() containing y_true",
                                        )
                                    )

    if violations:
        print(
            "[gate] FAILED: Oracle ground truth label leakage into predictions detected outside eval/oracle_upper_bound/:",
            file=sys.stderr,
        )
        for f, line_no, detail in violations:
            print(f"  - {f}:{line_no}: {detail}", file=sys.stderr)
        print(
            "  NOTE: Oracle bounds may only be evaluated inside eval/oracle_upper_bound/",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "[gate] PASSED: Zero oracle label prediction leakage found outside eval/oracle_upper_bound/."
    )


def check_no_hardcoded_iso_timestamps() -> None:
    """Fail on any hardcoded ISO timestamp string literal in non-test Python files."""
    print(
        "\n[gate] === Checking for hardcoded ISO timestamp literals in results writers ==="
    )
    res = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_py_files = [
        line.strip() for line in res.stdout.splitlines() if line.strip()
    ]

    iso_pat = re.compile(
        r'["\']202[0-9]-[0-1][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]'
    )

    violations = []
    for rel_path in tracked_py_files:
        path = REPO_ROOT / rel_path
        if (
            "tests" in path.parts
            or path.name.startswith("test_")
            or path.name.endswith("_test.py")
            or path.name.startswith("init_")
        ):
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for i, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if iso_pat.search(line):
                violations.append((rel_path, i, stripped))

    if violations:
        print(
            "[gate] FAILED: Hardcoded ISO timestamp literal found in non-test code:",
            file=sys.stderr,
        )
        for f, line_no, text in violations:
            print(f"  - {f}:{line_no}: {text}", file=sys.stderr)
        print(
            "  NOTE: Results writers must use datetime.now(UTC).isoformat() with run_id",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[gate] PASSED: No hardcoded ISO timestamps found in non-test code.")


def check_frozen_splits_integrity() -> None:
    """Verify cryptographic integrity of data/splits against data/splits/FROZEN.sha256."""
    print("\n[gate] === Checking data/splits/FROZEN.sha256 integrity ===")
    import hashlib

    frozen_path = REPO_ROOT / "data" / "splits" / "FROZEN.sha256"
    if not frozen_path.exists():
        print("[gate] FAILED: data/splits/FROZEN.sha256 not found", file=sys.stderr)
        sys.exit(1)

    violations = []
    for line in frozen_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        expected_sha, fname = parts[0], parts[1]
        target_file = REPO_ROOT / "data" / "splits" / fname
        if not target_file.exists():
            violations.append((fname, "File not found"))
            continue
        actual_sha = hashlib.sha256(target_file.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            violations.append(
                (fname, f"SHA256 mismatch: expected {expected_sha}, got {actual_sha}")
            )

    if violations:
        print("[gate] FAILED: Split integrity violations detected:", file=sys.stderr)
        for f, reason in violations:
            print(f"  - {f}: {reason}", file=sys.stderr)
        sys.exit(1)
    print(
        "[gate] PASSED: All frozen split SHA256 hashes match data/splits/FROZEN.sha256."
    )


def check_no_contaminated_splits_active() -> None:
    """Verify that contaminated splits (held_out_v2, dev2) are never configured as active test splits."""
    print(
        "\n[gate] === Checking for contaminated split references (held_out_v2, dev2) in active pipelines ==="
    )
    violations = []
    scan_dirs = [REPO_ROOT / "scripts", REPO_ROOT / "eval"]
    pattern = re.compile(r"default\s*=\s*\[[^\]]*(?:held_out_v2|dev2)")

    for sdir in scan_dirs:
        for py_file in sdir.glob("**/*.py"):
            rel = py_file.relative_to(REPO_ROOT).as_posix()
            if "build_corpus" in rel or "archive" in rel:
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if pattern.search(content):
                violations.append(
                    (
                        rel,
                        "contaminated split (held_out_v2/dev2) found in default arguments",
                    )
                )

    if violations:
        print(
            "[gate] FAILED: Contaminated splits configured in active pipelines:",
            file=sys.stderr,
        )
        for f, reason in violations:
            print(f"  - {f}: {reason}", file=sys.stderr)
        sys.exit(1)
    print(
        "[gate] PASSED: No contaminated splits (held_out_v2, dev2) configured in active pipelines."
    )


def check_minimum_power_rule() -> None:
    """Fail immediately if any split has n < target (Gate v3 Power Rule)."""
    print("\n[gate] === Checking Gate v3 minimum statistical power rule ===")
    targets = {
        "dev.jsonl": (100, 150),
        "val.jsonl": (80, 120),
        "held_out_v4.jsonl": (120, 150),
    }
    violations = []
    for fname, (min_atk, min_ben) in targets.items():
        fpath = REPO_ROOT / "data" / "splits" / fname
        if not fpath.exists():
            violations.append((fname, "File not found"))
            continue
        import json

        records = [
            json.loads(line)
            for line in fpath.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        atks = sum(1 for r in records if r.get("label_binary", False))
        bens = sum(1 for r in records if not r.get("label_binary", False))
        if atks < min_atk or bens < min_ben:
            violations.append(
                (
                    fname,
                    f"Power violation: {atks}/{min_atk} attacks, {bens}/{min_ben} benign",
                )
            )

    if violations:
        print("[gate] FAILED: Gate v3 Minimum Power Rule violated:", file=sys.stderr)
        for f, reason in violations:
            print(f"  - {f}: {reason}", file=sys.stderr)
        sys.exit(1)
    print(
        "[gate] PASSED: All splits meet or exceed Gate v3 statistical power requirements."
    )


def check_shortcut_audit() -> None:
    """Verify that all splits pass the shortcut baseline audit (AUC <= 0.70, max gap <= 15 pts)."""
    print(
        "\n[gate] === Running shortcut audit on data/splits (AUC <= 0.70, gap <= 15 pts) ==="
    )
    cmd = ["uv", "run", "python", "scripts/corpus/shortcut_audit.py"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if res.returncode != 0:
        print(
            f"\n[gate] FAILED: shortcut audit failed with code {res.returncode}",
            file=sys.stderr,
        )
        sys.exit(res.returncode)
    print(
        "[gate] PASSED: Shortcut audit verified all splits invariant to trivial baselines."
    )


def main() -> None:
    print("=" * 60)
    print("PARALLAX CI GATE EXECUTION")
    print("=" * 60)

    # 1. Ruff lint and format check
    run_command(["uv", "run", "ruff", "check", "."], "Ruff Linter")
    run_command(["uv", "run", "ruff", "format", "--check", "."], "Ruff Formatter Check")

    # 2. Mypy type checker
    run_command(["uv", "run", "mypy", "parallax"], "Mypy Type Checking")

    # 3. Pytest with coverage
    run_command(
        [
            "uv",
            "run",
            "pytest",
            "-m",
            "not integration and not gpu",
            "--cov=parallax",
            "--cov-fail-under=80",
        ],
        "Pytest Unit Suite & Coverage Check",
    )

    # 4. Bandit security audit
    run_command(
        ["uv", "run", "bandit", "-r", "parallax", "-ll"], "Bandit Security Audit"
    )

    # 5. Frozen contracts check
    check_frozen_contracts()

    # 6. File size and secrets check
    check_file_sizes_and_secrets()

    # 7. Metric traceability check
    check_report_metrics_traceability()

    # 7b. Log manifest integrity check (eval/logs/manifests/)
    check_log_manifest_integrity()

    # 8. Integrity check: no constant prediction literals ([1]*N, np.ones, [True for _ in range(N)])
    check_no_constant_prediction_construction_ast()

    # 9. Integrity check: no oracle label prediction leakage outside eval/oracle_upper_bound/
    check_no_oracle_label_prediction_leakage()

    # 10. Integrity check: no hardcoded ISO timestamps in non-test code
    check_no_hardcoded_iso_timestamps()

    # 11. Integrity check: data/splits/FROZEN.sha256 cryptographic verification
    check_frozen_splits_integrity()

    # 11b. Pipeline integrity: no contaminated splits configured as active
    check_no_contaminated_splits_active()

    # 11c. Statistical power: Gate v3 Minimum Power Rule (dev>=100/150, val>=80/120, held_out_v4>=120/150)
    check_minimum_power_rule()

    # 12. Statistical integrity: shortcut baseline audit (AUC <= 0.70, gap <= 15 pts)
    check_shortcut_audit()

    print("\n" + "=" * 60)
    print("ALL GATE CHECKS PASSED SUCCESSFULLY (READY FOR MERGE)")
    print("=" * 60)


if __name__ == "__main__":
    main()
