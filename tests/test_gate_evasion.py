"""Tests verifying the CI Quality Gate catches and prevents evasion attempts.

Validates that:
1. List-repeat constant predictions ([True]*N, [1]*N) fail the gate.
2. List comprehensions with constant literals ([True for _ in range(N)]) fail the gate.
3. Array constant predictions (np.ones, np.zeros) fail the gate.
4. Ground truth leakage (is_attack, label_binary, y_true) into predictions outside eval/oracle_upper_bound/
   fails the gate, even when tagged with comments like '# ORACLE UPPER BOUND ONLY'.
5. Multi-line assignments and method calls (.append) cannot evade the AST checks.
6. Legitimate evaluation constructs (y_true calculation, DetectorResult.is_attack) pass cleanly.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_ast_constant_prediction_code(code_str: str) -> list[str]:
    """Simulate gate AST constant prediction check on given code snippet."""
    tree = ast.parse(code_str)
    violations = []

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
                            violations.append("repeated constant list literal")
                    # List comprehension: [True for _ in range(N)]
                    elif isinstance(sub, ast.ListComp):
                        if isinstance(sub.elt, ast.Constant) and (
                            isinstance(sub.elt.value, bool) or sub.elt.value in (0, 1)
                        ):
                            violations.append("constant list comprehension literal")
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
                                f"constant array prediction call ({call_func.value.id}.{call_func.attr})"
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
                                    violations.append("repeated constant list literal")
                            elif (
                                isinstance(sub, ast.ListComp)
                                and isinstance(sub.elt, ast.Constant)
                                and (
                                    isinstance(sub.elt.value, bool)
                                    or sub.elt.value in (0, 1)
                                )
                            ):
                                violations.append("constant list comprehension literal")

    return violations


def check_ast_oracle_leakage_code(
    code_str: str, file_path: str = "eval/eval_arm.py"
) -> list[str]:
    """Simulate gate oracle prediction leakage check on given code snippet."""
    # Exclude tests and eval/oracle_upper_bound/
    if "tests" in file_path or "oracle_upper_bound" in file_path:
        return []

    tree = ast.parse(code_str)
    violations = []

    for node in ast.walk(tree):
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
                    for safe in ("y_true", "ground_truth", "actual", "test_y", "dev_y")
                )
                for tn in target_names
            )
            if is_pred_target:
                for sub in ast.walk(node.value):
                    if (
                        isinstance(sub, ast.Subscript)
                        and isinstance(sub.slice, ast.Constant)
                        and sub.slice.value in ("is_attack", "label", "label_binary")
                    ):
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
                                f"assignment to {target_names} from {sub.slice.value}"
                            )
                    elif isinstance(sub, ast.Name) and sub.id == "y_true":
                        violations.append(
                            f"assignment to {target_names} directly from y_true"
                        )

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
                                        f"{caller_id}.{node.func.attr}() containing {sub.slice.value}"
                                    )
                            elif isinstance(sub, ast.Name) and sub.id == "y_true":
                                violations.append(
                                    f"{caller_id}.{node.func.attr}() containing y_true"
                                )

    return violations


class TestGateEvasionPrevention:
    """Test suite ensuring evasion attempts are caught by gate checks."""

    def test_catches_list_repeat_multiplication_booleans(self) -> None:
        code = "y_pred = [True] * 100"
        violations = check_ast_constant_prediction_code(code)
        assert len(violations) > 0
        assert "repeated constant list literal" in violations[0]

    def test_catches_list_repeat_multiplication_integers(self) -> None:
        code = "preds = [1] * 50"
        violations = check_ast_constant_prediction_code(code)
        assert len(violations) > 0
        assert "repeated constant list literal" in violations[0]

    def test_catches_list_comprehension_constant_predictions(self) -> None:
        code = "y_pred = [True for _ in range(N)]"
        violations = check_ast_constant_prediction_code(code)
        assert len(violations) > 0
        assert "constant list comprehension literal" in violations[0]

    def test_catches_numpy_ones_assigned_to_predictions(self) -> None:
        code = "arm_b_preds = np.ones(len(samples))"
        violations = check_ast_constant_prediction_code(code)
        assert len(violations) > 0
        assert "constant array prediction call (np.ones)" in violations[0]

    def test_catches_numpy_zeros_assigned_to_predictions(self) -> None:
        code = "final_verdict = numpy.zeros(10)"
        violations = check_ast_constant_prediction_code(code)
        assert len(violations) > 0
        assert "constant array prediction call (numpy.zeros)" in violations[0]

    def test_catches_oracle_leakage_tagged_with_comment(self) -> None:
        # A comment or tag does NOT excuse oracle leakage outside eval/oracle_upper_bound/
        code = """
challenger_pred = (
    1 if rec["is_attack"] else 0
)  # ORACLE UPPER BOUND ONLY
"""
        violations = check_ast_oracle_leakage_code(
            code, "eval/router/recheck_router.py"
        )
        assert len(violations) > 0
        assert "is_attack" in violations[0]

    def test_catches_oracle_leakage_via_append(self) -> None:
        code = """
for s in held_out_samples:
    if condition:
        y_pred_arm_r.append(bool(s["label_binary"]))
"""
        violations = check_ast_oracle_leakage_code(code, "scripts/eval/eval_final.py")
        assert len(violations) > 0
        assert "label_binary" in violations[0]

    def test_catches_oracle_leakage_from_y_true(self) -> None:
        code = "rule_preds = list(y_true)"
        violations = check_ast_oracle_leakage_code(
            code, "data/external_eval/run_external_eval.py"
        )
        assert len(violations) > 0
        assert "y_true" in violations[0]

    def test_allows_oracle_in_oracle_upper_bound_directory(self) -> None:
        code = """
rule_oracle_preds = [
    y_true[idx] if idx in rule_routed else base_preds[idx]
    for idx in range(total_n)
]
"""
        violations = check_ast_oracle_leakage_code(
            code, "eval/oracle_upper_bound/run_oracle_upper_bound.py"
        )
        assert violations == []

    def test_allows_legitimate_ground_truth_assignments(self) -> None:
        code = """
y_true = [1 if r["is_attack"] else 0 for r in scored_records]
actual_label = [bool(s["label_binary"]) for s in samples]
"""
        violations = check_ast_oracle_leakage_code(
            code, "eval/router/recheck_router.py"
        )
        assert violations == []

    def test_allows_reading_detector_result_prediction(self) -> None:
        # DetectorResult.is_attack is a model prediction, not a ground truth label
        code = """
preds = [1 if r["promptforest_is_attack"] else 0 for r in scored_records]
pf_preds.append(bool(pf_res["is_attack"]))
"""
        violations = check_ast_oracle_leakage_code(
            code, "data/external_eval/run_external_eval.py"
        )
        assert violations == []


class TestGateSplitIntegrity:
    """Tests verifying that gate catches split tampering or hash evasion."""

    def test_catches_tampered_split_hash(self, tmp_path: Path) -> None:

        from scripts.gate import check_frozen_splits_integrity

        # Create temporary data/splits
        fake_splits = tmp_path / "data" / "splits"
        fake_splits.mkdir(parents=True)
        split_file = fake_splits / "held_out_v2.jsonl"
        split_file.write_text('{"text": "tampered"}', encoding="utf-8")

        # Write FROZEN.sha256 with wrong hash
        frozen_file = fake_splits / "FROZEN.sha256"
        frozen_file.write_text(f"{'0' * 64}  held_out_v2.jsonl\n", encoding="utf-8")

        # Monkeypatch REPO_ROOT and verify sys.exit(1)
        import scripts.gate as gate_mod

        orig_root = gate_mod.REPO_ROOT
        try:
            gate_mod.REPO_ROOT = tmp_path
            try:
                check_frozen_splits_integrity()
                assert False, "Should have exited with code 1"
            except SystemExit as exc:
                assert exc.code == 1
        finally:
            gate_mod.REPO_ROOT = orig_root

    def test_catches_shortcut_audit_threshold_violation(self) -> None:
        from scripts.corpus.shortcut_audit import (
            MAX_AUC_THRESHOLD,
            audit_split,
        )

        # Test with synthetic records that are trivially separable by length
        short_benign = [{"text": "cmd", "label_binary": False} for _ in range(30)]
        long_attack = [
            {
                "text": "Extremely long adversarial prompt text designed to trigger length shortcut "
                * 10,
                "label_binary": True,
            }
            for _ in range(30)
        ]
        res = audit_split("synthetic_tampered", short_benign + long_attack)
        # Length AUC should be ~1.0, failing the <= 0.70 threshold
        assert res["baselines"]["length_only"]["auc"] > MAX_AUC_THRESHOLD
        assert res["baselines"]["length_only"]["passed"] is False
        assert res["passed"] is False

    def test_catches_inverted_auc_shortcut(self) -> None:
        from scripts.corpus.shortcut_audit import MIN_AUC_THRESHOLD, audit_split

        # Twin pairs split across CV folds invert out-of-fold prediction -> ngram AUC < 0.30
        records = []
        for i in range(25):
            records.append(
                {
                    "text": f"unique token cluster {i} alpha beta gamma delta",
                    "label_binary": True,
                }
            )
            records.append(
                {
                    "text": f"unique token cluster {i} alpha beta gamma delta",
                    "label_binary": False,
                }
            )
        res = audit_split("synthetic_inverted", records)
        assert res["baselines"]["char_ngram_tfidf"]["auc"] < MIN_AUC_THRESHOLD
        assert res["baselines"]["char_ngram_tfidf"]["passed"] is False
        assert res["passed"] is False

    def test_catches_twin_pair_duplicate(self) -> None:
        from scripts.corpus.shortcut_audit import MAX_CROSS_LABEL_COSINE, audit_split

        # Build synthetic split with twin pairs (identical text with opposite labels)
        records = []
        for i in range(15):
            records.append(
                {
                    "text": f"Scada command isolate bus 4 system alert unit {i} execute immediate",
                    "label_binary": True,
                }
            )
            records.append(
                {
                    "text": f"Scada command isolate bus 4 system alert unit {i} execute immediate",
                    "label_binary": False,
                }
            )
        res = audit_split("synthetic_twin_pairs", records)
        assert res["max_cross_label_cosine"] >= MAX_CROSS_LABEL_COSINE
        assert res["cross_sim_pass"] is False
        assert res["passed"] is False

    def test_catches_nn_imbalanced_split(self) -> None:
        from scripts.corpus.shortcut_audit import audit_split

        # All attacks cluster together and all benign cluster together -> opp rate ~0.0 (< 0.35)
        attacks = [
            {
                "text": f"alpha beta gamma attack token sequence variant {i}",
                "label_binary": True,
            }
            for i in range(25)
        ]
        benign = [
            {
                "text": f"delta epsilon zeta benign token sequence variant {i}",
                "label_binary": False,
            }
            for i in range(25)
        ]
        res = audit_split("synthetic_clustered", attacks + benign)
        assert res["nn_opp_pass"] is False
        assert res["passed"] is False

    def test_catches_marker_stuffing_shortcut(self) -> None:
        from scripts.corpus.shortcut_audit import (
            MAX_MARKER_GAP_THRESHOLD,
            audit_split,
        )

        # Attacks have "CANARY" and "override" markers, benign do not -> gap > 15.0 pts threshold
        attacks = [
            {
                "text": f"CANARY-1234 execute override scada command variant {i}",
                "label_binary": True,
            }
            for i in range(25)
        ]
        benign = [
            {
                "text": f"Normal operational telemetry reading variant {i}",
                "label_binary": False,
            }
            for i in range(25)
        ]
        res = audit_split("synthetic_markers", attacks + benign)
        assert res["max_marker_gap_points"] > MAX_MARKER_GAP_THRESHOLD
        assert res["passed"] is False

    def test_catches_held_out_v3_tampering(self, tmp_path: Path) -> None:
        import scripts.gate as gate_mod
        from scripts.gate import check_frozen_splits_integrity

        fake_splits = tmp_path / "data" / "splits"
        fake_splits.mkdir(parents=True)
        split_file = fake_splits / "held_out_v3.jsonl"
        split_file.write_text('{"text": "tampered held out v3"}', encoding="utf-8")

        frozen_file = fake_splits / "FROZEN.sha256"
        frozen_file.write_text(f"{'1' * 64}  held_out_v3.jsonl\n", encoding="utf-8")

        orig_root = gate_mod.REPO_ROOT
        try:
            gate_mod.REPO_ROOT = tmp_path
            try:
                check_frozen_splits_integrity()
                assert False, "Should have exited with code 1"
            except SystemExit as exc:
                assert exc.code == 1
        finally:
            gate_mod.REPO_ROOT = orig_root

    def test_catches_contaminated_splits_in_active_pipeline(
        self, tmp_path: Path
    ) -> None:
        import scripts.gate as gate_mod
        from scripts.gate import check_no_contaminated_splits_active

        fake_scripts = tmp_path / "scripts" / "judge"
        fake_scripts.mkdir(parents=True)
        bad_script = fake_scripts / "run_eval.py"
        bad_script.write_text(
            'parser.add_argument("--splits", default=["dev", "held_out_v2"])',
            encoding="utf-8",
        )

        orig_root = gate_mod.REPO_ROOT
        try:
            gate_mod.REPO_ROOT = tmp_path
            try:
                check_no_contaminated_splits_active()
                assert False, "Should have exited with code 1"
            except SystemExit as exc:
                assert exc.code == 1
        finally:
            gate_mod.REPO_ROOT = orig_root
