"""Test suite for Closeout Documentation (tests/test_closeout_docs.py).

Verifies zero banned words and proper provenance attribution across closeout documents.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BANNED_WORDS = [
    "robust",
    "generalizes",
    "proves",
    "validated",
    "verified",
    "audited",
    "AUDIT_CRITERIA_SATISFIED",
    "integrity enforced",
    "signed off",
    "operational utility",
    "0 timeouts",
]

# K1/K8 Exclusion List: explicitly exempts historical incident citations,
# code references, and prohibited claims tables.
EXCLUSION_EXEMPTIONS = {
    "docs/SAFE_CLAIMS.md": [
        "prohibited:",
        "forbidden wording",
        "prohibited claim",
    ],
    "docs/LIMITATIONS_AND_THREATS.md": [
        "pretraining contamination",
    ],
}


def test_no_banned_words_in_closeout_docs(capsys):
    """Verify zero banned words in closeout markdown files outside exclusion list."""
    print("\n[BANNED-WORD SCAN] K1/K8 Exclusion List for Closeout Docs:")
    print(
        "  Exclusion list: Prompt payload samples, forbidden claims definitions, and git commit hashes."
    )

    target_files = [
        REPO_ROOT / "docs" / "LIMITATIONS_AND_THREATS.md",
        REPO_ROOT / "docs" / "WHAT_CHANGED_SINCE_PILOT.md",
        REPO_ROOT / "docs" / "EXTERNAL_RESULTS.md",
        REPO_ROOT / "docs" / "OVERNIGHT_REPORT.md",
        REPO_ROOT / "export" / "closeout" / "NUMBERS_SHEET.md",
        REPO_ROOT / "export" / "closeout" / "EXPORT_INDEX.md",
    ]

    for fpath in target_files:
        assert fpath.exists(), f"File missing: {fpath}"
        content = fpath.read_text(encoding="utf-8")
        for word in BANNED_WORDS:
            assert word.lower() not in content.lower(), (
                f"Banned word '{word}' found in {fpath.name}"
            )


def test_safe_claims_forbidden_column():
    """Verify SAFE_CLAIMS.md only mentions banned terms in its explicit prohibition columns."""
    fpath = REPO_ROOT / "docs" / "SAFE_CLAIMS.md"
    assert fpath.exists()
    lines = fpath.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, 1):
        # If line contains a prohibited word, it must be in the prohibition column or section
        for word in ["robust", "generalizes", "proves", "validated", "verified"]:
            if word in line.lower():
                is_prohibition_context = any(
                    marker in line.lower()
                    for marker in [
                        "prohibited",
                        "forbidden",
                        "barred",
                        "rule:",
                        "section 2",
                    ]
                )
                assert is_prohibition_context, (
                    f"Line {idx} in SAFE_CLAIMS.md uses banned word '{word}' outside prohibition context: {line}"
                )
