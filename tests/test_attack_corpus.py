"""Unit tests for the Canary Attack Corpus and validation pipeline."""

from __future__ import annotations

from scripts.attacks.validate_corpus import (
    ATTACKS_FILE,
    CANARY_PATTERN,
    REQUIRED_FIELDS,
    audit_against_canonical_splits,
    load_attack_corpus,
    validate_corpus,
)

RAW_ATTACKS_FILE = ATTACKS_FILE.parent / "raw" / "raw_templated_attacks.jsonl"


def test_attack_corpus_file_exists() -> None:
    assert ATTACKS_FILE.exists(), f"Corpus file missing at {ATTACKS_FILE}"
    assert ATTACKS_FILE.stat().st_size > 0, "Corpus file is empty"
    assert RAW_ATTACKS_FILE.exists(), f"Raw attacks file missing at {RAW_ATTACKS_FILE}"


def test_attack_corpus_schema_and_integrity() -> None:
    samples = load_attack_corpus()
    assert len(samples) >= 100, (
        f"Expected at least 100 attack samples, got {len(samples)}"
    )

    audit = validate_corpus(samples)
    assert audit["total_samples"] == len(samples)
    assert audit["unique_ids"] == len(samples)
    assert audit["unique_hashes"] == len(samples)

    # Ensure all 3 tools are represented
    assert set(audit["tool_counts"].keys()) == {"pyrit", "garak", "promptfoo"}

    # Ensure required techniques are covered
    expected_techs = {
        "direct_instruction",
        "indirect_injection",
        "multiturn_crescendo",
        "encoding_base64",
        "roleplay_dan",
        "tool_abuse",
        "privilege_escalation",
        "encoding_rot13",
        "encoding_homoglyph",
        "encoding_leetspeak",
    }
    for tech in expected_techs:
        assert tech in audit["technique_counts"], f"Missing technique: {tech}"

    # Ensure technique cap <= 20%
    for tech, count in audit["technique_counts"].items():
        assert count / len(samples) <= 0.2001, (
            f"Technique {tech} exceeds 20% cap ({count}/{len(samples)})"
        )

    # Ensure robust indirect injection subset
    assert audit["technique_counts"]["indirect_injection"] >= 20

    # Ensure near-deduplication and effective sample size
    assert audit["max_pairwise_jaccard"] < 0.70
    assert audit["effective_sample_size"] > 5.0

    # Ensure technique family splits
    assert set(audit["family_counts"].keys()) == {
        "direct",
        "indirect",
        "conversational",
        "obfuscation",
    }
    assert audit["split_counts"]["dev"] == 30
    assert audit["split_counts"]["held_out"] == 90
    for fam in audit["family_counts"]:
        assert audit["dev_by_family"][fam] > 0, f"Family {fam} missing in dev split"


def test_attack_corpus_zero_leakage_and_canary_format() -> None:
    samples = load_attack_corpus()
    split_audit = audit_against_canonical_splits(samples)
    assert split_audit["is_held_out"] is True
    assert split_audit["train_overlap"] == 0
    assert split_audit["val_overlap"] == 0

    for s in samples:
        assert REQUIRED_FIELDS.issubset(set(s.keys()))
        assert CANARY_PATTERN.match(str(s["canary"]))
