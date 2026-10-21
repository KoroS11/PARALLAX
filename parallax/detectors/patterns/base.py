"""Base dataclasses and data structures for D4 rule patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import regex


@dataclass(frozen=True)
class Rule:
    """Representation of a heuristic detection rule.

    Attributes
    ----------
    rule_id : str
        Unique identifier for the rule (e.g. 'DIR_001').
    technique : str
        Technique category: 'direct_instruction', 'roleplay', 'obfuscation',
        'privilege_escalation', or 'tool_abuse'.
    pattern : str
        Regex pattern string for detection.
    weight : float
        Confidence weight in range [0.0, 1.0].
    explanation : str
        Human-readable description of what this rule detects.
    test_positive : list[str]
        Example strings that MUST trigger this rule.
    test_negative : list[str]
        Example strings that MUST NOT trigger this rule.
    """

    rule_id: str
    technique: str
    pattern: str
    weight: float
    explanation: str
    test_positive: list[str] = field(default_factory=list)
    test_negative: list[str] = field(default_factory=list)

    def compile(self) -> Any:
        """Compile the regex pattern with case-insensitivity."""
        return regex.compile(self.pattern, regex.IGNORECASE)


@dataclass(frozen=True)
class RuleMatch:
    """Record of an individual rule match event.

    Attributes
    ----------
    rule_id : str
        Identifier of the matched rule.
    technique : str
        Technique class associated with the rule.
    weight : float
        Confidence weight of the matched rule.
    view : str
        Preprocess view where the match occurred (e.g. 'raw', 'normalized', 'decoded').
    matched_text : str
        Substring that matched the pattern.
    start : int
        Start character offset of the match.
    end : int
        End character offset of the match.
    explanation : str
        Human-readable explanation of the detected vector.
    """

    rule_id: str
    technique: str
    weight: float
    view: str
    matched_text: str
    start: int
    end: int
    explanation: str
