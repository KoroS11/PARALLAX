"""Rule patterns catalog categorized by technique class."""

from __future__ import annotations

from parallax.detectors.patterns.base import Rule, RuleMatch
from parallax.detectors.patterns.direct_instruction import DIRECT_INSTRUCTION_RULES
from parallax.detectors.patterns.obfuscation import OBFUSCATION_RULES
from parallax.detectors.patterns.privilege_escalation import PRIVILEGE_ESCALATION_RULES
from parallax.detectors.patterns.roleplay import ROLEPLAY_RULES
from parallax.detectors.patterns.tool_abuse import TOOL_ABUSE_RULES

ALL_RULES: list[Rule] = [
    *DIRECT_INSTRUCTION_RULES,
    *ROLEPLAY_RULES,
    *OBFUSCATION_RULES,
    *PRIVILEGE_ESCALATION_RULES,
    *TOOL_ABUSE_RULES,
]

RULES_BY_ID: dict[str, Rule] = {rule.rule_id: rule for rule in ALL_RULES}

RULES_BY_TECHNIQUE: dict[str, list[Rule]] = {
    "direct_instruction": DIRECT_INSTRUCTION_RULES,
    "roleplay": ROLEPLAY_RULES,
    "obfuscation": OBFUSCATION_RULES,
    "privilege_escalation": PRIVILEGE_ESCALATION_RULES,
    "tool_abuse": TOOL_ABUSE_RULES,
}

__all__ = [
    "ALL_RULES",
    "DIRECT_INSTRUCTION_RULES",
    "OBFUSCATION_RULES",
    "PRIVILEGE_ESCALATION_RULES",
    "ROLEPLAY_RULES",
    "RULES_BY_ID",
    "RULES_BY_TECHNIQUE",
    "TOOL_ABUSE_RULES",
    "Rule",
    "RuleMatch",
]
