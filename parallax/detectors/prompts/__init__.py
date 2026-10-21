"""Judge prompts and templates for D5 LLM Judge detector."""

from parallax.detectors.prompts.judge_prompt import (
    DEFAULT_ATTACK_CLASS,
    DEFAULT_ATTACK_CLASSES,
    JUDGE_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    USER_INPUT_END_FENCE,
    USER_INPUT_START_FENCE,
    create_repair_user_prompt,
    format_user_prompt,
    sanitize_untrusted_input,
)

__all__ = [
    "DEFAULT_ATTACK_CLASS",
    "DEFAULT_ATTACK_CLASSES",
    "JUDGE_SYSTEM_PROMPT",
    "REPAIR_SYSTEM_PROMPT",
    "USER_INPUT_END_FENCE",
    "USER_INPUT_START_FENCE",
    "create_repair_user_prompt",
    "format_user_prompt",
    "sanitize_untrusted_input",
]
