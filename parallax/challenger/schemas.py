"""Schemas and contract bridges for the Wave D Adversarial Challenger subsystem.

Note: parallax/contracts/ is permanently frozen. This module defines local
schemas for the Challenger and Judge and provides pure functions to map
them onto the frozen parallax.contracts.api.Decision enum.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from parallax.contracts.api import Decision


class ChallengerVerdict(str, Enum):
    """Verdicts produced by the Challenger / Judge adversarial reviewer."""

    ALLOW = "allow"
    BLOCK = "block"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"


class TriggerReason(str, Enum):
    """Categorical reasons triggering an adversarial review."""

    DISCREPANCY = "discrepancy"
    BORDERLINE = "borderline"
    LOW_CONFIDENCE = "low_confidence"
    BINARY_SPLIT = "binary_split"
    ENCODING_DETECTED = "encoding_detected"
    LOW_CONFIDENCE_ALLOWED = "low_confidence_allowed"
    RANDOM_AUDIT = "random_audit"
    MANUAL = "manual"
    ROUTER_UNCERTAINTY = "router_uncertainty"
    NONE = "none"


class TriggerResult(BaseModel):
    """Output from the rule-based Challenger trigger."""

    model_config = ConfigDict(extra="ignore")

    should_challenge: bool = Field(
        default=False,
        description="Whether an adversarial review is warranted",
    )
    reasons: list[TriggerReason] = Field(
        default_factory=list,
        description="List of trigger conditions satisfied",
    )
    discrepancy_score: float = Field(
        default=0.0,
        description="Absolute difference between highest and lowest detector scores",
    )
    min_score: float = Field(default=0.0, description="Minimum detector score observed")
    max_score: float = Field(default=0.0, description="Maximum detector score observed")
    fused_score: float = Field(
        default=0.0, description="Fused ensemble probability score"
    )
    confidence: float = Field(
        default=1.0, description="Confidence metric from ensemble"
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary details regarding detector scoring",
    )


class JudgeResult(BaseModel):
    """Structured response from the LLM Judge in the Challenger loop."""

    model_config = ConfigDict(extra="ignore")

    verdict: ChallengerVerdict = Field(
        default=ChallengerVerdict.ABSTAIN,
        description="Categorical judgment: allow, block, abstain, or escalate",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Judge confidence in the assessment [0.0, 1.0]",
    )
    reasoning: str = Field(
        default="",
        description="Explanatory rationale and identified attack indicators",
    )
    attack_class: str = Field(
        default="none",
        description="Identified injection technique or attack vector",
    )
    evidence_quote: str = Field(
        default="",
        description="Exact quote from input if block, else empty string",
    )
    model: str = Field(
        default="unknown",
        description="Identifier of the judge LLM model",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Execution time of the LLM judge in milliseconds",
    )
    extra_calls: int = Field(
        default=1,
        description="Number of extra LLM model calls consumed",
    )
    raw_response: str = Field(
        default="",
        description="Raw output text returned by the model before parsing",
    )
    error: str | None = Field(
        default=None,
        description="Error description if judgment encountered an issue",
    )


class ChallengerAuditRecord(BaseModel):
    """Complete audit record capturing the Challenger review lifecycle."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str | None = Field(default=None, description="Unique prompt identifier")
    original_decision: Decision = Field(
        ...,
        description="Original decision produced by standard ensemble fusion",
    )
    trigger_result: TriggerResult = Field(
        ...,
        description="Trigger evaluation results",
    )
    counter_evidence: str = Field(
        default="",
        description="Synthesized counter-evidence hypothesis presented to judge",
    )
    judge_verdict: ChallengerVerdict | None = Field(
        default=None,
        description="Verdict returned by the judge, if invoked",
    )
    final_decision: Decision = Field(
        ...,
        description="Final post-review decision mapped to frozen Decision enum",
    )
    is_escalated: bool = Field(
        default=False,
        description="True if original allow/flag decision was escalated to a stricter level",
    )
    extra_model_calls: int = Field(
        default=0,
        description="Count of additional model invocations incurred by the review",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Total duration of the challenger pipeline in milliseconds",
    )
    error: str | None = Field(
        default=None,
        description="Failure or timeout description, if any occurred",
    )
    timestamp_iso: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="UTC timestamp of the audit event",
    )


def map_challenger_verdict_to_decision(
    verdict: ChallengerVerdict | None,
    original_decision: Decision,
    escalate_to_block: bool = True,
    flag_on_abstain: bool = False,
) -> Decision:
    """Map ChallengerVerdict onto frozen parallax.contracts.api.Decision enum.

    ASYMMETRY INVARIANT:
    The Challenger has strictly asymmetric authority:
    1. It may escalate ALLOW -> FLAG or BLOCK.
    2. It may escalate FLAG -> BLOCK.
    3. It may NEVER downgrade a BLOCK to ALLOW or FLAG under any circumstances.
    4. On ABSTAIN, error, or None, the original decision is strictly preserved.
    """
    # 1. Asymmetry Invariant: BLOCK is never downgraded
    if original_decision == Decision.BLOCK:
        return Decision.BLOCK

    # 2. If judge was not invoked (None), preserve original decision
    if verdict is None:
        return original_decision

    # 3. If judge abstained:
    if verdict == ChallengerVerdict.ABSTAIN:
        return Decision.FLAG if flag_on_abstain else original_decision

    # 4. Explicit block verdict always escalates to Decision.BLOCK
    if verdict == ChallengerVerdict.BLOCK:
        return Decision.BLOCK

    # 4. Escalate verdict: upgrade ALLOW -> BLOCK (if configured) or FLAG
    if verdict == ChallengerVerdict.ESCALATE:
        return Decision.BLOCK if escalate_to_block else Decision.FLAG

    # 5. Allow verdict:
    # If original was ALLOW, remains ALLOW.
    # If original was FLAG, Challenger alone cannot downgrade without human review;
    # preserve FLAG for caution.
    if verdict == ChallengerVerdict.ALLOW:
        return (
            Decision.ALLOW if original_decision == Decision.ALLOW else original_decision
        )

    return original_decision
