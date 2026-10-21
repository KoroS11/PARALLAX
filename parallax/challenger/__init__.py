"""Wave D & Wave F Adversarial Challenger Subsystem."""

from parallax.challenger.challenger import ChallengerJudge, ChallengerReviewer
from parallax.challenger.challenger_v2 import (
    AdversarialProsecutor,
    AuthorityPolicy,
    ChallengerV2Reviewer,
    EvidenceSpan,
    ProsecutorHypothesis,
    SelfConsistencyResult,
)
from parallax.challenger.judge import (
    COMPARISON_JUDGE_MODEL,
    PRIMARY_JUDGE_MODEL,
    AdversarialJudge,
)
from parallax.challenger.schemas import (
    ChallengerAuditRecord,
    ChallengerVerdict,
    JudgeResult,
    TriggerReason,
    TriggerResult,
    map_challenger_verdict_to_decision,
)
from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig

__all__ = [
    "COMPARISON_JUDGE_MODEL",
    "PRIMARY_JUDGE_MODEL",
    "AdversarialJudge",
    "AdversarialProsecutor",
    "AuthorityPolicy",
    "ChallengerAuditRecord",
    "ChallengerJudge",
    "ChallengerReviewer",
    "ChallengerTrigger",
    "ChallengerV2Reviewer",
    "ChallengerVerdict",
    "EvidenceSpan",
    "JudgeResult",
    "ProsecutorHypothesis",
    "SelfConsistencyResult",
    "TriggerConfig",
    "TriggerReason",
    "TriggerResult",
    "map_challenger_verdict_to_decision",
]
