"""Wave F: Challenger V2 with Evidence Grounding, Self-Consistency, and Authority Ablation.

Enhancements over V1:
1. Two-Role Design:
   - Role 1 (Prosecutor): Analyzes prompt and detector signals, extracts specific
     evidence spans (character offsets + categories), synthesizes structured indictment.
   - Role 2 (Arbiter / Judge): Evaluates whether cited spans constitute active
     adversarial exploitation or benign context / false positive.
2. Evidence-Grounded Counter-Evidence:
   - Cites exact prompt spans and conflicting detector signals.
3. Self-Consistency N=3:
   - Evaluates N=3 independent judge passes (with varying seeds/framings).
   - Strict rule: ABSTAINS if there is any disagreement or divergence across the 3 passes.
4. Authority Policies (Ablated):
   - Policy A (Conservative Asymmetric): May escalate ALLOW -> FLAG/BLOCK, never downgrades BLOCK.
   - Policy B (Calibrated Downgrade for Confident FPs): Additionally may downgrade
     BLOCK -> FLAG (never straight to ALLOW) when self-consistent judge certifies benign context.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from parallax.challenger.schemas import (
    ChallengerAuditRecord,
    ChallengerVerdict,
    JudgeResult,
    TriggerResult,
)
from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig
from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult

logger = logging.getLogger(__name__)


class AuthorityPolicy(str, Enum):
    """Authority policies ablated in Wave F Challenger evaluation."""

    POLICY_A = "escalate_only"  # Conservative: ALLOW -> FLAG/BLOCK, never touches BLOCK
    POLICY_B = "calibrated_downgrade"  # Calibrated: ALLOW -> FLAG/BLOCK, BLOCK -> FLAG (never ALLOW)


class EvidenceSpan(BaseModel):
    """Identified suspicious text span in untrusted prompt."""

    model_config = ConfigDict(extra="ignore")

    start: int = Field(description="Start character offset")
    end: int = Field(description="End character offset")
    text: str = Field(description="Quoted substring from prompt")
    category: str = Field(description="Adversarial indicator category")
    explanation: str = Field(description="Specific indicator explanation")


class ProsecutorHypothesis(BaseModel):
    """Structured indictment synthesized by the Adversarial Prosecutor role."""

    model_config = ConfigDict(extra="ignore")

    spans: list[EvidenceSpan] = Field(default_factory=list)
    detector_signals: list[str] = Field(default_factory=list)
    argument: str = Field(default="")

    def format_counter_evidence(self) -> str:
        """Format hypothesis into markdown counter-evidence block for the arbiter."""
        lines = [
            "### ROLE 1 (PROSECUTOR) INDICTMENT",
            f"**Detector Signals Cited:** {len(self.detector_signals)} signals",
        ]
        for sig in self.detector_signals:
            lines.append(f"  - {sig}")

        lines.append(f"\n**Evidence Spans Cited:** {len(self.spans)} text spans")
        if not self.spans:
            lines.append(
                "  *(No explicit keyword or regex spans isolated; triggering on statistical signals)*"
            )
        else:
            for s in self.spans:
                lines.append(
                    f'  - [{s.start}:{s.end}] "{s.text}" ({s.category}): {s.explanation}'
                )

        lines.extend(
            [
                "\n**Prosecutor Argument:**",
                self.argument,
            ]
        )
        return "\n".join(lines)


class SelfConsistencyResult(BaseModel):
    """Aggregate result from N=3 judge self-consistency sampling."""

    model_config = ConfigDict(extra="ignore")

    final_verdict: ChallengerVerdict
    unanimous: bool
    verdict_votes: list[ChallengerVerdict]
    judge_results: list[JudgeResult]
    total_calls: int
    mean_confidence: float
    abstain_reason: str | None = None


# Adversarial pattern detectors for Prosecutor span extraction
SPAN_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "instruction_override",
        "Attempts to override system/assistant rules",
        re.compile(
            r"(?i)\b(ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)|disregard\s+all|override\s+system)\b"
        ),
    ),
    (
        "roleplay_jailbreak",
        "Forces unrestricted persona or roleplay",
        re.compile(
            r"(?i)\b(you\s+are\s+now\s+(DAN|unrestricted|an\s+AI\s+without\s+rules)|act\s+as\s+a\s+hacker|jailbreak\s+mode)\b"
        ),
    ),
    (
        "canary_probe",
        "Canary string or trip breaker probe",
        re.compile(r"(?i)\b(CANARY-[A-Za-z0-9_-]+|trip_breaker|mock_trip_breaker)\b"),
    ),
    (
        "delimiter_escape",
        "Suspicious delimiter or tag boundary injection",
        re.compile(r"(?i)(</untrusted_input>|<system>|\[SYSTEM\]|```system)"),
    ),
    (
        "credential_leak",
        "Extraction of hidden system prompts or credentials",
        re.compile(
            r"(?i)\b(reveal\s+(your\s+)?(system\s+prompt|secret\s+key|api_key|password)|print\s+instructions)\b"
        ),
    ),
]


class AdversarialProsecutor:
    """Role 1: Adversarial Prosecutor that isolates spans and synthesizes indictments."""

    def analyze(
        self,
        prompt: str,
        detector_results: Sequence[DetectorResult],
        trigger_result: TriggerResult,
    ) -> ProsecutorHypothesis:
        """Extract evidence spans and construct structured adversarial hypothesis."""
        spans: list[EvidenceSpan] = []

        for category, explanation, pattern in SPAN_PATTERNS:
            for match in pattern.finditer(prompt):
                spans.append(
                    EvidenceSpan(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(),
                        category=category,
                        explanation=explanation,
                    )
                )

        detector_signals: list[str] = []
        for det in detector_results:
            status = (
                "ATTACK"
                if det.is_attack is True
                else ("BENIGN" if det.is_attack is False else "ABSTAIN")
            )
            score_str = f"{det.score:.4f}" if det.score is not None else "N/A"
            if det.is_attack is True or (det.score is not None and det.score > 0.5):
                detector_signals.append(
                    f"{det.name} ({status}, score={score_str}, class={det.attack_class or 'none'})"
                )

        # Synthesize argument
        arg_parts = []
        if spans:
            arg_parts.append(
                f"Prompt contains {len(spans)} explicit adversarial indicator spans."
            )
        if detector_signals:
            arg_parts.append(
                f"Detectors {', '.join(detector_signals)} flagged malicious characteristics."
            )
        if trigger_result.discrepancy_score > 0.4:
            arg_parts.append(
                f"Significant detector discrepancy (delta={trigger_result.discrepancy_score:.3f}) suggests subtle evasion."
            )
        if not arg_parts:
            arg_parts.append(
                "Statistical ensemble confidence is low; prompt warrants manual security arbiter review."
            )

        argument = " ".join(arg_parts)

        return ProsecutorHypothesis(
            spans=spans,
            detector_signals=detector_signals,
            argument=argument,
        )


@runtime_checkable
class ArbiterJudge(Protocol):
    """Protocol for Arbiter Judge evaluated in self-consistency loop."""

    async def evaluate(
        self,
        prompt: str,
        counter_evidence_hypothesis: str,
        detector_summary: dict[str, Any],
        seed: int = 42,
    ) -> JudgeResult:
        """Evaluate prompt against prosecutor indictment."""
        ...


class ChallengerV2Reviewer:
    """Wave F Challenger V2 Reviewer with Evidence-Grounding and Self-Consistency."""

    def __init__(
        self,
        trigger: ChallengerTrigger | None = None,
        judge: ArbiterJudge | None = None,
        prosecutor: AdversarialProsecutor | None = None,
        policy: AuthorityPolicy = AuthorityPolicy.POLICY_A,
        self_consistency_n: int = 3,
        timeout_seconds: float = 10.0,
        judge_confidence_threshold: float = 0.70,
        downgrade_confidence_threshold: float = 0.65,
        learned_router: Any | None = None,
    ) -> None:
        self.trigger = trigger or ChallengerTrigger(TriggerConfig())
        self.judge = judge
        self.prosecutor = prosecutor or AdversarialProsecutor()
        self.policy = policy
        self.self_consistency_n = max(1, self_consistency_n)
        self.timeout_seconds = timeout_seconds
        self.judge_confidence_threshold = judge_confidence_threshold
        self.downgrade_confidence_threshold = downgrade_confidence_threshold
        self.learned_router = learned_router

    async def _evaluate_self_consistent(
        self,
        prompt: str,
        hypothesis: ProsecutorHypothesis,
        detector_summary: dict[str, Any],
    ) -> SelfConsistencyResult:
        """Query judge N=3 times with varying seeds and require unanimous agreement."""
        if self.judge is None:
            return SelfConsistencyResult(
                final_verdict=ChallengerVerdict.ABSTAIN,
                unanimous=False,
                verdict_votes=[],
                judge_results=[],
                total_calls=0,
                mean_confidence=0.0,
                abstain_reason="No judge configured",
            )

        counter_evidence = hypothesis.format_counter_evidence()
        tasks = []
        for i in range(self.self_consistency_n):
            seed = 42 + i * 17
            tasks.append(
                self.judge.evaluate(
                    prompt=prompt,
                    counter_evidence_hypothesis=counter_evidence,
                    detector_summary=detector_summary,
                    seed=seed,
                )
            )

        try:
            results: list[JudgeResult] = await asyncio.gather(*tasks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Self-consistency judge execution failed: %s", exc)
            return SelfConsistencyResult(
                final_verdict=ChallengerVerdict.ABSTAIN,
                unanimous=False,
                verdict_votes=[ChallengerVerdict.ABSTAIN],
                judge_results=[],
                total_calls=len(tasks),
                mean_confidence=0.0,
                abstain_reason=f"Execution exception: {exc}",
            )

        votes = [r.verdict for r in results]
        total_calls = sum(r.extra_calls for r in results)
        mean_conf = (
            sum(r.confidence for r in results) / len(results) if results else 0.0
        )

        # Disagreement or any abstain forces overall ABSTAIN
        if any(v == ChallengerVerdict.ABSTAIN for v in votes):
            return SelfConsistencyResult(
                final_verdict=ChallengerVerdict.ABSTAIN,
                unanimous=False,
                verdict_votes=votes,
                judge_results=results,
                total_calls=total_calls,
                mean_confidence=mean_conf,
                abstain_reason="One or more passes returned ABSTAIN / parse error",
            )

        # Check for unanimous agreement across all N votes
        if len(set(votes)) == 1:
            return SelfConsistencyResult(
                final_verdict=votes[0],
                unanimous=True,
                verdict_votes=votes,
                judge_results=results,
                total_calls=total_calls,
                mean_confidence=mean_conf,
                abstain_reason=None,
            )

        # Disagreement across passes -> ABSTAIN on disagreement
        return SelfConsistencyResult(
            final_verdict=ChallengerVerdict.ABSTAIN,
            unanimous=False,
            verdict_votes=votes,
            judge_results=results,
            total_calls=total_calls,
            mean_confidence=mean_conf,
            abstain_reason=f"Disagreement across {self.self_consistency_n} passes: {votes}",
        )

    def apply_authority_policy(
        self,
        verdict: ChallengerVerdict,
        original_decision: Decision,
        unanimous: bool,
        mean_confidence: float = 1.0,
        detector_results: Sequence[DetectorResult] | None = None,
        ensemble_confidence: float | None = None,
    ) -> tuple[Decision, bool]:
        """Apply Policy A or Policy B authority rules to translate verdict into Decision.

        Returns (final_decision, is_downgraded).
        """
        if verdict == ChallengerVerdict.ABSTAIN:
            # Unified Abstain Handling (Wave H.3): Unresolved timeout or model abstain
            # produces a distinct third outcome: FLAG (escalated to operator, not silent BLOCK or ALLOW).
            return Decision.FLAG, False

        # Invariant for both policies: ALLOW -> BLOCK or FLAG
        if original_decision == Decision.ALLOW:
            if self.policy == AuthorityPolicy.POLICY_A:
                if verdict == ChallengerVerdict.BLOCK and unanimous:
                    return Decision.BLOCK, False
                if verdict in (ChallengerVerdict.ESCALATE, ChallengerVerdict.BLOCK):
                    return Decision.FLAG, False
                return Decision.ALLOW, False

            if self.policy == AuthorityPolicy.POLICY_B:
                # Calibrated safe escalation: ALLOW -> FLAG (never straight to BLOCK)
                # Rescues system from judge false-blocks on allowed prompts
                if verdict in (ChallengerVerdict.ESCALATE, ChallengerVerdict.BLOCK):
                    logger.info(
                        "Challenger V2 Policy B: Escalating ALLOW to FLAG (quarantine/audit)"
                    )
                    return Decision.FLAG, False
                return Decision.ALLOW, False

        if original_decision == Decision.FLAG:
            if self.policy == AuthorityPolicy.POLICY_A:
                if verdict == ChallengerVerdict.BLOCK and unanimous:
                    return Decision.BLOCK, False
                return Decision.FLAG, False

            if self.policy == AuthorityPolicy.POLICY_B:
                # Under Policy B, if arbiter confirms BLOCK with unanimity and high confidence
                if (
                    verdict == ChallengerVerdict.BLOCK
                    and unanimous
                    and mean_confidence >= self.judge_confidence_threshold
                ):
                    return Decision.BLOCK, False
                return Decision.FLAG, False

        if original_decision == Decision.BLOCK:
            # Policy A: Never alter BLOCK under any circumstance
            if self.policy == AuthorityPolicy.POLICY_A:
                return Decision.BLOCK, False

            # Policy B: Calibrated downgrade: BLOCK -> FLAG (never straight to ALLOW)
            if self.policy == AuthorityPolicy.POLICY_B:
                # Invariant Guard 1: NEVER downgrade when any detector score >= 0.95
                if detector_results:
                    for det in detector_results:
                        if det.score is not None and det.score >= 0.95:
                            logger.info(
                                "Challenger V2 Policy B: Invariant guard — detector %s score %.4f >= 0.95, downgrade prohibited.",
                                det.name,
                                det.score,
                            )
                            return Decision.BLOCK, False

                        det_name = (det.name or "").lower()
                        # Invariant Guard 2: NEVER downgrade when D4 rules fired
                        if ("rule" in det_name or "d4" in det_name) and (
                            det.is_attack is True
                            or (det.score is not None and det.score > 0.5)
                        ):
                            logger.info(
                                "Challenger V2 Policy B: Invariant guard — D4 rules fired, downgrade prohibited."
                            )
                            return Decision.BLOCK, False

                # Invariant Guard 3: Ensemble confidence < threshold (if ensemble was overwhelmingly confident, prohibit downgrade)
                if ensemble_confidence is not None and ensemble_confidence >= 0.95:
                    logger.info(
                        "Challenger V2 Policy B: Invariant guard — ensemble confidence %.4f >= 0.95, downgrade prohibited.",
                        ensemble_confidence,
                    )
                    return Decision.BLOCK, False

                # If arbiter unanimously certifies benign context (ALLOW) with confidence:
                if (
                    verdict == ChallengerVerdict.ALLOW
                    and unanimous
                    and mean_confidence >= self.downgrade_confidence_threshold
                ):
                    logger.info(
                        "Challenger V2 Policy B: Downgrading BLOCK to FLAG (benign context certified)"
                    )
                    return Decision.FLAG, True

                # If arbiter verdict is ESCALATE (needs quarantine / human review rather than hard block)
                if (
                    verdict == ChallengerVerdict.ESCALATE
                    and mean_confidence >= self.downgrade_confidence_threshold
                ):
                    logger.info(
                        "Challenger V2 Policy B: Downgrading BLOCK to FLAG (quarantine certified)"
                    )
                    return Decision.FLAG, True

                return Decision.BLOCK, False

        return original_decision, False

    async def review(
        self,
        prompt: str,
        detector_results: Sequence[DetectorResult],
        fused_score: float,
        original_decision: Decision,
        confidence: float | None = None,
        prompt_id: str | None = None,
        encoding_present: bool = False,
        encoding_types: Sequence[str] | None = None,
        force_challenge: bool = False,
        router_prob_error: float | None = None,
        router_should_route: bool | None = None,
    ) -> ChallengerAuditRecord:
        """Execute evidence-grounded self-consistent review."""
        start_time = time.perf_counter()

        eff_router_prob = router_prob_error
        eff_router_route = router_should_route
        if (
            eff_router_prob is None
            and self.learned_router is not None
            and hasattr(self.learned_router, "route")
        ):
            try:
                r_decision = self.learned_router.route(
                    prompt=prompt,
                    results=list(detector_results),
                )
                eff_router_prob = getattr(r_decision, "prob_error", None)
                eff_router_route = getattr(r_decision, "should_route", None)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Learned router invocation error: %s", exc)

        trigger_res = self.trigger.evaluate(
            detector_results=detector_results,
            fused_score=fused_score,
            confidence=confidence,
            original_decision=original_decision,
            encoding_present=encoding_present,
            encoding_types=encoding_types,
            router_prob_error=eff_router_prob,
            router_should_route=eff_router_route,
        )

        should_challenge = trigger_res.should_challenge or force_challenge

        # Fast path if no challenge or judge unavailable
        if not should_challenge or self.judge is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ChallengerAuditRecord(
                prompt_id=prompt_id,
                original_decision=original_decision,
                trigger_result=trigger_res,
                counter_evidence="",
                judge_verdict=None,
                final_decision=original_decision,
                is_escalated=False,
                extra_model_calls=0,
                latency_ms=round(elapsed_ms, 2),
                error=None,
            )

        # Under Policy A, BLOCK cannot be altered, so fast path
        if (
            original_decision == Decision.BLOCK
            and self.policy == AuthorityPolicy.POLICY_A
        ):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ChallengerAuditRecord(
                prompt_id=prompt_id,
                original_decision=Decision.BLOCK,
                trigger_result=trigger_res,
                counter_evidence="",
                judge_verdict=None,
                final_decision=Decision.BLOCK,
                is_escalated=False,
                extra_model_calls=0,
                latency_ms=round(elapsed_ms, 2),
                error=None,
            )

        # Role 1: Prosecutor analysis and span extraction
        hypothesis = self.prosecutor.analyze(prompt, detector_results, trigger_res)

        detector_summary: dict[str, Any] = {
            "discrepancy_score": trigger_res.discrepancy_score,
            "fused_score": trigger_res.fused_score,
            "confidence": trigger_res.confidence,
            "detectors": [
                {
                    "name": d.name,
                    "is_attack": d.is_attack,
                    "score": d.score,
                    "class": d.attack_class,
                }
                for d in detector_results
            ],
            "prosecutor_spans": [s.model_dump() for s in hypothesis.spans],
        }

        # Role 2: Arbiter Judge evaluation with self-consistency N=3
        sc_res = await self._evaluate_self_consistent(
            prompt, hypothesis, detector_summary
        )

        # Apply authority policy
        final_decision, _is_downgraded = self.apply_authority_policy(
            verdict=sc_res.final_verdict,
            original_decision=original_decision,
            unanimous=sc_res.unanimous,
            mean_confidence=sc_res.mean_confidence,
            detector_results=detector_results,
            ensemble_confidence=confidence,
        )

        is_escalated = (
            original_decision == Decision.ALLOW
            and final_decision in (Decision.FLAG, Decision.BLOCK)
        ) or (original_decision == Decision.FLAG and final_decision == Decision.BLOCK)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ChallengerAuditRecord(
            prompt_id=prompt_id,
            original_decision=original_decision,
            trigger_result=trigger_res,
            counter_evidence=hypothesis.format_counter_evidence(),
            judge_verdict=sc_res.final_verdict,
            final_decision=final_decision,
            is_escalated=is_escalated,
            extra_model_calls=sc_res.total_calls,
            latency_ms=round(elapsed_ms, 2),
            error=sc_res.abstain_reason,
        )
