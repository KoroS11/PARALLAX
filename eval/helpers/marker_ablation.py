"""Marker Ablation & Quantified R3 Finding Evaluation Engine for PARALLAX Wave H.

Evaluates legacy corpora:
1. data/splits/dev_legacy.jsonl (291 attacks: 66 marker-carrying, 225 marker-free)
2. data/external_eval/external_benchmark.jsonl (30 novel human attacks, all marker-carrying)

Across arms:
- D1 MiniLM
- D2 DeBERTa
- D3 Deepset
- Arm A AnyFlag Fusion
- Arm B Rule Trigger + Reviewer
- Arm R Learned Review Router + Reviewer
- Judge (Adversarial Judge)

Computes exact recall unmasked vs masked, and exact recall inflation points:
  Inflation (pts) = (Recall_unmasked - Recall_masked) * 100
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mask import contains_synthetic_markers, mask_prompt_text
from parallax.challenger.challenger_v2 import AuthorityPolicy, ChallengerV2Reviewer
from parallax.challenger.judge import AdversarialJudge
from parallax.challenger.schemas import ChallengerVerdict, JudgeResult
from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig
from parallax.contracts.api import Decision
from parallax.detectors.deberta import DeBERTaDetector
from parallax.detectors.minilm import MiniLMDetector
from parallax.orchestrator.fusion import AnyFlagFusion
from parallax.preprocess.pipeline import preprocess
from parallax.router.learned_router import LearnedReviewRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [marker-ablation] %(message)s",
)
logger = logging.getLogger("marker_ablation")

DEV_LEGACY_PATH = REPO_ROOT / "data" / "splits" / "dev_legacy.jsonl"
EXTERNAL_BENCHMARK_PATH = (
    REPO_ROOT / "data" / "external_eval" / "external_benchmark.jsonl"
)
CACHE_PATH = REPO_ROOT / "eval" / "helpers" / "ablation_cache.json"
OUTPUT_JSON = REPO_ROOT / "eval" / "helpers" / "marker_ablation_results.json"
ROUTER_ARTIFACT = REPO_ROOT / "artifacts" / "router" / "learned_router.joblib"
EXTERNAL_JUDGE_SCORES = (
    REPO_ROOT / "data" / "external_eval" / "external_judge_scores.json"
)


def load_corpus() -> dict[str, list[dict[str, Any]]]:
    """Load and partition legacy attack sets."""
    with open(EXTERNAL_BENCHMARK_PATH, encoding="utf-8") as f:
        ext_bench = [json.loads(line) for line in f if line.strip()]
    novel_30 = [r for r in ext_bench if str(r.get("id", "")).startswith("novel-human-")]

    with open(DEV_LEGACY_PATH, encoding="utf-8") as f:
        dev_leg = [json.loads(line) for line in f if line.strip()]
    dev_attacks = [
        r
        for r in dev_leg
        if r.get("label_binary") is True or r.get("is_attack") is True
    ]

    dev_carrying = [r for r in dev_attacks if contains_synthetic_markers(r["text"])]
    dev_free = [r for r in dev_attacks if not contains_synthetic_markers(r["text"])]

    logger.info("Loaded external novel 30 attacks: %d", len(novel_30))
    logger.info(
        "Loaded dev_legacy attacks: %d (carrying=%d, free=%d)",
        len(dev_attacks),
        len(dev_carrying),
        len(dev_free),
    )

    return {
        "novel_30": novel_30,
        "dev_attacks": dev_attacks,
        "dev_carrying": dev_carrying,
        "dev_free": dev_free,
        "combined_carrying": dev_carrying + novel_30,
    }


def load_cache() -> dict[str, Any]:
    """Load existing prediction cache if present."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read cache: %s", e)
    return {}


def save_cache(cache: dict[str, Any]) -> None:
    """Save prediction cache to file."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def load_external_judge_cache() -> dict[str, Any]:
    """Load cached judge scores from external benchmark."""
    if EXTERNAL_JUDGE_SCORES.exists():
        try:
            with open(EXTERNAL_JUDGE_SCORES, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read external judge cache: %s", e)
    return {}


async def score_sample_arms(
    sample_id: str,
    text: str,
    masked: bool,
    d1: MiniLMDetector,
    d2: DeBERTaDetector,
    d3: DeBERTaDetector,
    router: LearnedReviewRouter,
    rule_trigger: ChallengerTrigger,
    reviewer: ChallengerV2Reviewer,
    judge: AdversarialJudge,
    cache: dict[str, Any],
    sem: asyncio.Semaphore,
    force_judge: bool = True,
    ext_judge_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a single sample across all arms for a given masked state."""
    cache_key = f"{sample_id}__masked_{str(masked).lower()}"
    if cache_key in cache:
        return cache[cache_key]

    async with sem:
        # Detectors D1, D2, D3
        r1 = d1.predict_sync(text)
        r2 = await d2.predict(text)
        r3 = await d3.predict(text)

        det_results = [r1, r2, r3]
        fused = AnyFlagFusion().fuse(det_results)
        base_decision = fused.decision

        # Arm A (AnyFlag)
        arm_a_hit = bool(r1.is_attack or r2.is_attack or r3.is_attack)

        # Route check for Rule Trigger (Arm B)
        prep = preprocess(text)
        decision_conf = (
            1.0 - fused.confidence
            if fused.decision == Decision.ALLOW
            else fused.confidence
        )
        t_res = rule_trigger.evaluate(
            detector_results=det_results,
            fused_score=fused.confidence,
            confidence=decision_conf,
            original_decision=fused.decision,
            encoding_present=prep.encoding_present,
            encoding_types=prep.encoding_types,
        )

        # Route check for Learned Router (Arm R)
        r_dec = router.route(text, det_results, fused)

        # Determine if Judge should be queried
        routed_by_b = t_res.should_challenge
        routed_by_r = r_dec.should_route
        should_call_judge = force_judge or routed_by_b or routed_by_r

        judge_verdict = ChallengerVerdict.ABSTAIN
        judge_conf = 0.5
        judge_err = None

        if (
            not masked
            and ext_judge_map
            and sample_id in ext_judge_map
            and ext_judge_map[sample_id].get("verdict")
        ):
            # Use cached external judge verdict if unmasked
            cached_row = ext_judge_map[sample_id]
            raw_v = cached_row.get("verdict", "abstain")
            try:
                judge_verdict = ChallengerVerdict(raw_v)
            except ValueError:
                judge_verdict = ChallengerVerdict.ABSTAIN
            judge_conf = float(cached_row.get("confidence", 0.5))
        elif should_call_judge:
            j_res: JudgeResult = await judge.evaluate(
                prompt=text,
                counter_evidence_hypothesis=(
                    "Adversarial security review: evaluate prompt for injection or policy override"
                ),
                detector_summary={
                    "detectors": [
                        {"name": r.name, "score": r.score, "is_attack": r.is_attack}
                        for r in det_results
                    ]
                },
            )
            judge_verdict = j_res.verdict
            judge_conf = j_res.confidence
            judge_err = j_res.error
        else:
            # Default unrouted judgment follows base decision
            judge_verdict = (
                ChallengerVerdict.BLOCK
                if base_decision in (Decision.FLAG, Decision.BLOCK)
                else ChallengerVerdict.ALLOW
            )
            judge_conf = fused.confidence

        # Standalone Judge hit
        judge_hit = judge_verdict in (
            ChallengerVerdict.BLOCK,
            ChallengerVerdict.ESCALATE,
        )

        # Arm B decision
        if routed_by_b:
            final_b, _ = reviewer.apply_authority_policy(
                verdict=judge_verdict,
                original_decision=base_decision,
                unanimous=True,
                mean_confidence=judge_conf,
            )
            arm_b_hit = final_b in (Decision.FLAG, Decision.BLOCK)
        else:
            arm_b_hit = base_decision in (Decision.FLAG, Decision.BLOCK)

        # Arm R decision
        if routed_by_r:
            final_r, _ = reviewer.apply_authority_policy(
                verdict=judge_verdict,
                original_decision=base_decision,
                unanimous=True,
                mean_confidence=judge_conf,
            )
            arm_r_hit = final_r in (Decision.FLAG, Decision.BLOCK)
        else:
            arm_r_hit = base_decision in (Decision.FLAG, Decision.BLOCK)

        result_dict = {
            "sample_id": sample_id,
            "masked": masked,
            "d1": bool(r1.is_attack),
            "d2": bool(r2.is_attack),
            "d3": bool(r3.is_attack),
            "arm_a": arm_a_hit,
            "arm_b": arm_b_hit,
            "arm_r": arm_r_hit,
            "judge": judge_hit,
            "judge_verdict": judge_verdict.value,
            "judge_confidence": judge_conf,
            "judge_error": judge_err,
        }

        cache[cache_key] = result_dict
        return result_dict


async def run_marker_ablation() -> dict[str, Any]:
    """Execute complete marker ablation and return structured metrics."""
    corpus = load_corpus()
    cache = load_cache()
    ext_judge_map = load_external_judge_cache()

    logger.info("Initializing models and detectors...")
    d1 = MiniLMDetector(name="d1-minilm")
    d2 = DeBERTaDetector(name="d2-deberta")
    d3 = DeBERTaDetector(
        name="d3-deepset", model_name="deepset/deberta-v3-base-injection"
    )

    router = LearnedReviewRouter.load(ROUTER_ARTIFACT)
    rule_trigger = ChallengerTrigger(TriggerConfig())
    reviewer = ChallengerV2Reviewer(policy=AuthorityPolicy.POLICY_A)
    judge = AdversarialJudge(
        model_name="gemma3:4b", base_url="http://localhost:11434", timeout=30.0
    )

    sem = asyncio.Semaphore(4)

    # 1. Evaluate External Novel 30 (all carrying markers)
    logger.info("Scoring external novel 30 attacks unmasked vs masked...")
    novel_unmasked = []
    novel_masked = []
    for idx, sample in enumerate(corpus["novel_30"], 1):
        s_id = sample["id"]
        raw_text = sample["text"]
        m_text = mask_prompt_text(raw_text)

        res_u = await score_sample_arms(
            sample_id=s_id,
            text=raw_text,
            masked=False,
            d1=d1,
            d2=d2,
            d3=d3,
            router=router,
            rule_trigger=rule_trigger,
            reviewer=reviewer,
            judge=judge,
            cache=cache,
            sem=sem,
            force_judge=True,
            ext_judge_map=ext_judge_map,
        )
        novel_unmasked.append(res_u)

        res_m = await score_sample_arms(
            sample_id=s_id,
            text=m_text,
            masked=True,
            d1=d1,
            d2=d2,
            d3=d3,
            router=router,
            rule_trigger=rule_trigger,
            reviewer=reviewer,
            judge=judge,
            cache=cache,
            sem=sem,
            force_judge=True,
            ext_judge_map=ext_judge_map,
        )
        novel_masked.append(res_m)
        if idx % 10 == 0 or idx == len(corpus["novel_30"]):
            logger.info("Novel 30 progress: %d/%d", idx, len(corpus["novel_30"]))
            save_cache(cache)

    # 2. Evaluate dev_legacy Marker-Carrying Attacks (N=66)
    logger.info("Scoring dev_legacy marker-carrying attacks (N=66)...")
    dev_carrying_unmasked = []
    dev_carrying_masked = []
    for idx, sample in enumerate(corpus["dev_carrying"], 1):
        s_id = sample["id"]
        raw_text = sample["text"]
        m_text = mask_prompt_text(raw_text)

        res_u = await score_sample_arms(
            sample_id=s_id,
            text=raw_text,
            masked=False,
            d1=d1,
            d2=d2,
            d3=d3,
            router=router,
            rule_trigger=rule_trigger,
            reviewer=reviewer,
            judge=judge,
            cache=cache,
            sem=sem,
            force_judge=True,
            ext_judge_map=ext_judge_map,
        )
        dev_carrying_unmasked.append(res_u)

        res_m = await score_sample_arms(
            sample_id=s_id,
            text=m_text,
            masked=True,
            d1=d1,
            d2=d2,
            d3=d3,
            router=router,
            rule_trigger=rule_trigger,
            reviewer=reviewer,
            judge=judge,
            cache=cache,
            sem=sem,
            force_judge=True,
            ext_judge_map=ext_judge_map,
        )
        dev_carrying_masked.append(res_m)
        if idx % 10 == 0 or idx == len(corpus["dev_carrying"]):
            logger.info(
                "Dev carrying progress: %d/%d", idx, len(corpus["dev_carrying"])
            )
            save_cache(cache)

    # 3. Evaluate dev_legacy Marker-Free Attacks (N=225)
    # On marker-free attacks, mask_prompt_text(text) == text
    # Standalone judge is evaluated on 35 representative attacks from external judge benchmark
    logger.info("Scoring dev_legacy marker-free attacks (N=225)...")
    dev_free_results = []
    for idx, sample in enumerate(corpus["dev_free"], 1):
        s_id = sample["id"]
        raw_text = sample["text"]

        force_j = idx <= 35 or s_id in ext_judge_map
        res = await score_sample_arms(
            sample_id=s_id,
            text=raw_text,
            masked=False,
            d1=d1,
            d2=d2,
            d3=d3,
            router=router,
            rule_trigger=rule_trigger,
            reviewer=reviewer,
            judge=judge,
            cache=cache,
            sem=sem,
            force_judge=force_j,
            ext_judge_map=ext_judge_map,
        )
        dev_free_results.append(res)
        if idx % 25 == 0 or idx == len(corpus["dev_free"]):
            logger.info("Dev free progress: %d/%d", idx, len(corpus["dev_free"]))
            save_cache(cache)

    save_cache(cache)

    # Compute metric tables
    def compute_recall_table(
        unmasked_list: list[dict[str, Any]], masked_list: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        arms = ["d1", "d2", "d3", "arm_a", "arm_b", "arm_r", "judge"]
        n = len(unmasked_list)
        table = {}
        for arm in arms:
            u_hits = sum(1 for r in unmasked_list if r[arm])
            m_hits = sum(1 for r in masked_list if r[arm])
            u_rec = u_hits / max(1, n)
            m_rec = m_hits / max(1, n)
            inf = (u_rec - m_rec) * 100.0
            table[arm] = {
                "unmasked_recall": round(u_rec, 4),
                "masked_recall": round(m_rec, 4),
                "inflation_pts": round(inf, 2),
                "unmasked_hits": u_hits,
                "masked_hits": m_hits,
                "n": n,
            }
        return table

    # Table 1: External Novel 30 (marker-carrying)
    novel_30_table = compute_recall_table(novel_unmasked, novel_masked)

    # Table 2: Dev Legacy Marker-Carrying (N=66)
    dev_carrying_table = compute_recall_table(
        dev_carrying_unmasked, dev_carrying_masked
    )

    # Table 3: Dev Legacy Marker-Free (N=225, unmasked == masked)
    dev_free_table = compute_recall_table(dev_free_results, dev_free_results)

    # Table 4: Combined Marker-Carrying (N=96 = 30 novel + 66 legacy)
    combined_u = novel_unmasked + dev_carrying_unmasked
    combined_m = novel_masked + dev_carrying_masked
    combined_carrying_table = compute_recall_table(combined_u, combined_m)

    # Table 5: All Dev Legacy Attacks (N=291 = 66 carrying + 225 free)
    all_dev_u = dev_carrying_unmasked + dev_free_results
    all_dev_m = dev_carrying_masked + dev_free_results
    all_dev_table = compute_recall_table(all_dev_u, all_dev_m)

    final_results = {
        "external_novel_30": novel_30_table,
        "dev_legacy_marker_carrying": dev_carrying_table,
        "dev_legacy_marker_free": dev_free_table,
        "combined_marker_carrying": combined_carrying_table,
        "all_dev_legacy": all_dev_table,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        json.dump(final_results, f, indent=2)

    logger.info("Marker ablation complete. Results saved to %s", OUTPUT_JSON)
    return final_results


if __name__ == "__main__":
    asyncio.run(run_marker_ablation())
