"""Offline script to populate the detector score cache for ablation evaluations.

Evaluates all held-out attacks and benign tasks through the real detectors
(RuleDetector, MiniLMDetector, DeBERTaDetector, and PromptForestAdapter) under
GPU file lock protection, saving results to `data/cache/detector_scores.json`.

Subsequent ablation evaluations and benchmarks can then run instantaneously and
offline with 100% deterministic reproducibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parallax.eval.real_ensemble import DEFAULT_CACHE_FILE, RealEvaluatorEnsemble
from parallax.sandbox.utility import BENIGN_TASK_CATALOG
from scripts.gpu_lock import gpu_lock

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

ATTACKS_FILE = REPO_ROOT / "data" / "attacks" / "canary_attacks.jsonl"


def load_all_prompts() -> list[tuple[str, str]]:
    """Load all attack prompts and benign tasks. Returns list of (prompt_text, category)."""
    prompts: list[tuple[str, str]] = []

    # Attacks
    if ATTACKS_FILE.exists():
        with open(ATTACKS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    prompts.append(
                        (item["text"], f"attack:{item.get('technique', 'unknown')}")
                    )

    # Benign tasks across all 16 scenarios
    for sc_id, tasks in BENIGN_TASK_CATALOG.items():
        for t in tasks:
            prompts.append((t.prompt, f"benign:{sc_id}"))

    return prompts


async def populate_cache(force_refresh: bool = False) -> None:
    """Populate detector score cache under GPU lock."""
    logger.info("Acquiring GPU lock for detector evaluation...")
    with gpu_lock(timeout_seconds=600):
        logger.info("GPU lock acquired. Initializing RealEvaluatorEnsemble...")
        ensemble = RealEvaluatorEnsemble(cache_path=DEFAULT_CACHE_FILE)

        prompts = load_all_prompts()
        logger.info(
            "Loaded %d total prompts (%d unique).",
            len(prompts),
            len({p[0] for p in prompts}),
        )

        t0 = time.perf_counter()
        processed = 0
        cached_hits = 0

        for idx, (text, cat) in enumerate(prompts, 1):
            if not force_refresh and ensemble.get_cached_results(text) is not None:
                cached_hits += 1
                continue

            logger.info(
                "[%d/%d] Scoring (%s): %s...",
                idx,
                len(prompts),
                cat,
                text[:50].replace("\n", " "),
            )
            await ensemble.evaluate_sample_async(text)
            processed += 1

            if idx % 10 == 0:
                logger.info(
                    "Progress: %d/%d processed (%d cache hits)",
                    idx,
                    len(prompts),
                    cached_hits,
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Cache population complete in %.2fs. Newly processed: %d, Cache hits: %d. Cache size: %d",
            elapsed,
            processed,
            cached_hits,
            len(ensemble._cache),
        )


if __name__ == "__main__":
    force = "--force" in sys.argv
    asyncio.run(populate_cache(force_refresh=force))
