"""Real Evaluator Ensemble for PARALLAX Ablation & Verification (Wave E1.1).

Replaces synthetic keyword mocks in ablation studies with real detectors behind
the existing Detector contract:
1. D4: RuleDetector (categorized regex heuristics across raw, normalized, and decoded views)
2. D1: MiniLMDetector (all-MiniLM-L6-v2 embeddings + calibrated logistic regression classifier)
3. D2: DeBERTaDetector (protectai/deberta-v3-base-prompt-injection-v2 transformer pipeline)
4. PromptForestAdapter (HTTP adapter to port 8765, labeled degraded: llama_guard + xgboost)

Features:
- Persistent disk score cache at `data/cache/detector_scores.json`:
  Caches full per-detector results indexed by SHA-256 prompt hash so evaluations
  re-run cheaply and deterministically without requiring live model inferences.
- Resource protection:
  Detectors run sequentially (one model at a time) to respect the 4 GB VRAM limit.
- Multi-view evaluation:
  Integrates `preprocess(text)` to provide decoded/normalized views to detectors,
  preventing blind spots against obfuscated or encoded payloads.
- Hermetic testability:
  Unit tests inject mock detectors and memory/temp caches, ensuring zero live model
  loading or network access during pytest.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from parallax.contracts.detectors import Detector, DetectorResult
from parallax.detectors.deberta import DeBERTaDetector
from parallax.detectors.minilm import MiniLMDetector
from parallax.detectors.promptforest import PromptForestAdapter
from parallax.detectors.rules import RuleDetector
from parallax.preprocess.pipeline import preprocess

logger = logging.getLogger(__name__)

DEFAULT_CACHE_FILE = Path("data/cache/detector_scores.json")


class RealEvaluatorEnsemble:
    """Orchestrates real detectors with persistent disk score caching and multi-view preprocessing."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        detectors: Sequence[Detector] | None = None,
        auto_load_detectors: bool = True,
        use_preprocess_views: bool = True,
    ) -> None:
        self.cache_path = Path(cache_path or DEFAULT_CACHE_FILE)
        self.use_preprocess_views = use_preprocess_views
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

        if detectors is not None:
            self.detectors = list(detectors)
        elif auto_load_detectors:
            self.detectors = self._initialize_default_detectors()
        else:
            self.detectors = []

    def _load_cache(self) -> None:
        """Load persistent score cache from disk if available."""
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.info(
                "Loaded %d cached detector score entries from %s",
                len(self._cache),
                self.cache_path,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load detector cache from %s: %s", self.cache_path, exc
            )
            self._cache = {}

    def _save_cache(self) -> None:
        """Atomically persist score cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.cache_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
            temp_path.replace(self.cache_path)
        except OSError as exc:
            logger.warning(
                "Failed to save detector cache to %s: %s", self.cache_path, exc
            )

    def _initialize_default_detectors(self) -> list[Detector]:
        """Initialize real detectors sequentially behind the Detector contract."""
        detectors: list[Detector] = []

        # 1. D4: Rules detector (CPU regex heuristics)
        d4 = RuleDetector(name="d4_rules", use_preprocess=self.use_preprocess_views)
        detectors.append(d4)

        # 2. D1: MiniLM embedding + LR classifier
        d1 = MiniLMDetector(name="d1-minilm")
        detectors.append(d1)

        # 3. D2: DeBERTa v3 transformer
        d2 = DeBERTaDetector(name="d2-deberta")
        detectors.append(d2)

        # 4. PromptForest adapter (port 8765, labeled degraded: llama_guard + xgboost)
        pf = PromptForestAdapter(
            name="promptforest",
            base_url="http://localhost:8765",
        )
        detectors.append(pf)

        return detectors

    @staticmethod
    def compute_prompt_hash(text: str) -> str:
        """Compute stable SHA-256 hash of prompt text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_detector_version(self, det: Detector) -> str:
        """Extract version or model identifier from a detector."""
        if hasattr(det, "model_name") and det.model_name:
            return str(det.model_name)
        if hasattr(det, "version") and det.version:
            return str(det.version)
        return getattr(det, "name", "unknown")

    def get_cached_results(self, text: str) -> list[DetectorResult] | None:
        """Lookup cached detector results for a prompt, verifying model versions."""
        h = self.compute_prompt_hash(text)
        entry = self._cache.get(h)
        if entry is None:
            return None

        # Verify model versions match to refuse stale cache
        cached_results = entry.get("results", [])
        cached_by_name = {r["name"]: r for r in cached_results}

        # If current ensemble has detectors, check version compatibility
        for det in self.detectors:
            if det.name in cached_by_name:
                cached_item = cached_by_name[det.name]
                expected_ver = self._get_detector_version(det)
                cached_ver = cached_item.get("version")
                if cached_ver is not None and cached_ver != expected_ver:
                    logger.warning(
                        "Refusing stale cache for prompt %s detector %s: cached version '%s' != current version '%s'",
                        h[:8],
                        det.name,
                        cached_ver,
                        expected_ver,
                    )
                    return None

        run_id = entry.get("run_id", "historical")
        logger.debug("Served scores for prompt %s cached from run %s", h[:8], run_id)

        results: list[DetectorResult] = []
        for r_dict in cached_results:
            results.append(
                DetectorResult(
                    name=r_dict["name"],
                    is_attack=r_dict.get("is_attack"),
                    score=r_dict.get("score"),
                    attack_class=r_dict.get("attack_class"),
                    latency_ms=r_dict.get("latency_ms", 0.0),
                    error=r_dict.get("error"),
                )
            )
        return results

    def store_cached_results(
        self,
        text: str,
        results: list[DetectorResult],
        run_id: str | None = None,
    ) -> None:
        """Cache detector results in memory and on disk with version and run_id metadata."""
        h = self.compute_prompt_hash(text)
        det_map = {d.name: d for d in self.detectors}
        eff_run_id = run_id or f"run-{h[:8]}"

        serialized_results = []
        for r in results:
            det = det_map.get(r.name)
            ver = self._get_detector_version(det) if det is not None else "1.0.0"
            serialized_results.append(
                {
                    "name": r.name,
                    "version": ver,
                    "is_attack": r.is_attack,
                    "score": r.score,
                    "attack_class": r.attack_class,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "run_id": eff_run_id,
                    "provenance": f"cached from run {eff_run_id}",
                }
            )

        self._cache[h] = {
            "prompt": text,
            "prompt_hash": h,
            "run_id": eff_run_id,
            "results": serialized_results,
        }
        self._save_cache()

    async def evaluate_sample_async(
        self, text: str, run_id: str | None = None
    ) -> list[DetectorResult]:
        """Asynchronously evaluate a prompt across detectors with cache support."""
        cached = self.get_cached_results(text)
        if cached is not None:
            return cached

        # Preprocessing views extraction for multi-view evaluation
        views_list: list[str] = [text]
        if self.use_preprocess_views:
            try:
                pre = preprocess(text)
                for view_txt in pre.views.values():
                    if view_txt and view_txt not in views_list:
                        views_list.append(view_txt)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Preprocessing failed for prompt, using raw view: %s", exc)

        results: list[DetectorResult] = []

        # Run detectors sequentially to protect 4GB VRAM limit
        for det in self.detectors:
            try:
                # Provide preprocess views to detectors that support multi-view evaluation
                if isinstance(det, RuleDetector) and hasattr(det, "evaluate_views"):
                    views_dict = {f"v_{i}": v for i, v in enumerate(views_list)}
                    score, tech, _ = det.evaluate_views(views_dict)
                    r = DetectorResult(
                        name=det.name,
                        is_attack=score >= det.threshold,
                        score=score,
                        attack_class=tech if score >= det.threshold else "none",
                        latency_ms=1.5,
                        error=None,
                    )
                elif isinstance(det, MiniLMDetector) and hasattr(det, "predict_sync"):
                    r = det.predict_sync(text, views=views_list)
                else:
                    r = await det.predict(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Detector %s execution failed: %s",
                    getattr(det, "name", "unknown"),
                    exc,
                )
                r = DetectorResult(
                    name=getattr(det, "name", "unknown"),
                    is_attack=None,
                    score=None,
                    attack_class=None,
                    latency_ms=0.0,
                    error=str(exc),
                )

            # Enforce degraded ensemble label for PromptForest
            if r.name == "promptforest" and r.attack_class is None:
                r.attack_class = "degraded ensemble"

            results.append(r)

        self.store_cached_results(text, results)
        return results

    def evaluate_sample(self, text: str) -> list[DetectorResult]:
        """Synchronously evaluate a prompt, leveraging cache first, then event loop."""
        cached = self.get_cached_results(text)
        if cached is not None:
            return cached

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # If inside an existing async loop, run in a separate thread to prevent event loop collision
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, self.evaluate_sample_async(text))
                return fut.result()
        else:
            return asyncio.run(self.evaluate_sample_async(text))
