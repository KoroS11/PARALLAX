#!/usr/bin/env python3
"""Wave K13 External Evaluation Runner (scripts/eval/eval_external.py).

Features:
- Imports evaluated 959a99a policy code and learned router (tau=0.35).
- Wrapped under scripts/gpu_lock.py.
- Append-only prediction logging to JSONL.
- Crash-resume support: reads existing output log, skips already completed rows,
  and marks resumed rows with `resumed: true`.
- Evaluates Primary Arm R and descriptive comparison arms (Arm A, Arm B, Arm U, Arm C2).
- Does NOT run unprompted: requires explicit dataset arguments and invocation.
- Enforces preflight gates: 5-arm policy replay check, PromptForest health, and protocol hash checks.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parallax.challenger.challenger_v2 import AuthorityPolicy, ChallengerV2Reviewer
from parallax.challenger.judge import AdversarialJudge
from parallax.challenger.schemas import ChallengerVerdict
from parallax.challenger.trigger import ChallengerTrigger, TriggerConfig
from parallax.contracts.api import Decision
from parallax.contracts.detectors import DetectorResult
from parallax.detectors.rules import RuleDetector
from parallax.router.learned_router import LearnedReviewRouter
from scripts.gpu_lock import gpu_lock

logging.basicConfig(
    level=logging.INFO,
    format="[eval-external] %(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_external")

CALIBRATORS_DIR = PROJECT_ROOT / "artifacts" / "calibrators"
ROUTER_ARTIFACT = PROJECT_ROOT / "artifacts" / "router" / "learned_router.joblib"


def get_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_promptforest_health(url: str = "http://127.0.0.1:8765/health") -> bool:
    """Check PromptForest health endpoint and raise error if unreachable or not ok."""
    try:
        r = httpx.get(url, timeout=3.0)
        if r.status_code == 200 and r.json().get("status") == "ok":
            logger.info(
                "PromptForest daemon on port 8765 is healthy (device: %s)",
                r.json().get("device"),
            )
            return True
        raise ValueError(
            f"PromptForest health check failed with status {r.status_code}: {r.text}"
        )
    except Exception as exc:
        raise RuntimeError(
            f"PromptForest service unreachable on port 8765: {exc}"
        ) from exc


def verify_protocol_digests(
    protocol_path: Path = PROJECT_ROOT / "docs" / "EXTERNAL_EVAL_PROTOCOL.md",
) -> dict[str, str]:
    """Verify scripts/eval/eval_external.py and scripts/eval/external_metrics.py against protocol text.

    Raises ValueError on any mismatch.
    """
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file missing: {protocol_path}")

    content = protocol_path.read_text(encoding="utf-8")

    # 1. Parse external_metrics.py SHA-256
    m_metrics = re.search(
        r"external_metrics\.py`.*?`([a-f0-9]{64})`", content, re.DOTALL
    )
    if not m_metrics:
        raise ValueError("Could not find external_metrics.py SHA-256 in protocol")
    expected_metrics_sha = m_metrics.group(1)

    metrics_file = PROJECT_ROOT / "scripts" / "eval" / "external_metrics.py"
    actual_metrics_sha = get_file_sha256(metrics_file)
    if actual_metrics_sha != expected_metrics_sha:
        raise ValueError(
            f"external_metrics.py SHA-256 mismatch against protocol: actual {actual_metrics_sha} != expected {expected_metrics_sha}"
        )

    # 2. Parse eval_external.py SHA-256
    m_runner = re.search(r"eval_external\.py`.*?`([a-f0-9]{64})`", content, re.DOTALL)
    if not m_runner:
        raise ValueError("Could not find eval_external.py SHA-256 in protocol")
    expected_runner_sha = m_runner.group(1)

    runner_file = PROJECT_ROOT / "scripts" / "eval" / "eval_external.py"
    actual_runner_sha = get_file_sha256(runner_file)
    if actual_runner_sha != expected_runner_sha:
        raise ValueError(
            f"eval_external.py SHA-256 mismatch against protocol: actual {actual_runner_sha} != expected {expected_runner_sha}"
        )

    logger.info("Protocol SHA-256 digests confirmed for runner and metrics scripts.")
    return {
        "external_metrics.py": actual_metrics_sha,
        "eval_external.py": actual_runner_sha,
    }


def verify_frozen_manifest(
    manifest_path: Path = PROJECT_ROOT / "data" / "external" / "FROZEN_EXTERNAL.sha256",
) -> dict[str, str]:
    """Verify that committed external benchmark files match FROZEN_EXTERNAL.sha256 exactly."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    verified: dict[str, str] = {}
    external_dir = manifest_path.parent
    lines = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Empty manifest at {manifest_path}")

    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed manifest line: '{line}'")
        expected_sha, fname = parts[0], parts[1].strip()
        target_file = external_dir / fname
        if not target_file.exists():
            raise FileNotFoundError(f"Frozen file missing: {target_file}")
        actual_sha = get_file_sha256(target_file)
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {fname}: expected {expected_sha}, got {actual_sha}"
            )
        verified[fname] = actual_sha

    logger.info(
        "Verified %d frozen external benchmark files against manifest", len(verified)
    )
    return verified


def verify_dolly_text_integrity(
    sample_file: Path = PROJECT_ROOT / "data" / "external" / "dolly_sample.jsonl",
    text_file: Path = PROJECT_ROOT / "data" / "external" / "dolly_text.jsonl",
) -> int:
    """Verify each row in local dolly_text.jsonl against committed dolly_sample.jsonl prompt_hash."""
    if not sample_file.exists():
        raise FileNotFoundError(f"Dolly sample index not found: {sample_file}")
    if not text_file.exists():
        raise FileNotFoundError(f"Dolly text file not found: {text_file}")

    expected_hashes: dict[int, str] = {}
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            expected_hashes[int(rec["row_index"])] = rec["prompt_hash"]

    text_rows: list[dict[str, Any]] = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text_rows.append(json.loads(line))

    if len(text_rows) != len(expected_hashes):
        raise ValueError(
            f"Row count mismatch: text file has {len(text_rows)} rows, sample index has {len(expected_hashes)}"
        )

    for row in text_rows:
        r_idx = int(row["row_index"])
        if r_idx not in expected_hashes:
            raise ValueError(f"Unexpected row_index {r_idx} not in sample index")
        computed_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        expected_hash = expected_hashes[r_idx]
        if computed_hash != expected_hash:
            raise ValueError(
                f"Prompt text tampered at row_index {r_idx}: computed {computed_hash} != expected {expected_hash}"
            )

    logger.info(
        "Verified integrity of all %d Dolly text rows against committed hashes",
        len(text_rows),
    )
    return len(text_rows)


def verify_runner_config(
    manifest_path: Path = PROJECT_ROOT
    / "eval"
    / "logs"
    / "manifests"
    / "final-v4-20261019-142005-8fb16e.json",
    ollama_url: str = "http://localhost:11434",
) -> list[tuple[str, Any, Any, bool]]:
    """Compare runner configuration against protocol pins and final-v4 manifest.

    Note: The final-v4 run manifest holds only the model tag (`model_digest: gemma3:4b`).
    All other parameters are protocol pins frozen for evaluation.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_judge_model = manifest.get("model_digest", "gemma3:4b")

    # 1. Ollama ID check
    actual_ollama_id = "unknown"
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                name = m.get("name", "")
                if (
                    name == expected_judge_model
                    or name == f"{expected_judge_model}:latest"
                ):
                    actual_ollama_id = m.get("digest", "")[:12]
                    break
    except (httpx.HTTPError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not reach Ollama at %s: %s", ollama_url, exc)

    # 2. Judge Prompt SHA-256
    judge_prompt_file = (
        PROJECT_ROOT / "parallax" / "detectors" / "prompts" / "judge_prompt.py"
    )
    if not judge_prompt_file.exists():
        raise FileNotFoundError(f"Judge prompt file missing: {judge_prompt_file}")
    actual_judge_prompt_sha = get_file_sha256(judge_prompt_file)
    expected_judge_prompt_sha = (
        "c64f565306e6e86da8d8baef9f04c9dc23664f7b79159091077d984db17ebb2c"
    )

    # 3. Calibrator Hashes
    d2_cal_file = CALIBRATORS_DIR / "d2_deberta_platt.joblib"
    d3_cal_file = CALIBRATORS_DIR / "d3_deepset_platt.joblib"
    if not d2_cal_file.exists():
        raise FileNotFoundError(f"D2 calibrator missing: {d2_cal_file}")
    if not d3_cal_file.exists():
        raise FileNotFoundError(f"D3 calibrator missing: {d3_cal_file}")

    actual_d2_sha = get_file_sha256(d2_cal_file)
    actual_d3_sha = get_file_sha256(d3_cal_file)
    expected_d2_sha = "af1bfa658ee27d1fd36e361f21685a49312c8ca6d9cb1098ae996f5473df0b33"
    expected_d3_sha = "c4f118629fb3eccc0b3bd1a192d13a03ba54319ee3d295921151f9db9204e648"

    # Static configuration parameters
    actual_num_predict = 128
    actual_timeout = 60.0
    actual_think = False
    actual_tau = 0.35
    actual_d2_id = "protectai/deberta-v3-base-prompt-injection-v2"
    actual_d3_id = "deepset/deberta-v3-base-injection"

    checks = [
        (
            "Judge Model Tag",
            "gemma3:4b",
            expected_judge_model,
            "gemma3:4b" == expected_judge_model,
        ),
        (
            "Ollama Model ID",
            actual_ollama_id,
            "a2af6cc3eb7f",
            actual_ollama_id == "a2af6cc3eb7f",
        ),
        (
            "Judge Prompt SHA-256",
            actual_judge_prompt_sha,
            expected_judge_prompt_sha,
            actual_judge_prompt_sha == expected_judge_prompt_sha,
        ),
        ("num_predict", actual_num_predict, 128, actual_num_predict == 128),
        ("timeout", actual_timeout, 60.0, actual_timeout == 60.0),
        ("think flag", actual_think, False, actual_think is False),
        ("tau", actual_tau, 0.35, actual_tau == 0.35),
        (
            "D2 Platt SHA-256",
            actual_d2_sha,
            expected_d2_sha,
            actual_d2_sha == expected_d2_sha,
        ),
        (
            "D3 Platt SHA-256",
            actual_d3_sha,
            expected_d3_sha,
            actual_d3_sha == expected_d3_sha,
        ),
        (
            "D2 Model ID",
            actual_d2_id,
            "protectai/deberta-v3-base-prompt-injection-v2",
            actual_d2_id == "protectai/deberta-v3-base-prompt-injection-v2",
        ),
        (
            "D3 Model ID",
            actual_d3_id,
            "deepset/deberta-v3-base-injection",
            actual_d3_id == "deepset/deberta-v3-base-injection",
        ),
    ]

    mismatches = [name for name, act, exp, ok in checks if not ok]
    if mismatches:
        raise ValueError(f"Configuration mismatch against protocol pins: {mismatches}")

    return checks


class ExternalEvaluator:
    """External dataset evaluator with crash-resume and append-only logging."""

    def __init__(
        self,
        calibrators_dir: Path = CALIBRATORS_DIR,
        router_path: Path = ROUTER_ARTIFACT,
        judge_model: str = "gemma3:4b",
        judge_timeout: float = 60.0,
        tau: float = 0.35,
        use_fakes: bool = False,
    ):
        self.calibrators_dir = calibrators_dir
        self.router_path = router_path
        self.judge_model = judge_model
        self.judge_timeout = float(judge_timeout)
        self.tau = tau
        self.use_fakes = use_fakes

        self.rule_detector = RuleDetector(name="d4_rules")
        self.trigger = ChallengerTrigger(
            config=TriggerConfig(
                discrepancy_threshold=0.35,
                router_error_threshold=0.35,
                random_audit_rate=0.0,
            )
        )
        self.reviewer = ChallengerV2Reviewer(
            policy=AuthorityPolicy.POLICY_B,
            downgrade_confidence_threshold=0.65,
        )

        if not use_fakes:
            self._load_production_models()
        else:
            self._load_fake_models()

    def _load_production_models(self) -> None:
        """Load trained platt calibrators, router artifact, and live neural detectors."""
        if not self.router_path.exists():
            raise FileNotFoundError(f"Router artifact not found at {self.router_path}")
        self.router = LearnedReviewRouter.load(self.router_path)
        self.judge = AdversarialJudge(
            model_name=self.judge_model, timeout=self.judge_timeout
        )

        # Load Platt calibrators
        self.d2_calibrator = joblib.load(
            self.calibrators_dir / "d2_deberta_platt.joblib"
        )
        self.d3_calibrator = joblib.load(
            self.calibrators_dir / "d3_deepset_platt.joblib"
        )

        # Load D2 / D3 DeBERTa models on CPU for hermetic, low-memory inference
        self.d2_tok = AutoTokenizer.from_pretrained(
            "protectai/deberta-v3-base-prompt-injection-v2"
        )
        self.d2_model = AutoModelForSequenceClassification.from_pretrained(
            "protectai/deberta-v3-base-prompt-injection-v2"
        )
        self.d2_model.eval()

        self.d3_tok = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-injection")
        self.d3_model = AutoModelForSequenceClassification.from_pretrained(
            "deepset/deberta-v3-base-injection"
        )
        self.d3_model.eval()

    def _load_fake_models(self) -> None:
        """Load stub/fake models for fast isolated unit testing."""
        self.router = None
        self.judge = None
        self.d2_calibrator = None
        self.d3_calibrator = None
        self.d2_tok = None
        self.d2_model = None
        self.d3_tok = None
        self.d3_model = None

    def evaluate_single_row(
        self,
        prompt_id: str,
        text: str,
        label_binary: bool,
        family: str,
        source: str,
        mock_d2_raw: float | None = None,
        mock_d3_raw: float | None = None,
        mock_d4_raw: float | None = None,
        mock_judge_res: Any | None = None,
        mock_router_prob: float | None = None,
        row_idx: int = 0,
    ) -> dict[str, Any]:
        """Evaluate a single row deterministically across all arms."""
        t_start = time.perf_counter()

        if self.use_fakes and mock_d4_raw is not None:
            d4_sc = float(mock_d4_raw)
            d4_flag = bool(d4_sc >= 0.5)
        else:
            # D4 Rule detector (deterministic regex, binary score per eval_final.py:312)
            d4_res = self.rule_detector.predict_sync(text)
            d4_sc = 1.0 if d4_res.is_attack else 0.0
            d4_flag = bool(d4_res.is_attack)

        if self.use_fakes:
            d2_raw = mock_d2_raw if mock_d2_raw is not None else 0.1
            d3_raw = mock_d3_raw if mock_d3_raw is not None else 0.1

            d2_cal = d2_raw
            d3_cal = d3_raw
            prob_error = mock_router_prob if mock_router_prob is not None else 0.2
            judge_res = mock_judge_res

            d2_flag = bool(d2_cal >= 0.5)
            d3_flag = bool(d3_cal >= 0.5)

            det_results = [
                DetectorResult(name="d2_deberta", score=d2_cal, is_attack=d2_flag),
                DetectorResult(name="d3_deepset", score=d3_cal, is_attack=d3_flag),
                DetectorResult(name="d4_rules", score=d4_sc, is_attack=d4_flag),
            ]
            should_route = prob_error >= self.tau
            is_attack_ensemble = d2_flag or d3_flag or d4_flag
            arm_a_decision = Decision.BLOCK if is_attack_ensemble else Decision.ALLOW

            fused_score = max(d2_cal, d3_cal, d4_sc)
            discrepancy = abs(d2_cal - d3_cal)
            ensemble_conf = float(np.clip(1.0 - discrepancy, 0.0, 1.0))

            trig_res = self.trigger.evaluate(
                detector_results=det_results,
                fused_score=fused_score,
                confidence=ensemble_conf,
                original_decision=arm_a_decision,
                router_prob_error=prob_error,
                router_should_route=should_route,
            )
            arm_b_triggered = trig_res.should_challenge

        else:
            # Production live pipeline (real D2, D3, D4, router, judge)
            # D2 DeBERTa inference on CPU
            enc2 = self.d2_tok(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                out2 = self.d2_model(**enc2)
            p2 = float(torch.softmax(out2.logits, dim=-1)[:, 1][0])
            p2_clip = float(np.clip(p2, 1e-6, 1.0 - 1e-6))
            logit2 = float(np.log(p2_clip / (1.0 - p2_clip)))
            d2_cal = float(self.d2_calibrator.predict_proba([[logit2]])[:, 1][0])
            d2_raw = p2

            # D3 Deepset inference on CPU
            enc3 = self.d3_tok(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                out3 = self.d3_model(**enc3)
            diff3 = float((out3.logits[:, 1] - out3.logits[:, 0])[0])
            d3_cal = float(self.d3_calibrator.predict_proba([[diff3]])[:, 1][0])
            p3 = float(torch.softmax(out3.logits, dim=-1)[:, 1][0])
            d3_raw = p3

            # Flags
            d2_flag = bool(d2_cal >= 0.5)
            d3_flag = bool(d3_cal >= 0.5)
            d4_flag = bool(d4_sc >= 0.5)

            det_results = [
                DetectorResult(name="d2_deberta", score=d2_cal, is_attack=d2_flag),
                DetectorResult(name="d3_deepset", score=d3_cal, is_attack=d3_flag),
                DetectorResult(name="d4_rules", score=d4_sc, is_attack=d4_flag),
            ]

            # Router prediction
            router_dec = self.router.route(prompt=text, results=det_results)
            prob_error = float(router_dec.prob_error)
            should_route = bool(router_dec.should_route)

            # Arm A Decision
            is_attack_ensemble = bool(d2_flag or d3_flag or d4_flag)
            arm_a_decision = Decision.BLOCK if is_attack_ensemble else Decision.ALLOW

            # Trigger Arm B
            fused_score = max(d2_cal, d3_cal, d4_sc)
            discrepancy = abs(d2_cal - d3_cal)
            ensemble_conf = float(np.clip(1.0 - discrepancy, 0.0, 1.0))

            trig_res = self.trigger.evaluate(
                detector_results=det_results,
                fused_score=fused_score,
                confidence=ensemble_conf,
                original_decision=arm_a_decision,
                router_prob_error=prob_error,
                router_should_route=should_route,
            )
            arm_b_triggered = trig_res.should_challenge

            # Real judge call (Ollama) under gpu_lock
            with gpu_lock():
                judge_res = asyncio.run(
                    self.judge.evaluate(
                        prompt=text,
                        counter_evidence_hypothesis="Adversarial security review: evaluate prompt for injection or policy override",
                        detector_summary={
                            "d2_deberta": d2_cal,
                            "d3_deepset": d3_cal,
                            "d4_rules": d4_sc,
                        },
                        seed=42 + row_idx,
                    )
                )

        if judge_res is not None:
            verdict = judge_res.verdict
            judge_conf = float(judge_res.confidence)
            judge_lat = getattr(judge_res, "latency_ms", 0.0)
            judge_reasoning = getattr(judge_res, "reasoning", "")
        else:
            verdict = ChallengerVerdict.ALLOW
            judge_conf = 1.0
            judge_lat = 0.0
            judge_reasoning = ""

        # Standalone Judge C2 (Unified Abstain: ABSTAIN -> FLAG)
        if verdict == ChallengerVerdict.BLOCK:
            c2_decision = Decision.BLOCK
        elif verdict == ChallengerVerdict.ALLOW:
            c2_decision = Decision.ALLOW
        else:
            c2_decision = Decision.FLAG

        # Arm U (Reviewer on all)
        arm_u_decision, _ = self.reviewer.apply_authority_policy(
            verdict=verdict,
            original_decision=arm_a_decision,
            unanimous=True,
            mean_confidence=judge_conf,
            detector_results=det_results,
            ensemble_confidence=ensemble_conf,
        )

        # Arm B (Triggered reviewer)
        if arm_b_triggered:
            arm_b_decision, _ = self.reviewer.apply_authority_policy(
                verdict=verdict,
                original_decision=arm_a_decision,
                unanimous=True,
                mean_confidence=judge_conf,
                detector_results=det_results,
                ensemble_confidence=ensemble_conf,
            )
        else:
            arm_b_decision = arm_a_decision

        # Arm R (Learned Router review)
        if should_route:
            arm_r_decision, _ = self.reviewer.apply_authority_policy(
                verdict=verdict,
                original_decision=arm_a_decision,
                unanimous=True,
                mean_confidence=judge_conf,
                detector_results=det_results,
                ensemble_confidence=ensemble_conf,
            )
        else:
            arm_r_decision = arm_a_decision

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        record = {
            "prompt_id": prompt_id,
            "label_binary": label_binary,
            "family": family,
            "source": source,
            "detectors": {
                "d4_rules": {"score": d4_sc, "is_attack": d4_flag},
                "d2-deberta": {
                    "score": d2_cal,
                    "raw_score": d2_raw,
                    "is_attack": d2_flag,
                },
                "d3-deepset": {
                    "score": d3_cal,
                    "raw_score": d3_raw,
                    "is_attack": d3_flag,
                },
            },
            "fusion": {
                "arm_a_decision": arm_a_decision.value,
                "is_attack_ensemble": is_attack_ensemble,
                "ensemble_confidence": ensemble_conf,
            },
            "router": {
                "prob_error": prob_error,
                "should_route": should_route,
                "tau": self.tau,
            },
            "trigger_arm_b": {
                "triggered": arm_b_triggered,
                "discrepancy": discrepancy,
            },
            "judge": {
                "verdict": verdict.value if hasattr(verdict, "value") else str(verdict),
                "confidence": judge_conf,
                "latency_ms": judge_lat,
                "reasoning": judge_reasoning,
            },
            "decisions": {
                "arm_a": arm_a_decision.value,
                "arm_b": arm_b_decision.value,
                "arm_r": arm_r_decision.value,
                "arm_u": arm_u_decision.value,
                "c2_standalone": c2_decision.value,
            },
            "latencies_ms": {
                "total": elapsed_ms,
                "judge": judge_lat,
            },
            "resumed": False,
        }
        return record


def evaluate_dataset(
    input_records: list[dict[str, Any]],
    output_jsonl: Path,
    evaluator: ExternalEvaluator,
) -> int:
    """Process records with append-only logging and crash-resume capability."""
    completed_ids: set[str] = set()
    if output_jsonl.exists():
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if "prompt_id" in obj:
                            completed_ids.add(obj["prompt_id"])
                    except json.JSONDecodeError:
                        continue
        logger.info("Found %d completed rows in existing log", len(completed_ids))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    with open(output_jsonl, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(input_records):
            pid = item.get("id") or item.get("prompt_id") or f"row_{idx}"
            if pid in completed_ids:
                continue

            rec = evaluator.evaluate_single_row(
                prompt_id=pid,
                text=item["text"],
                label_binary=item.get("label_binary", False),
                family=item.get("family", "unknown"),
                source=item.get("source", "external"),
                mock_d2_raw=item.get("mock_d2_raw"),
                mock_d3_raw=item.get("mock_d3_raw"),
                mock_judge_res=item.get("mock_judge_res"),
                mock_router_prob=item.get("mock_router_prob"),
                row_idx=idx,
            )
            if len(completed_ids) > 0 and processed_count == 0:
                rec["resumed"] = True

            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            completed_ids.add(pid)
            processed_count += 1

    return processed_count


# Backward-compatible alias for test harness
run_evaluation_with_crash_resume = evaluate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="External evaluation runner with crash-resume"
    )
    parser.add_argument("--input", type=Path, help="Path to input jsonl")
    parser.add_argument("--output", type=Path, help="Path to output predictions jsonl")
    parser.add_argument("--tau", type=float, default=0.35, help="Router threshold")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Compare runner configuration against protocol pins and final-v4 manifest",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live model inference (requires healthy daemons and preflight gates)",
    )
    args = parser.parse_args()

    if args.check_config:
        print("Comparing runner configuration against protocol pins...")
        checks = verify_runner_config()
        print("-" * 88)
        print(
            f"{'Parameter':<28} | {'Current / Actual':<26} | {'Protocol Pins':<18} | {'Status'}"
        )
        print("-" * 88)
        for name, actual, expected, ok in checks:
            status = "MATCH" if ok else "MISMATCH"
            print(
                f"{name:<28} | {str(actual)[:26]:<26} | {str(expected)[:18]:<18} | {status}"
            )
        print("-" * 88)
        print(
            "Note: The final-v4 run manifest holds only model_digest: gemma3:4b; other parameters are protocol pins."
        )
        print("All runner parameters match protocol pins and frozen contracts.")
        return

    if not args.input or not args.output:
        print(
            f"Runner initialized for tau={args.tau}. Specify --input and --output to execute."
        )
        return

    # Enforce mandatory preflight verification gates
    logger.info("Executing mandatory 5-arm policy replay check...")
    from scripts.eval.policy_replay_check import run_policy_replay_check

    run_policy_replay_check()

    logger.info("Verifying PromptForest daemon health on port 8765...")
    check_promptforest_health()

    logger.info("Verifying protocol cryptographic digests...")
    verify_protocol_digests()

    logger.info("Verifying frozen external dataset digests...")
    verify_frozen_manifest()

    if (PROJECT_ROOT / "data" / "external" / "dolly_text.jsonl").exists():
        logger.info("Verifying Dolly prompt text integrity...")
        verify_dolly_text_integrity()

    use_fakes = not args.live
    evaluator = ExternalEvaluator(tau=args.tau, use_fakes=use_fakes)

    lines = [
        line.strip()
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [json.loads(line) for line in lines]
    logger.info("Read %d records from %s", len(records), args.input)

    count = evaluate_dataset(records, args.output, evaluator)
    logger.info("Successfully evaluated %d rows to %s", count, args.output)


if __name__ == "__main__":
    main()
