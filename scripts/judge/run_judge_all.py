"""Wave H: Run Live LLM Judge on All Evaluation Prompts with Masked Inputs.

Executes live Ollama judge (comparing gemma3:4b, hlyn-labs-3b:latest, qwen3:8b)
with strict industrial prompt template, evidence quote validation, and masked inputs.

Splits:
- dev: data/splits/dev.jsonl (rebuilt N=48)
- val: data/splits/val.jsonl (rebuilt N=48)
- held_out_v3: data/splits/held_out_v3.jsonl (untouched N=46)
- dev_legacy: data/splits/dev_legacy.jsonl (demoted legacy corpus)
- external: data/external_eval/external_benchmark.jsonl (N=110)

Outputs detailed per-prompt audit logs to:
eval/logs/<run_id>/judge_<model_safe>.jsonl
eval/logs/latest_judge_<model_safe>.jsonl
eval/logs/sample_judge_<model_safe>.jsonl
Updates: eval/logs/MANIFEST.json

Cache key: (prompt_id, masked_text_hash, model_digest, template_hash)
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
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mask import mask_prompt_text
from parallax.challenger.judge import (
    DEFAULT_COUNTER_HYPOTHESIS,
    DEFAULT_OLLAMA_URL,
    JUDGE_SYSTEM_PROMPT,
    compute_template_hash,
    format_judge_evaluation_prompt,
    parse_raw_judge_output,
    validate_evidence,
)
from parallax.challenger.schemas import ChallengerVerdict
from scripts.gpu_lock import gpu_lock

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [judge-all] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("judge-all")

SPLIT_FILES = {
    "dev": REPO_ROOT / "data" / "splits" / "dev.jsonl",
    "val": REPO_ROOT / "data" / "splits" / "val.jsonl",
    "held_out_v3": REPO_ROOT / "data" / "splits" / "held_out_v3.jsonl",
    "dev_legacy": REPO_ROOT / "data" / "splits" / "dev_legacy.jsonl",
    "external": REPO_ROOT / "data" / "external_eval" / "external_benchmark.jsonl",
}

LOGS_DIR = REPO_ROOT / "eval" / "logs"
CACHE_FILE = REPO_ROOT / "eval" / "cache" / "judge_cache.json"
MANIFEST_FILE = LOGS_DIR / "MANIFEST.json"


def get_model_digest(model_name: str, base_url: str = DEFAULT_OLLAMA_URL) -> str:
    """Fetch exact model sha256 digest from Ollama tags/show endpoint."""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "PARALLAX/WaveH"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                name = m.get("name", "")
                if name == model_name or name.startswith(f"{model_name}:"):
                    digest = m.get("digest", "")
                    if digest:
                        return digest
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not query Ollama /api/tags for %s digest: %s", model_name, exc
        )

    # Fallback to /api/show
    try:
        url = f"{base_url.rstrip('/')}/api/show"
        req = urllib.request.Request(
            url,
            data=json.dumps({"name": model_name}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            digest = data.get("digest") or data.get("details", {}).get(
                "parent_model", ""
            )
            if digest:
                return digest
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not query Ollama /api/show for %s: %s", model_name, exc)

    # Known stable local digests on this machine as fallback
    known_digests = {
        "hlyn-labs-3b:latest": "d90b3206ce655d9415afd336bfde27cd344ba539ed12e6b393578091cb539e64",
        "hlyn-labs-3b": "d90b3206ce655d9415afd336bfde27cd344ba539ed12e6b393578091cb539e64",
        "gemma3:4b": "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",
        "qwen3:8b": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
    }
    return known_digests.get(model_name, "unknown_digest")


def load_cache() -> dict[str, dict[str, Any]]:
    """Load judge cache from disk, backfilling from existing log files if present."""
    cache: dict[str, dict[str, Any]] = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(
                "Loaded %d records from judge cache: %s", len(cache), CACHE_FILE
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error reading %s: %s", CACHE_FILE, exc)

    # Scan existing log files in eval/logs/
    if LOGS_DIR.exists():
        for jsonl_path in LOGS_DIR.glob("*/*.jsonl"):
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        p_id = row.get("prompt_id")
                        m_hash = row.get("masked_text_hash")
                        m_digest = row.get("model_digest")
                        t_hash = row.get("prompt_template_hash")
                        if p_id and m_hash and m_digest and t_hash:
                            key = f"{p_id}::{m_hash}::{m_digest}::{t_hash}"
                            if key not in cache:
                                cache[key] = row
            except Exception as err:  # noqa: BLE001
                logger.debug("Could not read prior log %s: %s", jsonl_path, err)
    return cache


def save_cache(cache: dict[str, dict[str, Any]]) -> None:
    """Save judge cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = CACHE_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    temp_file.replace(CACHE_FILE)
    logger.info("Saved %d records to judge cache: %s", len(cache), CACHE_FILE)


def load_split_prompts(split_name: str, split_path: Path) -> list[dict[str, Any]]:
    """Load samples from JSONL split file with masked text."""
    if not split_path.exists():
        raise FileNotFoundError(f"Split file missing: {split_path}")

    samples = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Label ground truth (for audit reporting only)
            if "label_binary" in data:
                lbl = int(data["label_binary"])
            elif "is_attack" in data:
                lbl = 1 if data["is_attack"] else 0
            else:
                lbl = 0

            raw_text = str(data.get("text") or data.get("prompt", ""))
            masked_text = mask_prompt_text(raw_text)
            masked_hash = hashlib.sha256(masked_text.encode("utf-8")).hexdigest()[:16]

            # Scenario family
            fam = data.get("family") or "unknown"
            scen = data.get("metadata", {}).get("scenario_id", "general")

            samples.append(
                {
                    "prompt_id": str(data["id"]),
                    "text": masked_text,
                    "raw_text": raw_text,
                    "masked_text_hash": masked_hash,
                    "label": lbl,
                    "split": split_name,
                    "family": fam,
                    "scenario_id": scen,
                    "raw_data": data,
                }
            )
    return samples


async def call_ollama_judge(
    client: httpx.AsyncClient,
    model_name: str,
    prompt: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    hypothesis: str = DEFAULT_COUNTER_HYPOTHESIS,
    timeout: float = 60.0,
) -> tuple[str, float, str | None]:
    """Execute raw Ollama chat call with strict options."""
    user_msg = format_judge_evaluation_prompt(
        untrusted_prompt=prompt,
        counter_evidence_hypothesis=hypothesis,
        detector_summary={},
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 128,
            "num_ctx": 1024,
            "stop": ["\n\n", "```"],
        },
    }

    t0 = time.perf_counter()
    raw_content = ""
    error_msg: str | None = None
    url = f"{base_url.rstrip('/')}/api/chat"

    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                raw_content = str(data["message"].get("content", ""))
            elif "response" in data:
                raw_content = str(data.get("response", ""))
            else:
                raw_content = str(data)
    except httpx.TimeoutException as exc:
        error_msg = f"Timeout after {timeout}s: {exc}"
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return raw_content, latency_ms, error_msg


def update_manifest(files_to_update: list[tuple[Path, int]]) -> None:
    """Update eval/logs/MANIFEST.json with newly created log files."""
    manifest_data: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "files": [],
    }
    existing_entries: dict[str, dict[str, Any]] = {}

    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                manifest_data["generated_at"] = data.get(
                    "generated_at", manifest_data["generated_at"]
                )
                for entry in data.get("files", []):
                    existing_entries[entry.get("rel_path", "")] = entry
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read MANIFEST.json: %s", exc)

    for p, num_rec in files_to_update:
        if not p.exists():
            continue
        rel = str(p.relative_to(LOGS_DIR))
        content = p.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        existing_entries[rel] = {
            "rel_path": rel,
            "size_bytes": len(content),
            "sha256": sha,
            "num_records": num_rec,
        }

    manifest_data["files"] = sorted(
        existing_entries.values(), key=lambda x: x["rel_path"]
    )
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(
        "Updated MANIFEST.json with %d tracked files", len(manifest_data["files"])
    )


async def run_evaluation_for_model(
    model_name: str,
    splits: list[str],
    run_id: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 60.0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run evaluation on all split prompts for the specified model."""
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    out_dir = LOGS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"judge_{safe_model}.jsonl"
    latest_jsonl = LOGS_DIR / f"latest_judge_{safe_model}.jsonl"
    sample_jsonl = LOGS_DIR / f"sample_judge_{safe_model}.jsonl"

    model_digest = get_model_digest(model_name, base_url)
    template_hash = compute_template_hash(DEFAULT_COUNTER_HYPOTHESIS)
    cache = load_cache()

    logger.info("=" * 60)
    logger.info("RUNNING LIVE JUDGE FOR MODEL: %s", model_name)
    logger.info("Model Digest: %s", model_digest)
    logger.info("Prompt Template Hash: %s", template_hash)
    logger.info("Target Splits: %s", ", ".join(splits))
    logger.info("Output Log: %s", out_jsonl)
    logger.info("=" * 60)

    # Gather prompts
    all_prompts: list[dict[str, Any]] = []
    for split_name in splits:
        if split_name not in SPLIT_FILES:
            logger.warning("Unknown split name: %s, skipping", split_name)
            continue
        split_path = SPLIT_FILES[split_name]
        split_prompts = load_split_prompts(split_name, split_path)
        if limit:
            split_prompts = split_prompts[:limit]
        all_prompts.extend(split_prompts)
        logger.info(
            "Split %s: loaded %d prompts from %s",
            split_name,
            len(split_prompts),
            split_path,
        )

    total_prompts = len(all_prompts)
    logger.info("Total prompts to process: %d", total_prompts)

    records: list[dict[str, Any]] = []
    live_count = 0
    cached_count = 0
    parse_ok_count = 0
    abstain_count = 0

    # Per-split confusion matrix tracking
    split_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "abstain": 0, "total": 0}
    )
    # Per-family benign tracking
    family_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"benign_total": 0, "false_blocks": 0, "allows": 0, "abstains": 0}
    )
    false_block_records: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        logger.info("Acquiring GPU lock for Ollama inference...")
        with gpu_lock(timeout_seconds=1800):
            logger.info("GPU lock acquired. Starting prompt evaluation...")

            for idx, item in enumerate(all_prompts, start=1):
                p_id = item["prompt_id"]
                split = item["split"]
                label = item["label"]
                masked_text = item["text"]
                m_hash = item["masked_text_hash"]
                family = item["family"]
                cache_key = f"{p_id}::{m_hash}::{model_digest}::{template_hash}"

                if cache_key in cache:
                    cached_row = cache[cache_key]
                    cached_run_id = cached_row.get(
                        "cached_from_run_id"
                    ) or cached_row.get("run_id", "cached_pre_wave_h")
                    record = {
                        "prompt_id": p_id,
                        "split": split,
                        "label": label,
                        "model": model_name,
                        "model_digest": model_digest,
                        "prompt_template_hash": template_hash,
                        "masked_text_hash": m_hash,
                        "raw_output": cached_row.get("raw_output", ""),
                        "parsed_verdict": cached_row.get("parsed_verdict", "abstain"),
                        "parse_ok": cached_row.get("parse_ok", False),
                        "abstain": cached_row.get("abstain", True),
                        "confidence": cached_row.get("confidence", 0.0),
                        "evidence_quote": cached_row.get("evidence_quote", ""),
                        "reasoning": cached_row.get("reasoning", ""),
                        "attack_class": cached_row.get("attack_class", "none"),
                        "latency_ms": cached_row.get("latency_ms", 0.0),
                        "timestamp": datetime.now(UTC).isoformat(),
                        "cached_from_run_id": cached_run_id,
                        "run_id": run_id,
                    }
                    cached_count += 1
                else:
                    raw_content, lat_ms, err = await call_ollama_judge(
                        client=client,
                        model_name=model_name,
                        prompt=masked_text,
                        base_url=base_url,
                        hypothesis=DEFAULT_COUNTER_HYPOTHESIS,
                        timeout=timeout,
                    )
                    live_count += 1

                    if err:
                        parse_ok = False
                        parsed_verdict = "abstain"
                        is_abstain = True
                        confidence = 0.0
                        reasoning = f"Error: {err}"
                        evidence_quote = ""
                        attack_class = "none"
                    else:
                        parsed = parse_raw_judge_output(raw_content)
                        if parsed is None:
                            parse_ok = False
                            parsed_verdict = "abstain"
                            is_abstain = True
                            confidence = 0.0
                            reasoning = "Failed to parse JSON"
                            evidence_quote = ""
                            attack_class = "none"
                        else:
                            parse_ok = True
                            parsed_verdict = parsed.verdict.value
                            is_abstain = parsed.verdict == ChallengerVerdict.ABSTAIN
                            confidence = parsed.confidence
                            reasoning = parsed.reasoning
                            evidence_quote = parsed.evidence_quote
                            attack_class = parsed.attack_class

                            # Mandatory Evidence Validation:
                            # If verdict is block or escalate, verify that evidence_quote is present verbatim
                            # in untrusted input. If missing, empty, or not found, REJECT verdict and abstain!
                            if parsed_verdict in (
                                "block",
                                "escalate",
                            ) and not validate_evidence(masked_text, evidence_quote):
                                parsed_verdict = "abstain"
                                is_abstain = True
                                confidence = 0.0
                                reasoning = (
                                    f"Evidence validation rejected block: quote '{evidence_quote}' "
                                    f"not found verbatim in untrusted input. Original reasoning: {reasoning}"
                                )

                    record = {
                        "prompt_id": p_id,
                        "split": split,
                        "label": label,
                        "model": model_name,
                        "model_digest": model_digest,
                        "prompt_template_hash": template_hash,
                        "masked_text_hash": m_hash,
                        "raw_output": raw_content,
                        "parsed_verdict": parsed_verdict,
                        "parse_ok": parse_ok,
                        "abstain": is_abstain,
                        "confidence": confidence,
                        "evidence_quote": evidence_quote,
                        "reasoning": reasoning,
                        "attack_class": attack_class,
                        "latency_ms": lat_ms,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "cached_from_run_id": None,
                        "run_id": run_id,
                    }
                    cache[cache_key] = record

                records.append(record)

                if record["parse_ok"]:
                    parse_ok_count += 1
                if record["abstain"]:
                    abstain_count += 1

                # Update stats
                v = record["parsed_verdict"]
                split_stats[split]["total"] += 1
                if record["abstain"]:
                    split_stats[split]["abstain"] += 1

                if label == 1:
                    if v in ("block", "escalate"):
                        split_stats[split]["tp"] += 1
                    else:
                        split_stats[split]["fn"] += 1
                else:
                    family_stats[family]["benign_total"] += 1
                    if v in ("block", "escalate"):
                        split_stats[split]["fp"] += 1
                        family_stats[family]["false_blocks"] += 1
                        false_block_records.append(
                            {
                                "prompt_id": p_id,
                                "split": split,
                                "family": family,
                                "evidence_quote": record["evidence_quote"],
                                "reasoning": record["reasoning"],
                                "text_snippet": masked_text[:120],
                            }
                        )
                    elif record["abstain"]:
                        family_stats[family]["abstains"] += 1
                        split_stats[split]["tn"] += 1
                    else:
                        family_stats[family]["allows"] += 1
                        split_stats[split]["tn"] += 1

                if idx % 25 == 0 or idx == total_prompts:
                    logger.info(
                        "Progress: %d/%d (Live: %d, Cached: %d, ParseOK: %d, Abstain: %d)",
                        idx,
                        total_prompts,
                        live_count,
                        cached_count,
                        parse_ok_count,
                        abstain_count,
                    )

    # Write out log file
    with open(out_jsonl, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        f.writelines(json.dumps(r) + "\n" for r in records)
    logger.info("Wrote %d records to %s", len(records), out_jsonl)

    # Write latest copy
    with open(latest_jsonl, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        f.writelines(json.dumps(r) + "\n" for r in records)
    logger.info("Updated latest log: %s", latest_jsonl)

    # Write sample copy (first 10 records)
    with open(sample_jsonl, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        f.writelines(json.dumps(r) + "\n" for r in records[:10])
    logger.info("Updated sample log: %s", sample_jsonl)

    # Update manifest
    update_manifest(
        [
            (out_jsonl, len(records)),
            (latest_jsonl, len(records)),
            (sample_jsonl, min(10, len(records))),
        ]
    )

    # Save cache
    save_cache(cache)

    # Calculate overall metrics
    tot_tp = sum(s["tp"] for s in split_stats.values())
    tot_fp = sum(s["fp"] for s in split_stats.values())
    tot_tn = sum(s["tn"] for s in split_stats.values())
    tot_fn = sum(s["fn"] for s in split_stats.values())
    tot_attacks = tot_tp + tot_fn
    tot_benigns = tot_tn + tot_fp
    raw_recall = (tot_tp / tot_attacks * 100.0) if tot_attacks > 0 else 0.0
    raw_fpr = (tot_fp / tot_benigns * 100.0) if tot_benigns > 0 else 0.0

    logger.info("-" * 50)
    logger.info("SUMMARY FOR MODEL %s (Run %s):", model_name, run_id)
    logger.info(
        "Total Evaluated: %d (Live: %d, Cached: %d)",
        total_prompts,
        live_count,
        cached_count,
    )
    logger.info(
        "Parse Success Rate: %.1f%% (%d/%d)",
        (parse_ok_count / total_prompts * 100.0) if total_prompts > 0 else 0.0,
        parse_ok_count,
        total_prompts,
    )
    logger.info(
        "Abstain Rate: %.1f%% (%d/%d)",
        (abstain_count / total_prompts * 100.0) if total_prompts > 0 else 0.0,
        abstain_count,
        total_prompts,
    )
    logger.info(
        "Overall Direct Recall on Attacks: %.2f%% (%d/%d)",
        raw_recall,
        tot_tp,
        tot_attacks,
    )
    logger.info(
        "Overall Direct FPR on Benign: %.2f%% (%d/%d)",
        raw_fpr,
        tot_fp,
        tot_benigns,
    )
    logger.info("-" * 50)

    # Split breakdown
    for s_name, s_val in split_stats.items():
        s_att = s_val["tp"] + s_val["fn"]
        s_ben = s_val["fp"] + s_val["tn"]
        rec = (s_val["tp"] / s_att * 100.0) if s_att > 0 else 0.0
        fpr = (s_val["fp"] / s_ben * 100.0) if s_ben > 0 else 0.0
        logger.info(
            "Split %s: TP=%d, FN=%d (Recall=%.1f%%) | FP=%d, TN=%d (FPR=%.2f%%) | Abstains=%d",
            s_name,
            s_val["tp"],
            s_val["fn"],
            rec,
            s_val["fp"],
            s_val["tn"],
            fpr,
            s_val["abstain"],
        )

    # Family breakdown
    logger.info("Benign Family Breakdown:")
    for f_name, f_val in family_stats.items():
        f_fpr = (
            (f_val["false_blocks"] / f_val["benign_total"] * 100.0)
            if f_val["benign_total"] > 0
            else 0.0
        )
        logger.info(
            "  Family %s: Total Benign=%d, False Blocks=%d (FPR=%.2f%%), Allows=%d, Abstains=%d",
            f_name,
            f_val["benign_total"],
            f_val["false_blocks"],
            f_fpr,
            f_val["allows"],
            f_val["abstains"],
        )

    return {
        "model": model_name,
        "run_id": run_id,
        "total_prompts": total_prompts,
        "recall": raw_recall,
        "fpr": raw_fpr,
        "split_stats": dict(split_stats),
        "family_stats": dict(family_stats),
        "false_blocks": false_block_records,
        "out_jsonl": str(out_jsonl),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Live LLM Judge on All Evaluation Prompts"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma3:4b"],
        help="Models to evaluate (e.g. gemma3:4b, hlyn-labs-3b:latest, qwen3:8b)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev", "val"],
        help="Splits to evaluate (dev, val; held_out_v3 evaluated only in final one-shot)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (defaults to auto-generated UTC ID)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Ollama request timeout in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of prompts per split (for dry runs)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help="Ollama base URL",
    )

    args = parser.parse_args()

    run_id = (
        args.run_id
        or f"judge-run-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    )
    logger.info("Starting Judge Run ID: %s", run_id)

    results = []
    for model in args.models:
        res = asyncio.run(
            run_evaluation_for_model(
                model_name=model,
                splits=args.splits,
                run_id=run_id,
                base_url=args.base_url,
                timeout=args.timeout,
                limit=args.limit,
            )
        )
        results.append(res)

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON SUMMARY:")
    for r in results:
        print(f"Model: {r['model']}")
        print(f"  Overall Recall: {r['recall']:.2f}% | Overall FPR: {r['fpr']:.2f}%")
        for s, st in r["split_stats"].items():
            s_att = st["tp"] + st["fn"]
            s_ben = st["fp"] + st["tn"]
            rec = (st["tp"] / s_att * 100.0) if s_att > 0 else 0.0
            fpr = (st["fp"] / s_ben * 100.0) if s_ben > 0 else 0.0
            print(
                f"  Split {s}: Recall={rec:.2f}% ({st['tp']}/{s_att}) | FPR={fpr:.2f}% ({st['fp']}/{s_ben}) | Abstain={st['abstain']}"
            )
    print("=" * 60)


if __name__ == "__main__":
    main()
