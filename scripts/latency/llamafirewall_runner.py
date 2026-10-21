"""Isolated runner for LlamaFirewall PromptGuardScanner latency benchmarking.

Executed in .venv_llamafirewall to ensure complete dependency isolation.
Measures cold latency (1st call) and warm latencies for N calls under gpu_lock.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path so gpu_lock can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llamafirewall import UserMessage
from llamafirewall.scanners.prompt_guard_scanner import PromptGuardScanner

from scripts.gpu_lock import gpu_lock


async def run_benchmark(prompts_file: Path, output_file: Path) -> None:
    data = json.loads(prompts_file.read_text(encoding="utf-8"))
    cold_prompt = data["cold_prompt"]
    warm_prompts = data["warm_prompts"]

    with gpu_lock(timeout_seconds=300):
        scanner = PromptGuardScanner(block_threshold=0.9)

        # 1. Cold latency (1st call)
        t0 = time.perf_counter()
        _ = await scanner.scan(UserMessage(cold_prompt))
        cold_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Warm latencies
        warm_latencies = []
        for prompt in warm_prompts:
            t_start = time.perf_counter()
            _ = await scanner.scan(UserMessage(prompt))
            elapsed = (time.perf_counter() - t_start) * 1000.0
            warm_latencies.append(round(elapsed, 2))

    result = {
        "cold_ms": round(cold_ms, 2),
        "warm_latencies": warm_latencies,
    }
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaFirewall Latency Runner")
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.prompts_file, args.output_file))


if __name__ == "__main__":
    main()
