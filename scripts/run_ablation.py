"""CLI execution runner for PARALLAX Wave D Ablation Study.

Runs Arms A, B, C, C2 on held-out test data, computes 95% bootstrap CIs,
and writes reports/ablation_results.json and reports/ABLATION_REPORT.md.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parallax.eval.ablation import run_full_ablation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


async def main() -> None:
    output_dir = Path("reports")
    await run_full_ablation(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
