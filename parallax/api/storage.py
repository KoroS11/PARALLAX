"""Storage provider and repository for scans and evaluation runs."""

import threading
from typing import Any

from parallax.contracts.api import EvalRunResponse, ScanResponse


class ApiStorage:
    """In-memory storage provider for API serving plane with thread-safe access."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url
        self._lock = threading.Lock()
        self._scans: list[dict[str, Any]] = []
        self._eval_runs: list[EvalRunResponse] = []

    def record_scan(self, scan: ScanResponse, prompt: str) -> None:
        """Record a scan result."""
        with self._lock:
            self._scans.append(
                {
                    "trace_id": scan.trace_id,
                    "prompt": prompt,
                    "decision": scan.decision.value,
                    "final_technique": scan.final_technique,
                    "disagreement": scan.disagreement,
                    "confidence": scan.confidence,
                    "latency_ms": scan.latency_ms,
                    "cached": scan.cached,
                }
            )

    def get_scans(self) -> list[dict[str, Any]]:
        """Return list of recorded scans."""
        with self._lock:
            return list(self._scans)

    def add_eval_run(self, run: EvalRunResponse) -> None:
        """Add an evaluation run record."""
        with self._lock:
            self._eval_runs.append(run)

    def get_eval_runs(self) -> list[EvalRunResponse]:
        """Retrieve all recorded evaluation runs."""
        with self._lock:
            return list(self._eval_runs)
