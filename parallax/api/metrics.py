"""Prometheus and aggregated metrics collection for PARALLAX API."""

import threading
from collections import deque

import numpy as np

from parallax.contracts.api import Decision, MetricsSummaryResponse, ScanResponse


class ApiMetrics:
    """Thread-safe in-memory metrics recorder for Prometheus and dashboard stats."""

    def __init__(self, max_latency_samples: int = 10000) -> None:
        self._lock = threading.Lock()
        self._total_scans = 0
        self._allow_count = 0
        self._flag_count = 0
        self._block_count = 0
        self._disagreement_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._errors = 0
        self._latencies_ms: deque[float] = deque(maxlen=max_latency_samples)

    def record_scan(self, scan: ScanResponse) -> None:
        """Record a completed scan response into metrics."""
        with self._lock:
            self._total_scans += 1
            if scan.cached:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            if scan.decision == Decision.ALLOW:
                self._allow_count += 1
            elif scan.decision == Decision.FLAG:
                self._flag_count += 1
            elif scan.decision == Decision.BLOCK:
                self._block_count += 1

            if scan.disagreement:
                self._disagreement_count += 1

            self._latencies_ms.append(scan.latency_ms)

    def record_cache_hit(self) -> None:
        """Increment cache hit counter."""
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        """Increment cache miss counter."""
        with self._lock:
            self._cache_misses += 1

    def record_error(self) -> None:
        """Increment unhandled error counter."""
        with self._lock:
            self._errors += 1

    def get_metrics_summary(self) -> MetricsSummaryResponse:
        """Compute summary metrics conforming to MetricsSummaryResponse schema."""
        with self._lock:
            total = self._total_scans
            allows = self._allow_count
            flags = self._flag_count
            blocks = self._block_count
            disagreements = self._disagreement_count
            latencies = list(self._latencies_ms)

        if total > 0:
            disagreement_rate = round(disagreements / total, 4)
        else:
            disagreement_rate = 0.0

        if latencies:
            p50 = round(float(np.percentile(latencies, 50)), 2)
            p95 = round(float(np.percentile(latencies, 95)), 2)
        else:
            p50 = 0.0
            p95 = 0.0

        return MetricsSummaryResponse(
            total_scans=total,
            allow_count=allows,
            flag_count=flags,
            block_count=blocks,
            disagreement_rate=disagreement_rate,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
        )

    def render_prometheus(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        with self._lock:
            allows = self._allow_count
            flags = self._flag_count
            blocks = self._block_count
            disagreements = self._disagreement_count
            hits = self._cache_hits
            misses = self._cache_misses
            errors = self._errors
            latencies = list(self._latencies_ms)

        latency_sum_sec = sum(latencies) / 1000.0 if latencies else 0.0
        latency_count = len(latencies)

        lines = [
            "# HELP parallax_scans_total Total number of scans processed.",
            "# TYPE parallax_scans_total counter",
            f'parallax_scans_total{{decision="allow"}} {allows}',
            f'parallax_scans_total{{decision="flag"}} {flags}',
            f'parallax_scans_total{{decision="block"}} {blocks}',
            "",
            "# HELP parallax_scan_disagreements_total Scans where detectors disagreed.",
            "# TYPE parallax_scan_disagreements_total counter",
            f"parallax_scan_disagreements_total {disagreements}",
            "",
            "# HELP parallax_cache_requests_total Total cache lookups.",
            "# TYPE parallax_cache_requests_total counter",
            f'parallax_cache_requests_total{{result="hit"}} {hits}',
            f'parallax_cache_requests_total{{result="miss"}} {misses}',
            "",
            "# HELP parallax_errors_total Total errors encountered.",
            "# TYPE parallax_errors_total counter",
            f"parallax_errors_total {errors}",
            "",
            "# HELP parallax_scan_duration_seconds Latency of scans in seconds.",
            "# TYPE parallax_scan_duration_seconds summary",
            f"parallax_scan_duration_seconds_count {latency_count}",
            f"parallax_scan_duration_seconds_sum {latency_sum_sec:.4f}",
            "",
        ]
        return "\n".join(lines)
