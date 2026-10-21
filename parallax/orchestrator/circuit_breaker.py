"""Circuit breaker pattern for detector error isolation and fail-fast behavior."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any

from parallax.contracts.detectors import DetectorResult


class CircuitState(str, Enum):
    """Lifecycle states of the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised or wrapped when a detector call is rejected due to an OPEN circuit."""

    def __init__(self, detector_name: str, retry_after: float) -> None:
        self.detector_name = detector_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker for detector '{detector_name}' is OPEN. "
            f"Retry after {retry_after:.2f}s."
        )


class CircuitBreaker:
    """Per-detector circuit breaker with CLOSED -> OPEN -> HALF-OPEN -> CLOSED cycle.

    Parameters
    ----------
    name : str
        Detector or resource identifier.
    failure_threshold : int, default=3
        Number of consecutive failures required to trip the breaker from CLOSED to OPEN.
    recovery_timeout : float, default=5.0
        Cooldown period in seconds before an OPEN circuit transitions to HALF-OPEN.
    success_threshold : int, default=1
        Number of consecutive successful trial probes required to reset HALF-OPEN to CLOSED.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 5.0,
        success_threshold: int = 1,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError(
                f"failure_threshold must be positive, got {failure_threshold}"
            )
        if recovery_timeout < 0:
            raise ValueError(
                f"recovery_timeout cannot be negative, got {recovery_timeout}"
            )

        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def consecutive_failures(self) -> int:
        """Current count of consecutive failures."""
        return self._consecutive_failures

    @property
    def consecutive_successes(self) -> int:
        """Current count of consecutive successes in HALF-OPEN state."""
        return self._consecutive_successes

    @property
    def last_failure_time(self) -> float:
        """Timestamp (monotonic) of the most recent failure."""
        return self._last_failure_time

    @property
    def state(self) -> CircuitState:
        """Current state of the circuit, auto-advancing from OPEN to HALF-OPEN on timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    def record_success(self) -> None:
        """Record a successful execution, potentially closing a HALF-OPEN circuit."""
        current_state = self.state
        if current_state == CircuitState.HALF_OPEN:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._consecutive_failures = 0
                self._consecutive_successes = 0
        elif current_state == CircuitState.CLOSED:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Record an execution failure, potentially opening a CLOSED or HALF-OPEN circuit."""
        self._last_failure_time = time.monotonic()
        self._consecutive_failures += 1

        current_state = self.state
        if current_state == CircuitState.HALF_OPEN:
            # Any failure while probing in HALF-OPEN re-trips back to OPEN
            self._state = CircuitState.OPEN
            self._consecutive_successes = 0
        elif current_state == CircuitState.CLOSED:
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._consecutive_successes = 0

    def trip(self) -> None:
        """Manually force the circuit breaker into the OPEN state."""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.monotonic()
        self._consecutive_successes = 0

    def reset(self) -> None:
        """Manually reset the circuit breaker to the CLOSED state."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time = 0.0

    def time_until_recovery(self) -> float:
        """Seconds remaining before recovery timeout expires for an OPEN circuit."""
        if self._state != CircuitState.OPEN:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        remaining = self.recovery_timeout - elapsed
        return max(0.0, remaining)

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, DetectorResult]],
        *args: Any,
        **kwargs: Any,
    ) -> DetectorResult:
        """Execute detector prediction through the circuit breaker.

        Returns an abstained DetectorResult if the circuit is currently OPEN.
        """
        async with self._lock:
            current_state = self.state
            if current_state == CircuitState.OPEN:
                remaining = self.time_until_recovery()
                return DetectorResult(
                    name=self.name,
                    is_attack=None,
                    score=None,
                    attack_class=None,
                    latency_ms=0.0,
                    error=(
                        f"Circuit breaker is OPEN for detector '{self.name}'. "
                        f"Retry in {remaining:.2f}s."
                    ),
                )

        # Execute target function
        t0 = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            # Check if detector result itself is an error / abstention
            if result.error is not None or result.is_attack is None:
                self.record_failure()
            else:
                self.record_success()
            return result
        except Exception as exc:  # noqa: BLE001
            self.record_failure()
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            return DetectorResult(
                name=self.name,
                is_attack=None,
                score=None,
                attack_class=None,
                latency_ms=elapsed,
                error=f"Execution failed: {exc}",
            )


class CircuitBreakerManager:
    """Registry managing circuit breakers across all configured detectors."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 5.0,
        success_threshold: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_breaker(
        self,
        name: str,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
    ) -> CircuitBreaker:
        """Get or initialize a CircuitBreaker for the given detector name."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold or self.failure_threshold,
                recovery_timeout=(
                    recovery_timeout
                    if recovery_timeout is not None
                    else self.recovery_timeout
                ),
                success_threshold=self.success_threshold,
            )
        return self._breakers[name]

    def reset_all(self) -> None:
        """Reset all managed circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_states(self) -> dict[str, CircuitState]:
        """Return a mapping of detector names to their current circuit states."""
        return {name: breaker.state for name, breaker in self._breakers.items()}
