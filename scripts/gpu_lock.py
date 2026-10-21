"""GPU concurrency lock manager for PARALLAX.

Ensures that only ONE process uses the GPU or a loaded Ollama model at a time.
Uses a file lock cross-platform (filelock).
"""

import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

LOCK_PATH = Path(
    os.environ.get(
        "PARALLAX_GPU_LOCK", Path(__file__).resolve().parent.parent / ".gpu.lock"
    )
)


@contextmanager
def gpu_lock(timeout_seconds: float = 3600.0) -> Generator[Path, None, None]:
    """Acquire exclusive access to the GPU/Ollama subsystem.

    Raises Timeout if lock cannot be acquired within timeout_seconds.
    """
    lock = FileLock(str(LOCK_PATH), timeout=timeout_seconds)
    start = time.perf_counter()
    try:
        lock.acquire(timeout=timeout_seconds)
        elapsed = time.perf_counter() - start
        if elapsed > 1.0:
            print(
                f"[gpu_lock] Acquired GPU lock after {elapsed:.1f}s wait (PID {os.getpid()})",
                file=sys.stderr,
            )
        yield LOCK_PATH
    except Timeout:
        raise TimeoutError(
            f"[gpu_lock] Failed to acquire GPU lock within {timeout_seconds}s. "
            f"Another process may be using the GPU (Lockfile: {LOCK_PATH})."
        ) from None
    finally:
        if lock.is_locked:
            lock.release()


def is_gpu_locked() -> bool:
    """Check if the GPU lock is currently held."""
    lock = FileLock(str(LOCK_PATH), timeout=0)
    try:
        lock.acquire(timeout=0)
        lock.release()
        return False
    except Timeout:
        return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        locked = is_gpu_locked()
        print(f"GPU Lock ({LOCK_PATH}): {'LOCKED' if locked else 'FREE'}")
    elif len(sys.argv) > 1:
        import subprocess

        cmd = sys.argv[1:]
        print(f"[gpu_lock] Running command under GPU lock: {' '.join(cmd)}")
        with gpu_lock():
            subprocess.run(cmd, check=True)
    else:
        print("Usage: python scripts/gpu_lock.py --status | <command...>")
