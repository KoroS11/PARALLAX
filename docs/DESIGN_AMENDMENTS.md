# PARALLAX: Design Amendments (v0.2)

These amendments supersede `DESIGN.md` where they conflict.

---

### D-1: Canonical data splits
The audit found exact duplicates across neuralchemy splits (38 train/val, 45 train/test). Build our own canonical splits: normalize text, hash it, remove cross-split duplicates (always drop from train), cluster near-duplicates (MinHash) so clusters never straddle splits, fix the seed, and write a split manifest with counts and hashes.

### D-2: Preprocessing stage (D0)
Before detection, run a deterministic, bounded preprocessing step: Unicode NFKC, zero-width and control-character stripping, basic homoglyph folding, whitespace collapse, detection and bounded decoding (depth <= 2, size-capped) of base64/hex/URL-encoding/ROT13, and a sliding-window chunker for long inputs so an injection is not diluted. It outputs several "views" of the prompt; each detector scores each view and the maximum counts. Evaluate with and without it (ablation); this directly targets the Mindgard evasions.

### D-3: D1 redesign
The audit found a ~70% false-positive rate on benign prompts and an optimistic metadata score (0.57 macro-F1 on a 1k subsample vs 0.44 on the full validation split). Likely cause: `class_weight="balanced"` on a 4k slice down-weights the largest class (`none`). New D1: train on the FULL canonical train split; two stages (binary attack-vs-none, then a technique head trained on attacks only); probability calibration on validation; threshold chosen so the false-positive rate on `none` is <= 5% on validation; all reported metrics from full val and test splits, never subsamples.

### D-4: Orchestration modes
`parallel_full` (all detectors at once), `cascade` (cheap detectors first, heavy ones only when the result is uncertain, with early exit and cancellation of the rest), and `sequential` (baseline). The experiment becomes a three-way latency/accuracy trade-off.

### D-5: Serving robustness
- Models load once at startup (lifespan) and are warmed up.
- Per-detector circuit breaker (closed/open/half-open), per-detector concurrency limits (bulkheads), a GPU semaphore of size 1, and micro-batching (max batch 16, max wait 5 ms) for the Hugging Face detectors.
- Abstain-aware fusion; `degraded` flag when any detector abstains.
- Configurable fail policy: `fail_closed` (default; if too few detectors respond, the decision is `flag`) or `fail_open`.
- API: input size limits, empty/invalid input rejection, request IDs, global exception handlers that never leak stack traces, concurrency limiter returning 503 with Retry-After, per-request timeout.
- Endpoints: `/health` (liveness), `/ready` (detectors loaded), `/metrics` (Prometheus text).
- Config through pydantic-settings and `PARALLAX_*` environment variables.

### D-6: Storage robustness
SQLAlchemy 2.x, SQLite with WAL and busy timeout, Alembic migrations that also work for Postgres. Writes go through a bounded write-behind queue so logging can never block or fail a request. Store a text hash and, only if enabled (`PARALLAX_STORE_TEXT`), a truncated copy of the text. Retention/pruning job.

### D-7: Observability
Structured JSON logs with request IDs, per-detector latency histograms, circuit-breaker state, cache hit rate, queue depths.

### D-8: Evaluation rigor
Bootstrap 95% confidence intervals on every headline metric, calibration error (ECE), recall at fixed FPR, ablations (normalization on/off, each fusion strategy, each orchestration mode), and a run manifest (git SHA, dataset hashes, seeds, library versions, hardware) written with every result.

### D-9: Reproducibility
One command each for: install, gate, run API, run eval, run demo. A fresh clone must pass the gate.
