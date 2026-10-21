# PARALLAX: agent rules

Project: multi-model prompt-injection/jailbreak detection with parallel detectors,
fusion, FastAPI, and a sandbox that measures attack success rate with guard on/off.
Full design: DESIGN.md (source of truth; propose changes, do not silently deviate).

## Stack and commands
- Python managed by uv. Add deps with `uv add <pkg>`; never hand-edit uv.lock.
- Run tools with `uv run ...`. In a new worktree, run `uv sync` first.
- Tests: `uv run pytest`. Lint: `uv run ruff check .`

## Hard rules
- Frozen contracts: parallax/contracts/ (Detector, DetectorResult, data schema,
  API schemas). Do not edit them; propose changes in your report instead.
- Stay inside your assigned directories. Never edit another stream's files.
- Unit tests must use fakes/mocks. Do NOT load real models in tests
  (machine has 15.7 GB RAM, 4 GB VRAM; several agents run at once).
- Only the detector-owning agent downloads its model.
- Sandbox payloads are harmless only (canary strings, mock tools). No real
  harmful content in the repo, data, or prompts.
- Reused third-party code (MIT): keep its copyright/license notice and log it
  in THIRD_PARTY.md. Datasets have their own licenses; record them in DATASETS.md.

## Reporting format (every subagent, when done)
1. Files created/changed  2. What works (with test output)  3. What does not
4. Contract changes you need  5. Next steps. Keep under 40 lines.

---

## 1. Operating rules

### 1.1 Autonomy
- Do NOT stop for approval between waves or after merges. Proceed automatically when the gates in section 3 pass.
- Ask the human ONLY when: (a) a frozen contract in parallax/contracts/ must change, (b) a destructive action is needed (deleting data, `git reset --hard`, force-push, deleting unmerged branches), (c) something outside this repository directory is needed, (d) a required secret/token is missing and there is no fallback.
- If a stream fails its gates after 2 fix attempts, mark it BLOCKED in docs/STATUS.md with the reason, continue with all streams that do not depend on it, and report at the end.
- Never push to any remote. Never use force flags. Commits go on stream branches; only the main agent merges into `main`.

### 1.2 Concurrency and resources
- At most 5 subagents running at once.
- The machine has 15.7 GB RAM and a 4 GB GPU. Only ONE process at a time may use the GPU or a loaded Ollama model. Use `scripts/gpu_lock.py` (a file lock context manager created in Wave 0) around every real-model run. Unit tests never load real models.
- Do not poll subagents in a loop. Wait for their completion messages; if you must check, check at most once every 5 minutes.

### 1.3 Ownership
- The main agent is the ONLY owner of pyproject.toml, uv.lock, scripts/gate.py, AGENTS.md and docs/STATUS.md. Subagents request dependencies in their report; the main agent adds them.
- Nobody edits parallax/contracts/. Nobody edits another stream's files. Bugs found in another stream go to docs/BUGS.md with a reproduction, and the main agent routes them to the owning stream.
- New modules may be added by the owning stream under its own directory.

### 1.4 Subagent report format (under 40 lines)
1. Files created/changed 2. What works (paste test output) 3. What does not
4. Dependencies or contract changes needed 5. Risks and follow-ups

### 1.5 Checkpointing
After every wave the main agent updates docs/STATUS.md (wave, per-stream status, commit hashes, blocked items, next wave) and commits it. This lets the run resume if the context or quota runs out.

### 1.6 Safety policy for the red-team parts
Payloads are harmless only: canary strings, mock tools, placeholders. No real harmful instructions in code, data, prompts, generated sets or logs. Local models only (Ollama on localhost).
