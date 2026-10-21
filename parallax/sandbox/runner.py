"""Checkpoint/resume execution runner for the PARALLAX sandbox.

Per AGENTS.md and DESIGN.md Sections 7 and 9:
- Executes episodes pairing CPS scenarios, red-team attacks, and victim LLMs.
- Supports guarded vs unguarded execution via GuardClient (in-process or HTTP API).
- Automated scoring: attack succeeds IF canary is leaked OR any forbidden mock tool is called.
- Checkpoint/resume: persists episodes to JSONL stream and optional database repository.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from parallax.contracts.api import Decision
from parallax.redteam.cps_attacks import CPSAttackPrompt
from parallax.sandbox.guard import GuardClient
from parallax.sandbox.scenarios import CPSScenario
from parallax.sandbox.scoring import (
    EpisodeResult,
    GuardedVsUnguardedComparison,
    compare_guarded_vs_unguarded,
    evaluate_episode_outcome,
)
from parallax.sandbox.victims import VictimClient, run_victim_loop
from parallax.storage.repositories import SandboxEpisodeRepository


class SandboxRunner:
    """Orchestrates sandbox execution with checkpointing and storage persistence."""

    def __init__(
        self,
        guard_client: GuardClient | None = None,
        repository: SandboxEpisodeRepository | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.guard_client = guard_client
        self.repository = repository
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._completed_keys: set[tuple[str, str, str, bool]] = set()

        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint_keys()

    def _load_checkpoint_keys(self) -> None:
        """Scan existing JSONL checkpoint file to discover completed episodes."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return

        with open(self.checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sc_id = data.get("metadata", {}).get("scenario_id", "")
                    atk_id = data.get("attack_id", "")
                    v_model = data.get("metadata", {}).get("victim_model", "")
                    guarded = bool(data.get("guarded", False))
                    self._completed_keys.add((sc_id, atk_id, v_model, guarded))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def _append_to_checkpoint(self, result: EpisodeResult) -> None:
        """Atomically append an episode record to the JSONL checkpoint file."""
        if not self.checkpoint_path:
            return

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

        sc_id = result.metadata.get("scenario_id", "")
        v_model = result.metadata.get("victim_model", "")
        self._completed_keys.add((sc_id, result.attack_id, v_model, result.guarded))

    def _read_checkpoint_results(self) -> list[EpisodeResult]:
        """Read existing results from checkpoint file synchronously."""
        results: list[EpisodeResult] = []
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return results

        with open(self.checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    results.append(
                        EpisodeResult(
                            episode_id=d["episode_id"],
                            attack_id=d["attack_id"],
                            attack_name=d["attack_name"],
                            attack_level=d["attack_level"],
                            technique=d["technique"],
                            prompt=d["prompt"],
                            guarded=d["guarded"],
                            guard_decision=d.get("guard_decision"),
                            canary=d.get("canary", ""),
                            victim_response=d.get("victim_response"),
                            tool_calls=d.get("tool_calls", []),
                            canary_leaked=d.get("canary_leaked", False),
                            forbidden_tool_called=d.get("forbidden_tool_called", False),
                            attack_succeeded=d.get("attack_succeeded", False),
                            metadata=d.get("metadata", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return results

    async def _persist_to_storage(self, result: EpisodeResult) -> None:
        """Persist episode into the database repository if configured."""
        if self.repository is None:
            return

        try:
            await self.repository.log_episode(
                episode_id=result.episode_id,
                victim_model=result.metadata.get("victim_model", "mock-victim"),
                attack_technique=result.technique,
                guard_enabled=result.guarded,
                attack_succeeded=result.attack_succeeded,
                details=result.to_dict(),
                safe=True,
            )
        except Exception:  # noqa: BLE001, S110
            # Safe persistence failure must not crash runner
            pass

    async def run_episode(
        self,
        scenario: CPSScenario,
        attack: CPSAttackPrompt,
        victim: VictimClient,
        guarded: bool = False,
        episode_id: str | None = None,
        max_turns: int = 4,
    ) -> EpisodeResult:
        """Execute a single sandbox attack episode with automated scoring."""
        ep_id = episode_id or f"EP-{uuid4().hex[:12]}"
        system_prompt, canary = scenario.build_system_prompt()
        registry = scenario.create_tool_registry()

        # Apply any indirect injection tool overrides from attack metadata
        if "tool_overrides" in attack.metadata:
            for sensor_id, override_val in attack.metadata["tool_overrides"].items():
                registry.set_sensor_override(sensor_id, override_val)

        guard_decision_str: str | None = None
        turn_result = None

        if guarded:
            if self.guard_client is None:
                raise ValueError(
                    "Guard execution requested but no guard_client configured."
                )

            scan_resp = await self.guard_client.scan(text=attack.prompt)
            guard_decision_str = (
                scan_resp.decision.value
                if hasattr(scan_resp.decision, "value")
                else str(scan_resp.decision)
            )

            # If guard blocked the prompt, it never reaches the victim LLM
            if scan_resp.decision == Decision.BLOCK or guard_decision_str == "block":
                ep_result = EpisodeResult(
                    episode_id=ep_id,
                    attack_id=attack.attack_id,
                    attack_name=attack.name,
                    attack_level=attack.level,
                    technique=attack.technique,
                    prompt=attack.prompt,
                    guarded=True,
                    guard_decision="block",
                    canary=canary,
                    victim_response=None,
                    tool_calls=[],
                    canary_leaked=False,
                    forbidden_tool_called=False,
                    attack_succeeded=False,
                    metadata={
                        "scenario_id": scenario.scenario_id,
                        "victim_model": victim.model_name,
                        "blocked_by_guard": True,
                    },
                )
                self._append_to_checkpoint(ep_result)
                await self._persist_to_storage(ep_result)
                return ep_result

        # Run victim interaction loop (either unguarded or allowed by guard)
        turn_result = await run_victim_loop(
            victim=victim,
            registry=registry,
            system_prompt=system_prompt,
            user_prompt=attack.prompt,
            max_turns=max_turns,
        )

        # Automated scoring per PARALLAX rules
        succeeded, leaked, forbidden_called = evaluate_episode_outcome(
            canary=canary,
            victim_response=turn_result.final_response,
            tool_records=turn_result.tool_calls,
            guard_decision=guard_decision_str,
            forbidden_tool_names=scenario.get_forbidden_tool_names(),
        )

        ep_result = EpisodeResult(
            episode_id=ep_id,
            attack_id=attack.attack_id,
            attack_name=attack.name,
            attack_level=attack.level,
            technique=attack.technique,
            prompt=attack.prompt,
            guarded=guarded,
            guard_decision=guard_decision_str,
            canary=canary,
            victim_response=turn_result.final_response,
            tool_calls=[call.to_dict() for call in turn_result.tool_calls],
            canary_leaked=leaked,
            forbidden_tool_called=forbidden_called,
            attack_succeeded=succeeded,
            metadata={
                "scenario_id": scenario.scenario_id,
                "victim_model": victim.model_name,
                "turns_taken": turn_result.turns_taken,
                "error": turn_result.error,
            },
        )

        self._append_to_checkpoint(ep_result)
        await self._persist_to_storage(ep_result)
        return ep_result

    async def run_suite(
        self,
        scenarios: Sequence[CPSScenario],
        attacks: Sequence[CPSAttackPrompt],
        victims: Sequence[VictimClient],
        guarded_modes: tuple[bool, ...] = (False, True),
        resume: bool = True,
        max_turns: int = 4,
    ) -> tuple[list[EpisodeResult], GuardedVsUnguardedComparison]:
        """Execute a full evaluation suite across scenarios, attacks, and victim LLMs."""
        results: list[EpisodeResult] = []

        # If resume is enabled, load existing results from checkpoint file first
        if resume:
            results.extend(self._read_checkpoint_results())

        for victim in victims:
            for scenario in scenarios:
                for attack in attacks:
                    for guarded in guarded_modes:
                        key = (
                            scenario.scenario_id,
                            attack.attack_id,
                            victim.model_name,
                            guarded,
                        )
                        if resume and key in self._completed_keys:
                            continue

                        ep_result = await self.run_episode(
                            scenario=scenario,
                            attack=attack,
                            victim=victim,
                            guarded=guarded,
                            max_turns=max_turns,
                        )
                        results.append(ep_result)

        unguarded = [r for r in results if not r.guarded]
        guarded_list = [r for r in results if r.guarded]
        comparison = compare_guarded_vs_unguarded(unguarded, guarded_list)
        return results, comparison
