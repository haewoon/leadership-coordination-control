from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.logging.schema import RunSummary, TeamState
from src.models.base_model import ModelOutput
from src.policies.base_policy import BasePolicy
from src.runner.team_runner import TeamRunner


class EpisodeRunner:
    def __init__(self, config: dict[str, Any], model: Any, run_logger: Any) -> None:
        self.config = config
        self.model = model
        self.run_logger = run_logger
        self.team_runner = TeamRunner(config=config, model=model, run_logger=run_logger)
        # When enabled, round 0 is generated once per (task_id, seed) with a
        # policy-agnostic prompt and reused across all conditions, so the
        # round-0-vote baseline is a true shared baseline. Cache persists for
        # the lifetime of the experiment run.
        self.shared_round0 = bool(self.config.get("shared_round0", False))
        self._round0_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._round0_base_policy = BasePolicy(self.config) if self.shared_round0 else None
        # `shared_round0_source` points at an existing run directory whose
        # logged round-0 outputs should be reused verbatim. This lets a later
        # condition (run as its own experiment) start from the *same* round 0
        # as the original matrix, preserving the shared-round-0 attribution
        # design across separately-executed runs.
        source = self.config.get("shared_round0_source")
        if self.shared_round0 and source:
            self._preload_round0_cache(Path(source))

    def _preload_round0_cache(self, source_dir: Path) -> None:
        summaries = source_dir / "run_summaries.jsonl"
        trace = source_dir / "trace.jsonl"
        if not summaries.exists() or not trace.exists():
            raise FileNotFoundError(
                f"shared_round0_source {source_dir} must contain run_summaries.jsonl and trace.jsonl"
            )
        # One condition suffices: the shared round 0 is identical across
        # conditions by construction. Prefer `flat`, else the first seen.
        chosen: dict[tuple[str, int], str] = {}
        with summaries.open() as fh:
            for line in fh:
                row = json.loads(line)
                key = (row["task_id"], int(row["seed"]))
                if key not in chosen or row["condition"] == "flat":
                    chosen[key] = row["run_id"]
        wanted = {run_id: key for key, run_id in chosen.items()}
        with trace.open() as fh:
            for line in fh:
                event = json.loads(line)
                if event.get("event_type") != "agent_output" or event.get("round_idx") != 0:
                    continue
                key = wanted.get(event["run_id"])
                if key is None:
                    continue
                self._round0_cache.setdefault(key, {})[event["source"]] = ModelOutput(
                    text=event["payload"],
                    prompt_tokens=int(event.get("prompt_tokens", 0)),
                    completion_tokens=int(event.get("completion_tokens", 0)),
                )

    def _shared_round0_outputs(self, task_instance: Any, team_state: Any, seed: int) -> dict[str, Any]:
        key = (task_instance.task_id, seed)
        cached = self._round0_cache.get(key)
        if cached is not None:
            return cached
        if self.config.get("shared_round0_source"):
            # With a preloaded source, regenerating would silently break the
            # identical-round-0 attribution design; fail loudly instead.
            raise KeyError(f"shared_round0_source has no round-0 outputs for {key}")
        prompts = self._round0_base_policy.neutral_round0_inputs(task_instance, team_state)
        outputs: dict[str, Any] = {}
        for agent_id, prompt in prompts.items():
            outputs[agent_id] = self.model.generate(
                prompt=prompt,
                metadata={"agent_id": agent_id, "round_idx": 0, "phase": "shared_round0"},
            )
        self._round0_cache[key] = outputs
        return outputs

    def run_episode(self, task_instance: Any, policy: Any, seed: int, condition: str) -> None:
        team_state = TeamState.create(
            task_instance=task_instance,
            condition=condition,
            seed=seed,
            team_size=self.config["team_size"],
            agent_profiles=self.config.get("agent_profiles"),
        )
        policy.initialize_team(task_instance, team_state)
        shared_round0 = (
            self._shared_round0_outputs(task_instance, team_state, seed)
            if self.shared_round0
            else None
        )
        final_answer = self.team_runner.run(
            task_instance=task_instance,
            team_state=team_state,
            policy=policy,
            shared_round0_outputs=shared_round0,
        )
        summary = RunSummary.from_team_state(team_state=team_state, final_answer=final_answer)
        self.run_logger.log_summary(asdict(summary))
