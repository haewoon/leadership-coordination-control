from __future__ import annotations

import concurrent.futures
from typing import Any

from src.logging.schema import EventRecord
from src.runner.stopping import should_stop_for_budget


class TeamRunner:
    def __init__(self, config: dict[str, Any], model: Any, run_logger: Any) -> None:
        self.config = config
        self.model = model
        self.run_logger = run_logger
        # Set to >1 to parallelize per-round agent generation calls. Default 1
        # preserves original sequential behavior for backward compatibility.
        # Per-config override via `runner.agent_parallelism`.
        self.agent_parallelism = int(self.config.get("runner", {}).get("agent_parallelism", 1))

    def run(
        self,
        task_instance: Any,
        team_state: Any,
        policy: Any,
        shared_round0_outputs: dict[str, Any] | None = None,
    ) -> str:
        round_budget = self.config["round_budget"][task_instance.task_type]

        # Shared-round-0 mode: round 0 was generated once (policy-agnostically)
        # and cached, so every condition starts from an identical round 0.
        # Inject those outputs here instead of generating, then run the policy
        # from round 1. This makes the round-0-vote baseline a true shared
        # baseline across conditions (removing the round-0 framing confound).
        start_round = 0
        if shared_round0_outputs is not None:
            team_state.round_idx = 0
            policy.before_round(task_instance, team_state)
            for agent_id, output in shared_round0_outputs.items():
                team_state.record_agent_output(agent_id=agent_id, output=output)
                self.run_logger.log_event(
                    EventRecord.from_model_output(
                        team_state=team_state,
                        task_id=task_instance.task_id,
                        event_type="agent_output",
                        source=agent_id,
                        payload=output.text,
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                    )
                )
            policy.after_agent_outputs(task_instance, team_state)
            for action in team_state.pending_actions:
                self.run_logger.log_event(
                    EventRecord.from_control_action(team_state, task_instance.task_id, action)
                )
            team_state.all_round_outputs.append(dict(team_state.latest_outputs))
            stopped = policy.should_stop(task_instance, team_state) or should_stop_for_budget(
                team_state, round_budget
            )
            start_round = round_budget if stopped else 1

        for round_idx in range(start_round, round_budget):
            team_state.round_idx = round_idx
            policy.before_round(task_instance, team_state)
            agent_inputs = policy.route_agent_inputs(task_instance, team_state)

            # Fire agent generations either sequentially (default) or in
            # parallel via ThreadPoolExecutor. Results are processed in the
            # original `agent_inputs` iteration order so logging order
            # (record_agent_output, log_event) is deterministic regardless of
            # which call returned first. Only the wall-clock generation phase
            # is parallelized; everything else (state updates, control actions,
            # finalize) stays sequential.
            if self.agent_parallelism > 1 and len(agent_inputs) > 1:
                max_workers = min(self.agent_parallelism, len(agent_inputs))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        agent_id: executor.submit(
                            self.model.generate,
                            prompt=prompt,
                            metadata={"agent_id": agent_id, "round_idx": round_idx},
                        )
                        for agent_id, prompt in agent_inputs.items()
                    }
                    outputs_by_agent = {agent_id: futures[agent_id].result() for agent_id in agent_inputs}
            else:
                outputs_by_agent = {}
                for agent_id, prompt in agent_inputs.items():
                    outputs_by_agent[agent_id] = self.model.generate(
                        prompt=prompt,
                        metadata={"agent_id": agent_id, "round_idx": round_idx},
                    )

            for agent_id, output in outputs_by_agent.items():
                team_state.record_agent_output(agent_id=agent_id, output=output)
                self.run_logger.log_event(
                    EventRecord.from_model_output(
                        team_state=team_state,
                        task_id=task_instance.task_id,
                        event_type="agent_output",
                        source=agent_id,
                        payload=output.text,
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                    )
                )
            policy.after_agent_outputs(task_instance, team_state)
            for action in team_state.pending_actions:
                self.run_logger.log_event(
                    EventRecord.from_control_action(team_state, task_instance.task_id, action)
                )
            # Snapshot this round's latest_outputs so finalize() can scan all
            # rounds for `Final answer: <X>` patterns (open-ended numeric).
            team_state.all_round_outputs.append(dict(team_state.latest_outputs))
            if policy.should_stop(task_instance, team_state):
                break
            if should_stop_for_budget(team_state, round_budget):
                break
        final_answer = policy.finalize(task_instance, team_state)
        team_state.final_answer = final_answer
        return final_answer
