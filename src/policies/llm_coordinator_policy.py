from __future__ import annotations

import re
from typing import Any

from src.logging.schema import EventRecord
from src.policies.base_policy import BasePolicy


_DECISION_RE = re.compile(r"decision\s*:\s*(continue|stop)", re.IGNORECASE)
_AGENT_MSG_RE = re.compile(r"^(agent_\d+)[ \t]*:[ \t]*(.+)$", re.MULTILINE)


class LLMCoordinatorPolicy(BasePolicy):
    """Unconstrained-coordinator baseline: a plain LLM coordinates the team with
    free-form messages and no leadership theory embedded. Unlike the leadership
    controllers it has no action vocabulary (explore/revise/accept/synthesize),
    no lexical or threshold decision rule, and no style-derived structure; the
    coordinator model reads the team's current answers each round and writes
    whatever per-agent guidance it wants, plus a CONTINUE/STOP decision (the
    minimal protocol needed to terminate). Final aggregation is held fixed at
    the same majority rule as the other conditions, so any gain is attributable
    to the coordinator's interaction, not a different aggregator.

    The coordinator's own generation call is logged to the trace as a
    `coordinator_output` event with token counts, so cost accounting includes
    it. Requires the runner to inject `model` and `run_logger` (see
    `needs_runtime` in ExperimentRunner).
    """

    needs_runtime = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model: Any = None
        self.run_logger: Any = None
        self._directives: dict[str, str] = {}
        self._stop = False

    def initialize_team(self, task_instance: Any, team_state: Any) -> None:
        super().initialize_team(task_instance, team_state)
        self._directives = {}
        self._stop = False

    def _coordinator_prompt(self, task_instance: Any, team_state: Any) -> str:
        choices = self.format_choices(task_instance)
        answers = "\n".join(
            f"{agent_id}: {output.text}"
            for agent_id, output in sorted(team_state.latest_outputs.items())
        )
        agent_ids = ", ".join(sorted(team_state.agents.keys()))
        return (
            "You are the coordinator of a small team of assistants who are answering "
            "a question together. Coordinate the team however you see fit.\n"
            f"Question: {task_instance.question}{choices}\n"
            f"Team answers this round:\n{answers}\n"
            "Decide how the team should proceed. If the team's answer is good enough, "
            "stop; otherwise tell each assistant what to do next round, in your own "
            "words. Reply in exactly this format:\n"
            "Decision: CONTINUE or STOP\n"
            + "\n".join(f"{aid}: <your message to {aid}>" for aid in sorted(team_state.agents.keys()))
            + f"\n(The assistants are {agent_ids}. Messages may be empty after STOP.)"
        )

    def after_agent_outputs(self, task_instance: Any, team_state: Any) -> None:
        prompt = self._coordinator_prompt(task_instance, team_state)
        output = self.model.generate(
            prompt=prompt,
            metadata={"agent_id": "coordinator", "round_idx": team_state.round_idx},
        )
        self.run_logger.log_event(
            EventRecord.from_model_output(
                team_state=team_state,
                task_id=task_instance.task_id,
                event_type="coordinator_output",
                source="coordinator",
                payload=output.text,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
            )
        )
        decision = _DECISION_RE.search(output.text)
        self._stop = bool(decision and decision.group(1).lower() == "stop")
        self._directives = {
            agent_id: message.strip()
            for agent_id, message in _AGENT_MSG_RE.findall(output.text)
            if agent_id in team_state.agents and message.strip()
        }

    def route_agent_inputs(self, task_instance: Any, team_state: Any) -> dict[str, str]:
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        prompts = {}
        for agent_id, agent in team_state.agents.items():
            directive = self._directives.get(agent_id, "")
            directive_line = f"Coordinator: {directive}\n" if directive else ""
            prompts[agent_id] = (
                f"[Round {team_state.round_idx}] {directive_line}"
                f"{self.agent_profile_instruction(agent)}\n"
                f"Question: {task_instance.question}{choices}\n"
                f"{answer_instruction}"
            )
        return prompts

    def should_stop(self, task_instance: Any, team_state: Any) -> bool:
        return self._stop

    def finalize(self, task_instance: Any, team_state: Any) -> str:
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        return self.majority_output(task_instance, team_state)
