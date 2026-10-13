from __future__ import annotations

from src.logging.schema import ControlAction
from src.policies.base_policy import BasePolicy


class TheoryFreeControlPolicy(BasePolicy):
    def route_agent_inputs(self, task_instance, team_state):
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        return {
            agent.agent_id: (
                f"[GenericControl][Round {team_state.round_idx}] Answer: {task_instance.question}{choices}\n"
                f"{self.agent_profile_instruction(agent)}\n"
                f"{answer_instruction}"
            )
            for agent in team_state.agents.values()
        }

    def after_agent_outputs(self, task_instance, team_state):
        if self.has_consensus(task_instance, team_state):
            team_state.pending_actions.append(
                ControlAction(action_type="synthesize", source="controller", target="team", reason="consensus")
            )
        else:
            for agent_id in team_state.agents:
                team_state.pending_actions.append(
                    ControlAction(
                        action_type="revise",
                        source="controller",
                        target=agent_id,
                        reason="generic disagreement threshold exceeded",
                    )
                )
    def should_stop(self, task_instance, team_state):
        return any(action.action_type == "synthesize" for action in team_state.pending_actions)

    def finalize(self, task_instance, team_state):
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        return self.majority_output(task_instance, team_state)
