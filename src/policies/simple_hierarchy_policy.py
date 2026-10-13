from __future__ import annotations

from src.logging.schema import ControlAction
from src.policies.base_policy import BasePolicy


class SimpleHierarchyPolicy(BasePolicy):
    def route_agent_inputs(self, task_instance, team_state):
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        prompts = {}
        for idx, agent in enumerate(team_state.agents.values()):
            profile_instruction = self.agent_profile_instruction(agent)
            if idx == 0:
                prompts[agent.agent_id] = (
                    f"[Leader] Provide high-level instructions for: {task_instance.question}{choices}\n"
                    f"{profile_instruction}\n"
                    f"{answer_instruction}"
                )
            else:
                prompts[agent.agent_id] = (
                    f"[Member][Round {team_state.round_idx}] Follow leader guidance and answer: "
                    f"{task_instance.question}{choices}\n"
                    f"{profile_instruction}\n"
                    f"{answer_instruction}"
                )
        return prompts

    def after_agent_outputs(self, task_instance, team_state):
        team_state.pending_actions.append(
            ControlAction(action_type="synthesize", source="leader", target="team", reason="leader decision")
        )

    def should_stop(self, task_instance, team_state):
        return True

    def finalize(self, task_instance, team_state):
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        member_outputs = {
            agent_id: output
            for agent_id, output in team_state.latest_outputs.items()
            if agent_id != "agent_0"
        }
        if not member_outputs:
            return next(iter(team_state.latest_outputs.values())).text
        return self.majority_output_from_mapping(task_instance, member_outputs, task_instance.gold_answer)
