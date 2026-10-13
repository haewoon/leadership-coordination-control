from __future__ import annotations

from src.logging.schema import ControlAction
from src.policies.base_policy import BasePolicy


class TransformationalPolicy(BasePolicy):
    DIRECTIVES = [
        "Explore edge cases.",
        "Generate an alternative interpretation.",
        "Integrate evidence into a unified answer."
    ]

    def route_agent_inputs(self, task_instance, team_state):
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        prompts = {}
        for idx, agent_id in enumerate(team_state.agents):
            directive = self.DIRECTIVES[idx % len(self.DIRECTIVES)]
            agent = team_state.agents[agent_id]
            prompts[agent_id] = (
                f"[Transformational][Round {team_state.round_idx}] Shared goal: solve collaboratively. "
                f"{directive} Question: {task_instance.question}{choices}\n"
                f"{self.agent_profile_instruction(agent)}\n"
                f"{answer_instruction}"
            )
        return prompts

    def after_agent_outputs(self, task_instance, team_state):
        if self.has_consensus(task_instance, team_state):
            team_state.pending_actions.append(
                ControlAction("synthesize", "leader", "team", "team reached consensus")
            )
        elif team_state.round_idx == 0:
            for agent_id in team_state.agents:
                team_state.pending_actions.append(
                    ControlAction("explore", "leader", agent_id, "maintain diverse solution paths")
                )
        else:
            team_state.pending_actions.append(
                ControlAction("synthesize", "leader", "team", "sufficient exploration completed")
            )

    def should_stop(self, task_instance, team_state):
        return any(action.action_type == "synthesize" for action in team_state.pending_actions)

    def finalize(self, task_instance, team_state):
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        return self.majority_output(task_instance, team_state)
