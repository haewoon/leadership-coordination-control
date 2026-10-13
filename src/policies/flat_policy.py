from __future__ import annotations

from src.evaluation.accuracy import normalize_prediction
from src.policies.base_policy import BasePolicy


class FlatPolicy(BasePolicy):
    def route_agent_inputs(self, task_instance, team_state):
        context = "\n".join(team_state.shared_memory[-3:])
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        return {
            agent.agent_id: (
                f"[Flat][Round {team_state.round_idx}] {context}\n"
                f"{self.agent_profile_instruction(agent)}\n"
                f"Question: {task_instance.question}{choices}\n"
                f"{answer_instruction}"
            )
            for agent in team_state.agents.values()
        }

    def after_agent_outputs(self, task_instance, team_state):
        team_state.shared_memory.extend(
            f"{agent_id}: {output.text}" for agent_id, output in team_state.latest_outputs.items()
        )

    def should_stop(self, task_instance, team_state):
        normalized_answers = {
            normalize_prediction(output.text, task_instance.gold_answer)
            for output in team_state.latest_outputs.values()
        }
        return len(normalized_answers) == 1 and bool(normalized_answers)

    def finalize(self, task_instance, team_state):
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        return self.majority_output(task_instance, team_state)
