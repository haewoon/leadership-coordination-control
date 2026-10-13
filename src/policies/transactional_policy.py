from __future__ import annotations

from collections import Counter

from src.evaluation.accuracy import normalize_prediction
from src.logging.schema import ControlAction
from src.policies.base_policy import BasePolicy


class TransactionalPolicy(BasePolicy):
    def route_agent_inputs(self, task_instance, team_state):
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        prompts = {}
        for agent_id, agent in team_state.agents.items():
            prompts[agent_id] = (
                f"[Transactional][Round {team_state.round_idx}] "
                f"Answer with concise justification. Question: {task_instance.question}{choices}\n"
                f"{self.agent_profile_instruction(agent)}\n"
                f"{answer_instruction}"
            )
        return prompts

    def after_agent_outputs(self, task_instance, team_state):
        team_state.accepted_outputs.clear()
        normalized_answers = {
            agent_id: normalize_prediction(output.text, task_instance.gold_answer)
            for agent_id, output in team_state.latest_outputs.items()
        }

        non_empty_answers = {agent_id: answer for agent_id, answer in normalized_answers.items() if answer}
        if not non_empty_answers:
            for agent_id in team_state.agents:
                team_state.pending_actions.append(
                    ControlAction("revise", "leader", agent_id, "no valid answer extracted")
                )
            return

        counts = Counter(non_empty_answers.values())
        majority_answer, majority_count = counts.most_common(1)[0]

        for agent_id, output in team_state.latest_outputs.items():
            answer = normalized_answers[agent_id]
            if answer == majority_answer:
                team_state.accepted_outputs[agent_id] = output
                team_state.pending_actions.append(
                    ControlAction("accept", "leader", agent_id, "matches majority answer")
                )
            else:
                team_state.pending_actions.append(
                    ControlAction("revise", "leader", agent_id, "deviates from majority answer")
                )

        if majority_count >= 2:
            team_state.pending_actions.append(
                ControlAction("synthesize", "leader", "team", "majority agreement reached")
            )

    def should_stop(self, task_instance, team_state):
        return any(action.action_type == "synthesize" for action in team_state.pending_actions)

    def finalize(self, task_instance, team_state):
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        if team_state.accepted_outputs:
            return self.majority_output_from_mapping(
                task_instance,
                team_state.accepted_outputs,
                task_instance.gold_answer,
            )
        return self.majority_output(task_instance, team_state)
