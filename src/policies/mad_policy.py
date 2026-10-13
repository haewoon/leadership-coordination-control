"""Multi-Agent Debate baseline (Du et al., 2024 style).

Peer-to-peer debate without an explicit leader, an acceptance criterion, or
control actions like accept/revise/explore. Every round each agent reads the
other agents' most recent answers and is invited to revise its own answer if
persuaded. Stops when all agents converge to the same normalized answer or the
round budget is exhausted; final answer is the majority vote.

This serves as a stronger non-leader baseline than the simple "revise everyone"
TheoryFreeControlPolicy, addressing the NFxC/qh4n review points about a
competitive non-theory-grounded multi-agent baseline.
"""

from __future__ import annotations

from src.evaluation.accuracy import normalize_prediction
from src.policies.base_policy import BasePolicy


class MADPolicy(BasePolicy):
    def route_agent_inputs(self, task_instance, team_state):
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        latest = team_state.latest_outputs
        prompts = {}
        for agent_id, agent in team_state.agents.items():
            if team_state.round_idx == 0 or not latest:
                debate_block = ""
            else:
                others = [
                    f"- {peer_id}: {output.text.strip()}"
                    for peer_id, output in latest.items()
                    if peer_id != agent_id
                ]
                debate_block = (
                    "\nPeer answers from the previous round:\n" + "\n".join(others) + "\n"
                )
            prompts[agent_id] = (
                f"[MAD][Round {team_state.round_idx}] "
                f"You are debating with peers; revise your answer only if a peer argument is convincing. "
                f"Question: {task_instance.question}{choices}{debate_block}\n"
                f"{self.agent_profile_instruction(agent)}\n"
                f"{answer_instruction}"
            )
        return prompts

    def after_agent_outputs(self, task_instance, team_state):
        team_state.shared_memory.extend(
            f"{agent_id}: {output.text}" for agent_id, output in team_state.latest_outputs.items()
        )

    def should_stop(self, task_instance, team_state):
        normalized = {
            normalize_prediction(output.text, task_instance.gold_answer)
            for output in team_state.latest_outputs.values()
        }
        return len(normalized) == 1 and bool(normalized) and "" not in normalized

    def finalize(self, task_instance, team_state):
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        return self.majority_output(task_instance, team_state)
