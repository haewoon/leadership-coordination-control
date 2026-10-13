from __future__ import annotations

import hashlib
from collections import Counter

from src.evaluation.accuracy import normalize_prediction
from src.logging.schema import ControlAction
from src.policies.base_policy import BasePolicy


class RandomActionPolicy(BasePolicy):
    """Non-theory matched control: an explicit-action controller that assigns each
    agent a uniformly random action from {accept, revise, explore} every round, with
    no leadership-derived structure. It shares the control vocabulary of the
    leadership controllers but its decision rule is chance. Used to test whether
    theory-grounding adds value beyond an arbitrary use of the same action set.

    Random choices are keyed deterministically by (run_id, round_idx, agent_id) so
    behavior is reproducible. `accept` agents are not re-generated (they keep their
    current answer); `revise`/`explore` agents re-answer. The team synthesizes when a
    2-of-3 majority agreement emerges or the round budget is exhausted.
    """

    ACTIONS = ["accept", "revise", "explore"]

    def _action_for(self, team_state, agent_id: str) -> str:
        salt = f"{team_state.run_id}::{team_state.round_idx}::{agent_id}"
        h = int(hashlib.sha256(salt.encode("utf-8")).hexdigest(), 16)
        return self.ACTIONS[h % len(self.ACTIONS)]

    def route_agent_inputs(self, task_instance, team_state):
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        prompts = {}
        for agent_id, agent in team_state.agents.items():
            act = self._action_for(team_state, agent_id)
            if act == "accept":
                continue  # keep current answer; no regeneration
            cue = ("Reconsider and revise your answer if warranted."
                   if act == "revise"
                   else "Explore a different line of reasoning or an alternative answer.")
            prompts[agent_id] = (
                f"[Control][Round {team_state.round_idx}] {cue} "
                f"Question: {task_instance.question}{choices}\n"
                f"{self.agent_profile_instruction(agent)}\n"
                f"{answer_instruction}"
            )
        return prompts

    def after_agent_outputs(self, task_instance, team_state):
        # Log the (random) per-agent actions for rounds 1+ only; round 0 is the
        # shared independent answer and carries no control action.
        if team_state.round_idx >= 1:
            for agent_id in team_state.agents:
                team_state.pending_actions.append(
                    ControlAction(self._action_for(team_state, agent_id), "leader", agent_id, "random control")
                )
        normalized = {
            agent_id: normalize_prediction(output.text, task_instance.gold_answer)
            for agent_id, output in team_state.latest_outputs.items()
        }
        non_empty = [a for a in normalized.values() if a]
        if non_empty:
            _, count = Counter(non_empty).most_common(1)[0]
            if count >= 2:
                team_state.pending_actions.append(
                    ControlAction("synthesize", "leader", "team", "majority agreement reached")
                )

    def should_stop(self, task_instance, team_state):
        return any(action.action_type == "synthesize" for action in team_state.pending_actions)

    def finalize(self, task_instance, team_state):
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        return self.majority_output(task_instance, team_state)
