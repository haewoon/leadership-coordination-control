"""Transactional ablation: accept-only (no management-by-exception revise step).

Bass's (1985) transactional leadership distinguishes two pillars:
- *Contingent reward*: positive reinforcement when behavior meets standards
  → in our framework, the `accept` action.
- *Management by exception*: active correction when behavior deviates
  → in our framework, the `revise` action.

`TransactionalPolicy` uses both pillars. This ablation preserves the
contingent-reward pillar (accept) and removes the management-by-exception
pillar (revise). Concretely:

- Round 0 runs exactly like transactional: each agent answers independently
  and the leader computes the round-0 majority.
- Agents whose answer matches the majority are `accept`-ed and added to
  `accepted_outputs`. Non-matching agents are *not* issued a revise action
  and the team does not run additional rounds.
- The team always `synthesize`s after round 0, whether or not a majority
  formed. If no clear majority exists, finalize falls back to
  `majority_output` over all latest outputs.

Compared to `TransactionalPolicy`, the difference is observable only on
cells where round-0 disagreement is common (no majority forms), since in
that case the original transactional controller runs additional revise
rounds to try to reach consensus. On most paper-12 cells, round-0 majority
forms immediately and accept-only behaves identically to transactional. On
hard reasoning regimes (e.g., MATH-500 Level 5) where round-0 disagreement
is the norm, the ablation isolates whether the revise step is responsible
for transactional's lock-in or whether contingent reward alone is sufficient.

This addresses Reviewer NFxC's request for action-level ablations within
transactional, and parallels the existing transformational two-pillar
decomposition (TransformationalDirectivesOnlyPolicy).
"""
from __future__ import annotations

from collections import Counter

from src.evaluation.accuracy import normalize_prediction
from src.logging.schema import ControlAction
from src.policies.base_policy import BasePolicy


class TransactionalAcceptOnlyPolicy(BasePolicy):
    def route_agent_inputs(self, task_instance, team_state):
        # Identical prompt to TransactionalPolicy so per-agent inputs are
        # comparable across the ablation.
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        prompts = {}
        for agent_id, agent in team_state.agents.items():
            prompts[agent_id] = (
                f"[TransactionalAcceptOnly][Round {team_state.round_idx}] "
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
        non_empty_answers = {agent_id: a for agent_id, a in normalized_answers.items() if a}

        if non_empty_answers:
            counts = Counter(non_empty_answers.values())
            majority_answer, majority_count = counts.most_common(1)[0]
            for agent_id, output in team_state.latest_outputs.items():
                if normalized_answers[agent_id] == majority_answer:
                    team_state.accepted_outputs[agent_id] = output
                    team_state.pending_actions.append(
                        ControlAction(
                            "accept",
                            "leader",
                            agent_id,
                            "matches majority answer (accept-only: no revise issued)",
                        )
                    )
                # NOTE: no revise action — non-majority agents are not pushed
                # to revise. This is the core of the accept-only ablation.

            if majority_count >= 2:
                reason = "majority agreement reached (accept-only)"
            else:
                reason = "no majority formed; accept-only synthesizes immediately without revise rounds"
        else:
            reason = "no valid answers; accept-only synthesizes immediately without revise rounds"

        # Always synthesize after round 0, with or without a majority.
        team_state.pending_actions.append(
            ControlAction("synthesize", "leader", "team", reason)
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
