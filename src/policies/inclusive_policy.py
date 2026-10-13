from __future__ import annotations

from collections import Counter

from src.evaluation.accuracy import normalize_prediction
from src.logging.schema import ControlAction
from src.policies.base_policy import BasePolicy


class InclusivePolicy(BasePolicy):
    GAP_MARKERS = (
        "gap claim:",
        "does not explain",
        "fails",
        "ignores",
        "contradict",
        "unlikely",
        "less plausible",
        "missing",
    )

    def route_agent_inputs(self, task_instance, team_state):
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        prompts = {}

        stage = self._stage(team_state)
        defended_labels = self._defended_labels(task_instance, team_state)
        defended_instruction = ""
        if defended_labels:
            labels_str = ", ".join(defended_labels)
            defended_instruction = (
                f" At this stage, keep your final label within the labels already defended by the team: {labels_str}."
            )
        if stage == "objection_round":
            majority_answer = getattr(team_state, "inclusive_majority_answer", "")
            minority_agents = set(getattr(team_state, "inclusive_minority_agents", []))
            majority_agents = set(getattr(team_state, "inclusive_majority_agents", []))
            minority_summary = getattr(team_state, "inclusive_minority_summary", "")

            for agent_id, agent in team_state.agents.items():
                if agent_id in minority_agents:
                    prompts[agent_id] = (
                        f"[Inclusive][Round {team_state.round_idx}] "
                        "Your role is to preserve a minority or dissenting view so it is not lost too early. "
                        f"The current leading answer is {majority_answer}. "
                        "State the strongest concise objection to that leading answer. "
                        "Begin with 'Gap claim:' and identify the key social or causal gap you think the majority is missing. "
                        "Do not introduce a completely new judgment label at this stage; strengthen the minority position already on the table."
                        f"Question: {task_instance.question}{choices}\n"
                        f"{self.agent_profile_instruction(agent)}\n"
                        "Respond in at most three short sentences and end with "
                        f"'Final answer: <label>'.{defended_instruction}"
                    )
                elif agent_id in majority_agents:
                    prompts[agent_id] = (
                        f"[Inclusive][Round {team_state.round_idx}] "
                        "Your role is to respond to a minority objection before the team converges. "
                        f"The current leading answer is {majority_answer}. "
                        f"Minority objection summary: {minority_summary}\n"
                        f"Question: {task_instance.question}{choices}\n"
                        f"{self.agent_profile_instruction(agent)}\n"
                        "First write either 'Gap assessment: objection exposes a gap.' or "
                        "'Gap assessment: no material gap.' Then give your concise answer and end with "
                        f"'Final answer: <label>'.{defended_instruction}"
                    )
                else:
                    prompts[agent_id] = (
                        f"[Inclusive][Round {team_state.round_idx}] "
                        f"Reassess the current team disagreement. Question: {task_instance.question}{choices}\n"
                        f"{self.agent_profile_instruction(agent)}\n"
                        f"{answer_instruction}{defended_instruction}"
                    )
            return prompts

        if stage == "reconsider_round":
            gap_summary = getattr(team_state, "inclusive_gap_summary", "")
            defense_summary = getattr(team_state, "inclusive_defense_summary", "")
            for agent_id, agent in team_state.agents.items():
                prompts[agent_id] = (
                    f"[Inclusive][Round {team_state.round_idx}] "
                    "The leader has judged that the minority objection is strong enough to reopen deliberation. "
                    f"Minority gap summary: {gap_summary}\n"
                    f"Majority defense summary: {defense_summary}\n"
                    f"Question: {task_instance.question}{choices}\n"
                    f"{self.agent_profile_instruction(agent)}\n"
                    f"{answer_instruction}{defended_instruction}"
                )
            return prompts

        for agent_id, agent in team_state.agents.items():
            prompts[agent_id] = (
                f"[Inclusive][Round {team_state.round_idx}] "
                "Answer independently, but be prepared to preserve minority objections if the team converges too quickly. "
                f"Question: {task_instance.question}{choices}\n"
                f"{self.agent_profile_instruction(agent)}\n"
                f"{answer_instruction}"
            )
        return prompts

    def after_agent_outputs(self, task_instance, team_state):
        stage = self._stage(team_state)
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
        majority_agents = [agent_id for agent_id, answer in non_empty_answers.items() if answer == majority_answer]
        minority_agents = [agent_id for agent_id, answer in non_empty_answers.items() if answer != majority_answer]

        if stage == "initial":
            setattr(team_state, "inclusive_initial_answers", dict(non_empty_answers))
            setattr(team_state, "inclusive_initial_majority_answer", majority_answer)
            setattr(team_state, "inclusive_supported_labels", sorted(set(non_empty_answers.values())))

        if stage == "objection_round":
            if self._strong_objection(team_state):
                self._set_stage(team_state, "reconsider_round")
                setattr(team_state, "inclusive_gap_summary", self._minority_gap_summary(team_state))
                setattr(team_state, "inclusive_defense_summary", self._majority_defense_summary(team_state))
                for agent_id in team_state.agents:
                    team_state.pending_actions.append(
                        ControlAction("reconsider", "leader", agent_id, "strong minority objection reopened deliberation")
                    )
                return
            team_state.pending_actions.append(
                ControlAction("synthesize", "leader", "team", "minority objection was heard before convergence")
            )
            return

        if stage == "reconsider_round":
            team_state.pending_actions.append(
                ControlAction("synthesize", "leader", "team", "reconsideration round completed")
            )
            return

        if not minority_agents:
            for agent_id in majority_agents:
                team_state.accepted_outputs[agent_id] = team_state.latest_outputs[agent_id]
                team_state.pending_actions.append(
                    ControlAction("accept", "leader", agent_id, "team already agrees")
                )
            team_state.pending_actions.append(
                ControlAction("synthesize", "leader", "team", "no dissent to preserve")
            )
            self._set_stage(team_state, "done")
            return

        self._set_stage(team_state, "objection_round")
        setattr(team_state, "inclusive_majority_answer", majority_answer)
        setattr(team_state, "inclusive_majority_agents", majority_agents)
        setattr(team_state, "inclusive_minority_agents", minority_agents)
        minority_summary = " | ".join(
            team_state.latest_outputs[agent_id].text.split("Final answer:")[0].strip()
            for agent_id in minority_agents
        )[:400]
        setattr(team_state, "inclusive_minority_summary", minority_summary)

        for agent_id in majority_agents:
            team_state.pending_actions.append(
                ControlAction("defend", "leader", agent_id, "respond to minority objection before finalizing")
            )
        for agent_id in minority_agents:
            team_state.pending_actions.append(
                ControlAction("object", "leader", agent_id, "preserve minority rationale before convergence")
            )

    def should_stop(self, task_instance, team_state):
        return any(action.action_type == "synthesize" for action in team_state.pending_actions)

    def finalize(self, task_instance, team_state):
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        initial_majority = getattr(team_state, "inclusive_initial_majority_answer", "")
        current_answers = self.normalized_answers(task_instance, team_state)
        non_empty = [answer for answer in current_answers.values() if answer]
        defended_labels = set(self._defended_labels(task_instance, team_state))

        if defended_labels:
            defended_only = [answer for answer in non_empty if answer in defended_labels]
        else:
            defended_only = non_empty

        # Preserve context-dependent judgments when the initial team signal was already
        # ambiguous and later rounds merely harden one side of the disagreement.
        if initial_majority == "B" and self._has_context_label(task_instance):
            if len(set(defended_only)) > 1 or "B" in defended_only:
                return "The team still faces a material context-sensitive disagreement, so the safest collective judgment is to preserve ambiguity. Final answer: B"

        if defended_labels and non_empty and not set(non_empty).issubset(defended_labels):
            filtered_outputs = {
                agent_id: output
                for agent_id, output in team_state.latest_outputs.items()
                if normalize_prediction(output.text, task_instance.gold_answer) in defended_labels
            }
            if filtered_outputs:
                return self.majority_output_from_mapping(
                    task_instance,
                    filtered_outputs,
                    task_instance.gold_answer,
                )

        return self.majority_output(task_instance, team_state)

    def _stage(self, team_state) -> str:
        return getattr(team_state, "inclusive_stage", "initial")

    def _set_stage(self, team_state, stage: str) -> None:
        setattr(team_state, "inclusive_stage", stage)

    def _strong_objection(self, team_state) -> bool:
        minority_agents = getattr(team_state, "inclusive_minority_agents", [])
        majority_agents = getattr(team_state, "inclusive_majority_agents", [])
        minority_strength = 0
        for agent_id in minority_agents:
            text = team_state.latest_outputs[agent_id].text.lower()
            minority_strength += sum(marker in text for marker in self.GAP_MARKERS)

        exposed_gap_votes = 0
        for agent_id in majority_agents:
            text = team_state.latest_outputs[agent_id].text.lower()
            if "gap assessment: objection exposes a gap" in text:
                exposed_gap_votes += 1

        return minority_strength >= 1 and exposed_gap_votes >= 1

    def _minority_gap_summary(self, team_state) -> str:
        minority_agents = getattr(team_state, "inclusive_minority_agents", [])
        return " | ".join(
            team_state.latest_outputs[agent_id].text.split("Final answer:")[0].strip()
            for agent_id in minority_agents
        )[:400]

    def _majority_defense_summary(self, team_state) -> str:
        majority_agents = getattr(team_state, "inclusive_majority_agents", [])
        return " | ".join(
            team_state.latest_outputs[agent_id].text.split("Final answer:")[0].strip()
            for agent_id in majority_agents
        )[:400]

    def _has_context_label(self, task_instance) -> bool:
        for choice in getattr(task_instance, "choices", []):
            if choice.get("label") == "B" and "context" in choice.get("text", "").lower():
                return True
        return False

    def _defended_labels(self, task_instance, team_state) -> list[str]:
        supported = getattr(team_state, "inclusive_supported_labels", None)
        if supported:
            return list(supported)
        initial_answers = getattr(team_state, "inclusive_initial_answers", {})
        labels = sorted({answer for answer in initial_answers.values() if answer})
        return labels
