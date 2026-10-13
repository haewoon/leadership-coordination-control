from __future__ import annotations

from collections import Counter

from src.evaluation.accuracy import normalize_prediction
from src.logging.schema import ControlAction
from src.policies.base_policy import BasePolicy


class SituationalPolicy(BasePolicy):
    DIRECTIVES = [
        "Stress-test the leading answer with a counterexample.",
        "Preserve a plausible alternative interpretation if one exists.",
        "Identify what evidence would justify convergence.",
    ]

    CONFLICT_MARKERS = (
        "however",
        "but",
        "although",
        "despite",
        "contradict",
        "unlikely",
        "less plausible",
        "fails",
        "surprising",
        "hard to believe",
        "does not explain",
        "missing",
    )

    # Trigger thresholds (overridable by sensitivity-analysis variants).
    EXPLORE_ON_TWO_ONE_SPLIT = True  # reopen on a 2-1 answer split
    RATIONALE_MIN = 3                # min distinct rationale openings for the unanimous-but-divergent branch
    CONFLICT_MIN = 2                 # min agents showing a conflict marker for that branch

    def route_agent_inputs(self, task_instance, team_state):
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        prompts = {}
        phase = self._phase(team_state)

        if phase == "explore":
            for idx, agent_id in enumerate(team_state.agents):
                agent = team_state.agents[agent_id]
                directive = self.DIRECTIVES[idx % len(self.DIRECTIVES)]
                prompts[agent_id] = (
                    f"[Situational:explore][Round {team_state.round_idx}] "
                    "Team state indicates unresolved ambiguity. "
                    f"{directive} "
                    f"Question: {task_instance.question}{choices}\n"
                    f"{self.agent_profile_instruction(agent)}\n"
                    f"{answer_instruction}"
                )
            return prompts

        if phase == "converge":
            leading_answer = getattr(team_state, "situational_leading_answer", "")
            for agent_id, agent in team_state.agents.items():
                prompts[agent_id] = (
                    f"[Situational:converge][Round {team_state.round_idx}] "
                    "Team state indicates it is time to converge. "
                    f"The current leading answer is {leading_answer}. "
                    "Only keep an alternative answer if you can clearly justify why the leading answer still fails. "
                    f"Question: {task_instance.question}{choices}\n"
                    f"{self.agent_profile_instruction(agent)}\n"
                    f"{answer_instruction}"
                )
            return prompts

        for agent_id, agent in team_state.agents.items():
            prompts[agent_id] = (
                f"[Situational:assessment][Round {team_state.round_idx}] "
                "Answer independently so the leader can assess whether the team should explore further or converge. "
                f"Question: {task_instance.question}{choices}\n"
                f"{self.agent_profile_instruction(agent)}\n"
                f"{answer_instruction}"
            )
        return prompts

    def after_agent_outputs(self, task_instance, team_state):
        phase = self._phase(team_state)
        if phase == "assessment":
            self._after_assessment(task_instance, team_state)
            return
        if phase == "explore":
            self._after_explore(task_instance, team_state)
            return
        self._after_converge(task_instance, team_state)

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

    def _phase(self, team_state) -> str:
        return getattr(team_state, "situational_phase", "assessment")

    def _set_phase(self, team_state, phase: str) -> None:
        setattr(team_state, "situational_phase", phase)

    def _after_assessment(self, task_instance, team_state):
        signals = self._signals(task_instance, team_state)
        setattr(team_state, "situational_signals", signals)
        setattr(team_state, "situational_leading_answer", signals["majority_answer"])

        if self._needs_exploration(signals):
            self._set_phase(team_state, "explore")
            for agent_id in team_state.agents:
                team_state.pending_actions.append(
                    ControlAction("explore", "leader", agent_id, "team state indicates unresolved ambiguity")
                )
            return

        self._set_phase(team_state, "converge")
        self._issue_transactional_actions(task_instance, team_state, "assessment indicates direct convergence")

    def _after_explore(self, task_instance, team_state):
        signals = self._signals(task_instance, team_state)
        setattr(team_state, "situational_signals", signals)
        setattr(team_state, "situational_leading_answer", signals["majority_answer"])

        # Exploration never synthesizes directly; it always hands off to a convergence step.
        self._set_phase(team_state, "converge")
        self._issue_transactional_actions(task_instance, team_state, "exploration completed, enter convergence mode")

    def _after_converge(self, task_instance, team_state):
        self._issue_transactional_actions(task_instance, team_state, "situational convergence step")

    def _issue_transactional_actions(self, task_instance, team_state, reason: str):
        team_state.accepted_outputs.clear()
        normalized_answers = {
            agent_id: normalize_prediction(output.text, task_instance.gold_answer)
            for agent_id, output in team_state.latest_outputs.items()
        }
        non_empty_answers = {agent_id: answer for agent_id, answer in normalized_answers.items() if answer}
        if not non_empty_answers:
            for agent_id in team_state.agents:
                team_state.pending_actions.append(
                    ControlAction("revise", "leader", agent_id, f"{reason}; no valid answer extracted")
                )
            return

        counts = Counter(non_empty_answers.values())
        majority_answer, majority_count = counts.most_common(1)[0]
        setattr(team_state, "situational_leading_answer", majority_answer)

        for agent_id, output in team_state.latest_outputs.items():
            answer = normalized_answers[agent_id]
            if answer == majority_answer:
                team_state.accepted_outputs[agent_id] = output
                team_state.pending_actions.append(
                    ControlAction("accept", "leader", agent_id, reason)
                )
            else:
                team_state.pending_actions.append(
                    ControlAction("revise", "leader", agent_id, reason)
                )

        if majority_count >= 2:
            team_state.pending_actions.append(
                ControlAction("synthesize", "leader", "team", f"{reason}; majority agreement reached")
            )

    def _signals(self, task_instance, team_state):
        normalized_answers = self.normalized_answers(task_instance, team_state)
        answers = [answer for answer in normalized_answers.values() if answer]
        counts = Counter(answers)
        majority_answer, majority_count = counts.most_common(1)[0] if counts else ("", 0)
        unique_answers = len(counts)
        rationale_diversity = self._rationale_diversity(team_state)
        conflict_intensity = self._conflict_intensity(team_state)
        return {
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "unique_answers": unique_answers,
            "rationale_diversity": rationale_diversity,
            "conflict_intensity": conflict_intensity,
        }

    def _needs_exploration(self, signals: dict[str, int | str]) -> bool:
        unique_answers = int(signals["unique_answers"])
        majority_count = int(signals["majority_count"])
        rationale_diversity = int(signals["rationale_diversity"])
        conflict_intensity = int(signals["conflict_intensity"])

        if unique_answers >= 3:
            return True
        if unique_answers == 2 and majority_count == 2 and self.EXPLORE_ON_TWO_ONE_SPLIT:
            return True
        if unique_answers == 1 and rationale_diversity >= self.RATIONALE_MIN and conflict_intensity >= self.CONFLICT_MIN:
            return True
        return False

    def _rationale_diversity(self, team_state) -> int:
        prefixes = set()
        for output in team_state.latest_outputs.values():
            text = output.text.strip()
            if not text:
                continue
            first_sentence = text.split(".")[0].strip().lower()
            prefixes.add(first_sentence[:100])
        return len(prefixes)

    def _conflict_intensity(self, team_state) -> int:
        count = 0
        for output in team_state.latest_outputs.values():
            text = output.text.lower()
            if any(marker in text for marker in self.CONFLICT_MARKERS):
                count += 1
        return count
