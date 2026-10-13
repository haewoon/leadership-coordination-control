from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.evaluation.accuracy import normalize_prediction


_FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*([^\n\r|]+)", re.IGNORECASE)


class BasePolicy:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def initialize_team(self, task_instance: Any, team_state: Any) -> None:
        team_state.shared_memory.append(f"Task: {task_instance.question}")

    def before_round(self, task_instance: Any, team_state: Any) -> None:
        team_state.pending_actions.clear()

    def format_choices(self, task_instance: Any) -> str:
        if not getattr(task_instance, "choices", None):
            return ""
        choice_lines = [f"{choice['label']}. {choice['text']}" for choice in task_instance.choices]
        return "\nChoices:\n" + "\n".join(choice_lines)

    def answer_instruction(self, task_instance: Any) -> str:
        if getattr(task_instance, "choices", None):
            labels = ", ".join(choice["label"] for choice in task_instance.choices)
            return (
                "Respond in at most two short sentences and end with "
                f"'Final answer: <label>' where <label> is one of {labels}."
            )
        metadata = getattr(task_instance, "metadata", {}) or {}
        dataset = metadata.get("dataset") or metadata.get("source")
        if dataset == "strategyqa":
            return "Respond in at most two short sentences and end with 'Final answer: yes' or 'Final answer: no'."
        if dataset == "gsm8k":
            return (
                "Solve the problem step by step in at most four short sentences and end with "
                "'Final answer: <number>' where <number> is the final numeric answer (just the number, no units)."
            )
        if dataset == "math500":
            return (
                "Solve the math problem step by step. Show key reasoning and arithmetic, and end with "
                "'Final answer: <number>' where <number> is the final numeric answer (an integer or decimal; "
                "no units, no LaTeX, no boxed expressions)."
            )
        return "Respond in at most two short sentences and end with 'Final answer: <answer>'."

    def normalized_answers(self, task_instance: Any, team_state: Any) -> dict[str, str]:
        return {
            agent_id: normalize_prediction(output.text, task_instance.gold_answer)
            for agent_id, output in team_state.latest_outputs.items()
        }

    def agent_profile_instruction(self, agent: Any) -> str:
        if not getattr(agent, "persona", None):
            return ""
        profile_name = getattr(agent, "profile_name", None)
        if profile_name:
            return (
                f"Profile ({profile_name}): {agent.persona} "
                "Keep this decision rule throughout your answer even if another interpretation also seems plausible."
            )
        return f"Profile: {agent.persona} Keep this decision rule throughout your answer."

    def has_consensus(self, task_instance: Any, team_state: Any) -> bool:
        normalized = set(self.normalized_answers(task_instance, team_state).values())
        return len(normalized) == 1 and bool(normalized)

    def majority_output(self, task_instance: Any, team_state: Any) -> str:
        return self.majority_output_from_mapping(
            task_instance,
            team_state.latest_outputs,
            task_instance.gold_answer,
        )

    def majority_output_from_mapping(self, task_instance: Any, outputs: dict[str, Any], gold_answer: str | None) -> str:
        normalized = {
            agent_id: normalize_prediction(output.text, gold_answer)
            for agent_id, output in outputs.items()
        }
        if not normalized:
            return ""

        majority_answer = Counter(normalized.values()).most_common(1)[0][0]
        for agent_id, answer in normalized.items():
            if answer == majority_answer:
                return outputs[agent_id].text
        return next(iter(outputs.values())).text

    def neutral_round0_inputs(self, task_instance: Any, team_state: Any) -> dict[str, str]:
        """Policy-agnostic round-0 prompt: independent answering with no policy
        framing. Used when `shared_round0` is enabled so every condition starts
        from an identical round-0 (generated once and reused), making the
        round-0-vote baseline a true shared baseline across conditions and
        removing the round-0 framing confound. Per-agent profile instructions
        are preserved so agent diversity (shared across conditions) is intact."""
        choices = self.format_choices(task_instance)
        answer_instruction = self.answer_instruction(task_instance)
        return {
            agent.agent_id: (
                f"[Round 0] {self.agent_profile_instruction(agent)}\n"
                f"Question: {task_instance.question}{choices}\n"
                f"{answer_instruction}"
            )
            for agent in team_state.agents.values()
        }

    def route_agent_inputs(self, task_instance: Any, team_state: Any) -> dict[str, str]:
        raise NotImplementedError

    def after_agent_outputs(self, task_instance: Any, team_state: Any) -> None:
        raise NotImplementedError

    def should_stop(self, task_instance: Any, team_state: Any) -> bool:
        return False

    def cross_round_majority_final_answer(self, task_instance: Any, team_state: Any) -> str | None:
        """For open-ended numeric tasks, scan every agent output across all
        completed rounds (plus any `accepted_outputs` collected by leadership
        controllers) for the `Final answer: <X>` pattern. Return a synthesized
        `Final answer: <majority>` string built from the most common normalized
        extraction.

        Returns None if the task type is not `open_ended_numeric` (so callers
        fall back to their existing finalize logic, preserving paper-12
        behavior on closed_ended / integration tasks) or if no agent output in
        any round contained a parseable `Final answer:` line.
        """
        task_type = getattr(task_instance, "task_type", None)
        if task_type != "open_ended_numeric":
            return None
        gold = getattr(task_instance, "gold_answer", None)
        extractions: list[str] = []

        history = getattr(team_state, "all_round_outputs", None) or []
        accepted = getattr(team_state, "accepted_outputs", None) or {}

        sources: list[dict[str, Any]] = list(history)
        if accepted:
            sources.append(dict(accepted))

        for round_outputs in sources:
            for output in round_outputs.values():
                text = getattr(output, "text", "")
                if not text or not _FINAL_ANSWER_RE.search(text):
                    continue
                norm = normalize_prediction(text, gold)
                if norm:
                    extractions.append(norm)

        if not extractions:
            return None
        majority = Counter(extractions).most_common(1)[0][0]
        return f"Final answer: {majority}"

    def finalize(self, task_instance: Any, team_state: Any) -> str:
        cross = self.cross_round_majority_final_answer(task_instance, team_state)
        if cross is not None:
            return cross
        return " | ".join(output.text for output in team_state.latest_outputs.values())
