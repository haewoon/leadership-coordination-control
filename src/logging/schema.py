from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from src.agents.llm_agent import LLMAgent


@dataclass
class ControlAction:
    action_type: str
    source: str
    target: str
    reason: str
    payload: dict[str, str] | None = None


@dataclass
class TeamState:
    run_id: str
    condition: str
    seed: int
    task_id: str
    round_idx: int
    agents: dict[str, LLMAgent]
    shared_memory: list[str] = field(default_factory=list)
    latest_outputs: dict[str, object] = field(default_factory=dict)
    accepted_outputs: dict[str, object] = field(default_factory=dict)
    pending_actions: list[ControlAction] = field(default_factory=list)
    final_answer: str | None = None
    # Snapshot of `latest_outputs` after each completed round. Used by policies'
    # `finalize` to scan all rounds for `Final answer: <X>` patterns on
    # open-ended numeric tasks. See `BasePolicy.cross_round_majority_final_answer`.
    all_round_outputs: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        task_instance,
        condition: str,
        seed: int,
        team_size: int,
        agent_profiles: list[dict[str, str]] | None = None,
    ) -> "TeamState":
        profiles = agent_profiles or []
        agents = {}
        for i in range(team_size):
            profile = profiles[i] if i < len(profiles) else {}
            agents[f"agent_{i}"] = LLMAgent(
                agent_id=f"agent_{i}",
                role="leader" if i == 0 else "member",
                profile_name=profile.get("name"),
                persona=profile.get("instruction"),
            )
        return cls(
            run_id=f"{task_instance.task_id}_{condition}_{seed}_{uuid4().hex[:8]}",
            condition=condition,
            seed=seed,
            task_id=task_instance.task_id,
            round_idx=0,
            agents=agents,
        )

    def record_agent_output(self, agent_id: str, output: object) -> None:
        self.latest_outputs[agent_id] = output


@dataclass
class EventRecord:
    run_id: str
    task_id: str
    condition: str
    round_idx: int
    event_type: str
    source: str
    target: str | None
    payload: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_model_output(
        cls,
        team_state: TeamState,
        task_id: str,
        event_type: str,
        source: str,
        payload: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> "EventRecord":
        return cls(
            run_id=team_state.run_id,
            task_id=task_id,
            condition=team_state.condition,
            round_idx=team_state.round_idx,
            event_type=event_type,
            source=source,
            target=None,
            payload=payload,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @classmethod
    def from_control_action(cls, team_state: TeamState, task_id: str, action: ControlAction) -> "EventRecord":
        return cls(
            run_id=team_state.run_id,
            task_id=task_id,
            condition=team_state.condition,
            round_idx=team_state.round_idx,
            event_type="control_action",
            source=action.source,
            target=action.target,
            payload=action.action_type,
        )


@dataclass
class RunSummary:
    run_id: str
    task_id: str
    condition: str
    seed: int
    final_answer: str
    rounds_used: int
    accepted_count: int

    @classmethod
    def from_team_state(cls, team_state: TeamState, final_answer: str) -> "RunSummary":
        return cls(
            run_id=team_state.run_id,
            task_id=team_state.task_id,
            condition=team_state.condition,
            seed=team_state.seed,
            final_answer=final_answer,
            rounds_used=team_state.round_idx + 1,
            accepted_count=len(team_state.accepted_outputs),
        )
