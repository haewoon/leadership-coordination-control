"""Situational ablation: random switching between explore and converge.

This policy has the same control space as SituationalPolicy (an explore step
followed by a transactional-style convergence step), but the decision to open
an explore phase is made randomly with a fixed probability rather than from
team-state signals (answer disagreement, rationale divergence, conflict
markers).

The probability is fixed at p_explore (default 0.45), which roughly matches the
empirical exploration rate of the genuine situational controller on the main
qwen3.5 AlphaNLI runs. This isolates the question raised by NFxC: does
situational's advantage come from the *signal-driven* switching, or merely from
having access to a richer control space?

Random switching is keyed by (run_id, round_idx) so behavior remains
deterministic for a fixed seed but varies across runs.
"""

from __future__ import annotations

import hashlib
import random

from src.logging.schema import ControlAction
from src.policies.situational_policy import SituationalPolicy


class SituationalRandomSwitchPolicy(SituationalPolicy):
    P_EXPLORE_DEFAULT = 0.45

    def _rng_for(self, team_state) -> random.Random:
        salt = f"{team_state.run_id}::{team_state.round_idx}"
        h = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:16], 16)
        return random.Random(h)

    def _after_assessment(self, task_instance, team_state):
        # Update signals only so logging stays comparable; do NOT use them.
        signals = self._signals(task_instance, team_state)
        setattr(team_state, "situational_signals", signals)
        setattr(team_state, "situational_leading_answer", signals["majority_answer"])

        p = float(self.config.get("p_explore_random", self.P_EXPLORE_DEFAULT))
        rng = self._rng_for(team_state)
        roll = rng.random()
        if roll < p:
            self._set_phase(team_state, "explore")
            for agent_id in team_state.agents:
                team_state.pending_actions.append(
                    ControlAction(
                        "explore",
                        "leader",
                        agent_id,
                        f"random switch (p={p:.2f}, roll={roll:.3f})",
                    )
                )
            return

        self._set_phase(team_state, "converge")
        self._issue_transactional_actions(
            task_instance,
            team_state,
            f"random switch to converge (p={p:.2f}, roll={roll:.3f})",
        )
