"""Sensitivity-analysis variants of the situational controller.

These perturb the (hand-set) reopening thresholds and conflict-marker lexicon to
test whether the situational controller's social-regime gain is robust to those
choices, rather than an artifact of one threshold setting.

  situational_split3  : reopen only on a full three-way split (drop the 2-1 trigger)
  situational_lenient : relax the unanimous-but-divergent branch (rationale>=2, conflict>=1)
  situational_altlex  : a different but comparable conflict-marker lexicon
"""
from __future__ import annotations

from src.policies.situational_policy import SituationalPolicy


class SituationalSplit3Policy(SituationalPolicy):
    """Reopen exploration only when all three agents disagree (no 2-1-split trigger)."""
    EXPLORE_ON_TWO_ONE_SPLIT = False


class SituationalLenientPolicy(SituationalPolicy):
    """Reopen more readily on a nominally-unanimous round (weaker evidence thresholds)."""
    RATIONALE_MIN = 2
    CONFLICT_MIN = 1


class SituationalAltLexPolicy(SituationalPolicy):
    """Same triggers, a different but comparable conflict-marker lexicon."""
    CONFLICT_MARKERS = (
        "nevertheless",
        "yet",
        "conversely",
        "disagree",
        "incorrect",
        "doubtful",
        "implausible",
        "overlooks",
        "inconsistent",
        "questionable",
        "not supported",
        "weak",
    )
