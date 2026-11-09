"""Uniform per-position analysis output for the GUI overlay (issue 0008)."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from alphigo.alphigo_types import Point


@dataclass
class AnalysisResult:
    """What a bot thinks about a position.

    ``top_moves`` is sorted descending by score; scores are percentages
    (0-100), already normalised over legal non-eye moves so they can be
    rendered directly.  ``position_winrate`` is in [0, 1] when the agent
    has a value head (AC, AZ); ``None`` otherwise.  ``pass_score`` is
    the raw percentage assigned to the pass move when it ranks in the
    top-5 by raw score, or ``None``.

    ``move_winrates`` holds the win rate of each entry in ``top_moves``,
    already converted to the side-to-move's perspective.  ``ownership``
    is a flat board_size**2 array in [-1, 1] where positive means the
    point is predicted to belong to the player to move; ``score_estimate``
    is the expected final margin in points, also from the perspective of
    the player to move.  Both are ``None`` for agents without those heads.

    ``perspective`` records *which* player "the player to move" was at the
    analysed position.  Every number above is relative to that player, so
    a consumer that renders the result one ply later (after the move has
    been applied) must use this field rather than the current side to move
    - otherwise win rates are attributed to the wrong player and the
    ownership map comes out colour-inverted.
    """

    top_moves: List[Tuple[Point, float]] = field(default_factory=list)
    position_winrate: Optional[float] = None
    pass_score: Optional[float] = None
    move_winrates: List[float] = field(default_factory=list)
    ownership: Optional[Any] = None
    score_estimate: Optional[float] = None
    perspective: Optional[Any] = None
