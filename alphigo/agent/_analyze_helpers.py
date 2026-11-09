"""Shared helpers for Agent.analyze implementations."""

import numpy as np

from alphigo import alphigo_board
from alphigo.agent.helpers import is_point_an_eye


def top_k_from_probs(probs, game_state, encoder, k=5):
    """Mask + renormalise a flat per-point distribution and pick top-k.

    Returns ``(top_moves, pass_score)`` where ``top_moves`` is a list of
    ``(Point, percentage)`` tuples (length <= k), and ``pass_score`` is
    the percentage on the pass move if its raw probability would rank
    in the top-k, else ``None``.

    Eye-filling moves and illegal moves are masked out.  The remaining
    mass is renormalised to 100%.
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    n = encoder.num_points()
    legal_mask = np.zeros(n, dtype=np.float64)
    for idx in range(n):
        point = encoder.decode_point_index(idx)
        move = alphigo_board.Move.play_move(point)
        if not game_state.is_valid_move(move):
            continue
        if is_point_an_eye(game_state.board, point,
                           game_state.next_player):
            continue
        legal_mask[idx] = 1.0

    masked = probs[:n] * legal_mask
    total = masked.sum()
    if total <= 0:
        return [], None
    pct = (masked / total) * 100.0

    order = np.argsort(-pct)[:k]
    top = []
    for idx in order:
        if pct[idx] <= 0:
            break
        top.append((encoder.decode_point_index(int(idx)), float(pct[idx])))

    # Pass score: only reported if pass would rank in the top-k by raw prob.
    # We don't know "pass" probability from the per-point output, so
    # callers that have one (AC, DL) supply it elsewhere if needed.
    return top, None


def softmax_normalise(values):
    """Numerically stable softmax over a 1D array."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    v = v - v.max()
    e = np.exp(v)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(v) / v.size
