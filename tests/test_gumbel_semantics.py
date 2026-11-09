"""Minimax-semantics test for GumbelZeroAgent.

Same fixed-weight construction as tests/test_puct_semantics.py: the
value head rates a position lost for the player to move iff the
opponent has a stone on A (tanh(-10 * x[A, plane 8])); the policy is
uniform. From an empty board the child where the mover just played A is
worth +1 to the mover. With m=81 every move is a root candidate, so a
sign-correct Gumbel search must select A and pi' must up-weight A.
"""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from test_puct_semantics import _build_fixed_model, A_RC


def _make_agent(sims, m):
    from alphigo.agent.gumbel_zero_agent import GumbelZeroAgent
    from alphigo.encoders.alphazero import AlphaZeroEncoder
    return GumbelZeroAgent(_build_fixed_model(), AlphaZeroEncoder((9, 9)),
                           num_simulations=sims, m=m)


def test_gumbel_search_selects_winning_move():
    from alphigo.alphigo_board import GameState, Point
    np.random.seed(11)
    a_point = Point(row=A_RC[0] + 1, col=A_RC[1] + 1)
    # m=81: all legal moves are candidates -> phase 1 visits every move
    # once; A's q=+1 then dominates sigma ranking regardless of noise.
    agent = _make_agent(sims=96, m=81)
    result = agent.search(GameState.new_game(9))

    assert result['move'].point == a_point, (
        f"Gumbel search must play the winning move {a_point}, played "
        f"{result['move'].point}. Likely a perspective/sign error in "
        f"_child_stats or _rank_score.")

    pt = result['policy_target']
    from alphigo.encoders.alphazero import AlphaZeroEncoder
    enc = AlphaZeroEncoder((9, 9))
    a_idx = enc.encode_point(a_point)
    assert int(np.argmax(pt)) == a_idx, (
        f"pi' must up-weight the winning move: argmax={int(np.argmax(pt))}, "
        f"A={a_idx}, pi'[A]={pt[a_idx]:.4f}, max={pt.max():.4f}")
    print('  [ok] gumbel search plays A and pi-prime up-weights A')


def test_gumbel_seeds_vary_but_stay_correct():
    """Across seeds the winner must always be A (value signal >> noise)."""
    from alphigo.alphigo_board import GameState, Point
    a_point = Point(row=A_RC[0] + 1, col=A_RC[1] + 1)
    agent = _make_agent(sims=96, m=81)
    for seed in (1, 2, 3):
        np.random.seed(seed)
        result = agent.search(GameState.new_game(9))
        assert result['move'].point == a_point, (
            f"seed {seed}: played {result['move'].point} instead of A")
    print('  [ok] winner is stable across Gumbel noise seeds')


if __name__ == '__main__':
    test_gumbel_search_selects_winning_move()
    test_gumbel_seeds_vary_but_stay_correct()
    print('tests/test_gumbel_semantics.py: all passed')
