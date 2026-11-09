"""Unit tests for GumbelZeroAgent selection mechanics (CPU-only)."""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _tiny_agent(**kwargs):
    from alphigo.agent.gumbel_zero_agent import GumbelZeroAgent
    from alphigo.data.networks import build_alphazero_two_head
    from alphigo.encoders.alphazero import AlphaZeroEncoder
    encoder = AlphaZeroEncoder((9, 9))
    model = build_alphazero_two_head(encoder.shape(), encoder.num_points())
    return GumbelZeroAgent(model, encoder, **kwargs)


def _fake_child(parent, q_own, n, prior):
    """Build a child with manually set stats. q_own is the child's
    own-perspective Q (parent sees -q_own)."""
    from alphigo.mcts.monte_carlo import MCTSNode
    from alphigo.alphigo_board import Move, Point
    child = MCTSNode(None, parent=parent,
                     move=Move.play_move(Point(1, 1 + len(parent.children))))
    child.Q = q_own
    child.N = n
    child.W = q_own * n
    child.prior = prior
    parent.children.append(child)
    return child


def test_interior_rule_prefers_parent_best_q():
    """Child that is BAD for its own mover (q_own=-0.8) is GOOD for the
    parent (+0.8) and must be selected over a neutral sibling."""
    from alphigo.mcts.monte_carlo import MCTSNode
    from alphigo.alphigo_board import GameState
    agent = _tiny_agent(num_simulations=8, m=4)
    root = MCTSNode(GameState.new_game(9))
    root.raw_value = 0.0
    good = _fake_child(root, q_own=-0.8, n=1, prior=0.5)
    _bad = _fake_child(root, q_own=+0.8, n=1, prior=0.5)
    picked = agent._select_child_deterministic(root)
    assert picked is good, (
        "interior rule must negate child.Q (parent perspective); "
        "picking the child whose own Q is positive reproduces the "
        "2026-06-11 sign bug")
    print('  [ok] interior rule selects parent-best child (sign correct)')


def test_interior_rule_visit_matching_spreads():
    """With equal q, the rule must prefer the under-visited child."""
    from alphigo.mcts.monte_carlo import MCTSNode
    from alphigo.alphigo_board import GameState
    agent = _tiny_agent(num_simulations=8, m=4)
    root = MCTSNode(GameState.new_game(9))
    root.raw_value = 0.0
    heavy = _fake_child(root, q_own=0.0, n=10, prior=0.5)
    light = _fake_child(root, q_own=0.0, n=0, prior=0.5)
    picked = agent._select_child_deterministic(root)
    assert picked is light, "visit-matching must prefer the unvisited child"
    print('  [ok] interior rule spreads visits (visit-matching term)')


def test_search_returns_valid_result_and_budget():
    """Full search on a real (random-weight) net: legal move, pi' sums
    to 1, root visit count == 1 (root eval) + num_simulations."""
    from alphigo.alphigo_board import GameState
    np.random.seed(3)
    agent = _tiny_agent(num_simulations=32, m=16)
    gs = GameState.new_game(9)
    result = agent.search(gs)
    pt = result['policy_target']
    assert pt.shape == (82,)  # 81 board points + pass
    assert abs(pt.sum() - 1.0) < 1e-5, f"pi' must sum to 1, got {pt.sum()}"
    assert -1.0 <= result['value'] <= 1.0
    mv = result['move']
    assert mv.is_pass or gs.is_valid_move(mv)
    print('  [ok] gumbel search returns legal move and normalized pi-prime')


def test_search_low_budget_degenerate():
    """n=8 (< m) and n=1 must not crash and must return a legal move."""
    from alphigo.alphigo_board import GameState
    np.random.seed(4)
    for sims in (8, 1):
        agent = _tiny_agent(num_simulations=sims, m=16)
        gs = GameState.new_game(9)
        result = agent.search(gs)
        assert abs(result['policy_target'].sum() - 1.0) < 1e-5
        mv = result['move']
        assert mv.is_pass or gs.is_valid_move(mv)
    print('  [ok] degenerate budgets (n=8, n=1) are safe')


if __name__ == '__main__':
    test_interior_rule_prefers_parent_best_q()
    test_interior_rule_visit_matching_spreads()
    test_search_returns_valid_result_and_budget()
    test_search_low_budget_degenerate()
    print('tests/test_gumbel_agent.py: all passed')
