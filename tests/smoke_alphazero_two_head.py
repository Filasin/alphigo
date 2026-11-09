"""Smoke test for AlphaZero two-headed network and PUCT MCTSNode (issue 0001)."""
import math

import numpy as np


def test_two_head_output_shapes_and_ranges():
    from alphigo.data.networks import build_alphazero_two_head

    board_size, num_classes = 9, 81
    # 17 planes matches the alphazero encoder, but the smoke test only
    # checks shapes/ranges so any plane count works.
    model = build_alphazero_two_head((board_size, board_size, 17), num_classes)

    # Two named outputs.
    assert len(model.outputs) == 2, \
        f"expected 2 outputs, got {len(model.outputs)}"

    x = np.zeros((4, board_size, board_size, 17), dtype='float32')
    policy, value = model(x)
    policy = policy.numpy()
    value = value.numpy()

    assert policy.shape == (4, num_classes), \
        f"policy shape {policy.shape} != (4, {num_classes})"
    assert value.shape == (4, 1), f"value shape {value.shape} != (4, 1)"

    row_sums = policy.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), \
        f"policy rows do not sum to 1: {row_sums}"

    assert np.all(value >= -1.0) and np.all(value <= 1.0), \
        "tanh value head must be in [-1, 1]"
    print('  [ok] two-head shapes', policy.shape, value.shape)


def test_build_alphazero_unmodified():
    """build_alphazero must remain a single-output policy model."""
    from alphigo.data.networks import build_alphazero
    m = build_alphazero((9, 9, 17), 81)
    assert len(m.outputs) == 1, \
        "build_alphazero must stay single-output (policy only)"
    print('  [ok] build_alphazero unchanged')


def test_mcts_node_has_puct_fields():
    from alphigo.alphigo_board import GameState
    from alphigo.mcts.monte_carlo import MCTSNode

    gs = GameState.new_game(9)
    node = MCTSNode(gs)
    assert hasattr(node, 'prior'), "MCTSNode missing prior"
    assert hasattr(node, 'Q'), "MCTSNode missing Q"
    assert hasattr(node, 'N'), "MCTSNode missing N"
    assert node.N == 0
    assert node.Q == 0.0
    assert node.prior == 0.0
    print('  [ok] MCTSNode has prior/Q/N')


def test_mcts_node_puct_score():
    from alphigo.alphigo_board import GameState
    from alphigo.mcts.monte_carlo import MCTSNode

    gs = GameState.new_game(9)
    node = MCTSNode(gs)
    node.prior = 0.25
    node.Q = 0.4
    node.N = 3
    total_visits = 10
    c_puct = 1.5
    # Q is child-perspective, so the parent-view score negates it.
    expected = -0.4 + 1.5 * 0.25 * math.sqrt(10) / (1 + 3)
    got = node.puct_score(c_puct, total_visits)
    assert math.isclose(got, expected, rel_tol=1e-9), \
        f"puct_score {got} != expected {expected}"

    # Unvisited child uses Q=0 so score equals exploration term only.
    fresh = MCTSNode(gs)
    fresh.prior = 0.5
    assert math.isclose(
        fresh.puct_score(c_puct, total_visits),
        1.5 * 0.5 * math.sqrt(10) / 1.0,
        rel_tol=1e-9,
    )
    print('  [ok] puct_score matches formula')


if __name__ == '__main__':
    test_mcts_node_has_puct_fields()
    test_mcts_node_puct_score()
    test_build_alphazero_unmodified()
    test_two_head_output_shapes_and_ranges()
    print('tests/smoke_alphazero_two_head.py: all passed')
