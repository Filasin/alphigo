"""Minimax-semantics test for ZeroAgent PUCT search.

Regression test for the sign-inversion bug found 2026-06-11: selection
maximized the child's own-perspective Q, so the search preferred moves
the value net rated best *for the opponent*.

Construction: a fixed-weight model whose value head says "if the
opponent (relative to the player to move) has a stone on point A, the
position is lost for the player to move" (tanh(-10 * x[A, plane 8])).
Policy head is uniform. From an empty board, the child where the mover
just played A is therefore worth +1 to the mover. A search with correct
minimax semantics must concentrate visits on A.
"""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

A_RC = (4, 4)                 # 0-based (row, col) of target point A


def _build_fixed_model():
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=(9, 9, 17))
    flat = layers.Flatten()(inp)
    pol = layers.Dense(81, use_bias=False, kernel_initializer='zeros',
                       trainable=False)(flat)
    pol = layers.Softmax()(pol)
    vdense = layers.Dense(1, use_bias=False, trainable=False)
    val = layers.Activation('tanh')(vdense(flat))
    model = Model(inp, [pol, val])
    w = np.zeros((9 * 9 * 17, 1), dtype=np.float32)
    idx = A_RC[0] * 9 * 17 + A_RC[1] * 17 + 8  # plane 8 = opponent stones, t
    w[idx, 0] = -10.0
    vdense.set_weights([w])
    return model


def test_search_concentrates_on_winning_move():
    from alphigo.agent.zero_agent import ZeroAgent
    from alphigo.encoders.alphazero import AlphaZeroEncoder
    from alphigo.alphigo_board import GameState, Point

    a_point = Point(row=A_RC[0] + 1, col=A_RC[1] + 1)
    agent = ZeroAgent(_build_fixed_model(), AlphaZeroEncoder((9, 9)),
                      num_simulations=200, c_puct=1.5)
    root = agent._run_search(GameState.new_game(9))

    ranked = sorted(root.children, key=lambda c: c.N, reverse=True)
    a_child = next(c for c in root.children if c.move.point == a_point)

    assert ranked[0].move.point == a_point, (
        f"search should concentrate visits on the winning move {a_point}, "
        f"but argmax-visits is {ranked[0].move.point} "
        f"(A got {a_child.N}/{root.N} visits). "
        f"Likely a perspective/sign error in PUCT selection or backup.")
    # The winning move should dominate, not merely edge out the rest.
    assert a_child.N >= 3 * ranked[1].N, (
        f"winning move should dominate visits: A={a_child.N}, "
        f"runner-up={ranked[1].N}")
    # Root Q is from the root mover's perspective. Only the single
    # A-transition carries value in this toy construction, so the mean
    # dilutes toward 1/N — but it must stay positive.
    assert root.Q > 0.0, f"root.Q should be positive for the mover, got {root.Q:.3f}"
    print('  [ok] PUCT search concentrates visits on the winning move')


def test_batched_search_same_semantics():
    """batch_size > 1 exercises the virtual-loss path; semantics must hold."""
    from alphigo.agent.zero_agent import ZeroAgent
    from alphigo.encoders.alphazero import AlphaZeroEncoder
    from alphigo.alphigo_board import GameState, Point

    a_point = Point(row=A_RC[0] + 1, col=A_RC[1] + 1)
    agent = ZeroAgent(_build_fixed_model(), AlphaZeroEncoder((9, 9)),
                      num_simulations=200, c_puct=1.5, batch_size=8)
    root = agent._run_search(GameState.new_game(9))
    ranked = sorted(root.children, key=lambda c: c.N, reverse=True)
    assert ranked[0].move.point == a_point, (
        f"batched search should also prefer {a_point}, got "
        f"{ranked[0].move.point}")
    print('  [ok] batched (virtual-loss) search has the same semantics')


if __name__ == '__main__':
    test_search_concentrates_on_winning_move()
    test_batched_search_same_semantics()
    print('tests/test_puct_semantics.py: all passed')
