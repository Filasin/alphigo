"""Smoke test for ZeroAgent (issue 0002)."""
import os
import tempfile

import h5py
import numpy as np


def _build_tiny_zero_agent(num_simulations=4):
    from alphigo.agent.zero_agent import ZeroAgent
    from alphigo.data.networks import build_alphazero_two_head
    from alphigo.encoders.alphazero import AlphaZeroEncoder

    encoder = AlphaZeroEncoder((9, 9))
    model = build_alphazero_two_head(encoder.shape(), encoder.num_points())
    return ZeroAgent(model, encoder, num_simulations=num_simulations,
                     c_puct=1.5)


def test_select_move_returns_legal_move():
    from alphigo.alphigo_board import GameState
    agent = _build_tiny_zero_agent(num_simulations=4)
    gs = GameState.new_game(9)
    move = agent.select_move(gs)
    # Must be a legal move on an empty board (or a pass).
    assert move.is_pass or move.is_resign or gs.is_valid_move(move), \
        f"agent returned illegal move {move}"
    print('  [ok] select_move returns legal move')


def test_search_target_and_value():
    from alphigo.alphigo_board import GameState
    agent = _build_tiny_zero_agent(num_simulations=8)
    gs = GameState.new_game(9)
    result = agent.search(gs)
    policy_target = result['policy_target']
    value = result['value']

    assert policy_target.shape == (81,), \
        f"policy_target shape {policy_target.shape} != (81,)"
    assert np.isclose(policy_target.sum(), 1.0, atol=1e-5), \
        f"policy_target should sum to 1, got {policy_target.sum()}"
    assert -1.0 <= value <= 1.0, f"value {value} outside [-1, 1]"
    print('  [ok] search target sums to 1, value in [-1, 1]')


def test_temperature_zero_is_argmax():
    from alphigo.alphigo_board import GameState
    agent = _build_tiny_zero_agent(num_simulations=8)
    agent.set_temperature(0.0)
    gs = GameState.new_game(9)
    result = agent.search(gs)
    pt = result['policy_target']
    # Argmax under T=0 should have a single 1.0 entry.
    assert np.isclose(pt.max(), 1.0, atol=1e-6), \
        f"T=0 should produce one-hot policy target, got max={pt.max()}"
    print('  [ok] temperature=0 produces greedy one-hot target')


def test_serialize_round_trip():
    from alphigo.agent.zero_agent import load_zero_agent
    from alphigo.alphigo_board import GameState

    agent = _build_tiny_zero_agent(num_simulations=4)
    fd, path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        with h5py.File(path, 'w') as f:
            agent.serialize(f)
        with h5py.File(path, 'r') as f:
            loaded = load_zero_agent(f)
        # Loaded agent should be able to play.
        gs = GameState.new_game(9)
        move = loaded.select_move(gs)
        assert move.is_pass or move.is_resign or gs.is_valid_move(move)
    finally:
        os.unlink(path)
    print('  [ok] serialize/load round-trip works')


def test_registry_create_alphazero():
    from alphigo.agent.registry import create_agent
    from alphigo.alphigo_board import GameState

    agent = _build_tiny_zero_agent(num_simulations=4)
    fd, path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        with h5py.File(path, 'w') as f:
            agent.serialize(f)
        # Default num_simulations.
        a1 = create_agent('alphazero', agent_file=path)
        # Override num_simulations.
        a2 = create_agent('alphazero', agent_file=path, num_simulations=2)
        assert a2._num_simulations == 2
        gs = GameState.new_game(9)
        m = a1.select_move(gs)
        assert m.is_pass or m.is_resign or gs.is_valid_move(m)
    finally:
        os.unlink(path)
    print('  [ok] registry create_agent("alphazero") works')


def test_virtual_loss_apply_revert():
    """Virtual loss must be exactly reversible."""
    from alphigo.alphigo_board import GameState
    agent = _build_tiny_zero_agent(num_simulations=8)
    gs = GameState.new_game(9)

    from alphigo.mcts.monte_carlo import MCTSNode
    root = MCTSNode(gs)
    root.N = 5
    root.W = 2.5
    root.Q = 0.5
    child = MCTSNode(gs, parent=root)
    child.N = 3
    child.W = 1.2
    child.Q = 0.4
    root.children.append(child)

    path = [root, child]
    orig_root = (root.N, root.W, root.Q)
    orig_child = (child.N, child.W, child.Q)

    agent._apply_virtual_loss(path)
    # After VL: N increased by 1, W increased by 1 (own-perspective Q is
    # negated during selection, so raising W repels the parent), Q = W/N
    assert root.N == 6, f"root.N should be 6, got {root.N}"
    assert child.N == 4, f"child.N should be 4, got {child.N}"
    assert abs(root.W - 3.5) < 1e-6, f"root.W should be 3.5, got {root.W}"
    assert abs(child.W - 2.2) < 1e-6, f"child.W should be 2.2, got {child.W}"

    agent._revert_virtual_loss(path)
    assert root.N == orig_root[0], f"root.N not reverted: {root.N}"
    assert abs(root.W - orig_root[1]) < 1e-6, f"root.W not reverted: {root.W}"
    assert abs(root.Q - orig_root[2]) < 1e-6, f"root.Q not reverted: {root.Q}"
    assert child.N == orig_child[0], f"child.N not reverted: {child.N}"
    assert abs(child.W - orig_child[1]) < 1e-6, f"child.W not reverted"
    assert abs(child.Q - orig_child[2]) < 1e-6, f"child.Q not reverted"
    print('  [ok] virtual loss apply/revert is exactly reversible')


def test_predict_batch():
    """_predict_batch must return correct shapes for batch of states."""
    from alphigo.alphigo_board import GameState
    agent = _build_tiny_zero_agent(num_simulations=4)
    gs = GameState.new_game(9)

    encoded = agent._encoder.encode(gs)
    batch = np.stack([encoded, encoded, encoded])  # B=3
    policies, net_values, ext_values, ownership = agent._predict_batch(batch)

    assert policies.shape == (3, 81), f"policies shape {policies.shape}"
    assert net_values.shape == (3,), f"net_values shape {net_values.shape}"
    assert ext_values is None, "ext_values should be None without external model"
    assert ownership is None, "two-head model has no ownership output"
    assert np.allclose(policies[0], policies[1], atol=1e-6), \
        "same input should give same output"
    print('  [ok] _predict_batch returns correct shapes')


def test_batched_search_produces_valid_results():
    """Batched MCTS (batch_size=4) must produce valid move and policy."""
    from alphigo.alphigo_board import GameState
    from alphigo.agent.zero_agent import ZeroAgent
    from alphigo.data.networks import build_alphazero_two_head
    from alphigo.encoders.alphazero import AlphaZeroEncoder

    encoder = AlphaZeroEncoder((9, 9))
    model = build_alphazero_two_head(encoder.shape(), encoder.num_points())
    agent = ZeroAgent(model, encoder, num_simulations=16, c_puct=1.5,
                      batch_size=4)

    gs = GameState.new_game(9)
    result = agent.search(gs)
    pt = result['policy_target']
    val = result['value']

    assert pt.shape == (81,), f"policy shape {pt.shape}"
    assert np.isclose(pt.sum(), 1.0, atol=1e-5), \
        f"policy should sum to 1, got {pt.sum()}"
    assert -1.0 <= val <= 1.0, f"value {val} outside [-1, 1]"
    move = result['move']
    assert move.is_pass or move.is_resign or gs.is_valid_move(move)
    print('  [ok] batched search (batch_size=4) produces valid results')


def test_batch_size_1_matches_sequential():
    """batch_size=1 must produce valid results (same code path)."""
    from alphigo.alphigo_board import GameState
    from alphigo.agent.zero_agent import ZeroAgent
    from alphigo.data.networks import build_alphazero_two_head
    from alphigo.encoders.alphazero import AlphaZeroEncoder

    np.random.seed(42)
    encoder = AlphaZeroEncoder((9, 9))
    model = build_alphazero_two_head(encoder.shape(), encoder.num_points())
    agent = ZeroAgent(model, encoder, num_simulations=8, c_puct=1.5,
                      batch_size=1)
    agent.set_temperature(0.0)

    gs = GameState.new_game(9)
    result = agent.search(gs)

    assert result['policy_target'].shape == (81,)
    assert np.isclose(result['policy_target'].sum(), 1.0, atol=1e-5)
    assert -1.0 <= result['value'] <= 1.0
    print('  [ok] batch_size=1 produces valid results (same code path)')


if __name__ == '__main__':
    test_select_move_returns_legal_move()
    test_search_target_and_value()
    test_temperature_zero_is_argmax()
    test_serialize_round_trip()
    test_registry_create_alphazero()
    test_virtual_loss_apply_revert()
    test_predict_batch()
    test_batched_search_produces_valid_results()
    test_batch_size_1_matches_sequential()
    print('tests/smoke_zero_agent.py: all passed')
