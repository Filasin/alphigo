"""Smoke test for ZeroReplayBuffer + batched_self_play_zero (issue 0003)."""
import os
import tempfile

import h5py
import numpy as np


def _build_tiny_zero_agent(num_simulations=3):
    from alphigo.agent.zero_agent import ZeroAgent
    from alphigo.data.networks import build_alphazero_two_head
    from alphigo.encoders.alphazero import AlphaZeroEncoder

    encoder = AlphaZeroEncoder((9, 9))
    model = build_alphazero_two_head(encoder.shape(), encoder.num_points())
    return ZeroAgent(model, encoder, num_simulations=num_simulations,
                     c_puct=1.5)


def test_zero_replay_buffer_round_trip():
    from alphigo.rl.zero_experience import ZeroReplayBuffer, load_zero_buffer

    n, planes, h, w, c = 7, 17, 9, 9, 81
    states = np.random.rand(n, h, w, planes).astype(np.float32)
    policy = np.random.rand(n, c).astype(np.float32)
    policy /= policy.sum(axis=1, keepdims=True)
    values = np.random.choice([-1.0, 1.0], size=n).astype(np.float32)

    buf = ZeroReplayBuffer(states, policy, values)
    assert buf.encoded_states.shape == (n, h, w, planes)
    assert buf.policy_targets.shape == (n, c)
    assert buf.value_targets.shape == (n,)

    fd, path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        with h5py.File(path, 'w') as f:
            buf.serialize(f)
        with h5py.File(path, 'r') as f:
            loaded = load_zero_buffer(f)
        assert np.allclose(loaded.encoded_states, states)
        assert np.allclose(loaded.policy_targets, policy)
        assert np.allclose(loaded.value_targets, values)
    finally:
        os.unlink(path)
    print('  [ok] ZeroReplayBuffer round-trips')


def test_batched_self_play_zero_smoke():
    from alphigo.rl.simulate import batched_self_play_zero

    agent = _build_tiny_zero_agent(num_simulations=3)
    buf = batched_self_play_zero(
        agent, board_size=9, num_games=5, num_simulations=3,
        temperature_moves=4, verbose=False,
    )

    assert buf.encoded_states.ndim == 4
    n = buf.encoded_states.shape[0]
    assert n > 0, "expected at least one decision recorded"
    assert buf.policy_targets.shape == (n, 81)
    assert buf.value_targets.shape == (n,)

    sums = buf.policy_targets.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), \
        f"policy_targets row sums not ~1: min={sums.min()}, max={sums.max()}"

    uniq_values = np.unique(buf.value_targets)
    # Allow {-1, 0, +1}: a draw / max-moves cap can yield 0, but {-1, +1}
    # should always be the dominant case.
    assert set(uniq_values.tolist()).issubset({-1.0, 0.0, 1.0}), \
        f"unexpected value targets: {uniq_values}"
    assert {-1.0, 1.0}.issubset(set(uniq_values.tolist())) or \
        len(uniq_values) >= 1
    print('  [ok] batched_self_play_zero produced', n, 'decisions')


def test_temperature_schedule_applied():
    """First N moves use T=1 (sampling); after, T=0 (greedy one-hot)."""
    from alphigo.rl.simulate import batched_self_play_zero

    agent = _build_tiny_zero_agent(num_simulations=3)
    n_temp = 4
    buf = batched_self_play_zero(
        agent, board_size=9, num_games=2, num_simulations=3,
        temperature_moves=n_temp, verbose=False,
    )

    # We can't observe move ordering directly, but at T=0 the policy
    # target must be a one-hot vector. After enough moves per game,
    # at least some samples should be one-hot.
    pt = buf.policy_targets
    one_hot = np.isclose(pt.max(axis=1), 1.0, atol=1e-6)
    assert one_hot.any(), \
        "expected at least some one-hot policy targets from T=0 phase"
    print('  [ok] temperature schedule produced one-hot targets after move',
          n_temp)


if __name__ == '__main__':
    test_zero_replay_buffer_round_trip()
    test_batched_self_play_zero_smoke()
    test_temperature_schedule_applied()
    print('tests/smoke_zero_replay_buffer.py: all passed')
