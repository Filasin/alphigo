"""Smoke test for AlphaZero --bootstrap from SL checkpoint (issue 0005)."""
import os
import tempfile

import h5py
import numpy as np
import pytest


def _write_sl_checkpoint(path, encoder_name='alphazero', board_size=9,
                         network_name='alphazero'):
    """Build a tiny pretend-SL checkpoint and overfit it on one example
    so the policy head is non-uniform when loaded for bootstrap."""
    from alphigo.agent.deep_learning import DeepLearningAgent
    from alphigo.data.networks import build_alphazero, NETWORK_MAP
    from alphigo.encoders.base import get_encoder_by_name
    from tensorflow.keras.optimizers import SGD

    encoder = get_encoder_by_name(encoder_name, board_size)
    if network_name == 'alphazero':
        model = build_alphazero(encoder.shape(), encoder.num_points())
    else:
        model = NETWORK_MAP[network_name](encoder.shape(),
                                          encoder.num_points())
    model.compile(loss='categorical_crossentropy', optimizer=SGD(0.5),
                  jit_compile=False)
    # One-step "training" so weights are not all-default-init garbage.
    x = np.random.RandomState(0).rand(
        4, *encoder.shape()).astype('float32')
    y = np.zeros((4, encoder.num_points()), dtype='float32')
    y[np.arange(4), [10, 20, 30, 40]] = 1.0
    model.fit(x, y, epochs=2, batch_size=4, verbose=0)

    agent = DeepLearningAgent(model, encoder)
    with h5py.File(path, 'w') as f:
        agent.serialize(f)


def test_bootstrap_compatible_sl_warm_starts_policy():
    fd, sl_path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        _write_sl_checkpoint(sl_path, encoder_name='alphazero',
                             board_size=9, network_name='alphazero')

        from alphigo.encoders.base import get_encoder_by_name
        from alphigo.training.method.alphazero import AlphaZeroMethod

        encoder = get_encoder_by_name('alphazero', 9)
        m = AlphaZeroMethod(num_simulations=3)
        agent = m.bootstrap_from_sl(sl_path, encoder, 9)

        # Two-headed model: policy head should be non-uniform after
        # warm start, value head still ~tanh(0)≈0.
        x = np.zeros((1, *encoder.shape()), dtype='float32')
        policy, value = agent._model(x, training=False)
        policy = policy.numpy()[0]
        assert policy.shape == (81,)
        # Non-uniform: max prob meaningfully exceeds uniform 1/81 ≈ 0.0123.
        assert policy.max() > 0.02, \
            f"bootstrapped policy looks uniform: max={policy.max()}"
        assert -1.0 <= float(value.numpy()[0, 0]) <= 1.0
    finally:
        os.unlink(sl_path)
    print('  [ok] compatible bootstrap warms policy head, value in range')


def test_bootstrap_wrong_encoder_raises():
    fd, sl_path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        _write_sl_checkpoint(sl_path, encoder_name='oneplane',
                             board_size=9, network_name='small')

        from alphigo.encoders.base import get_encoder_by_name
        from alphigo.training.method.alphazero import AlphaZeroMethod
        encoder = get_encoder_by_name('alphazero', 9)
        m = AlphaZeroMethod(num_simulations=3)
        with pytest.raises(ValueError, match="alphazero"):
            m.bootstrap_from_sl(sl_path, encoder, 9)
    finally:
        os.unlink(sl_path)
    print('  [ok] wrong-encoder SL checkpoint raises ValueError')


def test_bootstrap_wrong_network_raises():
    fd, sl_path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        # alphazero encoder but a different network architecture: shapes
        # diverge in the trunk so greedy match must fail.
        _write_sl_checkpoint(sl_path, encoder_name='alphazero',
                             board_size=9, network_name='small')

        from alphigo.encoders.base import get_encoder_by_name
        from alphigo.training.method.alphazero import AlphaZeroMethod
        encoder = get_encoder_by_name('alphazero', 9)
        m = AlphaZeroMethod(num_simulations=3)
        with pytest.raises(ValueError):
            m.bootstrap_from_sl(sl_path, encoder, 9)
    finally:
        os.unlink(sl_path)
    print('  [ok] wrong-network SL checkpoint raises ValueError')


def test_bootstrap_rejected_for_wrong_method():
    """train_rl --method q --bootstrap must error before training."""
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]

    fd, sl_path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        _write_sl_checkpoint(sl_path)
        proc = subprocess.run(
            [sys.executable, 'train_rl.py',
             '--method', 'q', '--board-size', '9',
             '--encoder', 'oneplane', '--network', 'small',
             '--bootstrap', sl_path, '--rounds', '1', '--games', '1'],
            cwd=str(repo), capture_output=True, text=True, timeout=120)
        assert proc.returncode != 0, \
            "train_rl should reject --bootstrap with --method q"
        combined = proc.stdout + proc.stderr
        assert 'bootstrap' in combined.lower()
    finally:
        os.unlink(sl_path)
    print('  [ok] --bootstrap rejected for non-alphazero/non-pg methods')


if __name__ == '__main__':
    test_bootstrap_compatible_sl_warm_starts_policy()
    test_bootstrap_wrong_encoder_raises()
    test_bootstrap_wrong_network_raises()
    test_bootstrap_rejected_for_wrong_method()
    print('tests/smoke_zero_bootstrap.py: all passed')
