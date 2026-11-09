"""Smoke test for Q-network builders."""
import numpy as np

from alphigo.data.networks import (
    build_small, build_q_two_input, NETWORK_MAP,
)


def test_linear_output():
    m = build_small((9, 9, 1), 81, output_activation='linear')
    y = m(np.zeros((2, 9, 9, 1), dtype='float32')).numpy()
    assert y.shape == (2, 81), f"expected (2, 81), got {y.shape}"
    # Linear outputs are unbounded — a random init will rarely all sum to 1
    row_sums = y.sum(axis=1)
    assert not np.allclose(row_sums, 1.0), \
        "linear-output model is still summing to 1 — softmax wasn't replaced"
    print('  [ok] linear output, shape', y.shape)


def test_softmax_output_default():
    m = build_small((9, 9, 1), 81)
    y = m(np.zeros((2, 9, 9, 1), dtype='float32')).numpy()
    assert y.shape == (2, 81)
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5), \
        "default output should still be softmax"
    print('  [ok] default softmax preserved')


def test_q_two_input():
    m = build_q_two_input((9, 9, 1), 81, network_name='small')
    board = np.zeros((3, 9, 9, 1), dtype='float32')
    action = np.zeros((3, 9, 9, 1), dtype='float32')
    action[0, 4, 4, 0] = 1.0  # centre move on sample 0
    q = m([board, action]).numpy()
    assert q.shape == (3, 1), f"expected (3, 1), got {q.shape}"
    assert np.all(q >= -1.0) and np.all(q <= 1.0), \
        "tanh head should bound Q in [-1, 1]"
    print('  [ok] two-input Q model, shape', q.shape, 'range OK')


if __name__ == '__main__':
    test_linear_output()
    test_softmax_output_default()
    test_q_two_input()
    print('tests/smoke_q_networks.py: all passed')
