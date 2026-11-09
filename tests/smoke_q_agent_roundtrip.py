"""Smoke test: QAgent construction, serialize, load round-trip."""
import os
import tempfile

import h5py
import numpy as np

from alphigo.agent.q_agent import QAgent, load_q_agent
from alphigo.data.networks import build_q_two_input
from alphigo.encoders.base import get_encoder_by_name


def test_construct_and_roundtrip():
    board_size = 9
    encoder = get_encoder_by_name('oneplane', board_size)
    model = build_q_two_input(encoder.shape(), encoder.num_points(),
                              network_name='small')
    agent = QAgent(model, encoder)
    agent.set_epsilon(0.15)
    assert agent._epsilon == 0.15

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'q.h5')
        with h5py.File(path, 'w') as f:
            agent.serialize(f)
        with h5py.File(path, 'r') as f:
            restored = load_q_agent(f)

    before = [w.numpy() for w in agent._model.weights]
    after = [w.numpy() for w in restored._model.weights]
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert np.allclose(b, a), "weights changed after round-trip"
    assert restored._encoder.name() == 'oneplane'
    # Confirm the model expects two inputs
    assert len(restored._model.inputs) == 2, \
        f"expected 2 inputs, got {len(restored._model.inputs)}"
    print('  [ok] QAgent round-trip, 2 inputs preserved')


if __name__ == '__main__':
    test_construct_and_roundtrip()
    print('tests/smoke_q_agent_roundtrip.py: all passed')
