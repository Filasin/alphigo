"""Smoke test: create_agent() builds the book-style Q variant from an h5 file."""
import os
import tempfile

import h5py

from alphigo.agent import create_agent, QAgent
from alphigo.agent.q_agent import load_q_agent
from alphigo.data.networks import build_q_two_input
from alphigo.encoders.base import get_encoder_by_name


def _serialize(agent, path):
    with h5py.File(path, 'w') as f:
        agent.serialize(f)


def test_create_q():
    enc = get_encoder_by_name('oneplane', 9)
    with tempfile.TemporaryDirectory() as d:
        q_model = build_q_two_input(enc.shape(), enc.num_points(),
                                    network_name='small')
        _serialize(QAgent(q_model, enc), os.path.join(d, 'q.h5'))
        q_agent = create_agent('q',
                               agent_file=os.path.join(d, 'q.h5'),
                               epsilon=0.2)
        assert isinstance(q_agent, QAgent)
        assert q_agent._epsilon == 0.2

    print('  [ok] create_agent("q") works')


if __name__ == '__main__':
    test_create_q()
    print('tests/smoke_q_registry.py: all passed')
