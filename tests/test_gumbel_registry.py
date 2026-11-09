"""Registry + serialize round-trip for GumbelZeroAgent."""
import os
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np


def test_serialize_load_and_registry():
    from alphigo.agent.gumbel_zero_agent import GumbelZeroAgent, \
        load_gumbel_agent
    from alphigo.agent.registry import create_agent
    from alphigo.data.networks import build_alphazero_two_head
    from alphigo.encoders.alphazero import AlphaZeroEncoder
    from alphigo.alphigo_board import GameState

    encoder = AlphaZeroEncoder((9, 9))
    model = build_alphazero_two_head(encoder.shape(), encoder.num_points())
    agent = GumbelZeroAgent(model, encoder, num_simulations=8, m=4)

    fd, path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        with h5py.File(path, 'w') as f:
            agent.serialize(f)
        with h5py.File(path, 'r') as f:
            loaded = load_gumbel_agent(f)
        assert loaded._gumbel_m == 4
        assert loaded._num_simulations == 8

        # Registry: explicit params override stored attrs.
        np.random.seed(5)
        reg_agent = create_agent('gumbel', agent_file=path,
                                 num_simulations=4, m=2)
        assert reg_agent._num_simulations == 4
        assert reg_agent._gumbel_m == 2
        gs = GameState.new_game(9)
        mv = reg_agent.select_move(gs)
        assert mv.is_pass or gs.is_valid_move(mv)

        # A plain ZeroAgent checkpoint must also load as gumbel.
        from alphigo.agent.zero_agent import ZeroAgent
        za = ZeroAgent(model, encoder, num_simulations=8)
        with h5py.File(path, 'w') as f:
            za.serialize(f)
        with h5py.File(path, 'r') as f:
            loaded2 = load_gumbel_agent(f, num_simulations=8, m=4)
        assert loaded2._gumbel_m == 4
    finally:
        os.unlink(path)
    print('  [ok] serialize/load round-trip + registry create work')


if __name__ == '__main__':
    test_serialize_load_and_registry()
    print('tests/test_gumbel_registry.py: all passed')
