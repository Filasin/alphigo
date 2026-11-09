"""_expand_batch must stash the node's network value as node.raw_value."""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_expand_stashes_raw_value():
    from alphigo.agent.zero_agent import ZeroAgent
    from alphigo.data.networks import build_alphazero_two_head
    from alphigo.encoders.alphazero import AlphaZeroEncoder
    from alphigo.alphigo_board import GameState
    from alphigo.mcts.monte_carlo import MCTSNode

    encoder = AlphaZeroEncoder((9, 9))
    model = build_alphazero_two_head(encoder.shape(), encoder.num_points())
    agent = ZeroAgent(model, encoder, num_simulations=2)

    node = MCTSNode(GameState.new_game(9))
    assert node.raw_value == 0.0, "raw_value must default to 0.0"
    values = agent._expand_batch([node])
    assert node.raw_value == values[0], (
        f"raw_value {node.raw_value} != returned value {values[0]}")
    print('  [ok] _expand_batch stashes raw_value')


if __name__ == '__main__':
    test_expand_stashes_raw_value()
    print('tests/test_raw_value_stash.py: all passed')
