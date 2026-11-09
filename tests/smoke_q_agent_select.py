"""Smoke test: QAgent.select_move returns legal moves and records decisions."""
import numpy as np

from alphigo.agent.q_agent import QAgent
from alphigo.alphigo_board import GameState, Move
from alphigo.data.networks import build_q_two_input
from alphigo.encoders.base import get_encoder_by_name
from alphigo.rl import ExperienceCollector


def _make_agent(board_size=9):
    encoder = get_encoder_by_name('oneplane', board_size)
    model = build_q_two_input(encoder.shape(), encoder.num_points(),
                              network_name='small')
    return QAgent(model, encoder)


def test_returns_legal_move_on_empty_board():
    np.random.seed(0)
    agent = _make_agent()
    state = GameState.new_game(9)
    move = agent.select_move(state)
    assert move.is_play, f"expected a play, got {move}"
    assert state.is_valid_move(Move.play_move(move.point))
    print('  [ok] empty-board move is legal:', move.point)


def test_records_to_collector():
    np.random.seed(0)
    agent = _make_agent()
    collector = ExperienceCollector()
    collector.begin_episode()
    agent.set_collector(collector)
    state = GameState.new_game(9)
    agent.select_move(state)
    assert len(collector._current_episode_actions) == 1
    print('  [ok] decision recorded')


def test_epsilon_one_is_random_over_legal():
    np.random.seed(42)
    agent = _make_agent()
    agent.set_epsilon(1.0)
    state = GameState.new_game(9)

    counts = np.zeros(81, dtype=int)
    for _ in range(200):
        move = agent.select_move(state)
        if move.is_play:
            counts[agent._encoder.encode_point(move.point)] += 1
    assert (counts > 0).sum() >= 60, \
        f"only {(counts > 0).sum()} distinct moves in 200 draws"
    print('  [ok] epsilon=1 gave', (counts > 0).sum(), 'distinct moves')


if __name__ == '__main__':
    test_returns_legal_move_on_empty_board()
    test_records_to_collector()
    test_epsilon_one_is_random_over_legal()
    print('tests/smoke_q_agent_select.py: all passed')
