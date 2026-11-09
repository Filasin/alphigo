"""Smoke tests for Agent.analyze (issue 0008).

Each neural agent (DL, PG, Q, AC, AZ) must return a populated AnalysisResult
on a fresh 9x9 game state.  Random / GTP agents return None (default).
"""
import numpy as np

from alphigo.alphigo_board import GameState, Move
from alphigo.agent.analysis import AnalysisResult


def _empty_game(board_size=9):
    return GameState.new_game(board_size)


def _assert_top_moves_legal(result, game, k=5):
    assert isinstance(result, AnalysisResult), type(result)
    assert len(result.top_moves) <= k
    assert len(result.top_moves) > 0
    # Sorted descending by score
    scores = [s for _, s in result.top_moves]
    assert scores == sorted(scores, reverse=True), scores
    for point, score in result.top_moves:
        assert 0.0 <= score <= 100.0, score
        assert game.is_valid_move(Move.play_move(point)), point


def test_random_agent_returns_none():
    from alphigo.agent.naive import RandomBot
    bot = RandomBot()
    assert bot.analyze(_empty_game()) is None
    print('  [ok] RandomBot.analyze returns None')


def test_dl_agent_analyze():
    from alphigo.agent.deep_learning import DeepLearningAgent
    from alphigo.encoders import get_encoder_by_name
    from alphigo.data.networks import build_small_orig

    enc = get_encoder_by_name('oneplane', 9)
    model = build_small_orig(enc.shape(), enc.num_points())
    agent = DeepLearningAgent(model, enc)
    g = _empty_game()
    r = agent.analyze(g)
    _assert_top_moves_legal(r, g)
    assert r.position_winrate is None
    print('  [ok] DeepLearningAgent.analyze populates top_moves, no winrate')


def test_pg_agent_analyze():
    from alphigo.agent.policy_agent import PolicyAgent
    from alphigo.encoders import get_encoder_by_name
    from alphigo.data.networks import build_small_orig

    enc = get_encoder_by_name('oneplane', 9)
    model = build_small_orig(enc.shape(), enc.num_points())
    agent = PolicyAgent(model, enc)
    g = _empty_game()
    r = agent.analyze(g)
    _assert_top_moves_legal(r, g)
    assert r.position_winrate is None
    print('  [ok] PolicyAgent.analyze populates top_moves, no winrate')


def test_q_agent_analyze():
    from alphigo.agent.q_agent import QAgent
    from alphigo.encoders import get_encoder_by_name
    from alphigo.data.networks import build_q_two_input

    enc = get_encoder_by_name('oneplane', 9)
    model = build_q_two_input(enc.shape(), enc.num_points())
    agent = QAgent(model, enc)
    g = _empty_game()
    r = agent.analyze(g)
    _assert_top_moves_legal(r, g)
    assert r.position_winrate is None
    print('  [ok] QAgent.analyze populates top_moves, no winrate')


def test_ac_agent_analyze():
    from alphigo.agent.ac_agent import ACAgent
    from alphigo.encoders import get_encoder_by_name
    from alphigo.data.networks import build_ac_two_output

    enc = get_encoder_by_name('oneplane', 9)
    model = build_ac_two_output(enc.shape(), enc.num_points())
    agent = ACAgent(model, enc)
    g = _empty_game()
    r = agent.analyze(g)
    _assert_top_moves_legal(r, g)
    assert r.position_winrate is not None
    assert 0.0 <= r.position_winrate <= 1.0, r.position_winrate
    print('  [ok] ACAgent.analyze populates top_moves and winrate in [0,1]')


def test_az_agent_analyze():
    from alphigo.agent.zero_agent import ZeroAgent
    from alphigo.data.networks import build_alphazero_two_head
    from alphigo.encoders.alphazero import AlphaZeroEncoder

    enc = AlphaZeroEncoder((9, 9))
    model = build_alphazero_two_head(enc.shape(), enc.num_points())
    agent = ZeroAgent(model, enc, num_simulations=8)
    g = _empty_game()
    r = agent.analyze(g)
    _assert_top_moves_legal(r, g)
    assert r.position_winrate is not None
    assert 0.0 <= r.position_winrate <= 1.0, r.position_winrate
    print('  [ok] ZeroAgent.analyze populates top_moves and winrate in [0,1]')


if __name__ == '__main__':
    test_random_agent_returns_none()
    test_dl_agent_analyze()
    test_pg_agent_analyze()
    test_q_agent_analyze()
    test_ac_agent_analyze()
    test_az_agent_analyze()
    print('tests/smoke_agent_analyze.py: all passed')
