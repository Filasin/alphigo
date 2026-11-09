"""BatchedSimulator must mirror GameState turn-by-turn on random games.

Per ADR-0002, the batched-board engine is a deliberate duplicate of the
canonical GameState (today, GameState-backed; in the future, potentially
numpy-vectorised). This test guards against drift if/when that backing
diverges.
"""
import numpy as np
import pytest

from alphigo.alphigo_board import GameState, Move
from alphigo.training.batched_board import BatchedSimulator
from alphigo.encoders.base import get_encoder_by_name


BOARD_SIZE = 9
NUM_GAMES = 200
SEED = 12345


def _legal_action_indices(state, encoder):
    """Return list of action-index ints that are legal plays in `state`."""
    legal = []
    n = encoder.board_width * encoder.board_height
    for idx in range(n):
        point = encoder.decode_point_index(idx)
        if state.is_valid_move(Move.play_move(point)):
            legal.append(idx)
    return legal


def test_batched_board_matches_gamestate_on_random_games():
    rng = np.random.default_rng(SEED)
    encoder = get_encoder_by_name('oneplane', BOARD_SIZE)

    sim = BatchedSimulator(
        encoder=encoder, board_size=BOARD_SIZE, num_games=NUM_GAMES,
    )
    states = [GameState.new_game(BOARD_SIZE) for _ in range(NUM_GAMES)]

    for _step in range(2 * BOARD_SIZE * BOARD_SIZE):
        # Pick the same action for both engines, deterministically per game.
        actions = np.full(NUM_GAMES, -1, dtype=np.int32)  # -1 == pass
        for gi in range(NUM_GAMES):
            if states[gi].is_over() or sim.is_finished(gi):
                continue
            legal = _legal_action_indices(states[gi], encoder)
            if not legal:
                continue  # both will pass
            actions[gi] = rng.choice(legal)

        # Apply to GameState
        for gi in range(NUM_GAMES):
            if states[gi].is_over():
                continue
            if actions[gi] == -1:
                states[gi] = states[gi].apply_move(Move.pass_turn())
            else:
                point = encoder.decode_point_index(int(actions[gi]))
                states[gi] = states[gi].apply_move(Move.play_move(point))

        # Apply to batched simulator
        sim.step(actions)

        # Compare board occupancies
        for gi in range(NUM_GAMES):
            assert sim.board_matches(gi, states[gi]), \
                f'divergence at game {gi}, step {_step}'


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_known_edge_cases(seed):
    """Hand-crafted positions: ko, suicide, eye-fill, scoring corners.

    Each scenario plays a fixed move sequence on both engines and asserts
    identical final state.
    """
    encoder = get_encoder_by_name('oneplane', BOARD_SIZE)
    sim = BatchedSimulator(encoder=encoder, board_size=BOARD_SIZE, num_games=1)
    state = GameState.new_game(BOARD_SIZE)

    # Each scenario is a list of (row, col) plays.
    scenarios = {
        1: [(4, 4), (3, 4), (4, 5), (3, 5), (5, 4), (4, 3), (5, 5)],   # ko-shape
        2: [(0, 0), (8, 8), (0, 8), (8, 0), (4, 4)],                   # corners
        3: [(2, 2), (2, 3), (3, 2), (3, 3), (4, 4), (5, 5), (4, 5)],   # eye
    }

    for (row, col) in scenarios[seed]:
        idx = row * BOARD_SIZE + col
        if not state.is_over():
            point = encoder.decode_point_index(idx)
            move = Move.play_move(point)
            if state.is_valid_move(move):
                state = state.apply_move(move)
                sim.step(np.array([idx], dtype=np.int32))
            else:
                # Both should treat this as pass.
                state = state.apply_move(Move.pass_turn())
                sim.step(np.array([-1], dtype=np.int32))
        assert sim.board_matches(0, state)
