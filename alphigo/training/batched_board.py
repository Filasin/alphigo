"""Batched Go-engine container for self-play throughput.

This module owns a `BatchedSimulator` whose external surface area is the
single-step `step(actions)` call plus query methods (`is_finished`,
`board_matches`). Today the implementation is backed by a
`list[GameState]` — the canonical Go engine in `alphigo.alphigo_board`.
Per ADR-0002, this module is the home for a future, genuinely vectorised
NumPy Go engine that would replace that backing while preserving the API.
The equivalence test in `tests/test_batched_board_equivalence.py` is the
regression guard against drift between this engine and `GameState`.

Note: the older self-play orchestrator in `alphigo.rl.simulate` is also
called `BatchedSimulator` but is a different class (model + encoder +
multi-game self-play loop). That class is scheduled for deletion once
Plan A's `train_rl.py` rewrite (Task 20) lands.
"""
from __future__ import annotations

import numpy as np

from alphigo.alphigo_board import GameState, Move, area_score
from alphigo.alphigo_types import Player
from alphigo.agent.helpers import is_point_an_eye


# Sentinel action: a non-negative encoded point index means "play that
# point"; any negative value (-1 by convention) means "pass turn".
PASS_ACTION = -1


class BatchedSimulator:
    """Holds N independent Go games and steps them in lock-step.

    The simulator owns one `GameState` per game slot plus a per-slot
    `active` flag. `step(actions)` applies one move per active game.
    Inactive games (those whose `GameState.is_over()` returned True at
    some point) are skipped on subsequent steps, even if the caller
    supplies a non-pass action for them.
    """

    def __init__(self, encoder, board_size: int, num_games: int):
        self._encoder = encoder
        self._board_size = int(board_size)
        self._num_games = int(num_games)
        self._games: list[GameState] = [
            GameState.new_game(self._board_size) for _ in range(self._num_games)
        ]
        self._active: list[bool] = [True] * self._num_games

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_games(self) -> int:
        return self._num_games

    @property
    def board_size(self) -> int:
        return self._board_size

    # ── Per-game queries ──────────────────────────────────────────────

    def is_finished(self, game_idx: int) -> bool:
        """True if game `game_idx` has ended."""
        return not self._active[game_idx]

    def state(self, game_idx: int) -> GameState:
        """The current `GameState` for game `game_idx`."""
        return self._games[game_idx]

    def state_tensor(self, game_idx: int) -> np.ndarray:
        """Encoded board for the side-to-move in game `game_idx`."""
        return self._encoder.encode(self._games[game_idx])

    def legal_mask(self, game_idx: int) -> np.ndarray:
        """Bool mask of length board_size**2: True iff playing that point
        is legal for the side to move and is not eye-filling."""
        game = self._games[game_idx]
        n = self._board_size * self._board_size
        mask = np.zeros(n, dtype=bool)
        for idx in range(n):
            point = self._encoder.decode_point_index(idx)
            if game.is_valid_move(Move.play_move(point)) and \
                    not is_point_an_eye(game.board, point, game.next_player):
                mask[idx] = True
        return mask

    def outcome(self, game_idx: int) -> int:
        """+1 if black won, -1 if white won, 0 if not over (treated as draw).

        run_batch assigns player A to black, so a positive outcome credits
        player A.
        """
        game = self._games[game_idx]
        if not game.is_over():
            return 0
        winner = game.winner()
        if winner == Player.black:
            return 1
        if winner == Player.white:
            return -1
        return 0

    def board_matches(self, game_idx: int, state: GameState) -> bool:
        """True iff the simulator's board for `game_idx` matches `state`."""
        sim_state = self._games[game_idx]
        bs = self._board_size
        for r in range(bs):
            for c in range(bs):
                point = self._encoder.decode_point_index(r * bs + c)
                if sim_state.board.get_player_on_point(point) != state.board.get_player_on_point(point):
                    return False
        return True

    # ── Stepping ──────────────────────────────────────────────────────

    def step(self, actions) -> None:
        """Apply one action per game.

        `actions` is an int array (or sequence) of length `num_games`.
        A negative entry (canonically -1) means pass; otherwise the
        entry is the encoder's flat point index.
        """
        actions = np.asarray(actions)
        if actions.shape[0] != self._num_games:
            raise ValueError(
                f'actions length {actions.shape[0]} != num_games {self._num_games}'
            )
        for gi in range(self._num_games):
            if not self._active[gi]:
                continue
            a = int(actions[gi])
            if a < 0:
                self._apply_pass(gi)
            else:
                self._apply_play(gi, a)

    # ── Internals ─────────────────────────────────────────────────────

    def _apply_pass(self, gi: int) -> None:
        self._games[gi] = self._games[gi].apply_move(Move.pass_turn())
        if self._games[gi].is_over():
            self._active[gi] = False

    def _apply_play(self, gi: int, action_idx: int) -> None:
        point = self._encoder.decode_point_index(action_idx)
        self._games[gi] = self._games[gi].apply_move(Move.play_move(point))
        if self._games[gi].is_over():
            self._active[gi] = False
