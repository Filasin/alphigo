import numpy as np

from alphigo.encoders.alphazero import AlphaZeroEncoder


class AlphaZeroKomiEncoder(AlphaZeroEncoder):
    """AlphaZero encoder + komi input plane — 18 planes.

    Planes 0-16 are identical to the ``alphazero`` encoder. Plane 17 is a
    constant plane carrying komi from the current player's perspective:

        plane 17 = (-komi if current player is Black else +komi) / 10

    i.e. positive when komi works in the mover's favor. With komi
    randomization in self-play, outcome prediction is only learnable as a
    function of (position, komi) — the network must estimate the board
    score instead of memorizing a fixed color base rate (the "always bet
    White" shortcut that a fixed 7.5 komi rewards).

    The komi value is held on the encoder instance (``self.komi``) rather
    than passed per call, so the ``encode(game_state)`` signature stays
    compatible with every existing caller. Self-play sets it per game;
    everything else (evals, probes) uses the standard default 7.5.
    """

    def __init__(self, board_size):
        super().__init__(board_size)
        self.num_planes = 18
        self.komi = 7.5

    def name(self):
        return 'alphazero_komi'

    def encode(self, game_state, out=None):
        from alphigo.alphigo_board import Player
        board_tensor = super().encode(game_state, out=out)
        if game_state.next_player == Player.black:
            board_tensor[:, :, 17] = -self.komi / 10.0
        else:
            board_tensor[:, :, 17] = self.komi / 10.0
        return board_tensor


def create(board_size):
    return AlphaZeroKomiEncoder(board_size)
