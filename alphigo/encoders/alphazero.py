import numpy as np

from alphigo.encoders.base import Encoder
from alphigo.alphigo_board import Point, Player


class AlphaZeroEncoder(Encoder):
    """AlphaZero-style encoder — 17 planes.

    Exactly the input representation used in Silver et al. 2017
    ("Mastering Chess and Shogi by Self-Play with a General Reinforcement
    Learning Algorithm"), adapted for Go.

    Layout
    ──────
    Planes  0– 7  Current player's stones at t, t-1, t-2, ..., t-7
    Planes  8–15  Opponent's stones       at t, t-1, t-2, ..., t-7
    Plane  16     Color to play: all-ones if current player is Black, else zeros

    History
    ───────
    Each history plane is a binary 19×19 image: 1 where that player had a
    stone at that point in time, 0 elsewhere.  Plane 0 / 8 are the current
    position; plane 7 / 15 are 7 moves ago.  Missing history (early in the
    game) is represented as all-zeros, which is the natural zero-padding used
    in the original paper.

    Why history matters
    ───────────────────
    A single-frame encoder cannot detect ko (the current position looks legal
    even when repeating a previous one).  With 8 frames the network can see
    the repeated position directly in its input, making ko naturally
    representable without a special ko plane.  History also reveals whether
    a group is growing or shrinking — useful for life-and-death judgement.
    """

    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 17

    def name(self):
        return 'alphazero'

    def encode(self, game_state, out=None):
        if out is None:
            board_tensor = np.zeros(self.shape(), dtype=np.float32)
        else:
            board_tensor = out
            board_tensor[:] = 0

        current_player = game_state.next_player
        cp = int(current_player)
        op = 3 - cp

        state = game_state
        for t in range(8):
            if state is None:
                break
            stones = state.board._stones
            board_tensor[:, :, t] = (stones == cp)
            board_tensor[:, :, t + 8] = (stones == op)
            state = state.previous_state

        if current_player == Player.black:
            board_tensor[:, :, 16] = 1

        return board_tensor



def create(board_size):
    return AlphaZeroEncoder(board_size)
