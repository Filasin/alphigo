import numpy as np

from alphigo.encoders.base import Encoder
from alphigo.alphigo_board import Move, Point


class AgPlaneEncoder(Encoder):
    """AlphaGo-inspired encoder — 22 planes.

    Inspired by the 48-plane encoder from the original AlphaGo SL policy
    network (Silver et al. 2016), scaled down to the features that carry the
    most signal at this data scale.  Omits features that require tree-search
    lookahead (ladder capture/escape) or expensive move simulation
    (self-atari, liberties-after-move).

    Layout
    ──────
    Planes  0– 7  Current player's stones: 1, 2, 3, 4, 5, 6, 7, 8+ liberties
    Planes  8–15  Opponent's stones:       1, 2, 3, 4, 5, 6, 7, 8+ liberties
    Plane  16     Ko — intersections illegal due to the ko rule
    Plane  17     Capture potential: playing here captures exactly 1 opponent stone
    Plane  18     Capture potential: playing here captures exactly 2 opponent stones
    Plane  19     Capture potential: playing here captures 3+ opponent stones
    Plane  20     Current player's stones on the board one move ago  (history)
    Plane  21     Opponent's stones on the board one move ago         (history)

    Notes
    ─────
    • Encoding is egocentric (player-relative), not absolute black/white.
    • Liberty buckets go up to 8 (vs 3 in sevenplane), capturing more
      fine-grained tactical state (e.g. a group with 2 libs is close to
      danger; 4 libs is comfortable; 8+ is virtually safe).
    • Capture planes are computed in O(board²) without any simulation:
      an opponent group in atari (1 liberty) will be captured by the move
      that plays on its sole remaining liberty.
    • History planes are zero at the opening move (no previous state).
    """

    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 22

    def name(self):
        return 'agplane'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape(), dtype=np.float32)
        current_player = game_state.next_player
        opponent = current_player.other

        # Planes 0-15: liberty buckets — iterate occupied points only
        for point, go_string in game_state.board.occupied_points():
            row, col = point.row - 1, point.col - 1
            liberty_plane = min(8, go_string.num_liberties) - 1
            if go_string.color == opponent:
                liberty_plane += 8
            board_tensor[row, col, liberty_plane] = 1

        # Plane 16: ko — check every empty point
        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row=row + 1, col=col + 1)
                if game_state.board.get_go_string(p) is None:
                    if game_state.does_move_violate_ko(current_player,
                                                       Move.play_move(p)):
                        board_tensor[row, col, 16] = 1

        # Planes 17-19: capture potential
        # For each empty point, count opponent stones in atari adjacent to it.
        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row=row + 1, col=col + 1)
                if game_state.board.get_go_string(p) is not None:
                    continue
                capture_count = 0
                for nb in p.neighbors():
                    if not (1 <= nb.row <= self.board_height and
                            1 <= nb.col <= self.board_width):
                        continue
                    nb_string = game_state.board.get_go_string(nb)
                    if (nb_string is not None and
                            nb_string.color == opponent and
                            nb_string.num_liberties == 1):
                        capture_count += len(nb_string.stones)
                if capture_count == 1:
                    board_tensor[row, col, 17] = 1
                elif capture_count == 2:
                    board_tensor[row, col, 18] = 1
                elif capture_count >= 3:
                    board_tensor[row, col, 19] = 1

        # Planes 20-21: history (one move ago)
        prev = game_state.previous_state
        if prev is not None:
            for point, go_string in prev.board.occupied_points():
                row, col = point.row - 1, point.col - 1
                plane = 20 if go_string.color == current_player else 21
                board_tensor[row, col, plane] = 1

        return board_tensor



def create(board_size):
    return AgPlaneEncoder(board_size)
