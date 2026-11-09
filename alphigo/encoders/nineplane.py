import numpy as np

from alphigo.encoders.base import Encoder
from alphigo.alphigo_board import Move, Point


class NinePlaneEncoder(Encoder):
    """Sevenplane + 2 history planes.

    Planes 0-2: current player's stones with 1, 2, 3+ liberties (egocentric)
    Planes 3-5: opponent's stones with 1, 2, 3+ liberties (egocentric)
    Plane  6:   ko — intersections illegal due to ko rule
    Plane  7:   current player's stones on the board one move ago
    Plane  8:   opponent's stones on the board one move ago

    History planes are zero when there is no previous state (opening move).
    """

    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 9

    def name(self):
        return 'nineplane'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape(), dtype=np.float32)
        current_player = game_state.next_player
        opponent = current_player.other
        base_plane = {current_player: 0, opponent: 3}

        # Planes 0-5: liberty buckets (current board)
        for point, go_string in game_state.board.occupied_points():
            row, col = point.row - 1, point.col - 1
            liberty_plane = min(3, go_string.num_liberties) - 1
            liberty_plane += base_plane[go_string.color]
            board_tensor[row, col, liberty_plane] = 1

        # Plane 6: ko
        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row=row + 1, col=col + 1)
                if game_state.board.get_go_string(p) is None:
                    if game_state.does_move_violate_ko(current_player,
                                                       Move.play_move(p)):
                        board_tensor[row, col, 6] = 1

        # Planes 7-8: previous board position
        prev = game_state.previous_state
        if prev is not None:
            for point, go_string in prev.board.occupied_points():
                row, col = point.row - 1, point.col - 1
                plane = 7 if go_string.color == current_player else 8
                board_tensor[row, col, plane] = 1

        return board_tensor



def create(board_size):
    return NinePlaneEncoder(board_size)
