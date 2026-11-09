import numpy as np

from alphigo.encoders.base import Encoder
from alphigo.alphigo_board import Move, Point


class SevenPlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 7

    def name(self):
        return 'sevenplane'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape(), dtype=np.float32)
        next_player = game_state.next_player
        base_plane = {next_player: 0, next_player.other: 3}

        # Planes 0-5: liberty buckets — iterate occupied points only
        for point, go_string in game_state.board.occupied_points():
            row, col = point.row - 1, point.col - 1
            liberty_plane = min(3, go_string.num_liberties) - 1
            liberty_plane += base_plane[go_string.color]
            board_tensor[row, col, liberty_plane] = 1

        # Plane 6: ko — must check every empty point
        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row=row + 1, col=col + 1)
                if game_state.board.get_go_string(p) is None:
                    if game_state.does_move_violate_ko(next_player,
                                                       Move.play_move(p)):
                        board_tensor[row, col, 6] = 1
        return board_tensor



def create(board_size):
    return SevenPlaneEncoder(board_size)
