import numpy as np

from alphigo.encoders.base import Encoder
from alphigo.alphigo_board import Point


class OnePlaneEncoder(Encoder):

    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 1

    def name(self):
        return 'oneplane'

    # fills matrix with 1 if the point containts one of the current
    # player's stones, -1 if the point containts one of the opponent's
    # stones, 0 otherwise
    def encode(self, game_state):
        board_matrix = np.zeros(self.shape(), dtype=np.float32)
        next_player = game_state.next_player
        for point, go_string in game_state.board.occupied_points():
            row, col = point.row - 1, point.col - 1
            board_matrix[row, col, 0] = 1 if go_string.color == next_player else -1
        return board_matrix



def create(board_size):
    return OnePlaneEncoder(board_size)
