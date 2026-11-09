import numpy as np

from alphigo.encoders.base import Encoder
from alphigo.alphigo_board import Move, Point, Player


class ElevenPlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        # 0 - 3. black stones with 1, 2, 3, 4+ liberties
        # 4 - 7. white stones with 1, 2, 3, 4+ liberties
        # 8. black plays next
        # 9. white plays next
        # 10. move would be illegal due to ko
        self.num_planes = 11

    def name(self):
        return 'elevenplane'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape(), dtype=np.float32)
        next_player = game_state.next_player

        # Planes 8-9: color to play
        if next_player == Player.black:
            board_tensor[:, :, 8] = 1
        else:
            board_tensor[:, :, 9] = 1

        # Planes 0-7: liberty buckets — iterate occupied points only
        for point, go_string in game_state.board.occupied_points():
            row, col = point.row - 1, point.col - 1
            liberty_plane = min(4, go_string.num_liberties) - 1
            if go_string.color == Player.white:
                liberty_plane += 4
            board_tensor[row, col, liberty_plane] = 1

        # Plane 10: ko — must check every empty point
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                if game_state.board.get_go_string(p) is None:
                    if game_state.does_move_violate_ko(next_player,
                                                       Move.play_move(p)):
                        board_tensor[r, c, 10] = 1
        return board_tensor



def create(board_size):
    return ElevenPlaneEncoder(board_size)
