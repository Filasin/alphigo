import abc
import importlib

import numpy as np

from alphigo.alphigo_board import Point


class Encoder(abc.ABC):
    """Base class for all board encoders.

    Subclasses must set self.board_width, self.board_height, and
    self.num_planes in __init__, then implement name() and encode().
    """

    @abc.abstractmethod
    def name(self) -> str:
        """Return the encoder's registered name string."""

    @abc.abstractmethod
    def encode(self, game_state) -> np.ndarray:
        """Encode a game state into a (H, W, planes) float32 array."""

    def encode_point(self, point: Point) -> int:
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index: int) -> Point:
        try:
            return self._point_table[index]
        except AttributeError:
            n = self.board_width * self.board_height
            self._point_table = [
                Point(row=i // self.board_width + 1,
                      col=i % self.board_width + 1)
                for i in range(n)
            ]
            return self._point_table[index]

    def num_points(self) -> int:
        return self.board_width * self.board_height

    def shape(self) -> tuple:
        return self.board_height, self.board_width, self.num_planes


def get_encoder_by_name(name, board_size):
    if isinstance(board_size, int):
        board_size = (board_size, board_size)
    module = importlib.import_module('alphigo.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size)
