import enum
from collections import namedtuple


class Player(enum.IntEnum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white

    __hash__ = int.__hash__


# namedtuple makes the accessing the point variable
# like point.row instead of point[0], which increases readability
class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        return [Point(self.row - 1, self.col),
                Point(self.row + 1, self.col),
                Point(self.row, self.col - 1),
                Point(self.row, self.col + 1)
                ]
