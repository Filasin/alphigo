import numpy as np

from alphigo import zobrist, alphigo_types
from alphigo.alphigo_types import Player, Point

# Precomputed neighbor/corner lists per point, keyed by board dimension (no alloc in hot path)
_neighbor_tables = {}
_corner_tables = {}


def _init_neighbor_table(dim):
    rows, cols = dim
    ntable = {}
    ctable = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            p = Point(row=r, col=c)
            ntable[p] = [
                n for n in p.neighbors()
                if 1 <= n.row <= rows and 1 <= n.col <= cols
            ]
            corners_all = [
                Point(r - 1, c - 1), Point(r - 1, c + 1),
                Point(r + 1, c - 1), Point(r + 1, c + 1),
            ]
            on = [cn for cn in corners_all if 1 <= cn.row <= rows and 1 <= cn.col <= cols]
            off = 4 - len(on)
            ctable[p] = (on, off)
    _neighbor_tables[dim] = ntable
    _corner_tables[dim] = ctable
    return ntable


class SuperkoTracker:
    """Tracks seen (player, board_hash) pairs for superko detection.

    Linked-list implementation: push() is O(1) (creates one node),
    contains() is O(n) walking the chain. This is faster overall
    because push() is called ~96× per MCTS search (every apply_move
    during descent) while contains() is only called during leaf
    expansion when a capture is detected (~2-5× per expansion).
    """
    __slots__ = ('_parent', '_key')

    def __init__(self, parent=None, key=None):
        self._parent = parent
        self._key = key

    def push(self, player, hash_val):
        """Return a new tracker with (player, hash_val) added."""
        return SuperkoTracker(self, (player, hash_val))

    def contains(self, player, hash_val):
        """Return True if (player, hash_val) has been seen before."""
        key = (player, hash_val)
        node = self
        while node._key is not None:
            if node._key == key:
                return True
            node = node._parent
        return False


class Move:
    __slots__ = ('point', 'is_play', 'is_pass', 'is_resign')

    _play_cache = {}
    _pass_move = None
    _resign_move = None

    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert sum([(point is not None), is_pass, is_resign]) == 1
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play_move(cls, point):
        m = cls._play_cache.get(point)
        if m is None:
            m = Move(point=point)
            cls._play_cache[point] = m
        return m

    @classmethod
    def pass_turn(cls):
        if cls._pass_move is None:
            cls._pass_move = Move(is_pass=True)
        return cls._pass_move

    @classmethod
    def resign(cls):
        if cls._resign_move is None:
            cls._resign_move = Move(is_resign=True)
        return cls._resign_move


# Strings are the stones with same color, which are connected
# Here we track GoString's stones and liberties (neighbor squares
# which are not occupied)
class GoString:
    __slots__ = ('color', 'stones', 'liberties', '_num_liberties')

    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = frozenset(stones) if not isinstance(stones, frozenset) else stones
        self.liberties = set(liberties) if not isinstance(liberties, set) else liberties
        self._num_liberties = len(self.liberties)

    def add_liberty(self, point):
        return self.with_liberty(point)

    def without_liberty(self, point):
        new_liberties = self.liberties.copy()
        new_liberties.discard(point)
        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point):
        new_liberties = self.liberties.copy()
        new_liberties.add(point)
        return GoString(self.color, self.stones, new_liberties)

    def merge_with(self, go_string):
        combined_stones = self.stones | go_string.stones
        combined_liberties = self.liberties | go_string.liberties
        combined_liberties -= combined_stones
        return GoString(self.color, combined_stones, combined_liberties)

    @property
    def num_liberties(self):
        return self._num_liberties

    def __eq__(self, other):
        return isinstance(other, GoString) and \
               self.color == other.color and \
               self.stones == other.stones and \
               self.liberties == other.liberties

    def __hash__(self):
        return hash((self.color, self.stones, frozenset(self.liberties)))


class Board:
    __slots__ = ('num_rows', 'num_cols', '_grid', '_hash', '_stones',
                 '_neighbor_table')

    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}
        self._hash = zobrist.EMPTY_BOARD
        self._stones = np.zeros((num_rows, num_cols), dtype=np.int8)
        dim = (num_rows, num_cols)
        if dim not in _neighbor_tables:
            _init_neighbor_table(dim)
        self._neighbor_table = _neighbor_tables[dim]

    def copy(self):
        """Shallow copy of the board. Safe because GoStrings are immutable."""
        board = Board.__new__(Board)
        board.num_rows = self.num_rows
        board.num_cols = self.num_cols
        board._grid = dict(self._grid)
        board._hash = self._hash
        board._stones = self._stones.copy()
        board._neighbor_table = self._neighbor_table
        return board

    def place_stone(self, player, point):
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None
        adjacent_same_color = set()
        adjacent_opposite_color = set()
        liberties = []
        for neighbor in self._neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                adjacent_same_color.add(neighbor_string)
            else:
                adjacent_opposite_color.add(neighbor_string)
        new_string = GoString(player, [point], liberties)

        # merging adjacent strings of same color
        for same_color_string in adjacent_same_color:
            new_string = new_string.merge_with(same_color_string)
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string

        self._stones[point.row - 1, point.col - 1] = int(player)

        # xor-ing the previous hash function, for storing new game state
        self._hash ^= zobrist.HASH_CODE[point, player]

        for other_color_string in adjacent_opposite_color:
            # reducing liberties of adjacent strings with opposite color
            replacement = other_color_string.without_liberty(point)
            if replacement.num_liberties:
                self._replace_string(replacement)
            else:
                # remove string if it has no liberties
                self._remove_string(other_color_string)

    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and \
               1 <= point.col <= self.num_cols

    def get_player_on_point(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    def occupied_points(self):
        """Yield (Point, GoString) for every occupied point.

        Used by encoders to iterate only stones instead of every board point.
        Each point in a GoString's stone set appears exactly once.
        Note: points where _grid[p] is None (removed stones) are excluded.
        """
        for point, go_string in self._grid.items():
            if go_string is not None:
                yield point, go_string

    def is_self_capture(self, player, point):
        """True if playing at point would be self-capture (no copy, logic only)."""
        friendly_strings = set()
        for neighbor in self._neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                return False  # has liberty
            if neighbor_string.color == player:
                friendly_strings.add(neighbor_string)
            else:
                if neighbor_string.num_liberties == 1:
                    return False  # would capture, not self-capture
        return all(s.num_liberties == 1 for s in friendly_strings)

    def will_capture(self, player, point):
        """True if playing at point would capture at least one opponent string."""
        for neighbor in self._neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None or neighbor_string.color == player:
                continue
            if neighbor_string.num_liberties == 1:
                return True
        return False

    def _replace_string(self, new_string):
        for point in new_string.stones:
            self._grid[point] = new_string

    def _remove_string(self, string):
        for point in string.stones:
            for neighbor in self._neighbor_table[point]:
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    # removing a string can create liberties
                    self._replace_string(neighbor_string.with_liberty(point))
            del self._grid[point]
            self._stones[point.row - 1, point.col - 1] = 0

            self._hash ^= zobrist.HASH_CODE[point, string.color]

    def zobrist_hash(self):
        return self._hash


def area_score(board, komi=7.5):
    """Area scoring (Chinese rules): stones + surrounded empty territory + komi.

    Returns (black_score, white_score) where white_score includes komi.

    Dead-stone handling: this function scores the board as-is without detecting
    dead stones. It is only used as a fallback for DL-vs-DL games where no GTP
    engine is present. In practice those games run long enough (~100 moves on 9x9)
    that dead stones are captured before both players pass. Games involving any
    GTP engine (GnuGo, Pachi) use the engine's own final_score command instead,
    which performs full life/death analysis. A proper L/D solver would be required
    to eliminate this limitation for DL-vs-DL games entirely.
    """
    size = board.num_rows
    black = 0
    white = 0
    visited = set()

    for row in range(1, size + 1):
        for col in range(1, size + 1):
            p = alphigo_types.Point(row=row, col=col)
            player = board.get_player_on_point(p)
            if player == alphigo_types.Player.black:
                black += 1
            elif player == alphigo_types.Player.white:
                white += 1
            elif p not in visited:
                # Flood-fill connected empty region
                region = []
                border = set()
                stack = [p]
                while stack:
                    curr = stack.pop()
                    if curr in visited:
                        continue
                    curr_player = board.get_player_on_point(curr)
                    if curr_player is not None:
                        border.add(curr_player)
                        continue
                    visited.add(curr)
                    region.append(curr)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = curr.row + dr, curr.col + dc
                        if 1 <= nr <= size and 1 <= nc <= size:
                            stack.append(alphigo_types.Point(row=nr, col=nc))
                if border == {alphigo_types.Player.black}:
                    black += len(region)
                elif border == {alphigo_types.Player.white}:
                    white += len(region)

    return black, white + komi


def ownership_map(board):
    """Per-intersection ownership: +1 black, -1 white, 0 neutral.

    Returns a numpy array of shape (board_size, board_size).
    Uses the same flood-fill logic as area_score but returns the full map
    instead of aggregate counts. No dead-stone detection.
    """
    import numpy as np
    size = board.num_rows
    omap = np.zeros((size, size), dtype=np.float32)
    visited = set()

    for row in range(1, size + 1):
        for col in range(1, size + 1):
            p = alphigo_types.Point(row=row, col=col)
            player = board.get_player_on_point(p)
            if player == alphigo_types.Player.black:
                omap[row - 1, col - 1] = 1.0
            elif player == alphigo_types.Player.white:
                omap[row - 1, col - 1] = -1.0
            elif p not in visited:
                region = []
                border = set()
                stack = [p]
                while stack:
                    curr = stack.pop()
                    if curr in visited:
                        continue
                    curr_player = board.get_player_on_point(curr)
                    if curr_player is not None:
                        border.add(curr_player)
                        continue
                    visited.add(curr)
                    region.append(curr)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = curr.row + dr, curr.col + dc
                        if 1 <= nr <= size and 1 <= nc <= size:
                            stack.append(alphigo_types.Point(row=nr, col=nc))
                if border == {alphigo_types.Player.black}:
                    for rp in region:
                        omap[rp.row - 1, rp.col - 1] = 1.0
                elif border == {alphigo_types.Player.white}:
                    for rp in region:
                        omap[rp.row - 1, rp.col - 1] = -1.0

    return omap


class GameState:
    __slots__ = ('board', 'next_player', 'previous_state', 'last_move',
                 '_tracker', '_is_over')

    def __init__(self, board, next_player, previous_state, move,
                 _tracker=None):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous_state
        self.last_move = move
        if _tracker is None:
            self._tracker = SuperkoTracker()
        else:
            self._tracker = _tracker
        if move is None:
            self._is_over = False
        elif move.is_resign:
            self._is_over = True
        else:
            self._is_over = (move.is_pass and previous_state is not None
                             and previous_state.last_move is not None
                             and previous_state.last_move.is_pass)

    def apply_move(self, move):
        if move.is_play:
            next_board = self.board.copy()
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        new_tracker = self._tracker.push(
            self.next_player, self.board.zobrist_hash()
        )
        return GameState(next_board, self.next_player.other, self, move,
                         _tracker=new_tracker)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    def move_history(self):
        """Full move history as [(player, move), ...] from oldest to newest."""
        moves = []
        state = self
        while state.last_move is not None:
            moves.append((state.next_player.other, state.last_move))
            state = state.previous_state
        moves.reverse()
        return moves

    def is_over(self):
        return self._is_over

    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False
        return self.board.is_self_capture(player, move.point)

    @property
    def situation(self):
        return self.next_player, self.board

    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        if not self.board.will_capture(player, move.point):
            return False  # ko only possible after a capture
        next_board = self.board.copy()
        next_board.place_stone(player, move.point)
        return self._tracker.contains(player.other, next_board.zobrist_hash())

    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (self.board.get_player_on_point(move.point) is None and
                not self.is_move_self_capture(self.next_player, move) and
                not self.does_move_violate_ko(self.next_player, move))

    def legal_moves(self):
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play_move(Point(row=row, col=col))
                if self.is_valid_move(move):
                    moves.append(move)
        moves.append(Move.pass_turn())
        moves.append(Move.resign())
        return moves

    def winner(self):
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.next_player
        black, white = area_score(self.board)
        if black > white:
            return alphigo_types.Player.black
        else:
            return alphigo_types.Player.white
