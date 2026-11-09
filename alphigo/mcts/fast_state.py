"""Fast numpy-only Go board for MCTS tree search.

Eliminates GoString objects and dict-based grid entirely.
Board state is a flat int8 numpy array; copies are 81-byte memcpy
(vs dict copy + GoString frozenset/set operations in Board).
Captures detected via flood-fill with temporary stone negation
(O(1) visited check, no set allocation).

When the Cython extension ``_cboard`` is available, place_stone and
compute_groups run at C speed (10-50x faster).
"""
import numpy as np

from alphigo import zobrist
from alphigo.alphigo_board import Move, SuperkoTracker
from alphigo.alphigo_types import Player, Point

try:
    from alphigo.mcts._cboard import (
        init_tables as _c_init_tables,
        place_stone as _c_place_stone,
        legal_moves as _c_legal_moves,
        c_select_child_deterministic,
        c_backup,
        c_descend,
        c_child_stats,
    )
    _HAS_CBOARD = True
except ImportError:
    _HAS_CBOARD = False

# ── Precomputed tables (per board size) ─────────────────────────────────

_NBR = {}
_CORNERS = {}
_HASH_FLAT = {}
_POINT_LUT = {}


def _ensure_tables(size):
    if size in _NBR:
        return
    n = size * size
    nbr = [None] * n
    corners = [None] * n
    hflat = [None] * n
    plut = [None] * n

    for idx in range(n):
        r, c = divmod(idx, size)
        row, col = r + 1, c + 1
        pt = Point(row=row, col=col)
        plut[idx] = pt

        nb = []
        if r > 0:
            nb.append(idx - size)
        if r < size - 1:
            nb.append(idx + size)
        if c > 0:
            nb.append(idx - 1)
        if c < size - 1:
            nb.append(idx + 1)
        nbr[idx] = nb

        on = []
        off = 0
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                on.append(nr * size + nc)
            else:
                off += 1
        corners[idx] = (on, off)

        hflat[idx] = {
            1: zobrist.HASH_CODE[pt, Player.black],
            2: zobrist.HASH_CODE[pt, Player.white],
        }

    _NBR[size] = nbr
    _CORNERS[size] = corners
    _HASH_FLAT[size] = hflat
    _POINT_LUT[size] = plut

    if _HAS_CBOARD:
        _c_init_tables(size, nbr, corners, hflat)


# ── FastBoard ───────────────────────────────────────────────────────────

class FastBoard:
    """Numpy-only Go board.  No GoStrings, no dict grid."""
    __slots__ = ('size', 'stones', '_hash', '_nbr', '_htbl',
                 '_corners', '_point_lut')

    def __init__(self, size):
        _ensure_tables(size)
        self.size = size
        self.stones = np.zeros(size * size, dtype=np.int8)
        self._hash = zobrist.EMPTY_BOARD
        self._nbr = _NBR[size]
        self._htbl = _HASH_FLAT[size]
        self._corners = _CORNERS[size]
        self._point_lut = _POINT_LUT[size]

    def copy(self):
        b = FastBoard.__new__(FastBoard)
        b.size = self.size
        b.stones = self.stones.copy()
        b._hash = self._hash
        b._nbr = self._nbr
        b._htbl = self._htbl
        b._corners = self._corners
        b._point_lut = self._point_lut
        return b

    @property
    def _stones(self):
        """2D view for encoder compatibility."""
        return self.stones.reshape(self.size, self.size)

    @property
    def num_rows(self):
        return self.size

    @property
    def num_cols(self):
        return self.size

    def zobrist_hash(self):
        return self._hash

    # ── stone placement + capture ────────────────────────────────────

    def place_stone(self, player_int, idx):
        """Place stone at flat index.  Handles captures in-place.

        Returns (num_captured, single_cap_idx).
        single_cap_idx is the flat index when exactly 1 stone captured,
        else -1.  Used for simple-ko short-circuit.
        """
        if _HAS_CBOARD:
            new_hash, total_captured, single_cap = _c_place_stone(
                self.stones, self._hash, player_int, idx)
            self._hash = new_hash
            return total_captured, single_cap

        stones = self.stones
        nbr = self._nbr
        htbl = self._htbl
        opp = 3 - player_int

        stones[idx] = player_int
        self._hash ^= htbl[idx][player_int]

        total_captured = 0
        single_cap = -1
        for n in nbr[idx]:
            if stones[n] == opp and not self._has_liberty(n):
                c = self._remove_group(n)
                total_captured += c
                if total_captured == 1 and c == 1:
                    single_cap = n
                else:
                    single_cap = -1

        return total_captured, single_cap

    def _has_liberty(self, idx):
        """Early-exit DFS using temporary stone negation as visited mark."""
        color = self.stones[idx]
        stones = self.stones
        nbr = self._nbr
        stack = [idx]
        touched = [idx]
        stones[idx] = -color
        found = False
        while stack and not found:
            p = stack.pop()
            for n in nbr[p]:
                s = stones[n]
                if s == 0:
                    found = True
                    break
                if s == color:
                    stones[n] = -color
                    touched.append(n)
                    stack.append(n)
        for p in touched:
            stones[p] = color
        return found

    def _remove_group(self, idx):
        """Remove dead group at idx in-place.  Returns count removed."""
        color = self.stones[idx]
        stones = self.stones
        nbr = self._nbr
        htbl = self._htbl
        count = 0
        stack = [idx]
        while stack:
            p = stack.pop()
            if stones[p] != color:
                continue
            stones[p] = 0
            self._hash ^= htbl[p][color]
            count += 1
            for n in nbr[p]:
                if stones[n] == color:
                    stack.append(n)
        return count

    # ── compatibility helpers ────────────────────────────────────────

    def get_player_on_point(self, point):
        idx = (point.row - 1) * self.size + (point.col - 1)
        v = self.stones[idx]
        if v == 0:
            return None
        return Player(v)


# ── FastGameState ───────────────────────────────────────────────────────

class FastGameState:
    """Lightweight game state wrapping FastBoard."""
    __slots__ = ('board', 'next_player', 'previous_state', 'last_move',
                 '_tracker', '_is_over')

    def __init__(self, board, next_player, previous_state, last_move,
                 tracker, is_over):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous_state
        self.last_move = last_move
        self._tracker = tracker
        self._is_over = is_over

    @classmethod
    def from_game_state(cls, gs):
        """Convert a regular GameState to FastGameState.

        The previous_state chain is shared (not deep-copied), so the
        encoder can still walk history frames through mixed types.
        """
        board = FastBoard(gs.board.num_rows)
        board.stones[:] = gs.board._stones.ravel()
        board._hash = gs.board.zobrist_hash()
        return cls(board, gs.next_player, gs.previous_state,
                   gs.last_move, gs._tracker, gs._is_over)

    def apply_move(self, move):
        if move.is_play:
            new_board = self.board.copy()
            idx = (move.point.row - 1) * self.board.size + (move.point.col - 1)
            new_board.place_stone(int(self.next_player), idx)
        else:
            new_board = self.board
        new_tracker = self._tracker.push(
            self.next_player, self.board.zobrist_hash())
        is_over = (move.is_resign
                   or (move.is_pass and self.last_move is not None
                       and self.last_move.is_pass))
        return FastGameState(new_board, self.next_player.other, self,
                             move, new_tracker, is_over)

    def is_over(self):
        return self._is_over

    def winner(self):
        if not self._is_over:
            return None
        if self.last_move is not None and self.last_move.is_resign:
            return self.next_player
        b, w = fast_area_score(self.board.stones, self.board.size,
                               self.board._nbr)
        return Player.black if b > w else Player.white


# ── Group info (for legal-move computation) ─────────────────────────────

def compute_groups(stones, nbr, size):
    """Compute group IDs and liberty counts from flat stones array.

    Returns (group_of, group_libs, group_members) where:
        group_of[idx] = group_id  (-1 for empty)
        group_libs[gid] = liberty count
        group_members[gid] = list of member flat indices
    """
    n = size * size
    group_of = [-1] * n
    group_libs = {}
    group_members = {}
    gid = 0

    for idx in range(n):
        color = stones[idx]
        if color == 0 or group_of[idx] >= 0:
            continue
        libs = set()
        members = []
        stack = [idx]
        while stack:
            p = stack.pop()
            if group_of[p] >= 0:
                continue
            group_of[p] = gid
            members.append(p)
            for nn in nbr[p]:
                sv = stones[nn]
                if sv == 0:
                    libs.add(nn)
                elif sv == color and group_of[nn] < 0:
                    stack.append(nn)
        group_libs[gid] = len(libs)
        group_members[gid] = members
        gid += 1

    return group_of, group_libs, group_members


# ── Fast area scoring ───────────────────────────────────────────────────

def fast_area_score(stones, size, nbr, komi=7.5):
    """Area scoring from flat int8 array.  No Point objects."""
    n = size * size
    black = 0
    white = 0
    visited = bytearray(n)

    for idx in range(n):
        v = stones[idx]
        if v == 1:
            black += 1
        elif v == 2:
            white += 1
        elif not visited[idx]:
            region_size = 0
            border = 0
            stack = [idx]
            while stack:
                p = stack.pop()
                if visited[p]:
                    continue
                pv = stones[p]
                if pv == 1:
                    border |= 1
                    continue
                if pv == 2:
                    border |= 2
                    continue
                visited[p] = 1
                region_size += 1
                for nn in nbr[p]:
                    if not visited[nn]:
                        stack.append(nn)
            if border == 1:
                black += region_size
            elif border == 2:
                white += region_size

    return black, white + komi
