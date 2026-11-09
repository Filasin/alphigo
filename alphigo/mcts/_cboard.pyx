# cython: boundscheck=False, wraparound=False, cdivision=True
"""C-level Go board + MCTS tree operations.

Board ops operate on flat int8 arrays with precomputed neighbor tables.
Tree ops (select_child, descend, backup) cache MCTSNode attributes into
C arrays so all arithmetic runs without Python object dispatch.
"""
from libc.string cimport memcpy, memset
from libc.math cimport log, exp

import numpy as np
cimport numpy as np

DEF MAX_N = 361        # 19*19
DEF MAX_NBR = 4
DEF MAX_GROUPS = 180   # upper bound on groups

# ── Precomputed tables (filled once per board size) ──────────────────

cdef int _nbr[MAX_N][MAX_NBR]
cdef int _nbr_count[MAX_N]
cdef int _corner_on[MAX_N][MAX_NBR]
cdef int _corner_on_count[MAX_N]
cdef int _corner_off[MAX_N]
cdef unsigned long long _htbl[MAX_N][3]   # [idx][color], color 1=B 2=W
cdef int _board_size = 0

cdef void _init_tables(int size,
                       list nbr_py,
                       list corners_py,
                       list htbl_py) noexcept:
    global _board_size
    cdef int n = size * size
    cdef int idx, k
    _board_size = size
    for idx in range(n):
        nb = nbr_py[idx]
        _nbr_count[idx] = len(nb)
        for k in range(len(nb)):
            _nbr[idx][k] = nb[k]

        on_grid, off_board = corners_py[idx]
        _corner_on_count[idx] = len(on_grid)
        _corner_off[idx] = off_board
        for k in range(len(on_grid)):
            _corner_on[idx][k] = on_grid[k]

        hdict = htbl_py[idx]
        _htbl[idx][0] = 0
        _htbl[idx][1] = <unsigned long long>hdict[1]
        _htbl[idx][2] = <unsigned long long>hdict[2]


def init_tables(int size, list nbr_py, list corners_py, list htbl_py):
    """Python entry point to initialise C tables."""
    _init_tables(size, nbr_py, corners_py, htbl_py)


# ── has_liberty ──────────────────────────────────────────────────────

cdef bint c_has_liberty(signed char* stones, int idx) noexcept nogil:
    cdef int stack[MAX_N]
    cdef int touched[MAX_N]
    cdef int sp = 0, tp = 0
    cdef signed char color = stones[idx]
    cdef int p, n, k
    cdef signed char s
    cdef bint found = 0

    stones[idx] = -color
    stack[sp] = idx; sp += 1
    touched[tp] = idx; tp += 1

    while sp > 0 and not found:
        sp -= 1
        p = stack[sp]
        for k in range(_nbr_count[p]):
            n = _nbr[p][k]
            s = stones[n]
            if s == 0:
                found = 1
                break
            if s == color:
                stones[n] = -color
                touched[tp] = n; tp += 1
                stack[sp] = n; sp += 1

    for k in range(tp):
        stones[touched[k]] = color
    return found


# ── remove_group ─────────────────────────────────────────────────────

cdef int c_remove_group(signed char* stones,
                        unsigned long long* board_hash,
                        int idx) noexcept nogil:
    cdef int stack[MAX_N]
    cdef int sp = 0
    cdef signed char color = stones[idx]
    cdef int p, n, k, count = 0

    stack[sp] = idx; sp += 1
    while sp > 0:
        sp -= 1
        p = stack[sp]
        if stones[p] != color:
            continue
        stones[p] = 0
        board_hash[0] ^= _htbl[p][<int>color]
        count += 1
        for k in range(_nbr_count[p]):
            n = _nbr[p][k]
            if stones[n] == color:
                stack[sp] = n; sp += 1
    return count


# ── place_stone ──────────────────────────────────────────────────────

def place_stone(np.ndarray[np.int8_t, ndim=1] stones_arr,
                unsigned long long board_hash,
                int player_int, int idx):
    """Place stone, handle captures.  Returns (new_hash, num_captured, single_cap)."""
    cdef signed char* stones = <signed char*>stones_arr.data
    cdef signed char opp = 3 - player_int
    cdef int n, k
    cdef int total_captured = 0
    cdef int single_cap = -1
    cdef int c
    cdef unsigned long long h = board_hash

    stones[idx] = <signed char>player_int
    h ^= _htbl[idx][player_int]

    for k in range(_nbr_count[idx]):
        n = _nbr[idx][k]
        if stones[n] == opp and not c_has_liberty(stones, n):
            c = c_remove_group(stones, &h, n)
            total_captured += c
            if total_captured == 1 and c == 1:
                single_cap = n
            else:
                single_cap = -1

    return h, total_captured, single_cap


# ── compute_groups ───────────────────────────────────────────────────

cdef struct GroupInfo:
    int group_of[MAX_N]
    int group_libs[MAX_GROUPS]
    int group_members[MAX_GROUPS][MAX_N]
    int group_size[MAX_GROUPS]
    int num_groups

cdef GroupInfo _gi

cdef void c_compute_groups(signed char* stones, int size) noexcept nogil:
    cdef int n = size * size
    cdef int idx, p, nn, k, gid = 0
    cdef signed char color, sv
    cdef int stack[MAX_N]
    cdef int sp
    cdef int lib_set[MAX_N]
    cdef int lib_count

    for idx in range(n):
        _gi.group_of[idx] = -1

    for idx in range(n):
        color = stones[idx]
        if color == 0 or _gi.group_of[idx] >= 0:
            continue

        lib_count = 0
        memset(lib_set, 0, n * sizeof(int))
        _gi.group_size[gid] = 0
        sp = 0
        stack[sp] = idx; sp += 1

        while sp > 0:
            sp -= 1
            p = stack[sp]
            if _gi.group_of[p] >= 0:
                continue
            _gi.group_of[p] = gid
            _gi.group_members[gid][_gi.group_size[gid]] = p
            _gi.group_size[gid] += 1
            for k in range(_nbr_count[p]):
                nn = _nbr[p][k]
                sv = stones[nn]
                if sv == 0:
                    if lib_set[nn] == 0:
                        lib_set[nn] = 1
                        lib_count += 1
                elif sv == color and _gi.group_of[nn] < 0:
                    stack[sp] = nn; sp += 1

        _gi.group_libs[gid] = lib_count
        gid += 1

    _gi.num_groups = gid


# ── legal_moves: combined group computation + legal move mask ────────

def legal_moves(np.ndarray[np.int8_t, ndim=1] stones_arr,
                int player_int,
                unsigned long long board_hash,
                int size,
                object tracker):
    """Compute legal move mask and indices.

    Returns (legal_mask, legal_indices) where:
        legal_mask: float32 array of shape (size*size,), 1.0 for legal
        legal_indices: list of legal flat indices
    """
    cdef signed char* stones = <signed char*>stones_arr.data
    cdef int n = size * size
    cdef signed char opp = 3 - player_int
    cdef int idx, k, nn, gid, si, c_idx
    cdef signed char sv
    cdef bint has_empty, has_enemy, can_capture, friendly_all_atari
    cdef unsigned long long new_hash
    cdef int fc
    cdef int seen[MAX_GROUPS]

    c_compute_groups(stones, size)

    cdef np.ndarray[np.float32_t, ndim=1] mask = np.zeros(n, dtype=np.float32)
    legal_idx_list = []

    for idx in range(n):
        if stones[idx] != 0:
            continue

        has_empty = 0
        has_enemy = 0
        can_capture = 0
        friendly_all_atari = 1

        for k in range(_nbr_count[idx]):
            nn = _nbr[idx][k]
            sv = stones[nn]
            if sv == 0:
                has_empty = 1
            elif sv == player_int:
                gid = _gi.group_of[nn]
                if _gi.group_libs[gid] > 1:
                    friendly_all_atari = 0
            else:
                has_enemy = 1
                gid = _gi.group_of[nn]
                if _gi.group_libs[gid] == 1:
                    can_capture = 1

        if not has_empty and not can_capture and friendly_all_atari:
            continue

        # superko check for capturing moves
        if can_capture:
            new_hash = board_hash ^ _htbl[idx][player_int]
            memset(seen, 0, _gi.num_groups * sizeof(int))
            for k in range(_nbr_count[idx]):
                nn = _nbr[idx][k]
                if stones[nn] == opp:
                    gid = _gi.group_of[nn]
                    if _gi.group_libs[gid] == 1 and seen[gid] == 0:
                        seen[gid] = 1
                        for si in range(_gi.group_size[gid]):
                            c_idx = _gi.group_members[gid][si]
                            new_hash ^= _htbl[c_idx][<int>opp]
            # tracker.contains is still Python — only called for captures
            other_player_int = 3 - player_int
            from alphigo.alphigo_types import Player
            other_player = Player(other_player_int)
            if tracker.contains(other_player, int(new_hash)):
                continue

        # eye-fill check (avoid filling own eyes)
        if not has_empty and not has_enemy:
            fc = 0
            for k in range(_corner_on_count[idx]):
                if stones[_corner_on[idx][k]] == player_int:
                    fc += 1
            if _corner_off[idx] > 0:
                if _corner_off[idx] + fc == 4:
                    continue
            elif fc >= 3:
                continue

        mask[idx] = 1.0
        legal_idx_list.append(idx)

    return mask, legal_idx_list


# ══════════════════════════════════════════════════════════════════════
#  MCTS tree operations — eliminate numpy overhead in the hot loops
# ══════════════════════════════════════════════════════════════════════

DEF MAX_KIDS = 82  # board_size^2 + 1 (pass)


def c_select_child_deterministic(list children, double c_visit,
                                 double c_scale, double node_raw_value):
    """Pure-C replacement for GumbelZeroAgent._select_child_deterministic.

    Same logic, but all arithmetic in typed C instead of numpy.
    """
    cdef int n = len(children)
    if n == 0:
        return None
    if n == 1:
        return children[0]

    cdef double q[MAX_KIDS]
    cdef double nvis[MAX_KIDS]
    cdef double pri[MAX_KIDS]
    cdef double scores_raw[MAX_KIDS]
    cdef int i
    cdef double sum_n = 0.0, max_n = 0.0, pi_vis = 0.0, wpq = 0.0
    cdef double v_mix, sigma, scores_max, exp_sum, exp_val, score
    cdef int best_idx = 0
    cdef double best_score

    for i in range(n):
        child = children[i]
        q[i] = -(<double>child.Q)
        nvis[i] = <double>child.N
        pri[i] = <double>child.prior

    for i in range(n):
        sum_n += nvis[i]
        if nvis[i] > max_n:
            max_n = nvis[i]
        if nvis[i] > 0:
            pi_vis += pri[i]
            wpq += pri[i] * q[i]

    if pi_vis > 0.0 and sum_n > 0.0:
        v_mix = (node_raw_value + sum_n * (wpq / pi_vis)) / (1.0 + sum_n)
    else:
        v_mix = node_raw_value

    sigma = (c_visit + max_n) * c_scale

    scores_max = -1e30
    for i in range(n):
        if nvis[i] > 0:
            scores_raw[i] = log(max(pri[i], 1e-12)) + sigma * q[i]
        else:
            scores_raw[i] = log(max(pri[i], 1e-12)) + sigma * v_mix
        if scores_raw[i] > scores_max:
            scores_max = scores_raw[i]

    exp_sum = 0.0
    for i in range(n):
        scores_raw[i] = exp(scores_raw[i] - scores_max)
        exp_sum += scores_raw[i]

    best_score = -1e30
    for i in range(n):
        score = (scores_raw[i] / exp_sum) - nvis[i] / (1.0 + sum_n)
        if score > best_score:
            best_score = score
            best_idx = i

    return children[best_idx]


def c_backup(list path, double leaf_value, int leaf_to_play):
    """Pure-C replacement for ZeroAgent._backup.

    Updates N, W, Q on each node in the path in-place.
    """
    cdef int i, n_count
    cdef double v, w
    for i in range(len(path)):
        node = path[i]
        n_count = node.N + 1
        if (<int>node.game_state.next_player) == leaf_to_play:
            v = leaf_value
        else:
            v = -leaf_value
        w = node.W + v
        node.N = n_count
        node.W = w
        node.Q = w / <double>n_count


def c_descend(object root, object root_child,
              double c_visit, double c_scale):
    """Descend from root_child to a leaf, applying moves lazily.

    Returns (leaf, path_list).  Inlines select_child_deterministic
    to avoid per-call Python function dispatch.
    """
    node = root_child
    if node.game_state is None:
        node.game_state = root.game_state.apply_move(node.move)
    path = [root, node]

    cdef int n, i, best_idx
    cdef double q[MAX_KIDS]
    cdef double nvis[MAX_KIDS]
    cdef double pri[MAX_KIDS]
    cdef double scores_raw[MAX_KIDS]
    cdef double sum_n, max_n, pi_vis, wpq, v_mix, sigma
    cdef double scores_max, exp_sum, score, best_score

    while node.children:
        # ── inlined select_child_deterministic ──
        children = node.children
        n = len(children)
        if n == 1:
            node = children[0]
        else:
            sum_n = 0.0; max_n = 0.0; pi_vis = 0.0; wpq = 0.0
            for i in range(n):
                child = children[i]
                q[i] = -(<double>child.Q)
                nvis[i] = <double>child.N
                pri[i] = <double>child.prior
                sum_n += nvis[i]
                if nvis[i] > max_n:
                    max_n = nvis[i]
                if nvis[i] > 0:
                    pi_vis += pri[i]
                    wpq += pri[i] * q[i]

            if pi_vis > 0.0 and sum_n > 0.0:
                v_mix = ((<double>node.raw_value) + sum_n * (wpq / pi_vis)) / (1.0 + sum_n)
            else:
                v_mix = <double>node.raw_value

            sigma = (c_visit + max_n) * c_scale
            scores_max = -1e30
            for i in range(n):
                if nvis[i] > 0:
                    scores_raw[i] = log(max(pri[i], 1e-12)) + sigma * q[i]
                else:
                    scores_raw[i] = log(max(pri[i], 1e-12)) + sigma * v_mix
                if scores_raw[i] > scores_max:
                    scores_max = scores_raw[i]

            exp_sum = 0.0
            for i in range(n):
                scores_raw[i] = exp(scores_raw[i] - scores_max)
                exp_sum += scores_raw[i]

            best_score = -1e30; best_idx = 0
            for i in range(n):
                score = (scores_raw[i] / exp_sum) - nvis[i] / (1.0 + sum_n)
                if score > best_score:
                    best_score = score
                    best_idx = i
            node = children[best_idx]
        # ── end inlined select ──

        if node.game_state is None:
            node.game_state = node.parent.game_state.apply_move(node.move)
        path.append(node)
        if node.game_state.is_over():
            break

    return node, path


def c_child_stats(list children):
    """Build (q, nvis, pri) arrays from MCTSNode children — typed loop."""
    cdef int n = len(children)
    cdef np.ndarray[np.float64_t, ndim=1] q_arr = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] nvis_arr = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] pri_arr = np.empty(n, dtype=np.float64)
    cdef int i
    for i in range(n):
        child = children[i]
        q_arr[i] = -(<double>child.Q)
        nvis_arr[i] = <double>child.N
        pri_arr[i] = <double>child.prior
    return q_arr, nvis_arr, pri_arr
