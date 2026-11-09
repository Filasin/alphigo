"""Fast Gumbel AlphaZero agent using numpy-only FastBoard.

Drop-in replacement for GumbelZeroAgent.  Converts the root GameState
to FastGameState at search entry, then runs the entire MCTS tree on
FastBoard (no GoStrings, no dict grid, 81-byte board copies).

Override summary:
    _run_gumbel_search — converts root to FastGameState
    _expand_batch      — uses compute_groups() for legal-move check
                         instead of GoString.num_liberties
"""
import numpy as np

from alphigo import zobrist
from alphigo.agent.gumbel_zero_agent import GumbelZeroAgent
from alphigo.alphigo_board import Move
from alphigo.alphigo_types import Player
from alphigo.mcts.fast_state import (
    FastGameState, compute_groups, _CORNERS, _HASH_FLAT, _POINT_LUT,
    _HAS_CBOARD,
)
from alphigo.mcts.monte_carlo import MCTSNode

if _HAS_CBOARD:
    from alphigo.mcts._cboard import legal_moves as _c_legal_moves
    from alphigo.mcts.fast_state import (
        c_select_child_deterministic as _c_select,
        c_descend as _c_descend,
        c_child_stats as _c_child_stats,
    )

__all__ = ['FastGumbelZeroAgent']

_NODE_NEW = MCTSNode.__new__
_PASS_MOVE = Move.pass_turn()
_PLAY_MOVE = Move.play_move


class FastGumbelZeroAgent(GumbelZeroAgent):
    """GumbelZeroAgent with FastBoard + Cython tree ops."""

    _encode_buf = None

    def _run_gumbel_search(self, game_state, pre_encoded=None):
        if not isinstance(game_state, FastGameState):
            game_state = FastGameState.from_game_state(game_state)
        return super()._run_gumbel_search(game_state, pre_encoded=pre_encoded)

    if _HAS_CBOARD:
        def _select_child_deterministic(self, node):
            return _c_select(node.children, self._c_visit,
                             self._c_scale, node.raw_value)

        def _descend(self, root, root_child):
            return _c_descend(root, root_child,
                              self._c_visit, self._c_scale)

        def _child_stats(self, node):
            return _c_child_stats(node.children)

    # ── fast _expand_batch ───────────────────────────────────────────

    def _expand_batch(self, nodes, pre_encoded=None):
        """Expand leaf nodes using FastBoard group-info for legality."""
        n = len(nodes)
        encoder = self._encoder

        if pre_encoded is not None and n == 1:
            encoded_batch = pre_encoded[np.newaxis]
        else:
            shape = encoder.shape()
            buf = self._encode_buf
            if buf is None or buf.shape[0] < n:
                buf = np.zeros((max(n, 32), *shape), dtype=np.float32)
                self.__class__._encode_buf = buf
            batch = buf[:n]
            for i in range(n):
                encoder.encode(nodes[i].game_state, out=batch[i])
            encoded_batch = batch

        policies, net_values, ext_values, ownership_out = \
            self._predict_batch(encoded_batch)

        values = []
        num_points = encoder.num_points()
        value_mode = self._value_mode
        node_new = _NODE_NEW
        pass_move = _PASS_MOVE
        play_move = _PLAY_MOVE
        _empty_list = []

        for i in range(n):
            node = nodes[i]
            policy = policies[i]

            if value_mode == 'zero':
                value = 0.0
            elif value_mode == 'rollout':
                value = self._rollout_value(node.game_state)
            elif value_mode == 'external' and ext_values is not None:
                value = float(ext_values[i])
            elif value_mode == 'ownership' and ownership_out is not None:
                raw = float(np.mean(ownership_out[i]))
                value = raw if node.game_state.next_player == Player.black \
                    else -raw
            else:
                value = float(net_values[i])

            gs = node.game_state
            board = gs.board
            stones = board.stones
            size = board.size
            plut = board._point_lut
            player_int = int(gs.next_player)
            tracker = gs._tracker
            board_hash = board._hash

            if _HAS_CBOARD:
                legal_mask, legal_indices = _c_legal_moves(
                    stones, player_int, board_hash, size, tracker)
            else:
                nbr = board._nbr
                htbl = board._htbl
                corners = board._corners
                opp = 3 - player_int
                other_player = gs.next_player.other

                group_of, group_libs, group_members = compute_groups(
                    stones, nbr, size)

                empty_indices = np.flatnonzero(stones == 0)

                legal_mask = np.zeros(num_points, dtype=np.float32)
                legal_indices = []

                for idx in empty_indices:
                    idx = int(idx)
                    has_empty = False
                    has_enemy = False
                    can_capture = False
                    friendly_all_atari = True

                    for nn in nbr[idx]:
                        sv = stones[nn]
                        if sv == 0:
                            has_empty = True
                        elif sv == player_int:
                            gid = group_of[nn]
                            if group_libs[gid] > 1:
                                friendly_all_atari = False
                        else:
                            has_enemy = True
                            gid = group_of[nn]
                            if group_libs[gid] == 1:
                                can_capture = True

                    if not has_empty and not can_capture and friendly_all_atari:
                        continue

                    if can_capture:
                        new_hash = board_hash ^ htbl[idx][player_int]
                        seen_gids = set()
                        for nn in nbr[idx]:
                            if stones[nn] == opp:
                                gid = group_of[nn]
                                if group_libs[gid] == 1 and gid not in seen_gids:
                                    seen_gids.add(gid)
                                    for si in group_members[gid]:
                                        new_hash ^= htbl[si][opp]
                        if tracker.contains(other_player, new_hash):
                            continue

                    if not has_empty and not has_enemy:
                        on_grid, off_board = corners[idx]
                        fc = 0
                        for c in on_grid:
                            if stones[c] == player_int:
                                fc += 1
                        if off_board > 0:
                            if off_board + fc == 4:
                                continue
                        elif fc >= 3:
                            continue

                    legal_mask[idx] = 1.0
                    legal_indices.append(idx)

            pass_prior = float(policy[num_points]) \
                if len(policy) > num_points else 0.0

            masked = policy[:num_points] * legal_mask
            total = float(masked.sum()) + pass_prior
            if total > 0.0:
                masked /= total
                pass_prior /= total
            else:
                if legal_indices:
                    u = 1.0 / (len(legal_indices) + 1)
                    for idx in legal_indices:
                        masked[idx] = u
                    pass_prior = u

            n_legal = len(legal_indices)
            children = [None] * (n_legal + 1)
            for j in range(n_legal):
                idx = legal_indices[j]
                c = node_new(MCTSNode)
                c.game_state = None
                c.parent = node
                c.move = play_move(plut[idx])
                c.prior = float(masked[idx])
                c.children = []
                c._unvisited_moves = None
                c.win_counts = None
                c.num_rollouts = 0
                c.Q = 0.0
                c.N = 0
                c.W = 0.0
                c.raw_value = 0.0
                children[j] = c

            pc = node_new(MCTSNode)
            pc.game_state = None
            pc.parent = node
            pc.move = pass_move
            pc.prior = pass_prior
            pc.children = []
            pc._unvisited_moves = None
            pc.win_counts = None
            pc.num_rollouts = 0
            pc.Q = 0.0
            pc.N = 0
            pc.W = 0.0
            pc.raw_value = 0.0
            children[n_legal] = pc

            node.children = children
            node._unvisited_moves = _empty_list
            node.raw_value = value
            values.append(value)

        return values
