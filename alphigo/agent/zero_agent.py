"""AlphaZero-style agent.

ZeroAgent runs a PUCT-guided MCTS using a two-headed (policy + value)
network as the leaf evaluator. After ``num_simulations`` simulations,
visit counts at the root are converted to a policy via temperature and
a move is sampled (or argmaxed when ``temperature == 0``).

Originally adapted from the companion code of Pumperla and Ferguson,
*Deep Learning and the Game of Go* (2019),
https://github.com/maxpumperla/deep_learning_and_the_game_of_go,
and substantially modified since.
"""

import math

import numpy as np

from alphigo import alphigo_board
from alphigo import encoders
from alphigo import kerasutil
from alphigo import zobrist
from alphigo.agent.base import Agent
from alphigo.agent.helpers import is_point_an_eye
from alphigo.alphigo_board import _corner_tables
from alphigo.alphigo_types import Player
from alphigo.mcts.monte_carlo import MCTSNode


__all__ = ['ZeroAgent', 'load_zero_agent']


class ZeroAgent(Agent):
    """AlphaZero-style PUCT search + two-headed network."""

    def __init__(self, model, encoder, num_simulations=50, c_puct=1.5,
                 value_mode='network', rollout_depth=80,
                 external_value_model=None, batch_size=1):
        Agent.__init__(self)
        # Sanity check: encoder feature shape must match the model input.
        # `model.input_shape` is (None, H, W, planes) for a single-input model.
        in_shape = tuple(model.input_shape[1:])
        if in_shape != tuple(encoder.shape()):
            raise ValueError(
                f"encoder/model mismatch: encoder produces shape "
                f"{tuple(encoder.shape())} but model expects {in_shape}. "
                f"Use a matching encoder+network combination.")
        if value_mode not in ('network', 'zero', 'rollout', 'external', 'ownership'):
            raise ValueError(
                f"value_mode must be 'network'|'zero'|'rollout'|'external'|'ownership'"
                f", got {value_mode!r}")
        if value_mode == 'external' and external_value_model is None:
            raise ValueError(
                "value_mode='external' requires external_value_model")
        self._model = model
        self._encoder = encoder
        self._num_simulations = int(num_simulations)
        self._c_puct = float(c_puct)
        self._value_mode = value_mode
        self._rollout_depth = int(rollout_depth)
        self._external_value_model = external_value_model
        self._temperature = 1.0
        self._collector = None
        # Compiled batch-1 inference graph (lazy: built on first _predict
        # call so the model is fully constructed when tf.function traces).
        # Eliminates per-call Python op dispatch overhead in eager mode.
        self._predict_fn = None
        self._value_predict_fn = None
        self._batch_size = max(1, int(batch_size))

    # ── exploration / sampling control ────────────────────────────────

    def set_temperature(self, temperature):
        """Set softmax/sampling temperature on the visit-count distribution.

        T=1.0 samples proportional to visit counts; T=0.0 picks argmax.
        """
        self._temperature = float(temperature)

    def set_collector(self, collector):
        self._collector = collector

    # ── inference ─────────────────────────────────────────────────────

    def _predict(self, game_state):
        encoded = self._encoder.encode(game_state)
        x = np.array([encoded], dtype=np.float32)
        if self._predict_fn is None:
            import tensorflow as tf
            input_sig = [tf.TensorSpec(shape=(None,) + tuple(self._encoder.shape()),
                                       dtype=tf.float32)]
            model = self._model
            @tf.function(input_signature=input_sig, reduce_retracing=True)
            def _traced_predict(inp):
                return model(inp, training=False)
            self._predict_fn = _traced_predict
        policy, value = self._predict_fn(x)
        return policy.numpy()[0], float(value.numpy()[0, 0])

    def _predict_external_value(self, game_state):
        encoded = self._encoder.encode(game_state)
        x = np.array([encoded], dtype=np.float32)
        if self._value_predict_fn is None:
            import tensorflow as tf
            input_sig = [tf.TensorSpec(
                shape=(None,) + tuple(self._encoder.shape()), dtype=tf.float32)]
            vmodel = self._external_value_model
            @tf.function(input_signature=input_sig, reduce_retracing=True)
            def _traced(inp):
                return vmodel(inp, training=False)
            self._value_predict_fn = _traced
        out = self._value_predict_fn(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return float(out.numpy()[0, 0])

    def _predict_batch(self, encoded_batch):
        """Batch inference: (B, H, W, C) -> (policies, net_values, ext_values).

        Returns:
            policies:   (B, num_points) numpy float32
            net_values: (B,) numpy float32
            ext_values: (B,) numpy float32, or None if no external model
        """
        x = np.asarray(encoded_batch, dtype=np.float32)
        if self._predict_fn is None:
            import tensorflow as tf
            input_sig = [tf.TensorSpec(
                shape=(None,) + tuple(self._encoder.shape()),
                dtype=tf.float32)]
            model = self._model
            @tf.function(input_signature=input_sig, reduce_retracing=True)
            def _traced_predict(inp):
                return model(inp, training=False)
            self._predict_fn = _traced_predict
        outputs = self._predict_fn(x)
        policy_out, value_out = outputs[0], outputs[1]
        policies = policy_out.numpy()
        net_values = value_out.numpy()[:, 0]
        n_out = len(outputs)
        if n_out == 4:
            ownership_out = outputs[3].numpy()
        elif n_out == 3:
            ownership_out = outputs[2].numpy()
        else:
            ownership_out = None

        ext_values = None
        if self._value_mode == 'external' and self._external_value_model is not None:
            if self._value_predict_fn is None:
                import tensorflow as tf
                input_sig = [tf.TensorSpec(
                    shape=(None,) + tuple(self._encoder.shape()),
                    dtype=tf.float32)]
                vmodel = self._external_value_model
                @tf.function(input_signature=input_sig, reduce_retracing=True)
                def _traced_value(inp):
                    return vmodel(inp, training=False)
                self._value_predict_fn = _traced_value
            vout = self._value_predict_fn(x)
            if isinstance(vout, (list, tuple)):
                vout = vout[0]
            ext_values = vout.numpy()[:, 0]

        return policies, net_values, ext_values, ownership_out

    def _rollout_value(self, game_state):
        """Random rollout from `game_state` to game end (or depth cap).

        Returns the outcome (+1 / 0 / -1) from the perspective of
        ``game_state.next_player``. Picks uniformly among legal non-eye
        moves; passes when no such move exists.
        """
        from alphigo.alphigo_board import GameState as _GS  # noqa: F401 (typecheck)
        rng = np.random
        state = game_state
        start_player = state.next_player
        for _ in range(self._rollout_depth):
            if state.is_over():
                break
            legal_points = []
            for idx in range(self._encoder.num_points()):
                point = self._encoder.decode_point_index(idx)
                move = alphigo_board.Move.play_move(point)
                if not state.is_valid_move(move):
                    continue
                if is_point_an_eye(state.board, point, state.next_player):
                    continue
                legal_points.append(point)
            if not legal_points:
                state = state.apply_move(alphigo_board.Move.pass_turn())
                continue
            pt = legal_points[rng.randint(len(legal_points))]
            state = state.apply_move(alphigo_board.Move.play_move(pt))
        # If we hit depth cap without is_over, use area scoring anyway.
        from alphigo.alphigo_board import area_score
        black, white = area_score(state.board)
        margin = black - white  # positive = black ahead
        if margin > 0:
            winner = Player.black
        elif margin < 0:
            winner = Player.white
        else:
            return 0.0
        return 1.0 if winner == start_player else -1.0

    def _expand(self, node, pre_encoded=None):
        """Expand a single node. Delegates to _expand_batch."""
        return self._expand_batch([node], pre_encoded=pre_encoded)[0]

    def _expand_batch(self, nodes, pre_encoded=None):
        """Expand multiple leaf nodes in a single batched inference call.

        Returns list of value estimates (one per node, from each node's
        next_player perspective).
        """
        if pre_encoded is not None and len(nodes) == 1:
            encoded_batch = pre_encoded[np.newaxis]
        else:
            encoded_batch = np.stack([
                self._encoder.encode(node.game_state) for node in nodes
            ])
        policies, net_values, ext_values, ownership_out = self._predict_batch(encoded_batch)

        values = []
        num_points = self._encoder.num_points()

        for i, node in enumerate(nodes):
            policy = policies[i]
            if self._value_mode == 'zero':
                value = 0.0
            elif self._value_mode == 'rollout':
                value = self._rollout_value(node.game_state)
            elif self._value_mode == 'external' and ext_values is not None:
                value = float(ext_values[i])
            elif self._value_mode == 'ownership' and ownership_out is not None:
                raw = float(np.mean(ownership_out[i]))
                from alphigo.alphigo_types import Player
                value = raw if node.game_state.next_player == Player.black else -raw
            else:
                value = float(net_values[i])

            legal_mask = np.zeros(num_points, dtype=np.float32)
            legal_moves_with_idx = []
            gs = node.game_state
            board = gs.board
            grid = board._grid
            player = gs.next_player
            neighbor_table = board._neighbor_table
            corner_table = _corner_tables[(board.num_rows, board.num_cols)]
            tracker = gs._tracker
            other_player = player.other
            grid_get = grid.get
            board_hash = board._hash
            hash_code = zobrist.HASH_CODE

            empty_indices = np.flatnonzero(board._stones == 0)
            _decode = self._encoder.decode_point_index
            for idx in empty_indices:
                point = _decode(int(idx))

                has_empty = False
                has_enemy = False
                can_capture = False
                friendly_all_atari = True

                for neighbor in neighbor_table[point]:
                    ns = grid_get(neighbor)
                    if ns is None:
                        has_empty = True
                    elif ns.color == player:
                        if ns.num_liberties > 1:
                            friendly_all_atari = False
                    else:
                        has_enemy = True
                        if ns.num_liberties == 1:
                            can_capture = True

                if not has_empty and not can_capture and friendly_all_atari:
                    continue

                if can_capture:
                    new_hash = board_hash ^ hash_code[point, player]
                    seen = set()
                    for neighbor in neighbor_table[point]:
                        ns = grid_get(neighbor)
                        if ns is not None and ns.color != player \
                                and ns.num_liberties == 1:
                            ns_id = id(ns)
                            if ns_id not in seen:
                                seen.add(ns_id)
                                for stone in ns.stones:
                                    new_hash ^= hash_code[stone, other_player]
                    if tracker.contains(other_player, new_hash):
                        continue

                if not has_empty and not has_enemy:
                    on_grid, off_board = corner_table[point]
                    friendly_corners = 0
                    for c in on_grid:
                        cs = grid_get(c)
                        if cs is not None and cs.color == player:
                            friendly_corners += 1
                    if off_board > 0:
                        if off_board + friendly_corners == 4:
                            continue
                    elif friendly_corners >= 3:
                        continue

                legal_mask[idx] = 1.0
                legal_moves_with_idx.append((idx, point))

            pass_prior = float(policy[num_points]) if len(policy) > num_points else 0.0

            masked = policy[:num_points] * legal_mask
            total = float(masked.sum()) + pass_prior
            if total > 0.0:
                masked /= total
                pass_prior /= total
            else:
                if legal_moves_with_idx:
                    u = 1.0 / (len(legal_moves_with_idx) + 1)
                    for idx, _pt in legal_moves_with_idx:
                        masked[idx] = u
                    pass_prior = u

            for idx, pt in legal_moves_with_idx:
                child = MCTSNode(None, parent=node,
                                 move=alphigo_board.Move.play_move(pt))
                child.prior = float(masked[idx])
                node.children.append(child)

            pass_child = MCTSNode(None, parent=node,
                                  move=alphigo_board.Move.pass_turn())
            pass_child.prior = pass_prior
            node.children.append(pass_child)

            node.unvisited_moves = []
            node.raw_value = value
            values.append(value)

        return values

    def _select_child_puct(self, node):
        """Select the child maximizing PUCT from the perspective of the
        player to move at ``node``.

        Convention: each node's Q is stored from its *own* next_player's
        perspective (see _backup). A child's next_player is the opponent
        of the player choosing here, so the child's Q must be negated.
        """
        c_puct = self._c_puct
        sqrt_total = c_puct * math.sqrt(node.N)
        best = None
        best_score = -1e18
        for c in node.children:
            score = -c.Q + sqrt_total * c.prior / (1 + c.N)
            if score > best_score:
                best_score = score
                best = c
        return best

    def _select_leaf(self, root):
        """PUCT descent from root to an unexpanded or terminal leaf.

        Returns (node, path) where path includes root.
        Children use lazy game_state: materialized on first selection.
        """
        node = root
        path = [node]
        while node.children:
            node = self._select_child_puct(node)
            if node.game_state is None:
                node.game_state = node.parent.game_state.apply_move(node.move)
            path.append(node)
            if node.game_state.is_over():
                break
        return node, path

    def _apply_virtual_loss(self, path):
        """Discourage re-selection of an in-flight path.

        Q is stored from each node's own next_player perspective and
        selection maximizes -child.Q, so making a node *less* attractive
        to its parent means *raising* its own-perspective W.
        """
        for n in path:
            n.N += 1
            n.W += 1.0
            n.Q = n.W / n.N

    def _revert_virtual_loss(self, path):
        for n in path:
            n.N -= 1
            n.W -= 1.0
            n.Q = n.W / n.N if n.N > 0 else 0.0

    def _backup(self, path, leaf_value, leaf_to_play):
        """Backup `leaf_value` (from leaf's next_player view) up the path."""
        for n in path:
            n.N += 1
            v = leaf_value if n.game_state.next_player == leaf_to_play \
                else -leaf_value
            n.W += v
            n.Q = n.W / n.N

    def _run_search(self, game_state):
        root = MCTSNode(game_state)
        if not root.is_terminal():
            root_value = self._expand(root)
            root.N = 1
            root.W = root_value
            root.Q = root_value
        else:
            return root

        sims_done = 0
        while sims_done < self._num_simulations:
            actual_batch = min(self._batch_size,
                               self._num_simulations - sims_done)
            leaves_to_expand = []
            terminal_backups = []

            for _ in range(actual_batch):
                leaf, path = self._select_leaf(root)

                if leaf.is_terminal():
                    winner = leaf.game_state.winner()
                    if winner is None:
                        lv = 0.0
                    else:
                        lv = 1.0 if winner == leaf.game_state.next_player \
                            else -1.0
                    terminal_backups.append(
                        (path, lv, leaf.game_state.next_player))
                elif leaf.children:
                    # Collision: leaf was already expanded by a prior
                    # selection in this batch. Treat as terminal using
                    # its current Q.
                    terminal_backups.append(
                        (path, leaf.Q, leaf.game_state.next_player))
                else:
                    self._apply_virtual_loss(path)
                    leaves_to_expand.append((leaf, path))

                sims_done += 1

            if leaves_to_expand:
                nodes = [leaf for leaf, _ in leaves_to_expand]
                values = self._expand_batch(nodes)
                for (leaf, path), value in zip(leaves_to_expand, values):
                    self._revert_virtual_loss(path)
                    self._backup(path, value, leaf.game_state.next_player)

            for path, value, player in terminal_backups:
                self._backup(path, value, player)

        return root

    # ── public search API (used by smoke test + collectors) ───────────

    def search(self, game_state):
        """Run search; return dict(move, policy_target, value).

        ``policy_target`` is a (num_points,) array — visit-count
        distribution after temperature is applied (sums to 1).
        ``value`` is the root's expected value from the current
        next_player's perspective.
        """
        root = self._run_search(game_state)
        self._last_root = root  # cache for GUI display (no extra search)
        policy_target = self._policy_target_from_root(root)
        move = self._sample_move(root, policy_target, game_state)
        return {
            'move': move,
            'policy_target': policy_target,
            'value': float(root.Q),
        }

    def forward_heads(self, game_state):
        """One forward pass at ``game_state``; return the auxiliary heads.

        Returns ``(ownership, score_estimate)`` from the perspective of the
        player to move, matching the self-play targets (which flip both for
        White).  ``ownership`` is a flat board_size**2 array in [-1, 1];
        ``score_estimate`` is the expected final margin in points, taken as
        the mean of the score-distribution head.  Either is ``None`` when
        the network does not have that head.
        """
        outputs = self._forward_raw(game_state)
        if outputs is None:
            return None, None

        ownership = None
        score_estimate = None

        # Head layouts produced by data/networks.py:
        #   2 heads -> policy, value
        #   3 heads -> policy, value, ownership
        #   4 heads -> policy, value, score_dist, ownership
        if len(outputs) == 3:
            ownership = np.asarray(outputs[2][0], dtype=np.float32)
        elif len(outputs) == 4:
            score_dist = np.asarray(outputs[2][0], dtype=np.float32)
            ownership = np.asarray(outputs[3][0], dtype=np.float32)
            # margin_to_scoredist() bins margins over [-30, +30], one bin
            # per point, so bin i corresponds to a margin of i + score_min.
            score_min = -(len(score_dist) // 2)
            bins = np.arange(len(score_dist), dtype=np.float32) + score_min
            total = float(score_dist.sum())
            if total > 0:
                score_estimate = float((score_dist * bins).sum() / total)

        return ownership, score_estimate

    def _forward_raw(self, game_state):
        """Raw network outputs for one position, as a list of numpy arrays."""
        x = np.array([self._encoder.encode(game_state)], dtype=np.float32)
        if self._predict_fn is None:
            import tensorflow as tf
            input_sig = [tf.TensorSpec(
                shape=(None,) + tuple(self._encoder.shape()), dtype=tf.float32)]
            model = self._model
            @tf.function(input_signature=input_sig, reduce_retracing=True)
            def _traced_predict(inp):
                return model(inp, training=False)
            self._predict_fn = _traced_predict
        outputs = self._predict_fn(x)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        return [o.numpy() for o in outputs]

    def analyze(self, game_state, k=5):
        """Run search; return AnalysisResult with visit-fraction top-K + WR."""
        from alphigo.agent.analysis import AnalysisResult
        root = self._run_search(game_state)
        if not root.children:
            return AnalysisResult()
        total_visits = sum(c.N for c in root.children) or 1
        ranked = sorted(root.children, key=lambda c: c.N, reverse=True)
        top = []
        move_winrates = []
        for child in ranked[:k]:
            mv = child.move
            if mv.is_pass or mv.is_resign or mv.point is None:
                continue
            top.append((mv.point, float(child.N) / total_visits * 100.0))
            # A child's Q is from the perspective of the player to move *at
            # the child*, i.e. the opponent - flip it so every win rate on
            # screen refers to the same side as position_winrate.
            child_wr = 1.0 - (float(child.Q) + 1.0) / 2.0
            move_winrates.append(max(0.0, min(1.0, child_wr)))
        winrate = (float(root.Q) + 1.0) / 2.0
        winrate = max(0.0, min(1.0, winrate))

        try:
            ownership, score_estimate = self.forward_heads(game_state)
        except Exception:
            ownership, score_estimate = None, None

        return AnalysisResult(top_moves=top, position_winrate=winrate,
                              pass_score=None, move_winrates=move_winrates,
                              ownership=ownership,
                              score_estimate=score_estimate,
                              perspective=game_state.next_player)

    def select_move(self, game_state):
        result = self.search(game_state)
        if self._collector is not None:
            encoded = self._encoder.encode(game_state)
            self._collector.record_decision(
                state=encoded,
                action=result['policy_target'],
                estimated_value=result['value'],
            )
        return result['move']

    # ── internals: visit-count → policy → move ────────────────────────

    def _policy_target_from_root(self, root):
        num_points = self._encoder.num_points()
        num_actions = num_points + 1
        target = np.zeros(num_actions, dtype=np.float32)
        if not root.children:
            return target
        for child in root.children:
            move = child.move
            if move.is_resign:
                continue
            if move.is_pass:
                target[num_points] = child.N
            elif move.point is not None:
                idx = self._encoder.encode_point(move.point)
                target[idx] = child.N

        if target.sum() == 0.0:
            return target

        if self._temperature == 0.0:
            best = int(np.argmax(target))
            target[:] = 0.0
            target[best] = 1.0
            return target

        powered = np.power(target, 1.0 / self._temperature)
        s = powered.sum()
        return powered / s if s > 0 else target

    def _sample_move(self, root, policy_target, game_state):
        if policy_target.sum() == 0.0:
            return alphigo_board.Move.pass_turn()

        num_points = self._encoder.num_points()

        if self._temperature == 0.0:
            idx = int(np.argmax(policy_target))
        else:
            idx = int(np.random.choice(len(policy_target), p=policy_target))

        if idx >= num_points:
            return alphigo_board.Move.pass_turn()

        point = self._encoder.decode_point_index(idx)
        move = alphigo_board.Move.play_move(point)
        if game_state.is_valid_move(move):
            return move
        return alphigo_board.Move.pass_turn()

    # ── serialization ─────────────────────────────────────────────────

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])
        h5file.create_group('agent')
        h5file['agent'].attrs['num_simulations'] = self._num_simulations
        h5file['agent'].attrs['c_puct'] = self._c_puct


def load_zero_agent(h5file, num_simulations=None, c_puct=None,
                    value_mode='network', rollout_depth=80,
                    external_value_model=None, batch_size=1):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = int(h5file['encoder'].attrs['board_width'])
    board_height = int(h5file['encoder'].attrs['board_height'])
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))

    if 'agent' in h5file:
        n_sim = int(h5file['agent'].attrs.get('num_simulations', 50))
        c = float(h5file['agent'].attrs.get('c_puct', 1.5))
    else:
        n_sim, c = 50, 1.5
    if num_simulations is not None:
        n_sim = int(num_simulations)
    if c_puct is not None:
        c = float(c_puct)
    return ZeroAgent(model, encoder, num_simulations=n_sim, c_puct=c,
                     value_mode=value_mode, rollout_depth=rollout_depth,
                     external_value_model=external_value_model,
                     batch_size=batch_size)
