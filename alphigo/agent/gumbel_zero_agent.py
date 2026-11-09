"""Gumbel AlphaZero agent (Danihelka et al., ICLR 2022).

Replaces PUCT root selection with Gumbel top-m sampling + sequential
halving, and interior selection with the deterministic visit-matching
rule. The training policy target is the improved policy
pi' = softmax(logits + sigma(completed Q)) instead of visit counts,
which stays informative at very low simulation counts (n = 16-32).

Differences from ZeroAgent:
- ``set_temperature(T)`` with T > 0 samples the move from the improved
  policy pi' instead of using the sequential-halving winner; T = 0
  (default at eval) picks the winner deterministically.
- No virtual loss: each visit wave descends from *distinct* root
  children, so in-flight paths are disjoint by construction.

Sign convention (matches ZeroAgent post 2026-06-11 fix): each node's Q
is stored from its OWN next_player perspective; viewed from the parent
it must be negated.
"""
import math
import os

import numpy as np

from alphigo import alphigo_board
from alphigo import encoders
from alphigo import kerasutil
from alphigo.agent.gumbel_math import (
    completed_q_values, improved_policy, mixed_value,
    sequential_halving_schedule, sigma_q)
from alphigo.agent.zero_agent import ZeroAgent
from alphigo.mcts.monte_carlo import MCTSNode

__all__ = ['GumbelZeroAgent', 'load_gumbel_agent']


class GumbelZeroAgent(ZeroAgent):
    """Gumbel AlphaZero search + (policy, value[, ownership]) network."""

    def __init__(self, model, encoder, num_simulations=32, m=16,
                 c_visit=50.0, c_scale=1.0, value_mode='network',
                 external_value_model=None, batch_size=16):
        ZeroAgent.__init__(
            self, model, encoder, num_simulations=num_simulations,
            c_puct=1.5, value_mode=value_mode,
            external_value_model=external_value_model,
            batch_size=batch_size)
        self._gumbel_m = int(m)
        self._c_visit = float(c_visit)
        self._c_scale = float(c_scale)
        # Root exploration for self-play: mix eps-uniform into the prior
        # used for Gumbel top-m candidate SAMPLING only. Halving ranks and
        # policy targets keep the true prior, so a sampled noise candidate
        # survives only if search finds its Q genuinely good. Off unless
        # ROOT_EXPLORE_EPS is set (the self-play generator sets it; evals,
        # h2h and puzzle probes never do).
        self._root_explore_eps = float(
            os.environ.get('ROOT_EXPLORE_EPS', '0') or 0.0)

    # ── interior (non-root) deterministic selection ───────────────────

    def _child_stats(self, node):
        """(q_parent_perspective, visit_counts, priors) arrays."""
        kids = node.children
        q = np.empty(len(kids))
        nvis = np.empty(len(kids))
        pri = np.empty(len(kids))
        for i, c in enumerate(kids):
            q[i] = -c.Q          # child Q is own-perspective: negate
            nvis[i] = c.N
            pri[i] = c.prior
        return q, nvis, pri

    def _select_child_deterministic(self, node):
        """argmax_a pi'(a) - N(a) / (1 + sum_b N(b))  (paper sec. 5)."""
        kids = node.children
        n = len(kids)
        q = np.empty(n)
        nvis = np.empty(n)
        pri = np.empty(n)
        for i in range(n):
            c = kids[i]
            q[i] = -c.Q
            nvis[i] = c.N
            pri[i] = c.prior

        logits = np.log(np.maximum(pri, 1e-12))

        sum_n = nvis.sum()
        max_n = nvis.max()
        visited = nvis > 0
        if visited.any() and sum_n > 0:
            pi_vis = pri[visited].sum()
            if pi_vis > 0.0:
                wq = (pri[visited] * q[visited]).sum() / pi_vis
                v_mix = (node.raw_value + sum_n * wq) / (1.0 + sum_n)
            else:
                v_mix = node.raw_value
        else:
            v_mix = node.raw_value
        cq = np.where(visited, q, v_mix)

        sigma = (self._c_visit + max_n) * self._c_scale
        scores_raw = logits + sigma * cq
        scores_raw -= scores_raw.max()
        exp = np.exp(scores_raw)
        pi_prime = exp / exp.sum()
        scores = pi_prime - nvis / (1.0 + sum_n)
        return kids[int(np.argmax(scores))]

    # ── descent and visit waves ───────────────────────────────────────

    def _descend(self, root, root_child):
        """Descend from a prescribed root child to a leaf.

        Returns (leaf, path) with path = [root, root_child, ...].
        """
        node = root_child
        if node.game_state is None:
            node.game_state = root.game_state.apply_move(node.move)
        path = [root, node]
        while node.children:
            node = self._select_child_deterministic(node)
            if node.game_state is None:
                node.game_state = node.parent.game_state.apply_move(
                    node.move)
            path.append(node)
            if node.game_state.is_over():
                break
        return node, path

    def _visit_wave(self, root, survivors):
        """One visit per surviving root child; leaf evals are batched.

        Paths are disjoint (distinct root children), so no virtual loss
        is needed.
        """
        leaves_to_expand = []
        terminal_backups = []
        for child in survivors:
            leaf, path = self._descend(root, child)
            if leaf.is_terminal():
                winner = leaf.game_state.winner()
                if winner is None:
                    lv = 0.0
                else:
                    lv = 1.0 if winner == leaf.game_state.next_player \
                        else -1.0
                terminal_backups.append(
                    (path, lv, leaf.game_state.next_player))
            else:
                leaves_to_expand.append((leaf, path))
        if leaves_to_expand:
            values = self._expand_batch(
                [leaf for leaf, _ in leaves_to_expand])
            for (leaf, path), value in zip(leaves_to_expand, values):
                self._backup(path, value, leaf.game_state.next_player)
        for path, value, player in terminal_backups:
            self._backup(path, value, player)

    # ── root: Gumbel top-m + sequential halving ───────────────────────

    def _root_v_mix(self, root):
        q, nvis, pri = self._child_stats(root)
        return mixed_value(root.raw_value, pri, q, nvis)

    def _run_gumbel_search(self, game_state, pre_encoded=None):
        """Returns (root, winner_child_or_None)."""
        root = MCTSNode(game_state)
        if root.is_terminal():
            return root, None
        root_value = self._expand(root, pre_encoded=pre_encoded)
        root.N = 1
        root.W = root_value
        root.Q = root_value
        children = root.children
        if not children:
            return root, None
        if len(children) == 1:
            self._visit_wave(root, [children[0]])
            return root, children[0]

        n = self._num_simulations
        m_eff = min(self._gumbel_m, len(children), max(1, n))
        priors = np.array([c.prior for c in children])
        eps = self._root_explore_eps
        if eps > 0.0:
            sel = (1.0 - eps) * priors + eps / len(children)
        else:
            sel = priors
        logits = np.log(np.maximum(sel, 1e-12))
        g = np.random.gumbel(size=len(children))
        order = np.argsort(-(g + logits))
        candidates = [children[i] for i in order[:m_eff]]
        g_by_id = {id(children[i]): float(g[i]) for i in order[:m_eff]}

        c_visit = self._c_visit
        c_scale = self._c_scale
        _log = math.log
        raw_v = root.raw_value
        _vm = raw_v
        _sm = 0.0

        def _rank_score(child):
            qhat = (-child.Q) if child.N > 0 else _vm
            return g_by_id[id(child)] + _log(max(child.prior, 1e-12)) + _sm * qhat

        for survivors_count, visits in sequential_halving_schedule(m_eff, n):
            survivors = candidates[:survivors_count]
            for _ in range(visits):
                self._visit_wave(root, survivors)

            max_n = 0
            sum_n = 0.0
            wpq = 0.0
            pi_vis = 0.0
            for c in children:
                ni = c.N
                if ni > max_n:
                    max_n = ni
                sum_n += ni
                if ni > 0:
                    pi_vis += c.prior
                    wpq += c.prior * (-c.Q)
            _vm = ((raw_v + sum_n * (wpq / pi_vis)) / (1.0 + sum_n)
                   if pi_vis > 0.0 and sum_n > 0.0 else raw_v)
            _sm = (c_visit + max_n) * c_scale

            candidates = sorted(survivors, key=_rank_score, reverse=True)

        return root, candidates[0]

    # ── public API (same contract as ZeroAgent.search) ────────────────

    def search(self, game_state, pre_encoded=None):
        """Run Gumbel search; return dict(move, policy_target, value).

        ``policy_target`` is pi' = softmax(logits + sigma(completed Q))
        over all legal root moves — NOT a visit-count distribution.
        """
        root, winner = self._run_gumbel_search(game_state, pre_encoded=pre_encoded)
        self._last_root = root  # cache for GUI display (no extra search)
        num_points = self._encoder.num_points()
        num_actions = num_points + 1
        target = np.zeros(num_actions, dtype=np.float32)
        if winner is None or not root.children:
            return {'move': alphigo_board.Move.pass_turn(),
                    'policy_target': target,
                    'value': float(root.Q)}

        q, nvis, pri = self._child_stats(root)
        logits = np.log(np.maximum(pri, 1e-12))
        cq = completed_q_values(q, nvis, pri, root.raw_value)
        pi_prime = improved_policy(logits, cq, nvis.max(),
                                   self._c_visit, self._c_scale)
        for child, p in zip(root.children, pi_prime):
            if child.move.is_pass:
                target[num_points] = p
            elif child.move.point is not None:
                idx = self._encoder.encode_point(child.move.point)
                target[idx] = p

        if self._temperature > 0:
            probs = pi_prime / pi_prime.sum()
            idx = np.random.choice(len(root.children), p=probs)
            move = root.children[idx].move
        else:
            move = winner.move

        return {'move': move,
                'policy_target': target,
                'value': float(root.Q)}

    # ── serialization ─────────────────────────────────────────────────

    def serialize(self, h5file):
        ZeroAgent.serialize(self, h5file)
        h5file['agent'].attrs['search'] = 'gumbel'
        h5file['agent'].attrs['gumbel_m'] = self._gumbel_m
        h5file['agent'].attrs['c_visit'] = self._c_visit
        h5file['agent'].attrs['c_scale'] = self._c_scale


def load_gumbel_agent(h5file, num_simulations=None, m=None,
                      c_visit=None, c_scale=None, value_mode='network',
                      external_value_model=None, batch_size=16):
    """Load any ZeroAgent-format checkpoint as a GumbelZeroAgent."""
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = int(h5file['encoder'].attrs['board_width'])
    board_height = int(h5file['encoder'].attrs['board_height'])
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))

    attrs = h5file['agent'].attrs if 'agent' in h5file else {}
    n_sim = int(attrs.get('num_simulations', 32))
    m_val = int(attrs.get('gumbel_m', 16))
    cv = float(attrs.get('c_visit', 50.0))
    cs = float(attrs.get('c_scale', 1.0))
    if num_simulations is not None:
        n_sim = int(num_simulations)
    if m is not None:
        m_val = int(m)
    if c_visit is not None:
        cv = float(c_visit)
    if c_scale is not None:
        cs = float(c_scale)
    return GumbelZeroAgent(model, encoder, num_simulations=n_sim,
                           m=m_val, c_visit=cv, c_scale=cs,
                           value_mode=value_mode,
                           external_value_model=external_value_model,
                           batch_size=batch_size)
