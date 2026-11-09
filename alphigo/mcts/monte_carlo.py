import random

import math

from alphigo.agent.base import Agent
from alphigo.agent.naive import RandomBot
from alphigo.alphigo_board import Move
from alphigo.alphigo_types import Player


class MCTSNode:
    __slots__ = (
        'game_state', 'parent', 'move', 'win_counts', 'num_rollouts',
        'children', '_unvisited_moves', 'prior', 'Q', 'N', 'W', 'raw_value',
    )

    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = None
        self.num_rollouts = 0
        self.children = []
        self._unvisited_moves = None
        self.prior = 0.0
        self.Q = 0.0
        self.N = 0
        self.W = 0.0
        self.raw_value = 0.0  # network value at expansion time; 0.0 until _expand_batch runs

    @property
    def unvisited_moves(self):
        if self._unvisited_moves is None:
            self._unvisited_moves = self.game_state.legal_moves()
        return self._unvisited_moves

    @unvisited_moves.setter
    def unvisited_moves(self, value):
        self._unvisited_moves = value

    def puct_score(self, c_puct, total_visits):
        """PUCT selection score from the *parent's* perspective:
        -Q + c_puct * P * sqrt(sum_N) / (1 + N).

        Q is stored from this node's own next_player perspective, which
        is the opponent of the parent player doing the selection, so it
        is negated here.
        """
        return -self.Q + c_puct * self.prior * math.sqrt(total_visits) / (1 + self.N)

    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner):
        if self.win_counts is None:
            self.win_counts = {Player.black: 0, Player.white: 0}
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    # tells if there is any move that can be added to a tree
    def can_add_children(self):
        return len(self.unvisited_moves) > 0

    # tells if this node is a terminal one
    def is_terminal(self):
        return self.game_state.is_over()

    def winning_frac(self, player):
        if self.num_rollouts == 0:
            return 0.0
        if self.win_counts is None:
            return 0.0
        return float(self.win_counts[player]) / float(self.num_rollouts)


class MCTSAgent(Agent):

    # RandomBot is stateless — reuse a single pair across all rollouts.
    _rollout_bots = {
        Player.black: RandomBot(),
        Player.white: RandomBot(),
    }

    def __init__(self, num_rounds, temperature):
        super().__init__()
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root
            # finding a node where we can add a child
            while (not node.can_add_children()) and (not node.is_terminal()):
                node = self.select_child(node)

            # add a new child node
            if node.can_add_children():
                node = node.add_random_child()

            # simulate a game from this node
            winner = self.simulate_random_game(node.game_state)

            # propagating a score, up the tree
            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_move = None
        best_frac = -1.0
        for child in root.children:
            child_frac = child.winning_frac(game_state.next_player)
            if best_frac < child_frac:
                best_frac = child_frac
                best_move = child.move
        if best_frac < 0.05:
            best_move = Move(is_resign=True)
        # Fallback if no children were expanded (e.g. num_rounds=0 or all terminal)
        if best_move is None:
            legal = game_state.legal_moves()
            best_move = legal[random.randint(0, len(legal) - 1)] if legal else Move.pass_turn()
        return best_move

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        if total_rollouts == 0:
            return random.choice(node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None

        for child in node.children:
            win_percentage = child.winning_frac(node.game_state.next_player)
            n = max(child.num_rollouts, 1)
            exploration_term = self.temperature * math.sqrt(log_rollouts / n)
            uct_score = win_percentage + exploration_term

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    @staticmethod
    def simulate_random_game(game):
        bots = MCTSAgent._rollout_bots

        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()
