class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()

    def analyze(self, game_state):
        """Return an AnalysisResult for the GUI overlay, or None if not supported.

        Default is None (Random / GTP no-op).  Neural agents override.
        """
        return None


class PassHackAgent(Agent):
    """Wrapper: passes when opponent passed and TT or value head says we're ahead.

    The underlying agent's policy network has no pass action (81 outputs for
    9x9, one per intersection).  Without this wrapper the agent fills dame
    until no legal moves remain, which costs ~1 point per extra move under
    dead-stone-aware scoring.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def select_move(self, game_state):
        if game_state.last_move is not None and game_state.last_move.is_pass:
            from alphigo.alphigo_board import area_score, Move
            from alphigo.alphigo_types import Player

            b, w = area_score(game_state.board)
            tt = b - w  # positive = Black ahead
            if game_state.next_player == Player.white:
                tt = -tt  # flip so positive = current player ahead

            val = 0.0
            if hasattr(self.inner, 'search'):
                result = self.inner.search(game_state)
                val = result['value']
                if tt > 0 or val > 0:
                    return Move.pass_turn()
                return result['move']

            if tt > 0:
                return Move.pass_turn()

        return self.inner.select_move(game_state)

    def analyze(self, game_state):
        return self.inner.analyze(game_state)
