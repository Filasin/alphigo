"""Headless game simulation (no GUI) for benchmarking agent vs agent."""

from alphigo.alphigo_board import GameState
from alphigo.alphigo_types import Player


MAX_MOVES_DEFAULT = 2000


def _gtp_final_score(black_agent, white_agent):
    """Get winner and score string from a GTP engine's final_score command.

    Returns (winner, score_str) where winner is Player.black/white/None
    and score_str is the raw GTP response (e.g. 'W+10.5', 'B+3.5', '0').
    """
    from alphigo.agent.gtp_agent import GtpAgent
    for agent in (black_agent, white_agent):
        if isinstance(agent, GtpAgent):
            try:
                result = agent.final_score().strip().upper()
                if result.startswith('B'):
                    return Player.black, result
                if result.startswith('W'):
                    return Player.white, result
                return None, result
            except Exception:
                pass
    return None, None


def simulate_game(black_agent, white_agent, board_size=9, max_moves=MAX_MOVES_DEFAULT):
    """Play one full game between two agents without any GUI.

    Returns a dict with:
        winner        – Player.black, Player.white, or None (draw / max-moves)
        reason        – "resign", "pass", "max_moves"
        num_moves     – total moves played
        move_log      – list of (player, move) tuples
    """
    game = GameState.new_game(board_size)
    bots = {
        Player.black: black_agent,
        Player.white: white_agent,
    }
    move_log = []
    num_moves = 0

    while not game.is_over():
        if num_moves >= max_moves:
            return {
                "winner": None,
                "reason": "max_moves",
                "num_moves": num_moves,
                "move_log": move_log,
            }

        move = bots[game.next_player].select_move(game)
        move_log.append((game.next_player, move))
        game = game.apply_move(move)
        num_moves += 1

    score_str = None
    if game.last_move and game.last_move.is_resign:
        reason = "resign"
        winner = game.winner()  # next_player after resign = winner
    else:
        reason = "pass"
        winner, score_str = _gtp_final_score(black_agent, white_agent)
        if winner is None:
            winner = game.winner()

    return {
        "winner": winner,
        "reason": reason,
        "num_moves": num_moves,
        "move_log": move_log,
        "score": score_str,
        "final_state": game,
    }
