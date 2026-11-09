#!/usr/bin/env python
"""Unified CLI for AlphiGo – benchmark, play, list agents.

Examples:

  # List available agent types
  python run.py agents

  # Benchmark: 20 headless games, MCTS vs Random on 9x9
  python run.py benchmark --black mcts --white random --board-size 9 -n 20

  # MCTS with custom params
  python run.py benchmark --black mcts:num_rounds=1000,temperature=1.2 \\
                           --white random -n 50 --board-size 9

  # Deep-learning agent from file
  python run.py benchmark --black dl:agent_file=models/bot.h5 \\
                           --white mcts -n 10

  # Save results to JSON
  python run.py benchmark --black mcts --white random -n 20 -o results/run1.json

  # Play a GUI game: human vs MCTS
  python run.py play --black human --white mcts:num_rounds=500 --board-size 9
"""

import argparse
import sys


def parse_agent_spec(spec):
    """Parse 'name:key=val,key=val' into (name, kwargs_dict).

    Examples:
        "random"                     -> ("random", {})
        "mcts:num_rounds=500"        -> ("mcts", {"num_rounds": "500"})
        "dl:agent_file=path/to/f.h5" -> ("dl", {"agent_file": "path/to/f.h5"})
    """
    if ":" not in spec:
        return spec, {}
    name, params_str = spec.split(":", 1)
    kwargs = {}
    for pair in params_str.split(","):
        if "=" not in pair:
            raise ValueError(f"Bad param '{pair}' in agent spec '{spec}'. "
                             f"Expected key=value.")
        k, v = pair.split("=", 1)
        kwargs[k.strip()] = v.strip()
    return name, kwargs


# ── Subcommands ──────────────────────────────────────────────────────

def cmd_agents(_args):
    from alphigo.agent.registry import list_agents
    print("\nAvailable agents:\n")
    for name, help_text in list_agents():
        print(f"  {name:20s} {help_text}")
    print("\nPass params with  name:key=val,key=val  (e.g. mcts:num_rounds=1000)")
    print()


def _print_move_log(move_log, board_size):
    from alphigo.alphigo_board import GameState
    from alphigo.alphigo_types import Player, Point

    COLS = 'ABCDEFGHJKLMNOPQRST'

    def pt_str(pt):
        return COLS[pt.col - 1] + str(pt.row)

    def print_board(board):
        print('  ' + ' '.join(COLS[c] for c in range(board_size)))
        for r in range(board_size, 0, -1):
            row = []
            for c in range(1, board_size + 1):
                p = board.get_player_on_point(Point(row=r, col=c))
                row.append('X' if p == Player.black else 'O' if p == Player.white else '.')
            print(f'{r:2} ' + ' '.join(row))

    gs = GameState.new_game(board_size)
    for i, (player, move) in enumerate(move_log, 1):
        c = 'B' if player == Player.black else 'W'
        label = 'PASS' if move.is_pass else ('RESIGN' if move.is_resign else pt_str(move.point))
        print(f'  Move {i:3d} {c}: {label}')
        gs = gs.apply_move(move)
        print_board(gs.board)
        print()


def cmd_benchmark(args):
    from alphigo.agent.registry import create_agent
    from alphigo.benchmark import run_benchmark
    from alphigo.simulate import simulate_game
    from alphigo.alphigo_types import Player

    _DL_NAMES = {'dl', 'deep', 'deeplearning'}

    b_name, b_kwargs = parse_agent_spec(args.black)
    w_name, w_kwargs = parse_agent_spec(args.white)

    if args.temperature is not None:
        if b_name in _DL_NAMES:
            b_kwargs.setdefault('temperature', str(args.temperature))
        if w_name in _DL_NAMES:
            w_kwargs.setdefault('temperature', str(args.temperature))

    black_agent = create_agent(b_name, **b_kwargs)
    white_agent = create_agent(w_name, **w_kwargs)

    black_label = args.black
    white_label = args.white

    if args.moves:
        # Single-game mode with move-by-move output
        result = simulate_game(black_agent, white_agent,
                               board_size=args.board_size,
                               max_moves=args.max_moves)
        winner = result['winner']
        winner_label = (black_label if winner == Player.black
                        else white_label if winner == Player.white
                        else 'draw')
        print(f"Black: {black_label}  |  White: {white_label}")
        print(f"Board: {args.board_size}x{args.board_size}\n")
        _print_move_log(result['move_log'], args.board_size)
        print(f"Result: {winner_label} wins ({result['reason']}, {result['num_moves']} moves)")
        return

    results = run_benchmark(
        black_agent,
        white_agent,
        black_label=black_label,
        white_label=white_label,
        board_size=args.board_size,
        num_games=args.num_games,
        max_moves=args.max_moves,
        swap_colors=not args.no_swap,
        verbose=True,
    )

    if args.output:
        results.save(args.output)


def cmd_tournament(args):
    import itertools
    from alphigo.agent.registry import create_agent
    from alphigo.benchmark import run_benchmark, TournamentResults

    _DL_NAMES = {'dl', 'deep', 'deeplearning'}

    agents = {}
    for spec in args.agents:
        name, kwargs = parse_agent_spec(spec)
        if args.temperature is not None and name in _DL_NAMES:
            kwargs.setdefault('temperature', str(args.temperature))
        agents[spec] = create_agent(name, **kwargs)

    labels = list(agents.keys())
    tournament = TournamentResults(
        agent_labels=labels,
        board_size=args.board_size,
        games_per_matchup=args.num_games,
    )

    pairs = list(itertools.combinations(labels, 2))
    total_matchups = len(pairs)

    for idx, (a_label, b_label) in enumerate(pairs, 1):
        print(f"Matchup {idx}/{total_matchups}: {a_label} vs {b_label} ... ",
              end="", flush=True)

        results = run_benchmark(
            agents[a_label],
            agents[b_label],
            black_label=a_label,
            white_label=b_label,
            board_size=args.board_size,
            num_games=args.num_games,
            max_moves=args.max_moves,
            swap_colors=True,
            verbose=False,
        )
        s = results.summary_dict()
        ms = s['avg_ms_per_move']
        ms_str = f"{ms:.1f}ms/move" if ms is not None else "n/a"
        print(f"{a_label} {s['a_wins']}W  {b_label} {s['b_wins']}W  "
              f"draws {s['draws']}  ({s['a_win_pct']}% / {s['b_win_pct']}%)  {ms_str}")
        tournament.matchups.append(results)

    tournament.print_standings()
    tournament.print_matrix()

    if args.output:
        tournament.save(args.output)


def cmd_play(args):
    from alphigo.agent.registry import create_agent
    from gui.game_runner import run_gui_game, get_human_move, get_bot_move_async

    def make_move_fn(spec):
        if spec == "human":
            return get_human_move, None
        name, kwargs = parse_agent_spec(spec)
        bot = create_agent(name, **kwargs)
        return (lambda game, gui, on_key=None:
                get_bot_move_async(bot, game, gui, on_key=on_key)), bot

    black_fn, black_agent = make_move_fn(args.black)
    white_fn, white_agent = make_move_fn(args.white)

    winner = run_gui_game(args.board_size, black_fn, white_fn,
                          delay_after_draw=args.delay,
                          black_agent=black_agent,
                          white_agent=white_agent)
    if winner is not None:
        # Player is an IntEnum: print the name, not the underlying int.
        print(f"\nWinner: {winner.name}")
    else:
        print("\nGame was quit.")


# ── Argument parser ──────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="AlphiGo – unified CLI for agent comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # agents
    sub.add_parser("agents", help="List available agent types")

    # benchmark
    bp = sub.add_parser("benchmark", help="Run headless games and collect stats")
    bp.add_argument("--black", required=True,
                    help="Black agent spec  (e.g. mcts:num_rounds=500)")
    bp.add_argument("--white", required=True,
                    help="White agent spec  (e.g. random)")
    bp.add_argument("--board-size", type=int, default=9)
    bp.add_argument("-n", "--num-games", type=int, default=10)
    bp.add_argument("--max-moves", type=int, default=2000)
    bp.add_argument("--no-swap", action="store_true",
                    help="Don't alternate colors between games")
    bp.add_argument("-o", "--output", help="Save results JSON to this path")
    bp.add_argument("--moves", action="store_true",
                    help="Print move-by-move board for a single game")
    bp.add_argument("--temperature", type=float, default=None,
                    help="Move sampling temperature for DL agents (0=greedy, "
                         "0.5=recommended for varied games)")

    # tournament
    tp = sub.add_parser("tournament",
                        help="Round-robin tournament between multiple agents")
    tp.add_argument("--agents", nargs="+", required=True,
                    help="Agent specs  (e.g. mcts:num_rounds=500 random)")
    tp.add_argument("--board-size", type=int, default=9)
    tp.add_argument("-n", "--num-games", type=int, default=50,
                    help="Games per matchup (default: 50)")
    tp.add_argument("--max-moves", type=int, default=2000)
    tp.add_argument("--temperature", type=float, default=None,
                    help="Move sampling temperature for DL agents (0=greedy, "
                         "0.5=recommended for varied games)")
    tp.add_argument("-o", "--output",
                    help="Save tournament results JSON to this path")

    # play
    pp = sub.add_parser("play", help="Play a GUI game")
    pp.add_argument("--black", default="human",
                    help="Black agent spec or 'human'")
    pp.add_argument("--white", default="random",
                    help="White agent spec or 'human'")
    pp.add_argument("--board-size", type=int, default=9)
    pp.add_argument("--delay", type=float, default=0.5,
                    help="Seconds to pause between moves for bot-vs-bot (default: 0.5)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "agents": cmd_agents,
        "benchmark": cmd_benchmark,
        "tournament": cmd_tournament,
        "play": cmd_play,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
