"""Benchmark runner: pit two agents against each other over N games.

Usage (programmatic):
    from alphigo.benchmark import run_benchmark

    results = run_benchmark(
        black_agent=bot_a,
        white_agent=bot_b,
        board_size=9,
        num_games=50,
        swap_colors=True,   # play half as black, half as white
    )
    results.print_summary()
    results.save("results/mcts_vs_random.json")
"""

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from alphigo.alphigo_types import Player
from alphigo.simulate import simulate_game


def elo_diff(win_rate):
    """Return Elo difference implied by win_rate (0-1). None if undefined."""
    if win_rate <= 0 or win_rate >= 1:
        return None
    return round(400 * math.log10(win_rate / (1 - win_rate)), 1)


def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> tuple:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) as fractions in [0, 1].
    Works correctly at small sample sizes and extreme win rates.
    """
    if n == 0:
        return (0.0, 1.0)
    # z = 1.96 for 95%, 1.645 for 90%, 2.576 for 99%
    z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_table.get(confidence, 1.96)
    p = wins / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class GameResult:
    game_id: int
    winner: str  # "black", "white", or "none"
    reason: str
    num_moves: int
    elapsed_sec: float
    black_agent: str
    white_agent: str


@dataclass
class BenchmarkResults:
    black_label: str
    white_label: str
    board_size: int
    num_games: int
    swap_colors: bool
    games: list = field(default_factory=list)

    # ── Computed stats ───────────────────────────────────────────────

    def _agent_wins(self, label):
        return sum(
            1 for g in self.games
            if (g.winner == "black" and g.black_agent == label) or
               (g.winner == "white" and g.white_agent == label)
        )

    def summary_dict(self):
        a_wins = self._agent_wins(self.black_label)
        b_wins = self._agent_wins(self.white_label)
        draws = sum(1 for g in self.games if g.winner == "none")
        total_moves = sum(g.num_moves for g in self.games)
        total_time = sum(g.elapsed_sec for g in self.games)
        n = len(self.games) or 1
        a_wr = a_wins / n
        b_wr = b_wins / n
        avg_ms_per_move = (round(1000 * total_time / total_moves, 2)
                           if total_moves else None)
        a_ci = wilson_ci(a_wins, n)
        b_ci = wilson_ci(b_wins, n)
        return {
            "agent_a": self.black_label,
            "agent_b": self.white_label,
            "board_size": self.board_size,
            "total_games": len(self.games),
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "a_win_pct": round(100 * a_wr, 1),
            "b_win_pct": round(100 * b_wr, 1),
            "a_win_pct_ci": (round(100 * a_ci[0], 1), round(100 * a_ci[1], 1)),
            "b_win_pct_ci": (round(100 * b_ci[0], 1), round(100 * b_ci[1], 1)),
            "a_elo_diff": elo_diff(a_wr),
            "b_elo_diff": elo_diff(b_wr),
            "avg_moves": round(total_moves / n, 1),
            "avg_time_sec": round(total_time / n, 2),
            "avg_ms_per_move": avg_ms_per_move,
            "total_time_sec": round(total_time, 2),
        }

    def print_summary(self):
        s = self.summary_dict()
        w = 50
        print("=" * w)
        print("BENCHMARK RESULTS".center(w))
        print("=" * w)
        print(f"  {s['agent_a']}  vs  {s['agent_b']}")
        print(f"  Board: {s['board_size']}x{s['board_size']}   "
              f"Games: {s['total_games']}   "
              f"Swap colors: {self.swap_colors}")
        print("-" * w)
        def _elo_str(v):
            if v is None:
                return "n/a"
            return f"{v:+.0f}"

        print(f"  {s['agent_a']:>20s} wins: {s['a_wins']:>4d}  ({s['a_win_pct']}%)  Elo diff: {_elo_str(s['a_elo_diff'])}")
        print(f"  {s['agent_b']:>20s} wins: {s['b_wins']:>4d}  ({s['b_win_pct']}%)  Elo diff: {_elo_str(s['b_elo_diff'])}")
        print(f"  {'draws':>20s}:      {s['draws']:>4d}")
        print("-" * w)
        print(f"  Avg moves/game: {s['avg_moves']}")
        print(f"  Avg time/game:  {s['avg_time_sec']}s")
        print(f"  Total time:     {s['total_time_sec']}s")
        print("=" * w)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary_dict(),
            "config": {
                "black_label": self.black_label,
                "white_label": self.white_label,
                "board_size": self.board_size,
                "num_games": self.num_games,
                "swap_colors": self.swap_colors,
            },
            "games": [asdict(g) for g in self.games],
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Results saved to {path}")


@dataclass
class TournamentResults:
    agent_labels: list
    board_size: int
    games_per_matchup: int
    matchups: list = field(default_factory=list)

    def standings(self):
        stats = {label: {"wins": 0, "losses": 0, "draws": 0}
                 for label in self.agent_labels}
        for br in self.matchups:
            for g in br.games:
                if g.winner == "none":
                    stats[g.black_agent]["draws"] += 1
                    stats[g.white_agent]["draws"] += 1
                elif g.winner == "black":
                    stats[g.black_agent]["wins"] += 1
                    stats[g.white_agent]["losses"] += 1
                elif g.winner == "white":
                    stats[g.white_agent]["wins"] += 1
                    stats[g.black_agent]["losses"] += 1

        rows = []
        for label in self.agent_labels:
            s = stats[label]
            total = s["wins"] + s["losses"] + s["draws"]
            win_pct = round(100 * s["wins"] / total, 1) if total else 0.0
            rows.append({
                "agent": label,
                "wins": s["wins"],
                "losses": s["losses"],
                "draws": s["draws"],
                "win_pct": win_pct,
            })
        rows.sort(key=lambda r: (-r["win_pct"], r["agent"]))
        return rows

    def pairwise_matrix(self):
        agent_wins = {a: {b: 0 for b in self.agent_labels}
                      for a in self.agent_labels}
        agent_games = {a: {b: 0 for b in self.agent_labels}
                       for a in self.agent_labels}

        for br in self.matchups:
            for g in br.games:
                ba, wa = g.black_agent, g.white_agent
                agent_games[ba][wa] += 1
                agent_games[wa][ba] += 1
                if g.winner == "black":
                    agent_wins[ba][wa] += 1
                elif g.winner == "white":
                    agent_wins[wa][ba] += 1

        matrix = {}
        for a in self.agent_labels:
            matrix[a] = {}
            for b in self.agent_labels:
                if a == b:
                    matrix[a][b] = None
                else:
                    total = agent_games[a][b]
                    matrix[a][b] = (round(100 * agent_wins[a][b] / total, 1)
                                    if total else None)
        return matrix

    def print_standings(self):
        rows = self.standings()

        header = (f"TOURNAMENT STANDINGS "
                  f"({self.board_size}x{self.board_size}, "
                  f"{self.games_per_matchup} games/matchup)")
        sep = "=" * len(header)
        print(f"\n{header}")
        print(sep)
        print(f"  {'Rank':<6}{'Agent':<24}"
              f"{'W':>5}{'L':>5}{'D':>5}{'Win%':>8}")
        print("-" * len(header))
        for i, r in enumerate(rows, 1):
            print(f"  {i:<6}{r['agent']:<24}"
                  f"{r['wins']:>5}{r['losses']:>5}{r['draws']:>5}"
                  f"{r['win_pct']:>7.1f}%")
        print(sep)

    def print_matrix(self):
        matrix = self.pairwise_matrix()
        agents = [r["agent"] for r in self.standings()]

        col_w = max(max(len(a) for a in agents), 6) + 2
        table_w = col_w + len(agents) * col_w

        print(f"\nPAIRWISE WIN RATES")
        print("=" * table_w)
        print(" " * col_w + "".join(f"{a:>{col_w}}" for a in agents))

        for a in agents:
            cells = []
            for b in agents:
                if a == b:
                    cells.append(f"{'-':>{col_w}}")
                elif matrix[a][b] is not None:
                    cells.append(f"{matrix[a][b]:.0f}%".rjust(col_w))
                else:
                    cells.append(f"{'n/a':>{col_w}}")
            print(f"{a:>{col_w}}" + "".join(cells))
        print()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "standings": self.standings(),
            "pairwise": self.pairwise_matrix(),
            "config": {
                "agent_labels": self.agent_labels,
                "board_size": self.board_size,
                "games_per_matchup": self.games_per_matchup,
            },
            "matchups": [
                {
                    "agents": [br.black_label, br.white_label],
                    "summary": br.summary_dict(),
                    "games": [asdict(g) for g in br.games],
                }
                for br in self.matchups
            ],
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Tournament results saved to {path}")


def run_benchmark(
    black_agent,
    white_agent,
    *,
    black_label="black_agent",
    white_label="white_agent",
    board_size=9,
    num_games=10,
    max_moves=2000,
    swap_colors=True,
    verbose=True,
):
    """Run num_games between two agents and return BenchmarkResults.

    If swap_colors is True, agents alternate who plays black/white each game
    so that first-move advantage is neutralized.
    """
    results = BenchmarkResults(
        black_label=black_label,
        white_label=white_label,
        board_size=board_size,
        num_games=num_games,
        swap_colors=swap_colors,
    )

    for i in range(num_games):
        swapped = swap_colors and (i % 2 == 1)
        if swapped:
            b_agent, w_agent = white_agent, black_agent
            b_label, w_label = white_label, black_label
        else:
            b_agent, w_agent = black_agent, white_agent
            b_label, w_label = black_label, white_label

        if verbose:
            tag = " (swapped)" if swapped else ""
            print(f"Game {i + 1}/{num_games}: "
                  f"{b_label} (B) vs {w_label} (W){tag} ... ",
                  end="", flush=True)

        t0 = time.perf_counter()
        result = simulate_game(b_agent, w_agent,
                               board_size=board_size,
                               max_moves=max_moves)
        elapsed = time.perf_counter() - t0

        winner_str = "none"
        if result["winner"] == Player.black:
            winner_str = "black"
        elif result["winner"] == Player.white:
            winner_str = "white"

        gr = GameResult(
            game_id=i + 1,
            winner=winner_str,
            reason=result["reason"],
            num_moves=result["num_moves"],
            elapsed_sec=round(elapsed, 3),
            black_agent=b_label,
            white_agent=w_label,
        )
        results.games.append(gr)

        if verbose:
            winner_name = (
                b_label if winner_str == "black"
                else w_label if winner_str == "white"
                else "draw"
            )
            print(f"{winner_name} wins "
                  f"({result['reason']}, {result['num_moves']} moves, "
                  f"{elapsed:.1f}s)")

    if verbose:
        print()
        results.print_summary()

    return results
