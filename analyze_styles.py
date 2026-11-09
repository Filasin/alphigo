#!/usr/bin/env python3
"""Comparative style analysis between two DL agents.

Plays both agents against a common opponent (default: GnuGo) and against
each other, recording full probability distributions at every move.
Produces a report with:

  1. Move agreement rate (how often agents pick the same top-1 move)
  2. Policy entropy by game phase (opening / midgame / endgame)
  3. Spatial move heatmaps (where each agent tends to play)
  4. KL divergence between the two policies on shared positions
  5. Game-length and resignation analysis

Usage:
    python analyze_styles.py \\
        --agent-a checkpoints/supervised_sevenplane_large_adam_19x19_e15.h5 \\
        --agent-b checkpoints/supervised_elevenplane_large_adam_19x19_e15.h5 \\
        --opponent gnugo:level=1 \\
        --board-size 19 \\
        --num-games 30 \\
        --output results/style_analysis
"""

import argparse
import json
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_from_path(path):
    """Extract a short label from a checkpoint path."""
    import re
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'(?:supervised|mcts)_([^_]+)_([^_]+)_', base)
    if m:
        return f'{m.group(1)}/{m.group(2)}'
    return base[:30]


def _game_phase(move_num, total_moves):
    """Classify a move into opening / midgame / endgame."""
    frac = move_num / max(total_moves, 1)
    if frac < 0.25:
        return 'opening'
    elif frac < 0.70:
        return 'midgame'
    else:
        return 'endgame'


def _entropy(probs):
    """Shannon entropy of a probability distribution."""
    p = probs[probs > 1e-10]
    return -np.sum(p * np.log(p))


def _kl_divergence(p, q):
    """KL(P || Q) — how different P is from Q."""
    mask = (p > 1e-10) & (q > 1e-10)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


# ---------------------------------------------------------------------------
# Core: play a game and record policy distributions at each move
# ---------------------------------------------------------------------------

def play_and_record(agent, opponent, board_size, agent_color='black',
                    max_moves=2000):
    """Play one game, recording the agent's policy distribution at every move.

    Returns:
        game_record: dict with keys:
            winner: 'agent', 'opponent', or 'draw'
            reason: 'resign', 'pass', 'max_moves'
            num_moves: int
            agent_moves: list of dicts, each with:
                move_num: int (1-based, within the full game)
                point: (row, col) or 'pass'
                probs: np.array of shape (board_size*board_size,)
                entropy: float
    """
    from alphigo.alphigo_board import GameState, Move
    from alphigo.alphigo_types import Player
    from alphigo.agent.gtp_agent import GtpAgent

    player_map = {
        'black': Player.black,
        'white': Player.white,
    }
    agent_player = player_map[agent_color]
    opp_player = agent_player.other

    bots = {
        agent_player: agent,
        opp_player: opponent,
    }

    game = GameState.new_game(board_size)
    agent_moves = []
    move_num = 0

    while not game.is_over() and move_num < max_moves:
        current = game.next_player
        if current == agent_player:
            # Record full probability distribution before selecting
            probs = agent.predict(game)
            probs = np.clip(probs, 1e-10, None)
            probs = probs / probs.sum()

            move = agent.select_move(game)
            point = (move.point.row, move.point.col) if move.point else 'pass'
            agent_moves.append({
                'move_num': move_num + 1,
                'point': point,
                'probs': probs,
                'entropy': float(_entropy(probs)),
            })
        else:
            move = bots[current].select_move(game)

        game = game.apply_move(move)
        move_num += 1

    # Determine winner
    if game.last_move and game.last_move.is_resign:
        reason = 'resign'
        raw_winner = game.winner()
    else:
        reason = 'pass' if move_num < max_moves else 'max_moves'
        # Prefer GTP scoring
        raw_winner = None
        if isinstance(opponent, GtpAgent):
            try:
                result = opponent.final_score().strip().upper()
                if result.startswith('B'):
                    raw_winner = Player.black
                elif result.startswith('W'):
                    raw_winner = Player.white
            except Exception:
                pass
        if raw_winner is None:
            raw_winner = game.winner()

    if raw_winner == agent_player:
        winner = 'agent'
    elif raw_winner == opp_player:
        winner = 'opponent'
    else:
        winner = 'draw'

    return {
        'winner': winner,
        'reason': reason,
        'num_moves': move_num,
        'agent_moves': agent_moves,
    }


# ---------------------------------------------------------------------------
# Head-to-head: record both agents' policies on the same positions
# ---------------------------------------------------------------------------

def play_head_to_head(agent_a, agent_b, board_size, max_moves=2000,
                      h2h_temperature=0.3):
    """Play agent_a (black) vs agent_b (white), recording both policies.

    At each position, BOTH agents produce a probability distribution,
    but only the current player's move is actually played.

    A small h2h_temperature is applied during move selection to ensure
    game diversity (greedy play produces identical games every time).
    The recorded probability distributions are always from the raw
    (temperature=0) policy for accurate comparison.

    Returns a dict with move-level comparison data.
    """
    from alphigo.alphigo_board import GameState
    from alphigo.alphigo_types import Player

    game = GameState.new_game(board_size)
    bots = {Player.black: agent_a, Player.white: agent_b}

    # Save original temperatures, apply h2h sampling temperature
    orig_temp_a = agent_a.temperature
    orig_temp_b = agent_b.temperature
    agent_a.temperature = h2h_temperature
    agent_b.temperature = h2h_temperature

    comparisons = []
    move_num = 0

    while not game.is_over() and move_num < max_moves:
        current = game.next_player

        # Both agents evaluate this position (raw policy, no temperature)
        probs_a = agent_a.predict(game)
        probs_a = np.clip(probs_a, 1e-10, None)
        probs_a = probs_a / probs_a.sum()

        probs_b = agent_b.predict(game)
        probs_b = np.clip(probs_b, 1e-10, None)
        probs_b = probs_b / probs_b.sum()

        top1_a = int(np.argmax(probs_a))
        top1_b = int(np.argmax(probs_b))

        comparisons.append({
            'move_num': move_num + 1,
            'current_player': 'a' if current == Player.black else 'b',
            'agree': top1_a == top1_b,
            'kl_a_b': float(_kl_divergence(probs_a, probs_b)),
            'kl_b_a': float(_kl_divergence(probs_b, probs_a)),
            'entropy_a': float(_entropy(probs_a)),
            'entropy_b': float(_entropy(probs_b)),
            'top1_a': top1_a,
            'top1_b': top1_b,
            'top5_a': np.argsort(-probs_a)[:5].tolist(),
            'top5_b': np.argsort(-probs_b)[:5].tolist(),
        })

        # The current player's agent makes the actual move
        move = bots[current].select_move(game)
        game = game.apply_move(move)
        move_num += 1

    # Restore original temperatures
    agent_a.temperature = orig_temp_a
    agent_b.temperature = orig_temp_b

    winner = None
    if game.last_move and game.last_move.is_resign:
        raw = game.winner()
        winner = 'a' if raw == Player.black else 'b'
    else:
        raw = game.winner()
        if raw == Player.black:
            winner = 'a'
        elif raw == Player.white:
            winner = 'b'

    return {
        'winner': winner,
        'num_moves': move_num,
        'comparisons': comparisons,
    }


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------

def analyze_vs_opponent(records, label, board_size):
    """Aggregate stats from games against a common opponent."""
    wins = sum(1 for r in records if r['winner'] == 'agent')
    losses = sum(1 for r in records if r['winner'] == 'opponent')
    draws = len(records) - wins - losses

    # Aggregate per-phase entropy
    phase_entropy = {'opening': [], 'midgame': [], 'endgame': []}
    all_entropy = []
    move_heatmap = np.zeros((board_size, board_size))
    win_heatmap = np.zeros((board_size, board_size))
    loss_heatmap = np.zeros((board_size, board_size))
    game_lengths = []
    win_lengths = []
    loss_lengths = []
    resign_count = 0

    for rec in records:
        total = rec['num_moves']
        game_lengths.append(total)
        if rec['reason'] == 'resign':
            resign_count += 1
        if rec['winner'] == 'agent':
            win_lengths.append(total)
        elif rec['winner'] == 'opponent':
            loss_lengths.append(total)

        for am in rec['agent_moves']:
            phase = _game_phase(am['move_num'], total)
            phase_entropy[phase].append(am['entropy'])
            all_entropy.append(am['entropy'])
            if am['point'] != 'pass':
                r, c = am['point']
                move_heatmap[r - 1, c - 1] += 1
                if rec['winner'] == 'agent':
                    win_heatmap[r - 1, c - 1] += 1
                elif rec['winner'] == 'opponent':
                    loss_heatmap[r - 1, c - 1] += 1

    return {
        'label': label,
        'games': len(records),
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': wins / max(len(records), 1),
        'resign_rate': resign_count / max(len(records), 1),
        'avg_game_length': np.mean(game_lengths) if game_lengths else 0,
        'avg_win_length': np.mean(win_lengths) if win_lengths else 0,
        'avg_loss_length': np.mean(loss_lengths) if loss_lengths else 0,
        'entropy_overall': np.mean(all_entropy) if all_entropy else 0,
        'entropy_by_phase': {
            k: np.mean(v) if v else 0 for k, v in phase_entropy.items()
        },
        'move_heatmap': move_heatmap,
        'win_heatmap': win_heatmap,
        'loss_heatmap': loss_heatmap,
    }


def analyze_head_to_head(h2h_records, board_size):
    """Aggregate stats from head-to-head games."""
    all_agree = []
    all_kl = []
    phase_agree = {'opening': [], 'midgame': [], 'endgame': []}
    phase_kl = {'opening': [], 'midgame': [], 'endgame': []}
    phase_entropy_a = {'opening': [], 'midgame': [], 'endgame': []}
    phase_entropy_b = {'opening': [], 'midgame': [], 'endgame': []}
    disagree_positions = []

    for rec in h2h_records:
        total = rec['num_moves']
        for c in rec['comparisons']:
            phase = _game_phase(c['move_num'], total)
            all_agree.append(c['agree'])
            sym_kl = (c['kl_a_b'] + c['kl_b_a']) / 2
            all_kl.append(sym_kl)
            phase_agree[phase].append(c['agree'])
            phase_kl[phase].append(sym_kl)
            phase_entropy_a[phase].append(c['entropy_a'])
            phase_entropy_b[phase].append(c['entropy_b'])
            if not c['agree']:
                disagree_positions.append(c)

    # Sort disagreements by KL divergence to find the most polarizing positions
    disagree_positions.sort(key=lambda x: (x['kl_a_b'] + x['kl_b_a']) / 2,
                            reverse=True)

    return {
        'total_positions': len(all_agree),
        'agreement_rate': np.mean(all_agree) if all_agree else 0,
        'mean_kl_divergence': np.mean(all_kl) if all_kl else 0,
        'agreement_by_phase': {
            k: np.mean(v) if v else 0 for k, v in phase_agree.items()
        },
        'kl_by_phase': {
            k: np.mean(v) if v else 0 for k, v in phase_kl.items()
        },
        'entropy_a_by_phase': {
            k: np.mean(v) if v else 0 for k, v in phase_entropy_a.items()
        },
        'entropy_b_by_phase': {
            k: np.mean(v) if v else 0 for k, v in phase_entropy_b.items()
        },
        'top_disagreements': disagree_positions[:20],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(stats_a, stats_b, h2h_stats, board_size, output_dir):
    """Generate all comparison figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print('[analyze] matplotlib not installed — skipping figures.')
        return

    os.makedirs(output_dir, exist_ok=True)
    label_a = stats_a['label']
    label_b = stats_b['label']

    # ── 1. Entropy by game phase ──────────────────────────────────────────
    phases = ['opening', 'midgame', 'endgame']
    ent_a = [stats_a['entropy_by_phase'][p] for p in phases]
    ent_b = [stats_b['entropy_by_phase'][p] for p in phases]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(phases))
    w = 0.35
    ax.bar(x - w / 2, ent_a, w, label=label_a, color='#2196F3')
    ax.bar(x + w / 2, ent_b, w, label=label_b, color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel('Mean Policy Entropy (nats)')
    ax.set_title(f'Policy Entropy by Game Phase — vs opponent')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'entropy_by_phase.png'))
    plt.close(fig)
    print(f'[analyze] entropy_by_phase.png saved.')

    # ── 2. Move heatmaps ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, stats, label in [(axes[0], stats_a, label_a),
                              (axes[1], stats_b, label_b)]:
        hm = stats['move_heatmap']
        if hm.sum() > 0:
            hm = hm / hm.sum()  # normalize to frequency
        im = ax.imshow(hm, cmap='YlOrRd', interpolation='nearest',
                       origin='lower')
        ax.set_title(f'{label}\n(move frequency)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'Spatial Move Distribution — {board_size}x{board_size}',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'move_heatmaps.png'))
    plt.close(fig)
    print(f'[analyze] move_heatmaps.png saved.')

    # ── 3. Win vs loss heatmap difference ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, stats, label in [(axes[0], stats_a, label_a),
                              (axes[1], stats_b, label_b)]:
        win_h = stats['win_heatmap']
        loss_h = stats['loss_heatmap']
        total = win_h + loss_h
        # Win rate per cell (where agent played), NaN where no moves
        with np.errstate(divide='ignore', invalid='ignore'):
            wr = np.where(total > 0, win_h / total, np.nan)
        im = ax.imshow(wr, cmap='RdYlGn', interpolation='nearest',
                       origin='lower', vmin=0, vmax=1)
        ax.set_title(f'{label}\n(local win rate where played)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'Win Rate by Board Region — {board_size}x{board_size}',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'win_loss_heatmaps.png'))
    plt.close(fig)
    print(f'[analyze] win_loss_heatmaps.png saved.')

    # ── 4. Head-to-head agreement by phase ────────────────────────────────
    if h2h_stats:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Agreement rate
        agree_vals = [h2h_stats['agreement_by_phase'][p] for p in phases]
        ax1.bar(phases, [v * 100 for v in agree_vals], color='#4CAF50')
        ax1.set_ylabel('Move Agreement (%)')
        ax1.set_title('Top-1 Move Agreement by Phase')
        ax1.set_ylim(0, 100)
        ax1.axhline(100 / (board_size * board_size), color='gray',
                     linestyle='--', alpha=0.5, label='random baseline')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # KL divergence
        kl_vals = [h2h_stats['kl_by_phase'][p] for p in phases]
        ax2.bar(phases, kl_vals, color='#E91E63')
        ax2.set_ylabel('Symmetric KL Divergence (nats)')
        ax2.set_title('Policy Divergence by Phase')
        ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'{label_a} vs {label_b} — Policy Comparison', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'h2h_agreement.png'))
        plt.close(fig)
        print(f'[analyze] h2h_agreement.png saved.')

    # ── 5. Game length distributions ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    all_a = [r['num_moves'] for r in _global_records_a]
    all_b = [r['num_moves'] for r in _global_records_b]
    if all_a and all_b:
        bins = np.linspace(0, max(max(all_a), max(all_b)) + 10, 30)
        ax.hist(all_a, bins=bins, alpha=0.6, label=label_a, color='#2196F3')
        ax.hist(all_b, bins=bins, alpha=0.6, label=label_b, color='#FF9800')
        ax.set_xlabel('Game Length (moves)')
        ax.set_ylabel('Count')
        ax.set_title('Game Length Distribution vs Opponent')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'game_lengths.png'))
    plt.close(fig)
    print(f'[analyze] game_lengths.png saved.')


# Global storage for game records (used by figure generation)
_global_records_a = []
_global_records_b = []


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(stats_a, stats_b, h2h_stats, output_dir):
    """Write a text summary report."""
    path = os.path.join(output_dir, 'style_report.md')
    lines = []
    lines.append('# Style Analysis Report\n')

    la, lb = stats_a['label'], stats_b['label']

    lines.append(f'## Agent A: {la}')
    lines.append(f'## Agent B: {lb}\n')

    # -- vs opponent --
    lines.append('## Performance vs Opponent\n')
    lines.append(f'| Metric | {la} | {lb} |')
    lines.append(f'|--------|------|------|')
    for key, fmt in [('win_rate', '.1%'), ('resign_rate', '.1%'),
                     ('avg_game_length', '.1f'), ('avg_win_length', '.1f'),
                     ('avg_loss_length', '.1f'), ('entropy_overall', '.3f')]:
        va = format(stats_a[key], fmt)
        vb = format(stats_b[key], fmt)
        lines.append(f'| {key} | {va} | {vb} |')

    # -- entropy by phase --
    lines.append('\n## Entropy by Game Phase\n')
    lines.append(f'| Phase | {la} | {lb} |')
    lines.append(f'|-------|------|------|')
    for phase in ['opening', 'midgame', 'endgame']:
        ea = f"{stats_a['entropy_by_phase'][phase]:.3f}"
        eb = f"{stats_b['entropy_by_phase'][phase]:.3f}"
        lines.append(f'| {phase} | {ea} | {eb} |')

    # -- head-to-head --
    if h2h_stats:
        lines.append('\n## Head-to-Head Policy Comparison\n')
        lines.append(f'- Positions analyzed: {h2h_stats["total_positions"]}')
        lines.append(f'- Overall move agreement: '
                     f'{h2h_stats["agreement_rate"]:.1%}')
        lines.append(f'- Mean symmetric KL divergence: '
                     f'{h2h_stats["mean_kl_divergence"]:.4f}')

        lines.append(f'\n| Phase | Agreement | KL Div | Entropy {la} | Entropy {lb} |')
        lines.append(f'|-------|-----------|--------|-------------|-------------|')
        for phase in ['opening', 'midgame', 'endgame']:
            ag = f"{h2h_stats['agreement_by_phase'][phase]:.1%}"
            kl = f"{h2h_stats['kl_by_phase'][phase]:.4f}"
            ea = f"{h2h_stats['entropy_a_by_phase'][phase]:.3f}"
            eb = f"{h2h_stats['entropy_b_by_phase'][phase]:.3f}"
            lines.append(f'| {phase} | {ag} | {kl} | {ea} | {eb} |')

    lines.append('')
    os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'[analyze] {path} saved.')


def _play_series(agent, opponent, board_size, num_games, label):
    """Play a series of games, restarting the GTP opponent on crash."""
    from alphigo.agent.gtp_agent import GtpAgent
    records = []
    for i in range(num_games):
        color = 'black' if i % 2 == 0 else 'white'
        print(f'  Game {i+1}/{num_games} ({label} as {color}) ... ',
              end='', flush=True)
        t0 = time.time()
        try:
            rec = play_and_record(agent, opponent, board_size,
                                  agent_color=color)
        except RuntimeError as exc:
            if isinstance(opponent, GtpAgent):
                print(f'GTP crash, restarting: {exc}')
                opponent.restart()
                t0 = time.time()
                rec = play_and_record(agent, opponent, board_size,
                                      agent_color=color)
            else:
                raise
        elapsed = time.time() - t0
        records.append(rec)
        print(f'{rec["winner"]} ({rec["reason"]}, {rec["num_moves"]} moves, '
              f'{elapsed:.1f}s)')
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _global_records_a, _global_records_b

    parser = argparse.ArgumentParser(
        description='Compare playing styles of two DL agents')
    parser.add_argument('--agent-a', required=True,
                        help='Path to first agent .h5 checkpoint')
    parser.add_argument('--agent-b', required=True,
                        help='Path to second agent .h5 checkpoint')
    parser.add_argument('--opponent', default='gnugo:level=1',
                        help='GTP opponent spec (default: gnugo:level=1)')
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--num-games', type=int, default=30,
                        help='Games per agent vs opponent (default: 30)')
    parser.add_argument('--h2h-games', type=int, default=20,
                        help='Head-to-head games between A and B (default: 20)')
    parser.add_argument('--output', default='results/style_analysis',
                        help='Output directory')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature for vs-opponent games (0=greedy)')
    parser.add_argument('--h2h-temperature', type=float, default=0.5,
                        help='Sampling temperature for head-to-head games '
                             '(default: 0.5, must be >0 for game diversity)')
    args = parser.parse_args()

    # -- Load agents --
    import h5py
    from alphigo.agent.deep_learning import load_prediction_agent
    from alphigo.agent.registry import create_agent
    from run import parse_agent_spec

    def _load(spec, temperature):
        # Raw .h5 path → DL agent. spec like "pg:agent_file=..." → registry.
        if ':' in spec and '=' in spec:
            name, kwargs = parse_agent_spec(spec)
            ag = create_agent(name, **kwargs)
        else:
            with h5py.File(spec, 'r') as h5:
                ag = load_prediction_agent(h5)
        # All agents in scope expose set_temperature (alias to set_epsilon
        # for PG/Q/AC); fall back to direct attribute for DL.
        if hasattr(ag, 'set_temperature'):
            ag.set_temperature(float(temperature))
        ag.temperature = float(temperature)  # used by h2h block
        # Pull a label from agent_file= if present, else from spec text.
        label_src = spec
        if '=' in spec:
            for part in spec.split(':', 1)[1].split(','):
                if part.startswith('agent_file='):
                    label_src = part.split('=', 1)[1]
                    break
        return ag, _label_from_path(label_src)

    print(f'Loading agent A: {args.agent_a}')
    agent_a, label_a = _load(args.agent_a, args.temperature)

    print(f'Loading agent B: {args.agent_b}')
    agent_b, label_b = _load(args.agent_b, args.temperature)

    opp_name, opp_kwargs = parse_agent_spec(args.opponent)
    print(f'Loading opponent: {args.opponent}')
    opponent_a = create_agent(opp_name, **opp_kwargs)
    # Need a separate opponent instance for agent B (fresh GTP process)
    opponent_b = create_agent(opp_name, **opp_kwargs)

    bs = args.board_size

    # -- Play agent A vs opponent --
    print(f'\n=== {label_a} vs {args.opponent} ({args.num_games} games) ===')
    records_a = _play_series(agent_a, opponent_a, bs, args.num_games, label_a)

    # -- Play agent B vs opponent --
    print(f'\n=== {label_b} vs {args.opponent} ({args.num_games} games) ===')
    records_b = _play_series(agent_b, opponent_b, bs, args.num_games, label_b)

    _global_records_a = records_a
    _global_records_b = records_b

    # -- Head-to-head --
    print(f'\n=== {label_a} vs {label_b} ({args.h2h_games} games) ===')
    h2h_records = []
    for i in range(args.h2h_games):
        # Alternate who plays black
        if i % 2 == 0:
            a, b = agent_a, agent_b
            tag = f'{label_a}(B) vs {label_b}(W)'
        else:
            a, b = agent_b, agent_a
            tag = f'{label_b}(B) vs {label_a}(W)'
        print(f'  Game {i+1}/{args.h2h_games}: {tag} ... ', end='', flush=True)
        t0 = time.time()
        rec = play_head_to_head(a, b, bs,
                               h2h_temperature=args.h2h_temperature)
        elapsed = time.time() - t0
        h2h_records.append(rec)
        print(f'winner={"a" if rec["winner"] == "a" else "b" if rec["winner"] == "b" else "draw"} '
              f'({rec["num_moves"]} moves, {elapsed:.1f}s)')

    # -- Analyze --
    print('\nAnalyzing ...')
    stats_a = analyze_vs_opponent(records_a, label_a, bs)
    stats_b = analyze_vs_opponent(records_b, label_b, bs)
    h2h_stats = analyze_head_to_head(h2h_records, bs) if h2h_records else None

    # -- Output --
    os.makedirs(args.output, exist_ok=True)
    write_report(stats_a, stats_b, h2h_stats, args.output)
    generate_figures(stats_a, stats_b, h2h_stats, bs, args.output)

    # Save raw stats (without numpy arrays) for later use
    def _serialize(stats):
        s = dict(stats)
        for k in ('move_heatmap', 'win_heatmap', 'loss_heatmap'):
            s[k] = s[k].tolist()
        return s

    summary = {
        'agent_a': _serialize(stats_a),
        'agent_b': _serialize(stats_b),
        'head_to_head': {
            k: v for k, v in (h2h_stats or {}).items()
            if k != 'top_disagreements'
        },
        'config': {
            'agent_a_path': args.agent_a,
            'agent_b_path': args.agent_b,
            'opponent': args.opponent,
            'board_size': bs,
            'num_games': args.num_games,
            'h2h_games': args.h2h_games,
            'temperature': args.temperature,
        },
    }
    json_path = os.path.join(args.output, 'style_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[analyze] {json_path} saved.')

    # -- Print summary --
    print(f'\n{"=" * 60}')
    print(f'STYLE ANALYSIS SUMMARY')
    print(f'{"=" * 60}')
    print(f'  {label_a}: {stats_a["wins"]}W/{stats_a["losses"]}L '
          f'({stats_a["win_rate"]:.0%}) vs {args.opponent}')
    print(f'  {label_b}: {stats_b["wins"]}W/{stats_b["losses"]}L '
          f'({stats_b["win_rate"]:.0%}) vs {args.opponent}')
    print(f'  Avg entropy: {label_a}={stats_a["entropy_overall"]:.3f}, '
          f'{label_b}={stats_b["entropy_overall"]:.3f}')
    if h2h_stats:
        print(f'  Move agreement: {h2h_stats["agreement_rate"]:.1%}')
        print(f'  Mean KL divergence: {h2h_stats["mean_kl_divergence"]:.4f}')
    print(f'  Output: {args.output}/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
