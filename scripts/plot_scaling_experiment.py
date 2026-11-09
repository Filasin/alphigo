"""Generate thesis figures from the data scaling experiment.

Produces:
  1. Sample efficiency curve: win rate vs number of training games
  2. Loss curves: policy/value/ownership loss vs data volume
  3. Policy entropy decay: entropy on first-move position vs data volume

Usage:
  python scripts/plot_scaling_experiment.py [--csv results/scaling_experiment_b9.csv]
"""
import argparse
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def thousands_formatter(x, pos):
    if x >= 1000:
        return f'{x/1000:.0f}K'
    return f'{x:.0f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results/scaling_experiment_b9.csv')
    parser.add_argument('--out-dir', default='results/figures')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run the scaling experiment first.",
              file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    # Format: games,positions,chunks_used,policy_loss,value_loss,ownership_loss,total_loss,vs_random_winpct,vs_gnugo_winpct,policy_entropy_first_move
    data = np.genfromtxt(args.csv, delimiter=',', names=True, dtype=None,
                         encoding='utf-8')

    games = data['games']
    policy_loss = data['policy_loss']
    value_loss = data['value_loss']
    ownership_loss = data['ownership_loss']
    vs_random = data['vs_random_winpct']
    vs_gnugo1 = data['vs_gnugo1_winpct']
    vs_gnugo5 = data['vs_gnugo5_winpct']
    entropy = data['policy_entropy_first_move']

    # ── Figure 1: Sample Efficiency Curve ───────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(games, vs_random, 'b-o', label='vs Random Bot', linewidth=2, markersize=8)
    if np.any(vs_gnugo1 >= 0):
        mask = vs_gnugo1 >= 0
        ax.plot(games[mask], vs_gnugo1[mask], 'r-s', label='vs GnuGo L1',
                linewidth=2, markersize=8)
    if np.any(vs_gnugo5 >= 0):
        mask = vs_gnugo5 >= 0
        ax.plot(games[mask], vs_gnugo5[mask], 'g-^', label='vs GnuGo L5',
                linewidth=2, markersize=8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.set_xlabel('Training Games (rollout MCTS, 800 sims/move)', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Sample Efficiency: Win Rate vs Training Data Volume\n'
                 '(9×9, 3 res blocks, 64 filters, pure policy)', fontsize=13)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out_dir, 'scaling_win_rate.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close()

    # ── Figure 2: Loss Curves ───────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(games, policy_loss, 'b-o', label='Policy (CE)', linewidth=2)
    ax.plot(games, value_loss, 'r-s', label='Value (MSE)', linewidth=2)
    ax.plot(games, ownership_loss, 'g-^', label='Ownership (MSE)', linewidth=2)
    ax.set_xlabel('Training Games', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss vs Training Data Volume\n'
                 '(held-out chunk, same distribution)', fontsize=13)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out_dir, 'scaling_loss_curves.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close()

    # ── Figure 3: Policy Entropy Decay ──────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    uniform_entropy = np.log(81)  # 4.394 for 9x9
    ax.axhline(y=uniform_entropy, color='gray', linestyle='--', alpha=0.5,
               label=f'Uniform random ({uniform_entropy:.2f})')
    ax.plot(games, entropy, 'purple', marker='D', linewidth=2, markersize=8,
            label='Trained model')
    ax.set_xlabel('Training Games', fontsize=12)
    ax.set_ylabel('Policy Entropy (nats)', fontsize=12)
    ax.set_title('First-Move Policy Entropy vs Training Data Volume\n'
                 '(lower = more confident opening preferences)', fontsize=13)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out_dir, 'scaling_entropy.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close()

    # ── Figure 4: Combined (for thesis single-figure) ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Win rate
    ax = axes[0]
    ax.plot(games, vs_random, 'b-o', label='vs Random', linewidth=2, markersize=6)
    if np.any(vs_gnugo1 >= 0):
        mask = vs_gnugo1 >= 0
        ax.plot(games[mask], vs_gnugo1[mask], 'r-s', label='vs GnuGo L1',
                linewidth=2, markersize=6)
    if np.any(vs_gnugo5 >= 0):
        mask = vs_gnugo5 >= 0
        ax.plot(games[mask], vs_gnugo5[mask], 'g-^', label='vs GnuGo L5',
                linewidth=2, markersize=6)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel('Training Games')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('(a) Sample Efficiency')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Loss
    ax = axes[1]
    ax.plot(games, policy_loss, 'b-o', label='Policy', linewidth=2, markersize=6)
    ax.plot(games, value_loss, 'r-s', label='Value', linewidth=2, markersize=6)
    ax.plot(games, ownership_loss, 'g-^', label='Ownership', linewidth=2, markersize=6)
    ax.set_xlabel('Training Games')
    ax.set_ylabel('Val Loss')
    ax.set_title('(b) Validation Loss')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Entropy
    ax = axes[2]
    ax.axhline(y=uniform_entropy, color='gray', linestyle='--', alpha=0.4,
               label='Uniform')
    ax.plot(games, entropy, 'purple', marker='D', linewidth=2, markersize=6,
            label='Model')
    ax.set_xlabel('Training Games')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('(c) First-Move Entropy')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Data Scaling Experiment — 9×9 Go, Rollout MCTS Bootstrap',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(args.out_dir, 'scaling_combined.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close()

    # ── Print summary table ─────────────────────────────────────────
    print("\n=== Summary Table ===")
    print(f"{'Games':>8} {'Policy':>8} {'Value':>8} {'Own':>8} {'vsRand':>8} {'vsGnuL1':>8} {'vsGnuL5':>8} {'Entropy':>8}")
    print("-" * 72)
    for i in range(len(games)):
        print(f"{games[i]:>8} {policy_loss[i]:>8.3f} {value_loss[i]:>8.3f} "
              f"{ownership_loss[i]:>8.3f} {vs_random[i]:>7.1f}% "
              f"{vs_gnugo1[i]:>7.1f}% {vs_gnugo5[i]:>7.1f}% {entropy[i]:>8.3f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
