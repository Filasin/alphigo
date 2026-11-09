"""Generate thesis figures from the bootstrapped self-play experiment.

Produces a combined figure with:
  (a) Win rate vs Random, GnuGo L1, GnuGo L5 per round
  (b) Policy, value, ownership loss per round
  (c) First-move entropy per round

All panels include 80K bootstrap baseline as dashed lines and
markers indicating promotion vs rejection.

Usage:
  python scripts/plot_selfplay_experiment.py \
      [--csv results/selfplay_bootstrap_loop_b9.csv]
"""
import argparse
import sys
import os

import numpy as np
import matplotlib.pyplot as plt


BASELINE = {
    'policy_loss': 3.7479,
    'value_loss': 0.5868,
    'ownership_loss': 0.5561,
    'vs_random': 100.0,
    'vs_gnugo1': 0.0,
    'vs_gnugo5': 0.0,
    'entropy': 4.3924,
}
UNIFORM_ENTROPY = np.log(81)  # 4.394


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results/selfplay_bootstrap_loop_b9.csv')
    parser.add_argument('--out-dir', default='results/figures')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.genfromtxt(args.csv, delimiter=',', names=True, dtype=None,
                         encoding='utf-8')

    rounds = data['round']
    policy_loss = data['policy_loss']
    value_loss = data['value_loss']
    ownership_loss = data['ownership_loss']
    vs_random = data['vs_random']
    vs_gnugo1 = data['vs_gnugo1']
    vs_gnugo5 = data['vs_gnugo5']
    entropy = data['entropy']
    promoted = data['promoted']

    promoted_mask = np.array([p == 'yes' for p in promoted])
    rejected_mask = np.array([p == 'no' for p in promoted])

    # ── Combined Figure (for thesis) ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Win rate
    ax = axes[0]
    valid = vs_random >= 0
    if np.any(valid):
        ax.plot(rounds[valid], vs_random[valid], 'b-o',
                label='vs Random', linewidth=2, markersize=5)
    valid = vs_gnugo1 >= 0
    if np.any(valid):
        ax.plot(rounds[valid], vs_gnugo1[valid], 'r-s',
                label='vs GnuGo L1', linewidth=2, markersize=5)
    valid = vs_gnugo5 >= 0
    if np.any(valid):
        ax.plot(rounds[valid], vs_gnugo5[valid], 'g-^',
                label='vs GnuGo L5', linewidth=2, markersize=5)
    ax.axhline(y=BASELINE['vs_random'], color='b', linestyle='--', alpha=0.3)
    ax.axhline(y=BASELINE['vs_gnugo1'], color='r', linestyle='--', alpha=0.3)
    for r in rounds[promoted_mask]:
        ax.axvline(x=r, color='green', alpha=0.15, linewidth=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('(a) Win Rate')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Loss
    ax = axes[1]
    valid_loss = policy_loss >= 0
    if np.any(valid_loss):
        ax.plot(rounds[valid_loss], policy_loss[valid_loss], 'b-o',
                label='Policy', linewidth=2, markersize=5)
        ax.plot(rounds[valid_loss], value_loss[valid_loss], 'r-s',
                label='Value', linewidth=2, markersize=5)
        ax.plot(rounds[valid_loss], ownership_loss[valid_loss], 'g-^',
                label='Ownership', linewidth=2, markersize=5)
    ax.axhline(y=BASELINE['policy_loss'], color='b', linestyle='--',
               alpha=0.4, label=f'Baseline pol ({BASELINE["policy_loss"]:.2f})')
    ax.axhline(y=BASELINE['value_loss'], color='r', linestyle='--', alpha=0.4)
    ax.axhline(y=BASELINE['ownership_loss'], color='g', linestyle='--', alpha=0.4)
    for r in rounds[promoted_mask]:
        ax.axvline(x=r, color='green', alpha=0.15, linewidth=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Val Loss')
    ax.set_title('(b) Validation Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Entropy
    ax = axes[2]
    valid_ent = entropy > 0
    if np.any(valid_ent):
        ax.plot(rounds[valid_ent], entropy[valid_ent], 'purple',
                marker='D', linewidth=2, markersize=5, label='Model')
    ax.axhline(y=UNIFORM_ENTROPY, color='gray', linestyle='--', alpha=0.4,
               label=f'Uniform ({UNIFORM_ENTROPY:.2f})')
    ax.axhline(y=BASELINE['entropy'], color='purple', linestyle='--',
               alpha=0.4, label=f'Baseline ({BASELINE["entropy"]:.2f})')
    for r in rounds[promoted_mask]:
        ax.axvline(x=r, color='green', alpha=0.15, linewidth=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('(c) First-Move Entropy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Bootstrapped Self-Play Experiment — 9×9 Go, sims=800',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(args.out_dir, 'selfplay_bootstrap_combined.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close()

    # ── Print summary table ─────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'Round':>5} {'Policy':>8} {'Value':>8} {'Own':>8} "
          f"{'vsRand':>8} {'vsGL1':>8} {'vsGL5':>8} "
          f"{'Entropy':>8} {'H2H%':>6} {'Promo':>6}")
    print("-" * 88)
    for i in range(len(rounds)):
        print(f"{rounds[i]:>5} {policy_loss[i]:>8.3f} {value_loss[i]:>8.3f} "
              f"{ownership_loss[i]:>8.3f} {vs_random[i]:>7.1f}% "
              f"{vs_gnugo1[i]:>7.1f}% {vs_gnugo5[i]:>7.1f}% "
              f"{entropy[i]:>8.3f} {data['h2h_winpct'][i]:>5}% "
              f"{promoted[i]:>6}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
