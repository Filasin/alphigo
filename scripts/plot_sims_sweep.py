"""Plot the sims-sweep CSV as a publishable figure.

Reads results/sims_sweep_*.csv (sims, n_games, wins, win_pct,
wilson_ci_low, wilson_ci_high, wall_seconds) and produces a log-x
win-rate-vs-sims plot with Wilson 95% CI error bars.

Usage:
  python scripts/plot_sims_sweep.py [csv_path] [--out OUT.png]

The default csv is results/sims_sweep_r0_value_trained.csv. Adds
horizontal reference lines for the pure-policy baseline (20% from
bootstrap r0) and SL alone (17%), so the curve can be read against
"the best non-MCTS agent" without flipping pages.
"""
import argparse
import csv
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs', nargs='*',
                        default=['results/sims_sweep_r0_value_trained.csv'],
                        help='One or more CSVs; each plotted as its own curve.')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='Optional curve labels (one per csv; defaults to filenames).')
    parser.add_argument('--out', default='results/sims_sweep_plot.png',
                        help='Output PNG path.')
    parser.add_argument('--title',
                        default='AZ win-rate vs num_simulations (9x9, vs GnuGo level=1)')
    parser.add_argument('--opponent-label', default='GnuGo level=1')
    parser.add_argument('--baseline-pure-policy', type=float, default=20.0,
                        help='Reference horizontal line: pure-policy r0 (no MCTS).')
    parser.add_argument('--baseline-sl', type=float, default=17.0,
                        help='Reference horizontal line: SL alone.')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # Load all CSVs.
    curves = []
    for i, p in enumerate(args.csvs):
        cp = Path(p)
        if not cp.exists():
            print(f'ERROR: csv not found: {cp}', file=sys.stderr)
            return 1
        sims, pct, ci_lo, ci_hi = [], [], [], []
        with open(cp) as f:
            for row in csv.DictReader(f):
                sims.append(int(row['sims']))
                pct.append(float(row['win_pct']))
                ci_lo.append(float(row['wilson_ci_low']))
                ci_hi.append(float(row['wilson_ci_high']))
        label = (args.labels[i] if args.labels and i < len(args.labels)
                 else cp.stem)
        curves.append({
            'label': label,
            'sims': np.array(sims), 'pct': np.array(pct),
            'ci_lo': np.array(ci_lo), 'ci_hi': np.array(ci_hi),
        })

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Reference horizontal lines first so points sit on top.
    if args.baseline_pure_policy is not None:
        ax.axhline(args.baseline_pure_policy, ls='--', color='#1f77b4',
                   alpha=0.55, lw=1.2,
                   label=f'pure policy (no MCTS): {args.baseline_pure_policy:.0f}%')
    if args.baseline_sl is not None:
        ax.axhline(args.baseline_sl, ls=':', color='#888',
                   alpha=0.55, lw=1.2,
                   label=f'SL alone: {args.baseline_sl:.0f}%')

    palette = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    for i, c in enumerate(curves):
        color = palette[i % len(palette)]
        err_lo = c['pct'] - c['ci_lo']
        err_hi = c['ci_hi'] - c['pct']
        ax.errorbar(c['sims'], c['pct'], yerr=[err_lo, err_hi],
                    fmt='o-', color=color, lw=2, ms=7,
                    ecolor=color, elinewidth=1.2, capsize=4,
                    label=c['label'])
        for x, y in zip(c['sims'], c['pct']):
            ax.annotate(f'{y:.0f}%', (x, y),
                        textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=8, color=color)

    # X-axis log scale, ticks from union of sims.
    all_sims = sorted(set(int(s) for c in curves for s in c['sims']))
    ax.set_xscale('log')
    ax.set_xlabel('num_simulations (PUCT rollouts per move)')
    ax.set_ylabel(f'Win rate vs {args.opponent_label} (%)')
    ax.set_title(args.title, fontsize=11)
    ax.set_xticks(all_sims)
    ax.set_xticklabels([str(s) for s in all_sims])
    ymax = max((c['pct'].max() if len(c['pct']) else 0) for c in curves)
    ax.set_ylim(-3, max(40, ymax + 12))
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'[saved] {args.out}')

    print('\nSummary:')
    for c in curves:
        print(f'  {c["label"]}:')
        for s, p, lo, hi in zip(c['sims'], c['pct'], c['ci_lo'], c['ci_hi']):
            print(f'    sims={s:>3d}  {p:5.1f}%  [Wilson 95% CI: {lo:.1f}, {hi:.1f}]')


if __name__ == '__main__':
    sys.exit(main())
