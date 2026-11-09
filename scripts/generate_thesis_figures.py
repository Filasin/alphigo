"""Generate thesis figures from experiment CSV data.

Usage (from WSL):
    python scripts/generate_thesis_figures.py

Outputs to figures/.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS = 'results'
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (7, 4.5),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def fig_collapse_allchunks():
    """Figure: sims sweep showing monotonic collapse (all-chunks value head)."""
    rows = read_csv(f'{RESULTS}/sims_sweep_b19_allchunks.csv')
    sims = [int(r['sims']) for r in rows]
    win = [float(r['win_pct']) for r in rows]
    lo = [float(r['wilson_ci_low']) for r in rows]
    hi = [float(r['wilson_ci_high']) for r in rows]
    err_lo = [max(0, w - l) for w, l in zip(win, lo)]
    err_hi = [max(0, h - w) for h, w in zip(win, hi)]

    fig, ax = plt.subplots()
    ax.errorbar(sims, win, yerr=[err_lo, err_hi], fmt='o-',
                capsize=4, color='#d62728', linewidth=2, markersize=7,
                label='Joint model (all-chunks)')
    ax.set_xscale('log')
    ax.set_xlabel('MCTS Simulations')
    ax.set_ylabel('Win Rate vs GNU Go L5 (%)')
    ax.set_title('Value-Head Collapse: More Search = Worse Play')
    ax.set_ylim(0, 100)
    ax.axhline(50, ls='--', color='gray', alpha=0.5, label='Break-even')
    ax.set_xticks(sims)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{OUT}/collapse_allchunks.pdf')
    fig.savefig(f'{OUT}/collapse_allchunks.png')
    plt.close(fig)
    print(f'  wrote collapse_allchunks')


def fig_mcts_correctness():
    """Figure: value_mode comparison (network vs zero vs rollout)."""
    data = {
        'network':  {'sims': [1, 25, 50, 200], 'win': [93, 73, 73, 40]},
        'zero':     {'sims': [1, 25, 50, 200], 'win': [100, 80, 67, 73]},
        'rollout':  {'sims': [1, 25, 50, 200], 'win': [87, 53, 46, 53]},
    }
    colors = {'network': '#d62728', 'zero': '#2ca02c', 'rollout': '#1f77b4'}
    labels = {'network': 'Trained value (collapse)',
              'zero': 'No value (flat — MCTS correct)',
              'rollout': 'Random rollout'}

    fig, ax = plt.subplots()
    for mode, d in data.items():
        ax.plot(d['sims'], d['win'], 'o-', color=colors[mode],
                label=labels[mode], linewidth=2, markersize=7)
    ax.set_xscale('log')
    ax.set_xlabel('MCTS Simulations')
    ax.set_ylabel('Win Rate vs GNU Go L5 (%)')
    ax.set_title('MCTS Correctness: Degradation Is the Value Network')
    ax.set_ylim(0, 105)
    ax.axhline(50, ls='--', color='gray', alpha=0.5)
    ax.set_xticks([1, 25, 50, 200])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{OUT}/mcts_correctness.pdf')
    fig.savefig(f'{OUT}/mcts_correctness.png')
    plt.close(fig)
    print(f'  wrote mcts_correctness')


def fig_seed_variance():
    """Figure: 5-seed sims sweep showing seed fragility."""
    seeds = {
        1: {'sims': [1, 25, 50, 200], 'win': [76.7, 80.0, 83.3, 40.0], 'cat': 'collapse'},
        2: {'sims': [1, 25, 50, 200], 'win': [83.3, 93.3, 86.7, 70.0], 'cat': 'hold'},
        3: {'sims': [1, 25, 50, 200], 'win': [86.7, 83.3, 86.7, 66.7], 'cat': 'hold'},
        4: {'sims': [1, 25, 50, 200], 'win': [83.3, 70.0, 66.7, 53.3], 'cat': 'collapse'},
        5: {'sims': [1, 25, 50, 200], 'win': [86.7, 80.0, 73.3, 56.7], 'cat': 'collapse'},
    }
    cat_colors = {'collapse': '#d62728', 'hold': '#2ca02c'}
    cat_styles = {'collapse': '--', 'hold': '-'}

    fig, ax = plt.subplots()
    for seed, d in seeds.items():
        cat = d['cat']
        ax.plot(d['sims'], d['win'], f'o{cat_styles[cat]}',
                color=cat_colors[cat], alpha=0.7, linewidth=1.5,
                markersize=6, label=f'Seed {seed} ({cat})')
    ax.set_xscale('log')
    ax.set_xlabel('MCTS Simulations')
    ax.set_ylabel('Win Rate vs GNU Go L5 (%)')
    ax.set_title('Seed Fragility: 3/5 Seeds Collapse at sims=200')
    ax.set_ylim(20, 100)
    ax.axhline(50, ls=':', color='gray', alpha=0.5)
    ax.set_xticks([1, 25, 50, 200])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{OUT}/seed_variance.pdf')
    fig.savefig(f'{OUT}/seed_variance.png')
    plt.close(fig)
    print(f'  wrote seed_variance')


def fig_separate_vs_joint():
    """Figure: separate value net vs joint model comparison."""
    sims = [1, 25, 50, 200, 800]
    separate = [100.0, 90.0, 83.3, 88.5, 46.7]
    joint_s1 = [76.7, 80.0, 83.3, 40.0, 32.0]
    zero = [100, 80, 67, 73, None]

    fig, ax = plt.subplots()
    ax.plot(sims, separate, 'o-', color='#2ca02c', linewidth=2,
            markersize=7, label='Separate value net')
    ax.plot(sims, joint_s1, 's--', color='#d62728', linewidth=2,
            markersize=7, label='Joint model (seed-1)')
    # zero only has 4 points
    ax.plot([1, 25, 50, 200], zero[:4], '^:', color='#1f77b4',
            linewidth=1.5, markersize=6, label='No value (zero)')
    ax.set_xscale('log')
    ax.set_xlabel('MCTS Simulations')
    ax.set_ylabel('Win Rate vs GNU Go L5 (%)')
    ax.set_title('Architectural Separation Delays Collapse')
    ax.set_ylim(0, 105)
    ax.axhline(50, ls='--', color='gray', alpha=0.4)
    ax.set_xticks(sims)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate the key comparison
    ax.annotate('48 pp\nimprovement',
                xy=(200, 88.5), xytext=(350, 65),
                arrowprops=dict(arrowstyle='->', color='#2ca02c'),
                fontsize=9, color='#2ca02c', ha='center')
    ax.annotate('', xy=(200, 40), xytext=(350, 62),
                arrowprops=dict(arrowstyle='->', color='#d62728'))

    fig.savefig(f'{OUT}/separate_vs_joint.pdf')
    fig.savefig(f'{OUT}/separate_vs_joint.png')
    plt.close(fig)
    print(f'  wrote separate_vs_joint')


def fig_sl_accuracy():
    """Figure: SL validation accuracy by encoder/network (19x19)."""
    configs = [
        ('alphazero/alphazero', 45.5),
        ('agplane/large', 40.6),
        ('agplane/medium', 39.0),
        ('sevenplane/large', 33.7),
        ('elevenplane/large', 33.5),
        ('sevenplane/medium', 32.4),
        ('elevenplane/medium', 31.1),
        ('agplane/small', 23.7),
        ('elevenplane/small', 13.0),
        ('sevenplane/small', 12.1),
        ('oneplane/large', 6.2),
        ('oneplane/small', 2.1),
        ('oneplane/medium', 1.0),
    ]
    names = [c[0] for c in configs]
    accs = [c[1] for c in configs]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ca02c' if 'alphazero' in n else
              '#1f77b4' if 'agplane' in n else
              '#ff7f0e' if 'sevenplane' in n else
              '#9467bd' if 'elevenplane' in n else
              '#d62728' for n in names]
    bars = ax.barh(range(len(names)), accs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Validation Accuracy (%)')
    ax.set_title('19×19 Supervised Learning: Validation Accuracy')
    ax.invert_yaxis()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    fig.savefig(f'{OUT}/sl_accuracy.pdf')
    fig.savefig(f'{OUT}/sl_accuracy.png')
    plt.close(fig)
    print(f'  wrote sl_accuracy')


def fig_diagnostic_chain():
    """Figure: diagnostic chain summary (bar chart of sims=200 results)."""
    experiments = ['Joint\n(seed-1)', 'Joint\n(5-seed avg)', 'No value\n(zero)',
                   'Separate\nvalue net']
    values = [40.0, 57.3, 73.0, 88.5]
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(experiments, values, color=colors, width=0.6, alpha=0.85)
    ax.set_ylabel('Win Rate at sims=200 (%)')
    ax.set_title('Diagnostic Chain: Progressive Mitigation of Collapse')
    ax.set_ylim(0, 100)
    ax.axhline(50, ls='--', color='gray', alpha=0.5, label='Break-even')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    fig.savefig(f'{OUT}/diagnostic_chain.pdf')
    fig.savefig(f'{OUT}/diagnostic_chain.png')
    plt.close(fig)
    print(f'  wrote diagnostic_chain')


if __name__ == '__main__':
    print('Generating thesis figures...')
    fig_collapse_allchunks()
    fig_mcts_correctness()
    fig_seed_variance()
    fig_separate_vs_joint()
    fig_sl_accuracy()
    fig_diagnostic_chain()
    print(f'Done. {len(os.listdir(OUT))} files in {OUT}/')
