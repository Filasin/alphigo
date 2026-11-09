#!/usr/bin/env python
"""Generate thesis figures from bench JSONL logs.

Reads `logs/phase_a_bench.jsonl` (one JSON record per line: method, seed,
round, opponent, pct) and produces:

  docs/figures/fig1_learning_curves.png    -- rounds × win-rate vs Random
                                              one panel per method, seed bands
  docs/figures/fig4_strength_bar.png       -- bar chart of round-50 win-rate
                                              vs Random + mcts100, mean ± std

Phase B/C produce additional figures via this same script (extended later).
"""

import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


JSONL_PATH = 'logs/phase_a_bench.jsonl'
FIGURES_DIR = 'docs/figures'

METHOD_COLORS = {
    'q':  '#2ca02c',  # green
    'pg': '#1f77b4',  # blue
    'ac': '#d62728',  # red
}
METHOD_LABELS = {
    'q':  'Q (book Ch 11)',
    'pg': 'PG (REINFORCE)',
    'ac': 'AC (book Ch 12)',
}


def load_records(path=JSONL_PATH):
    records = []
    if not os.path.exists(path):
        print(f'[error] {path} not found — run scripts/phase_a_bench.sh first',
              file=sys.stderr)
        sys.exit(1)
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'[warn] skipping malformed line: {line[:80]} ({e})',
                      file=sys.stderr)
    return records


def _seed_from_label(label):
    m = re.search(r'seed\s*(\d+)', label)
    return int(m.group(1)) if m else 1


def fig1_learning_curves(records, out_path):
    """Per-method learning curve, mean ± std across seeds, vs Random."""
    by_method = defaultdict(lambda: defaultdict(list))   # method -> round -> list
    for r in records:
        if r['opponent'] != 'random':
            continue
        by_method[r['method']][int(r['round'])].append(r['pct'])

    if not by_method:
        print('[warn] no random-opponent records found', file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    for method, rounds_dict in sorted(by_method.items()):
        rounds = sorted(rounds_dict.keys())
        means = np.array([np.mean(rounds_dict[r]) for r in rounds])
        stds  = np.array([np.std(rounds_dict[r])  for r in rounds])
        ax.plot(rounds, means, marker='o', linewidth=2.0,
                color=METHOD_COLORS.get(method, 'gray'),
                label=f'{METHOD_LABELS.get(method, method)}  '
                      f'(n={len(rounds_dict[rounds[0]])} seeds)')
        ax.fill_between(rounds, means - stds, means + stds, alpha=0.18,
                        color=METHOD_COLORS.get(method, 'gray'))
    ax.axhline(50, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('RL training round (1000 self-play games per round)')
    ax.set_ylabel('win-rate vs Random  (%, N=100 per checkpoint)')
    ax.set_ylim(0, 100)
    ax.set_title('Phase A learning curves: Q / PG / AC at 50×1000 budget on 9×9')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[ok] wrote {out_path}')


def fig4_strength_bar(records, out_path):
    """Bar chart of method strength at the final round, vs Random + mcts-100."""
    final_round = max((int(r['round']) for r in records), default=0)
    if final_round == 0:
        return

    by_method_opp = defaultdict(lambda: defaultdict(list))  # method -> opp -> [pct, ...]
    for r in records:
        if int(r['round']) != final_round:
            continue
        by_method_opp[r['method']][r['opponent']].append(r['pct'])

    methods = sorted(by_method_opp.keys())
    if not methods:
        return
    opponents = ['random', 'mcts100']

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for i, opp in enumerate(opponents):
        means = [np.mean(by_method_opp[m].get(opp, [0])) for m in methods]
        stds  = [np.std(by_method_opp[m].get(opp, [0]))  for m in methods]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=f'vs {opp}', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_ylabel(f'win-rate at round {final_round}  (%)')
    ax.set_title(f'Final-round strength (Phase A, mean ± std across seeds)')
    ax.set_ylim(0, 100)
    ax.axhline(50, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.legend(frameon=False)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[ok] wrote {out_path}')


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    records = load_records()
    print(f'[info] loaded {len(records)} bench records')

    fig1_learning_curves(records, os.path.join(FIGURES_DIR,
                                                'fig1_learning_curves.png'))
    fig4_strength_bar(records, os.path.join(FIGURES_DIR,
                                             'fig4_strength_bar.png'))


if __name__ == '__main__':
    main()
