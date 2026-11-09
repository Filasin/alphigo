#!/usr/bin/env python
"""thesis_report.py — generate tables and figures from experiments.jsonl.

Reads results/experiments.jsonl and produces:
  - CSV + LaTeX booktabs tables (encoder x network accuracy, Elo, optimizer ablation)
  - Matplotlib figures (learning curves, heatmaps, grouped bar charts)

Gracefully skips any figure/table if insufficient data exists.

Usage:
    python thesis_report.py
    python thesis_run.py --report   (delegates here)
"""

import json
import os
import sys
from collections import defaultdict

RESULTS_DIR = 'results'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'experiments.jsonl')
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')
FIGURE_DIR = os.path.join(RESULTS_DIR, 'figures')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiments():
    """Load all experiments from JSONL. Returns list of dicts."""
    if not os.path.exists(RESULTS_FILE):
        print(f'[report] {RESULTS_FILE} not found.')
        return []
    entries = []
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def filter_entries(entries, board_size=None, training_method=None):
    out = entries
    if board_size is not None:
        out = [e for e in out if e.get('board_size') == board_size]
    if training_method is not None:
        out = [e for e in out if e.get('training_method') == training_method]
    return out


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _latex_bold_best(values, fmt='.4f', higher_is_better=True):
    """Return list of formatted strings with best value bolded."""
    numeric = []
    for v in values:
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            numeric.append(None)

    if all(v is None for v in numeric):
        return [_fmt_val(v, fmt) for v in values]

    valid = [v for v in numeric if v is not None]
    if not valid:
        return [_fmt_val(v, fmt) for v in values]

    best = max(valid) if higher_is_better else min(valid)

    result = []
    for v in numeric:
        s = _fmt_val(v, fmt)
        if v is not None and abs(v - best) < 1e-9:
            s = f'\\textbf{{{s}}}'
        result.append(s)
    return result


def _fmt_val(v, fmt='.4f'):
    if v is None:
        return '--'
    return f'{v:{fmt}}'


def write_latex_table(filepath, headers, rows, caption, label,
                      bold_cols=None, higher_is_better=True, fmt='.4f'):
    """Write a LaTeX booktabs table."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Bold best per column if requested
    if bold_cols:
        # Transpose to column-wise
        cols_data = list(zip(*[r[1:] for r in rows]))  # skip row label
        formatted_cols = []
        for i, col in enumerate(cols_data):
            if i in bold_cols:
                formatted_cols.append(_latex_bold_best(col, fmt, higher_is_better))
            else:
                formatted_cols.append([_fmt_val(v, fmt) for v in col])
        # Transpose back
        for ri, row in enumerate(rows):
            for ci in range(len(formatted_cols)):
                rows[ri][ci + 1] = formatted_cols[ci][ri]

    col_spec = 'l' + 'r' * (len(headers) - 1)

    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
        ' & '.join(headers) + ' \\\\',
        '\\midrule',
    ]

    for row in rows:
        cells = [str(row[0])]  # row label
        for v in row[1:]:
            cells.append(str(v) if not isinstance(v, float) else _fmt_val(v, fmt))
        lines.append(' & '.join(cells) + ' \\\\')

    lines += [
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_csv(filepath, headers, rows):
    """Write a simple CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in rows:
            f.write(','.join(str(v) if v is not None else '' for v in row) + '\n')


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def gen_accuracy_table(entries, board_size):
    """Encoder x Network val_accuracy table."""
    data = filter_entries(entries, board_size=board_size, training_method='supervised')
    if not data:
        print(f'[report] No supervised data for {board_size}x{board_size} accuracy table.')
        return

    # Collect encoders and networks present
    encoders = sorted(set(e['encoder'] for e in data))
    networks = sorted(set(e['network'] for e in data))

    # Build lookup
    lookup = {}
    for e in data:
        lookup[(e['encoder'], e['network'])] = e.get('final_val_accuracy')

    headers = ['Encoder'] + networks
    rows = []
    for enc in encoders:
        row = [enc] + [lookup.get((enc, net)) for net in networks]
        rows.append(row)

    suffix = '' if board_size == 9 else f'_{board_size}x{board_size}'
    write_csv(os.path.join(TABLE_DIR, f'encoder_accuracy{suffix}.csv'), headers, rows)
    write_latex_table(
        os.path.join(TABLE_DIR, f'encoder_accuracy{suffix}.tex'),
        headers, [list(r) for r in rows],
        caption=f'Supervised learning validation accuracy ({board_size}x{board_size})',
        label=f'tab:encoder_accuracy{suffix}',
        bold_cols=list(range(len(networks))),
        higher_is_better=True,
    )
    print(f'[report] encoder_accuracy{suffix} table written.')


def gen_elo_table(entries, board_size):
    """Encoder x Network Elo table (vs first opponent found)."""
    data = filter_entries(entries, board_size=board_size, training_method='supervised')
    data = [e for e in data if e.get('eval')]
    if not data:
        print(f'[report] No eval data for {board_size}x{board_size} Elo table.')
        return

    # Find most common opponent
    opp_counts = defaultdict(int)
    for e in data:
        for opp in e['eval']:
            opp_counts[opp] += 1
    primary_opp = max(opp_counts, key=opp_counts.get)

    encoders = sorted(set(e['encoder'] for e in data))
    networks = sorted(set(e['network'] for e in data))

    lookup = {}
    for e in data:
        elo_info = e.get('eval', {}).get(primary_opp)
        if elo_info:
            lookup[(e['encoder'], e['network'])] = elo_info.get('elo_delta')

    headers = ['Encoder'] + networks
    rows = []
    for enc in encoders:
        row = [enc] + [lookup.get((enc, net)) for net in networks]
        rows.append(row)

    suffix = '' if board_size == 9 else f'_{board_size}x{board_size}'
    write_csv(os.path.join(TABLE_DIR, f'elo_results{suffix}.csv'), headers, rows)
    write_latex_table(
        os.path.join(TABLE_DIR, f'elo_results{suffix}.tex'),
        headers, [list(r) for r in rows],
        caption=f'Elo difference vs {primary_opp} ({board_size}x{board_size})',
        label=f'tab:elo_results{suffix}',
        bold_cols=list(range(len(networks))),
        higher_is_better=True,
        fmt='.0f',
    )
    print(f'[report] elo_results{suffix} table written.')


def gen_optimizer_ablation(entries):
    """Adam vs SGD comparison table."""
    data = filter_entries(entries, training_method='supervised')
    adam = {(e['board_size'], e['encoder'], e['network']): e
            for e in data if e.get('optimizer') == 'adam'}
    sgd = {(e['board_size'], e['encoder'], e['network']): e
           for e in data if e.get('optimizer') == 'sgd'}

    common = sorted(set(adam.keys()) & set(sgd.keys()))
    if not common:
        print('[report] No optimizer ablation data (need both adam and sgd runs).')
        return

    headers = ['Board', 'Encoder', 'Network', 'Adam acc', 'SGD acc', 'Adam loss', 'SGD loss']
    rows = []
    for key in common:
        a, s = adam[key], sgd[key]
        rows.append([
            f'{key[0]}x{key[0]}', key[1], key[2],
            a.get('final_val_accuracy'), s.get('final_val_accuracy'),
            a.get('final_val_loss'), s.get('final_val_loss'),
        ])

    write_csv(os.path.join(TABLE_DIR, 'optimizer_ablation.csv'), headers, rows)
    write_latex_table(
        os.path.join(TABLE_DIR, 'optimizer_ablation.tex'),
        headers, [list(r) for r in rows],
        caption='Optimizer ablation: Adam vs SGD',
        label='tab:optimizer_ablation',
        bold_cols=None,
    )
    print('[report] optimizer_ablation table written.')


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def _short_label(agent_spec):
    """Convert a dl:agent_file=checkpoints/... spec to encoder/network."""
    import re
    m = re.search(r'(?:supervised|mcts)_([^_]+)_([^_]+)_', agent_spec)
    if m:
        return f'{m.group(1)}/{m.group(2)}'
    if 'agent_file=' in agent_spec:
        path = agent_spec.split('agent_file=', 1)[1]
        return os.path.splitext(os.path.basename(path))[0][:24]
    return agent_spec[:24]


def _setup_plt():
    """Import matplotlib with non-interactive backend."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
        })
        return plt
    except ImportError:
        print('[report] matplotlib not installed — skipping figures.')
        return None


def gen_learning_curves(entries, board_size):
    """Val_loss per epoch for all combos."""
    plt = _setup_plt()
    if plt is None:
        return

    data = filter_entries(entries, board_size=board_size, training_method='supervised')
    data = [e for e in data if e.get('history', {}).get('val_loss')]
    if not data:
        print(f'[report] No history data for {board_size}x{board_size} learning curves.')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.tab10

    for i, entry in enumerate(data):
        label = f'{entry["encoder"]}/{entry["network"]}'
        color = cmap(i % 10)
        epochs = range(1, len(entry['history']['val_loss']) + 1)
        ax1.plot(epochs, entry['history']['val_loss'], label=label, color=color)
        if 'val_accuracy' in entry['history']:
            ax2.plot(epochs, entry['history']['val_accuracy'], label=label, color=color)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title(f'Learning Curves — Val Loss ({board_size}x{board_size})')
    ax1.legend(fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title(f'Learning Curves — Val Accuracy ({board_size}x{board_size})')
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    suffix = '' if board_size == 9 else f'_{board_size}x{board_size}'
    path = os.path.join(FIGURE_DIR, f'learning_curves{suffix}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[report] {path} saved.')


def gen_elo_heatmap(entries, board_size):
    """Seaborn heatmap of Elo, encoder x network."""
    plt = _setup_plt()
    if plt is None:
        return
    try:
        import seaborn as sns
    except ImportError:
        print('[report] seaborn not installed — skipping heatmap.')
        return
    import numpy as np

    data = filter_entries(entries, board_size=board_size, training_method='supervised')
    data = [e for e in data if e.get('eval')]
    if not data:
        print(f'[report] No eval data for {board_size}x{board_size} heatmap.')
        return

    # Use first opponent
    opp_counts = defaultdict(int)
    for e in data:
        for opp in e['eval']:
            opp_counts[opp] += 1
    primary_opp = max(opp_counts, key=opp_counts.get)

    encoders = sorted(set(e['encoder'] for e in data))
    networks = sorted(set(e['network'] for e in data))

    matrix = np.full((len(encoders), len(networks)), np.nan)
    for e in data:
        elo_info = e.get('eval', {}).get(primary_opp)
        if elo_info and elo_info.get('elo_delta') is not None:
            ei = encoders.index(e['encoder'])
            ni = networks.index(e['network'])
            matrix[ei, ni] = elo_info['elo_delta']

    if np.all(np.isnan(matrix)):
        print(f'[report] All NaN Elo values for {board_size}x{board_size} heatmap.')
        return

    fig, ax = plt.subplots(figsize=(max(8, len(networks) * 1.2), max(5, len(encoders) * 0.8)))
    sns.heatmap(matrix, annot=True, fmt='.0f', xticklabels=networks,
                yticklabels=encoders, cmap='RdYlGn', center=0, ax=ax,
                mask=np.isnan(matrix))
    ax.set_title(f'Elo vs {primary_opp} ({board_size}x{board_size})')
    ax.set_xlabel('Network')
    ax.set_ylabel('Encoder')
    fig.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    suffix = '' if board_size == 9 else f'_{board_size}x{board_size}'
    path = os.path.join(FIGURE_DIR, f'elo_heatmap{suffix}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[report] {path} saved.')


def gen_encoder_comparison(entries):
    """Grouped bar by encoder family (average across networks)."""
    plt = _setup_plt()
    if plt is None:
        return
    import numpy as np

    data = filter_entries(entries, training_method='supervised')
    data = [e for e in data if e.get('final_val_accuracy') is not None]
    if len(data) < 2:
        print('[report] Insufficient data for encoder comparison.')
        return

    # Group by (board_size, encoder) -> list of accuracies
    groups = defaultdict(list)
    for e in data:
        groups[(e['board_size'], e['encoder'])].append(e['final_val_accuracy'])

    board_sizes = sorted(set(k[0] for k in groups))
    encoders = sorted(set(k[1] for k in groups))

    fig, ax = plt.subplots(figsize=(max(8, len(encoders) * 1.5), 5))
    x = np.arange(len(encoders))
    width = 0.8 / max(len(board_sizes), 1)

    for i, bs in enumerate(board_sizes):
        means = [np.mean(groups.get((bs, enc), [np.nan])) for enc in encoders]
        ax.bar(x + i * width, means, width, label=f'{bs}x{bs}',
               color=plt.cm.tab10(i))

    ax.set_xlabel('Encoder')
    ax.set_ylabel('Mean Validation Accuracy')
    ax.set_title('Encoder Comparison (avg across networks)')
    ax.set_xticks(x + width * (len(board_sizes) - 1) / 2)
    ax.set_xticklabels(encoders, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, 'encoder_progression.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[report] {path} saved.')


def gen_network_comparison(entries):
    """Grouped bar by network depth (average across encoders)."""
    plt = _setup_plt()
    if plt is None:
        return
    import numpy as np

    data = filter_entries(entries, training_method='supervised')
    data = [e for e in data if e.get('final_val_accuracy') is not None]
    if len(data) < 2:
        print('[report] Insufficient data for network comparison.')
        return

    groups = defaultdict(list)
    for e in data:
        groups[(e['board_size'], e['network'])].append(e['final_val_accuracy'])

    board_sizes = sorted(set(k[0] for k in groups))
    networks = sorted(set(k[1] for k in groups))

    fig, ax = plt.subplots(figsize=(max(8, len(networks) * 1.5), 5))
    x = np.arange(len(networks))
    width = 0.8 / max(len(board_sizes), 1)

    for i, bs in enumerate(board_sizes):
        means = [np.mean(groups.get((bs, net), [np.nan])) for net in networks]
        ax.bar(x + i * width, means, width, label=f'{bs}x{bs}',
               color=plt.cm.tab10(i))

    ax.set_xlabel('Network')
    ax.set_ylabel('Mean Validation Accuracy')
    ax.set_title('Network Comparison (avg across encoders)')
    ax.set_xticks(x + width * (len(board_sizes) - 1) / 2)
    ax.set_xticklabels(networks, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, 'network_depth.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[report] {path} saved.')


def gen_params_vs_accuracy(entries):
    """Scatter plot: parameter count vs validation accuracy, one dot per combo."""
    plt = _setup_plt()
    if plt is None:
        return
    import numpy as np

    data = filter_entries(entries, training_method='supervised')
    data = [e for e in data if e.get('final_val_accuracy') is not None
            and e.get('total_params') is not None]
    if len(data) < 2:
        print('[report] Insufficient data for params_vs_accuracy scatter.')
        return

    board_sizes = sorted(set(e['board_size'] for e in data))
    markers = ['o', 's', '^', 'D', 'v', '*']
    encoder_list = sorted(set(e['encoder'] for e in data))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, bs in enumerate(board_sizes):
        bs_data = [e for e in data if e['board_size'] == bs]
        for j, enc in enumerate(encoder_list):
            enc_data = [e for e in bs_data if e['encoder'] == enc]
            if not enc_data:
                continue
            xs = [e['total_params'] for e in enc_data]
            ys = [e['final_val_accuracy'] for e in enc_data]
            labels = [e['network'] for e in enc_data]
            color = plt.cm.tab10(j)
            ax.scatter(xs, ys, marker=markers[i % len(markers)],
                       color=color, s=80,
                       label=f'{enc} ({bs}x{bs})')
            for x, y, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, (x, y), textcoords='offset points',
                            xytext=(4, 2), fontsize=7)

    # Highlight Pareto frontier (best accuracy for each param count threshold)
    if data:
        sorted_data = sorted(data, key=lambda e: e['total_params'])
        best_acc = -1
        pareto_x, pareto_y = [], []
        for e in sorted_data:
            if e['final_val_accuracy'] > best_acc:
                best_acc = e['final_val_accuracy']
                pareto_x.append(e['total_params'])
                pareto_y.append(e['final_val_accuracy'])
        ax.step(pareto_x, pareto_y, where='post',
                color='black', linewidth=1, linestyle='--',
                alpha=0.5, label='Pareto frontier')

    ax.set_xlabel('Total Parameters')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Parameter Efficiency: Accuracy vs Model Size')
    ax.legend(fontsize=7, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, 'params_vs_accuracy.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[report] {path} saved.')


def gen_tournament_figures():
    """Win-rate heatmap + standings chart from any tournament*.json in results/."""
    plt = _setup_plt()
    if plt is None:
        return
    try:
        import seaborn as sns
    except ImportError:
        print('[report] seaborn not installed — skipping tournament figures.')
        return
    import json
    import numpy as np

    if not os.path.isdir(RESULTS_DIR):
        return
    paths = sorted(
        os.path.join(RESULTS_DIR, f)
        for f in os.listdir(RESULTS_DIR)
        if f.startswith('tournament') and f.endswith('.json')
    )
    if not paths:
        print('[report] No tournament*.json found — skipping tournament figures.')
        return

    for path in paths:
        with open(path) as f:
            data = json.load(f)

        standings = data.get('standings', [])
        pairwise = data.get('pairwise', {})
        config = data.get('config', {})
        if not standings or not pairwise:
            print(f'[report] {path}: missing standings or pairwise — skipped.')
            continue

        games_per_matchup = config.get('games_per_matchup', '?')
        board_size = config.get('board_size', '?')
        stem = os.path.splitext(os.path.basename(path))[0]

        # standings is already sorted by win_pct (strongest first)
        raw_labels = [r['agent'] for r in standings]
        short = {raw: _short_label(raw) for raw in raw_labels}
        tick_labels = [short[l] for l in raw_labels]
        n = len(raw_labels)

        # ── Heatmap: row agent win rate vs column agent ──────────────────────
        matrix = np.full((n, n), np.nan)
        for i, a in enumerate(raw_labels):
            for j, b in enumerate(raw_labels):
                if a != b:
                    v = pairwise.get(a, {}).get(b)
                    if v is not None:
                        matrix[i, j] = float(v)

        fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.3)))
        sns.heatmap(
            matrix, annot=True, fmt='.0f',
            xticklabels=tick_labels, yticklabels=tick_labels,
            cmap='RdYlGn', center=50, vmin=0, vmax=100,
            ax=ax, mask=np.isnan(matrix), linewidths=0.5,
        )
        ax.set_title(
            f'Head-to-Head Win Rate (%) — {board_size}×{board_size}, '
            f'{games_per_matchup} games/matchup\n'
            f'Each cell = row agent win % vs column agent'
        )
        ax.set_xlabel('Opponent')
        ax.set_ylabel('Agent')
        fig.tight_layout()

        os.makedirs(FIGURE_DIR, exist_ok=True)
        heatmap_path = os.path.join(FIGURE_DIR, f'{stem}_winrate.png')
        fig.savefig(heatmap_path)
        plt.close(fig)
        print(f'[report] {heatmap_path} saved.')

        # ── Standings bar chart ───────────────────────────────────────────────
        win_pcts = [r['win_pct'] for r in standings]
        colors = plt.cm.RdYlGn([v / 100.0 for v in win_pcts])

        fig2, ax2 = plt.subplots(figsize=(max(6, n * 1.4), 4))
        bars = ax2.bar(tick_labels, win_pcts, color=colors)
        ax2.axhline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

        for bar, r in zip(bars, standings):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f'{r["wins"]}W / {r["losses"]}L',
                ha='center', va='bottom', fontsize=8,
            )

        ax2.set_ylabel('Win Rate (%)')
        ax2.set_ylim(0, min(115, max(win_pcts) + 18))
        ax2.set_title(
            f'Tournament Standings — {board_size}×{board_size} '
            f'(all matchups combined, ranked by win rate)'
        )
        ax2.grid(True, alpha=0.3, axis='y')
        fig2.tight_layout()

        standings_path = os.path.join(FIGURE_DIR, f'{stem}_standings.png')
        fig2.savefig(standings_path)
        plt.close(fig2)
        print(f'[report] {standings_path} saved.')

        # ── CSV table ─────────────────────────────────────────────────────────
        os.makedirs(TABLE_DIR, exist_ok=True)
        csv_path = os.path.join(TABLE_DIR, f'{stem}_standings.csv')
        write_csv(
            csv_path,
            ['Rank', 'Agent', 'Wins', 'Losses', 'Draws', 'WinPct'],
            [
                [i + 1, short[r['agent']], r['wins'], r['losses'],
                 r['draws'], f'{r["win_pct"]:.1f}']
                for i, r in enumerate(standings)
            ],
        )
        print(f'[report] {csv_path} saved.')


def gen_elo_ranking(entries, board_size):
    """Horizontal bar chart of all models ranked by Elo vs primary opponent."""
    plt = _setup_plt()
    if plt is None:
        return
    import numpy as np

    data = filter_entries(entries, board_size=board_size, training_method='supervised')
    data = [e for e in data if e.get('eval')]
    if not data:
        print(f'[report] No eval data for {board_size}x{board_size} ranking chart.')
        return

    opp_counts = defaultdict(int)
    for e in data:
        for opp in e['eval']:
            opp_counts[opp] += 1
    primary_opp = max(opp_counts, key=opp_counts.get)

    # Flatten to (label, elo) pairs, skip entries with no Elo for this opponent
    bots = []
    for e in data:
        elo_info = e.get('eval', {}).get(primary_opp)
        if elo_info and elo_info.get('elo_delta') is not None:
            label = f'{e["encoder"]}/{e["network"]}'
            bots.append((label, elo_info['elo_delta']))

    if not bots:
        print(f'[report] No Elo values for {board_size}x{board_size} ranking chart.')
        return

    bots.sort(key=lambda x: x[1], reverse=True)
    labels, elos = zip(*bots)
    elos = np.array(elos, dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(4, len(bots) * 0.35)))
    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in elos]
    bars = ax.barh(range(len(bots)), elos, color=colors)

    # Annotate each bar with the Elo value
    for i, (bar, val) in enumerate(zip(bars, elos)):
        x = bar.get_width()
        offset = 5 if x >= 0 else -5
        ha = 'left' if x >= 0 else 'right'
        ax.text(x + offset, i, f'{val:+.0f}', va='center', ha=ha, fontsize=8)

    ax.set_yticks(range(len(bots)))
    ax.set_yticklabels([f'#{i+1}  {lbl}' for i, lbl in enumerate(labels)], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel(f'Elo delta vs {primary_opp}')
    ax.set_title(f'Bot Ranking by Elo — {board_size}×{board_size}')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    suffix = '' if board_size == 9 else f'_{board_size}x{board_size}'
    path = os.path.join(FIGURE_DIR, f'elo_ranking{suffix}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[report] {path} saved.')

    # Also write a ranked CSV for the thesis
    csv_path = os.path.join(TABLE_DIR, f'elo_ranking{suffix}.csv')
    write_csv(csv_path,
              ['Rank', 'Encoder', 'Network', f'Elo_vs_{primary_opp}'],
              [[i + 1, lbl.split('/')[0], lbl.split('/')[1], f'{elo:.0f}']
               for i, (lbl, elo) in enumerate(bots)])
    print(f'[report] {csv_path} saved.')


def gen_report_md(entries):
    """Generate a narrative scaffold report.md from experiment results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'report.md')

    sup = filter_entries(entries, training_method='supervised')

    def _best(data, metric='final_val_accuracy', board_size=None):
        if board_size:
            data = [e for e in data if e.get('board_size') == board_size]
        data = [e for e in data if e.get(metric) is not None]
        if not data:
            return None
        return max(data, key=lambda e: e[metric])

    lines = [
        '# AlphiGo Experiment Report',
        '',
        f'Generated from `{RESULTS_FILE}` — {len(entries)} experiment(s).',
        '',
        '---',
        '',
        '## 1. Encoder Comparison',
        '',
        'Research question: does adding more board features (planes) '
        'improve move prediction?',
        '',
    ]

    best_9 = _best(sup, board_size=9)
    if best_9:
        lines += [
            f'Best 9×9 result: **{best_9["encoder"]}/{best_9["network"]}** — '
            f'val accuracy {best_9["final_val_accuracy"]:.4f} '
            f'({best_9["total_params"]:,} params).',
            '',
        ]

    lines += [
        'See Table: `results/tables/encoder_accuracy.csv`',
        'See Figure: `results/figures/encoder_progression.png`',
        '',
        '---',
        '',
        '## 2. Network Depth',
        '',
        'Research question: does a deeper network improve accuracy '
        'proportional to its parameter cost?',
        '',
        'See Table: `results/tables/network_accuracy.csv`',
        'See Figure: `results/figures/network_depth.png`',
        '',
        '---',
        '',
        '## 3. Parameter Efficiency',
        '',
        'Research question: which encoder/network combination gives the '
        'best accuracy per parameter (Pareto-optimal)?',
        '',
        'See Table: `results/tables/params_vs_accuracy.csv`',
        'See Figure: `results/figures/params_vs_accuracy.png`',
        '',
        '---',
        '',
        '## 4. Training Dynamics',
        '',
        'Learning curves showing validation loss and accuracy per epoch '
        'for all trained models.',
        '',
        'See Figure: `results/figures/learning_curves.png`',
        '',
        '---',
        '',
        '## 5. Strength vs Real Bots',
        '',
        'Round-robin tournament results. Relative Elo: best model = 0, '
        'all others shown relative to it.',
        '',
        'See Table: `results/tables/tournament.csv`',
        'See Figure: `results/figures/tournament_relative_elo.png`',
        '',
        '---',
        '',
    ]

    # RL section
    rl = filter_entries(entries, training_method='pg')
    if rl:
        # Find latest round per config
        rl_configs = defaultdict(list)
        for e in rl:
            k = f"{e.get('board_size')}x{e.get('board_size')} " \
                f"{e.get('encoder')}/{e.get('network')}"
            rl_configs[k].append(e)

        lines += [
            '## 6. Reinforcement Learning (Policy Gradient)',
            '',
            'Research question: can self-play with policy gradient learning '
            'improve on supervised training?',
            '',
        ]

        for config, rounds in sorted(rl_configs.items()):
            rounds = sorted(rounds, key=lambda x: x.get('rl_round', 0))
            latest = rounds[-1]
            lines.append(
                f'- **{config}**: {len(rounds)} rounds, '
                f'latest win rate vs previous = '
                f'{latest.get("vs_previous_win_rate", 0):.1%}'
            )
            ev = latest.get('eval', {})
            for opp, res in sorted(ev.items()):
                wr = res.get('win_rate', 0)
                lines.append(f'  - vs {opp}: {wr:.1%}')

        lines += [
            '',
            'See Table: `results/tables/rl_progress.csv`',
            'See Table: `results/tables/supervised_vs_rl.csv`',
            'See Figure: `results/figures/rl_progress.png`',
            '',
            '---',
            '',
        ]

    # Optimizer ablation section if data exists
    adam = [e for e in sup if e.get('optimizer') == 'adam']
    sgd = [e for e in sup if e.get('optimizer') == 'sgd']
    if adam and sgd:
        next_section = 7 if rl else 6
        lines += [
            f'## {next_section}. Optimizer Ablation',
            '',
            'Adam vs SGD comparison on matched encoder/network combinations.',
            '',
            'See Table: `results/tables/optimizer_ablation.csv`',
            'See Figure: `results/figures/optimizer_ablation.png`',
            '',
            '---',
            '',
        ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'[report] {path} written.')


def gen_optimizer_ablation_figure(entries):
    """Adam vs SGD bar chart."""
    plt = _setup_plt()
    if plt is None:
        return
    import numpy as np

    data = filter_entries(entries, training_method='supervised')
    adam = {(e['board_size'], e['encoder'], e['network']): e
            for e in data if e.get('optimizer') == 'adam'}
    sgd = {(e['board_size'], e['encoder'], e['network']): e
           for e in data if e.get('optimizer') == 'sgd'}

    common = sorted(set(adam.keys()) & set(sgd.keys()))
    if not common:
        print('[report] No optimizer ablation data for figure.')
        return

    labels = [f'{k[1]}/{k[2]}\n({k[0]}x{k[0]})' for k in common]
    adam_accs = [adam[k].get('final_val_accuracy', 0) or 0 for k in common]
    sgd_accs = [sgd[k].get('final_val_accuracy', 0) or 0 for k in common]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar(x - width / 2, adam_accs, width, label='Adam', color=plt.cm.tab10(0))
    ax.bar(x + width / 2, sgd_accs, width, label='SGD', color=plt.cm.tab10(1))

    ax.set_xlabel('Encoder/Network')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Optimizer Ablation: Adam vs SGD')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, 'optimizer_ablation.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[report] {path} saved.')


# ---------------------------------------------------------------------------
# RL-specific tables and figures
# ---------------------------------------------------------------------------

def gen_rl_progress_table(entries):
    """CSV/LaTeX table showing RL win rate progression per round."""
    rl = filter_entries(entries, training_method='pg')
    if not rl:
        print('[report] No pg data for RL progress table.')
        return

    os.makedirs(TABLE_DIR, exist_ok=True)

    # Group by (board_size, encoder, network)
    groups = defaultdict(list)
    for e in rl:
        k = (e.get('board_size'), e.get('encoder'), e.get('network'))
        groups[k].append(e)

    headers = ['Config', 'Round', 'vs_prev_win%', 'Games', 'Decisions',
               'Self-play(s)', 'Train(s)']
    rows = []
    for (bs, enc, net), rounds in sorted(groups.items()):
        rounds = sorted(rounds, key=lambda x: x.get('rl_round', 0))
        for r in rounds:
            label = f'{bs}x{bs} {enc}/{net}'
            rows.append([
                label,
                r.get('rl_round', '?'),
                f"{r.get('vs_previous_win_rate', 0):.1%}",
                r.get('games_per_round', '?'),
                r.get('num_decisions', '?'),
                r.get('self_play_time_seconds', '?'),
                r.get('training_time_seconds', '?'),
            ])

    csv_path = os.path.join(TABLE_DIR, 'rl_progress.csv')
    write_csv(csv_path, headers, rows)

    # Also emit external eval if available
    has_eval = any(r.get('eval') for r in rl)
    if has_eval:
        eval_headers = ['Config', 'Round']
        all_opps = set()
        for r in rl:
            all_opps.update(r.get('eval', {}).keys())
        all_opps = sorted(all_opps)
        eval_headers += [f'vs {opp}' for opp in all_opps]

        eval_rows = []
        for (bs, enc, net), rounds in sorted(groups.items()):
            rounds = sorted(rounds, key=lambda x: x.get('rl_round', 0))
            for r in rounds:
                label = f'{bs}x{bs} {enc}/{net}'
                row = [label, r.get('rl_round', '?')]
                ev = r.get('eval', {})
                for opp in all_opps:
                    wr = ev.get(opp, {}).get('win_rate')
                    row.append(f'{wr:.1%}' if wr is not None else '-')
                eval_rows.append(row)

        eval_path = os.path.join(TABLE_DIR, 'rl_eval_progress.csv')
        write_csv(eval_path, eval_headers, eval_rows)


def gen_rl_progress_figure(entries):
    """Line plot of RL win rate vs previous agent over rounds."""
    plt = _setup_plt()
    if plt is None:
        return

    rl = filter_entries(entries, training_method='pg')
    if not rl:
        print('[report] No pg data for RL progress figure.')
        return

    # Group by (board_size, encoder, network)
    groups = defaultdict(list)
    for e in rl:
        k = (e.get('board_size'), e.get('encoder'), e.get('network'))
        groups[k].append(e)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: win rate vs previous per round
    ax = axes[0]
    for (bs, enc, net), rounds in sorted(groups.items()):
        rounds = sorted(rounds, key=lambda x: x.get('rl_round', 0))
        x = [r.get('rl_round', 0) for r in rounds]
        y = [r.get('vs_previous_win_rate', 0.5) for r in rounds]
        label = f'{enc}/{net} ({bs}x{bs})'
        ax.plot(x, y, 'o-', label=label, markersize=4)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.set_xlabel('RL Round')
    ax.set_ylabel('Win Rate vs Previous')
    ax.set_title('RL Training Progress')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Right: external eval Elo over rounds (if available)
    ax = axes[1]
    has_eval = False
    for (bs, enc, net), rounds in sorted(groups.items()):
        rounds = sorted(rounds, key=lambda x: x.get('rl_round', 0))
        # Collect all opponent names
        all_opps = set()
        for r in rounds:
            all_opps.update(r.get('eval', {}).keys())

        for opp in sorted(all_opps):
            x = []
            y = []
            for r in rounds:
                ev = r.get('eval', {}).get(opp, {})
                if ev.get('win_rate') is not None:
                    x.append(r.get('rl_round', 0))
                    y.append(ev['win_rate'])
                    has_eval = True
            if x:
                label = f'{enc}/{net} vs {opp}'
                ax.plot(x, y, 's-', label=label, markersize=4)

    if has_eval:
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('RL Round')
        ax.set_ylabel('Win Rate vs Opponent')
        ax.set_title('RL Strength vs External Opponents')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'No external eval data yet',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('RL Strength vs External Opponents')

    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, 'rl_progress.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[report] {path} saved.')


def gen_supervised_vs_rl_table(entries):
    """Side-by-side comparison of supervised and RL agents on the same
    encoder/network combos, using external eval win rates."""
    sup = filter_entries(entries, training_method='supervised')
    rl = filter_entries(entries, training_method='pg')
    if not sup or not rl:
        print('[report] Need both supervised and RL data for comparison table.')
        return

    os.makedirs(TABLE_DIR, exist_ok=True)

    # For RL, take the latest round per (bs, enc, net)
    rl_best = {}
    for e in rl:
        k = (e.get('board_size'), e.get('encoder'), e.get('network'))
        rn = e.get('rl_round', 0)
        if k not in rl_best or rn > rl_best[k].get('rl_round', 0):
            rl_best[k] = e

    # For supervised, index by (bs, enc, net)
    sup_map = {}
    for e in sup:
        k = (e.get('board_size'), e.get('encoder'), e.get('network'))
        sup_map[k] = e

    # Find common configs
    common = sorted(set(sup_map.keys()) & set(rl_best.keys()))
    if not common:
        # No exact match — still show all RL results with their eval
        common_rl = sorted(rl_best.keys())
        if not common_rl:
            return

    # Collect all opponents
    all_opps = set()
    for e in list(sup_map.values()) + list(rl_best.values()):
        all_opps.update(e.get('eval', {}).keys())
    all_opps = sorted(all_opps)

    if not all_opps:
        print('[report] No eval data for supervised vs RL comparison.')
        return

    headers = ['Config', 'Method']
    for opp in all_opps:
        headers.append(f'vs {opp} win%')

    rows = []
    configs = common if common else sorted(rl_best.keys())
    for k in configs:
        bs, enc, net = k
        label = f'{bs}x{bs} {enc}/{net}'

        # Supervised row
        if k in sup_map:
            row = [label, 'supervised']
            ev = sup_map[k].get('eval', {})
            for opp in all_opps:
                wr = ev.get(opp, {}).get('win_rate')
                row.append(f'{wr:.1%}' if wr is not None else '-')
            rows.append(row)

        # RL row
        if k in rl_best:
            rnd = rl_best[k].get('rl_round', '?')
            row = [label, f'RL (round {rnd})']
            ev = rl_best[k].get('eval', {})
            for opp in all_opps:
                wr = ev.get(opp, {}).get('win_rate')
                row.append(f'{wr:.1%}' if wr is not None else '-')
            rows.append(row)

    csv_path = os.path.join(TABLE_DIR, 'supervised_vs_rl.csv')
    write_csv(csv_path, headers, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report():
    """Generate all tables and figures."""
    entries = load_experiments()
    if not entries:
        print('[report] No experiments found. Run thesis_run.py first.')
        return

    print(f'[report] Loaded {len(entries)} experiment(s).\n')

    # Tables
    for bs in (9, 19):
        gen_accuracy_table(entries, bs)
        gen_elo_table(entries, bs)
    gen_optimizer_ablation(entries)

    # Figures
    for bs in (9, 19):
        gen_learning_curves(entries, bs)
        gen_elo_heatmap(entries, bs)
        gen_elo_ranking(entries, bs)
    gen_encoder_comparison(entries)
    gen_network_comparison(entries)
    gen_params_vs_accuracy(entries)
    gen_optimizer_ablation_figure(entries)
    gen_tournament_figures()

    # RL tables and figures
    gen_rl_progress_table(entries)
    gen_rl_progress_figure(entries)
    gen_supervised_vs_rl_table(entries)

    # Narrative scaffold
    gen_report_md(entries)

    print('\n[report] Done.')


if __name__ == '__main__':
    generate_report()
