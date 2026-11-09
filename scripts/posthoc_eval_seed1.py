"""Post-hoc external eval for Phase A seed=1.

Plays each method's checkpoint (q/pg/ac/alphazero) against gnugo:level=1
with color swap, 50 games per match. Appends results to
results/experiments.jsonl as external_eval rows so the thesis report
script can pick them up alongside the per-round JSONL.

Run AFTER Phase A seed=1 completes. Requires GPU (don't run while
phase_a_seed1.sh is alive).

Usage:
    python scripts/posthoc_eval_seed1.py
    python scripts/posthoc_eval_seed1.py --rounds 50 --methods q,pg,ac,alphazero
    python scripts/posthoc_eval_seed1.py --rounds 1,10,20,30,40,50 --opponent gnugo:level=1
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── Method specs (must match phase_a_*.sh slugs) ────────────────────
METHODS = {
    'q':         ('q_oneplane_book_2layer_lr1e-1_b9_seed1',         'oneplane', 'book_2layer', 0.1),
    'pg':        ('pg_oneplane_small_lr1e-4_b9_seed1',              'oneplane', 'small',       1e-4),
    'ac':        ('ac_oneplane_book_2layer_lr1e-2_b9_seed1',        'oneplane', 'book_2layer', 1e-2),
    'alphazero': ('alphazero_alphazero_alphazero_lr1e-4_b9_seed1',  'alphazero','alphazero',   1e-4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default='q,pg,ac,alphazero',
                    help='Comma list (subset of q,pg,ac,alphazero)')
    ap.add_argument('--rounds', default='1,10,20,30,40,50',
                    help='Comma list of round numbers to evaluate')
    ap.add_argument('--opponent', default='gnugo:level=1',
                    help='Opponent spec (registry style)')
    ap.add_argument('--games', type=int, default=50,
                    help='Games per match (with --swap, half black half white)')
    ap.add_argument('--board-size', type=int, default=9)
    ap.add_argument('--out-jsonl', default='results/experiments.jsonl')
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    rounds = [int(r) for r in args.rounds.split(',') if r.strip()]

    from alphigo.agent.registry import create_agent
    from alphigo.benchmark import run_benchmark
    from alphigo.training.baseline import REFERENCE_ELOS

    opp_elo = REFERENCE_ELOS.get(args.opponent)
    print(f'Opponent: {args.opponent}  (reference Elo = {opp_elo})')
    opp_agent = create_agent(*_split(args.opponent))
    out_path = ROOT / args.out_jsonl

    for method in methods:
        if method not in METHODS:
            print(f'  [skip] unknown method {method!r}'); continue
        slug, encoder, network, lr = METHODS[method]
        run_dir = ROOT / 'runs' / slug / 'checkpoints'
        if not run_dir.exists():
            print(f'  [skip] no checkpoints for {method} ({run_dir})'); continue

        for r in rounds:
            ckpt = run_dir / f'round_{r:03d}.h5'
            if not ckpt.exists():
                print(f'  [skip] {method} round_{r:03d} (no checkpoint)'); continue

            print(f'\n=== {method} round_{r:03d}  vs  {args.opponent} ===')
            agent = create_agent(method, agent_file=str(ckpt))

            label_a = f'{method}_round{r:03d}'
            results = run_benchmark(
                black_agent=agent, white_agent=opp_agent,
                black_label=label_a, white_label=args.opponent,
                board_size=args.board_size, num_games=args.games,
                swap_colors=True, verbose=False,
            )
            summary = results.summary()
            wins_a = summary['a_wins']
            wins_b = summary['b_wins']
            draws = args.games - wins_a - wins_b
            wr = wins_a / max(wins_a + wins_b, 1)

            row = {
                'key': f'posthoc__{method}__seed1__round_{r:03d}__vs__{args.opponent}',
                'kind': 'external_posthoc',
                'method': method, 'seed': 1, 'rl_round': r,
                'encoder': encoder, 'network': network, 'lr': lr,
                'board_size': args.board_size,
                'opponent': args.opponent,
                'games': args.games, 'wins_a': wins_a, 'wins_b': wins_b,
                'draws': draws, 'win_rate': wr,
                'opp_reference_elo': opp_elo,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'checkpoint_path': str(ckpt.relative_to(ROOT)),
            }
            print(f'   → wr={wr*100:.1f}% ({wins_a}/{wins_a+wins_b})')
            with out_path.open('a') as f:
                f.write(json.dumps(row) + '\n')


def _split(spec):
    """Parse 'name:k=v,k=v' into (name, kwargs_dict)."""
    if ':' not in spec:
        return spec, {}
    name, rest = spec.split(':', 1)
    kw = {}
    for pair in rest.split(','):
        if '=' in pair:
            k, v = pair.split('=', 1)
            try: kw[k] = int(v)
            except ValueError:
                try: kw[k] = float(v)
                except ValueError: kw[k] = v
    return name, kw


if __name__ == '__main__':
    main()
