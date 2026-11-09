#!/usr/bin/env python
"""Evaluate a series of model checkpoints against fixed opponents and track Elo.

Examples:
  # Evaluate all checkpoints vs GnuGo, 50 games each
  python eval_checkpoints.py checkpoints/*.h5 --opponents gnugo -n 50

  # vs GnuGo and Pachi at different strengths, save results
  python eval_checkpoints.py checkpoints/*.h5 \\
      --opponents gnugo pachi:playouts=1000 pachi:playouts=5000 \\
      -n 100 -o results/elo_progress.json

  # Include known opponent Elo so absolute estimates are shown
  python eval_checkpoints.py checkpoints/*.h5 \\
      --opponents gnugo pachi:playouts=1000 \\
      --reference-elo gnugo=1300 pachi:playouts=1000=1600 \\
      -n 50 --plot

  # Sort checkpoints by epoch number in filename instead of mtime
  python eval_checkpoints.py checkpoints/*.h5 --opponents gnugo -n 50 --sort-by epoch
"""

import argparse
import json
import re
import sys
from pathlib import Path

from alphigo.benchmark import wilson_ci


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_agent_spec(spec):
    """Parse 'name:key=val,key=val' into (name, kwargs_dict)."""
    if ":" not in spec:
        return spec, {}
    name, params_str = spec.split(":", 1)
    kwargs = {}
    for pair in params_str.split(","):
        if "=" not in pair:
            raise ValueError(f"Bad param '{pair}' in agent spec '{spec}'.")
        k, v = pair.split("=", 1)
        kwargs[k.strip()] = v.strip()
    return name, kwargs


def extract_epoch(path):
    """Try to extract epoch number from checkpoint filename, else return inf."""
    m = re.search(r'epoch[_\-]?(\d+)', Path(path).stem, re.IGNORECASE)
    return int(m.group(1)) if m else float('inf')


def _elo_str(v):
    if v is None:
        return "  n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.0f}"


def _abs_elo_str(elo_diff_val, ref_elo):
    if elo_diff_val is None or ref_elo is None:
        return "  n/a"
    return f"~{ref_elo + elo_diff_val:.0f}"


# ── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_checkpoints(
    checkpoint_paths,
    opponent_specs,
    num_games=50,
    board_size=9,
    max_moves=2000,
    reference_elos=None,
    sort_by="epoch",
    verbose=True,
):
    """Run benchmarks for each checkpoint against each opponent.

    Returns list of dicts with keys:
        checkpoint, opponent, win_pct, elo_diff, abs_elo, num_games
    """
    from alphigo.agent.registry import create_agent
    from alphigo.benchmark import run_benchmark, elo_diff as calc_elo

    reference_elos = reference_elos or {}

    # Sort checkpoints
    if sort_by == "epoch":
        checkpoint_paths = sorted(checkpoint_paths, key=extract_epoch)
    else:
        checkpoint_paths = sorted(checkpoint_paths, key=lambda p: Path(p).stat().st_mtime)

    # Pre-build opponents (keep alive across checkpoints)
    opponents = {}
    for spec in opponent_specs:
        name, kwargs = parse_agent_spec(spec)
        opponents[spec] = create_agent(name, **kwargs)

    rows = []
    total = len(checkpoint_paths) * len(opponent_specs)
    done = 0

    for ckpt_path in checkpoint_paths:
        ckpt_label = Path(ckpt_path).name

        import h5py
        from alphigo.agent.deep_learning import load_prediction_agent
        with h5py.File(ckpt_path, "r") as h5:
            dl_agent = load_prediction_agent(h5)

        for opp_spec, opp_agent in opponents.items():
            done += 1
            if verbose:
                print(f"[{done}/{total}] {ckpt_label}  vs  {opp_spec} ... ",
                      end="", flush=True)

            results = run_benchmark(
                dl_agent,
                opp_agent,
                black_label=ckpt_label,
                white_label=opp_spec,
                board_size=board_size,
                num_games=num_games,
                max_moves=max_moves,
                swap_colors=True,
                verbose=False,
            )

            s = results.summary_dict()
            if verbose:
                ms = s['avg_ms_per_move']
                ms_str = f"{ms:.1f}ms/move" if ms is not None else "n/a"
                ci = s['a_win_pct_ci']
                print(f"{s['a_win_pct']}% [{ci[0]}-{ci[1]}]  "
                      f"Elo diff: {_elo_str(s['a_elo_diff'])}  {ms_str}")
            elo_d = s["a_elo_diff"]
            ref = reference_elos.get(opp_spec)
            abs_elo = (ref + elo_d) if (elo_d is not None and ref is not None) else None

            a_ci = wilson_ci(s["a_wins"], s["total_games"])
            rows.append({
                "checkpoint": ckpt_label,
                "checkpoint_path": str(ckpt_path),
                "opponent": opp_spec,
                "wins": s["a_wins"],
                "losses": s["b_wins"],
                "draws": s["draws"],
                "win_pct": s["a_win_pct"],
                "win_pct_ci_low": round(100 * a_ci[0], 1),
                "win_pct_ci_high": round(100 * a_ci[1], 1),
                "elo_diff": elo_d,
                "abs_elo": abs_elo,
                "num_games": s["total_games"],
            })

    return rows, checkpoint_paths


# ── Output ───────────────────────────────────────────────────────────────────

def print_table(rows, opponent_specs, reference_elos=None):
    reference_elos = reference_elos or {}
    has_abs = any(reference_elos.get(opp) is not None for opp in opponent_specs)

    col_w = 12
    ckpt_w = max((len(r["checkpoint"]) for r in rows), default=20)

    # Group rows by checkpoint (preserve order)
    from collections import OrderedDict
    by_ckpt = OrderedDict()
    for r in rows:
        by_ckpt.setdefault(r["checkpoint"], {})[r["opponent"]] = r

    # Header
    header = f"\n{'Checkpoint':<{ckpt_w}}"
    for opp in opponent_specs:
        if has_abs and reference_elos.get(opp) is not None:
            header += f"  {'W%':>5} {'EloΔ':>6} {'~Elo':>6}"
        else:
            header += f"  {'W%':>5} {'EloΔ':>6}"
    print(header)
    print("-" * len(header.rstrip()))

    for ckpt, opp_map in by_ckpt.items():
        line = f"{ckpt:<{ckpt_w}}"
        for opp in opponent_specs:
            if opp not in opp_map:
                if has_abs and reference_elos.get(opp) is not None:
                    line += f"  {'n/a':>5} {'n/a':>6} {'n/a':>6}"
                else:
                    line += f"  {'n/a':>5} {'n/a':>6}"
                continue
            r = opp_map[opp]
            w = f"{r['win_pct']:.1f}%"
            e = _elo_str(r["elo_diff"])
            if has_abs and reference_elos.get(opp) is not None:
                a = _abs_elo_str(r["elo_diff"], reference_elos.get(opp))
                line += f"  {w:>5} {e:>6} {a:>6}"
            else:
                line += f"  {w:>5} {e:>6}"
        print(line)

    # Column legend
    print()
    for opp in opponent_specs:
        ref = reference_elos.get(opp)
        ref_str = f"  (ref Elo: {ref})" if ref is not None else ""
        print(f"  {opp}{ref_str}")


def plot_progress(rows, opponent_specs, checkpoint_paths, output_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. pip install matplotlib")
        return

    ckpt_names = [Path(p).name for p in checkpoint_paths]
    x = list(range(1, len(ckpt_names) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for opp in opponent_specs:
        opp_rows = [r for r in rows if r["opponent"] == opp]
        elo_vals = [r["elo_diff"] for r in opp_rows]
        win_vals = [r["win_pct"] for r in opp_rows]

        ax1.plot(x, win_vals, marker='o', label=opp)
        ax2.plot(x, elo_vals, marker='o', label=opp)

    ax1.axhline(50, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_ylabel("Win % vs opponent")
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_ylabel("Elo difference vs opponent")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ckpt_names, rotation=30, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Checkpoint Elo Progress")
    fig.tight_layout()

    if output_path:
        plot_path = Path(output_path).with_suffix('.png')
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def compute_relative_elo(rows, opponent_specs):
    """Compute relative Elo from round-robin results.

    Best-performing model is anchored at 0. All others shown relative to it.
    Returns dict mapping checkpoint_label -> relative_elo.
    """
    # Sum Elo diffs across all opponents for each checkpoint
    scores = {}
    counts = {}
    for r in rows:
        ckpt = r["checkpoint"]
        elo_d = r.get("elo_diff")
        if elo_d is not None:
            scores[ckpt] = scores.get(ckpt, 0.0) + elo_d
            counts[ckpt] = counts.get(ckpt, 0) + 1

    if not scores:
        return {}

    # Average Elo diff per opponent for each checkpoint
    avg_scores = {ckpt: scores[ckpt] / counts[ckpt] for ckpt in scores}

    # Anchor best model at 0
    best_score = max(avg_scores.values())
    return {ckpt: round(score - best_score, 1)
            for ckpt, score in avg_scores.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model checkpoints against GTP opponents and track Elo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoints", nargs="+",
        help="Checkpoint .h5 files (supports glob expansion by shell)",
    )
    parser.add_argument(
        "--opponents", nargs="+", required=True,
        help="Opponent agent specs  (e.g. gnugo  pachi:playouts=1000)",
    )
    parser.add_argument(
        "-n", "--num-games", type=int, default=50,
        help="Games per checkpoint per opponent (default: 50)",
    )
    parser.add_argument(
        "--board-size", type=int, default=9,
        help="Board size (default: 9)",
    )
    parser.add_argument(
        "--max-moves", type=int, default=2000,
    )
    parser.add_argument(
        "--reference-elo", nargs="*", default=[],
        metavar="OPPONENT=ELO",
        help="Known Elo of opponents for absolute Elo estimates  "
             "(e.g. gnugo=1300  'pachi:playouts=1000=1600')",
    )
    parser.add_argument(
        "--sort-by", choices=["epoch", "mtime"], default="epoch",
        help="How to sort checkpoints (default: epoch number in filename)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Save results JSON to this path",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show/save Elo progress plot (requires matplotlib)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-game output",
    )
    parser.add_argument(
        "--self-anchor", action="store_true",
        help="Compute relative Elo using round-robin between all checkpoints "
             "and opponents. Best model = 0, all others relative to it. "
             "No external Elo reference needed.",
    )

    args = parser.parse_args()

    # Parse reference Elos: "gnugo=1300" → {"gnugo": 1300}
    reference_elos = {}
    for entry in args.reference_elo:
        # Last '=' separates opponent spec from Elo value
        idx = entry.rfind("=")
        if idx == -1:
            parser.error(f"Bad --reference-elo entry '{entry}'. Expected OPPONENT=ELO")
        opp, elo_val = entry[:idx], entry[idx + 1:]
        try:
            reference_elos[opp] = int(elo_val)
        except ValueError:
            parser.error(f"Elo value must be an integer in '{entry}'")

    rows, sorted_paths = evaluate_checkpoints(
        args.checkpoints,
        args.opponents,
        num_games=args.num_games,
        board_size=args.board_size,
        max_moves=args.max_moves,
        reference_elos=reference_elos,
        sort_by=args.sort_by,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 70)
    print("ELO PROGRESS SUMMARY")
    print("=" * 70)
    print_table(rows, args.opponents, reference_elos)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "config": {
                "opponents": args.opponents,
                "num_games": args.num_games,
                "board_size": args.board_size,
                "reference_elos": reference_elos,
            },
            "results": rows,
        }, indent=2))
        print(f"\nResults saved to {out}")

    if args.self_anchor:
        rel_elo = compute_relative_elo(rows, args.opponents)
        print("\n" + "=" * 50)
        print("RELATIVE ELO (best model = 0)")
        print("=" * 50)
        for ckpt, elo in sorted(rel_elo.items(), key=lambda x: -x[1]):
            print(f"  {ckpt:<50s}  {elo:+.0f}")
        # Attach to rows for JSON output
        for r in rows:
            r["relative_elo"] = rel_elo.get(r["checkpoint"])

    if args.plot:
        plot_progress(rows, args.opponents, sorted_paths,
                      output_path=args.output)


if __name__ == "__main__":
    main()
