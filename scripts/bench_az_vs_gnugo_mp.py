"""Multiprocess AZ-vs-gnugo benchmark.

Sequential `run.py benchmark` plays one game at a time; with sims=200 a
single game takes ~10-15 minutes wall, so a 30-game sweep cell is ~5-8h.
Most of that time is Python MCTS overhead + GTP I/O, not the GPU forward
pass (the model is small enough that the 3090 inference is amortised).

This script spawns N worker processes (default 4) that each:
- load their own ZeroAgent from disk (clean GPU state via `spawn` context)
- play their share of the N games, alternating colours
- each game gets a fresh gnugo subprocess (created/closed by the worker)
- enable TF `memory_growth` so workers don't OOM each other's GPU memory
- return (wins, losses, draws, list of game records)

Results are aggregated in the parent and written to a CSV row identical
in shape to `scripts/sims_sweep.sh`'s output, so downstream plot/analysis
code is unchanged.

Smoke usage:
    python scripts/bench_az_vs_gnugo_mp.py \\
        --agent runs/.../r0_joint_smargin_b19.h5 \\
        --opponent gnugo:level=5 \\
        --num-simulations 25 \\
        --board-size 19 \\
        --num-games 8 \\
        --num-workers 4
"""
import argparse
import multiprocessing
import os
import sys
import time

import numpy as np


def _worker(worker_id, conn, agent_path, opponent_spec, board_size,
            sims_per_call, game_ids, seed, value_model_path=None):
    """Play `len(game_ids)` games. Sends each result via `conn` as it
    completes, then sends sentinel None. Surviving a mid-batch crash
    means already-completed games are not lost."""
    np.random.seed(seed)
    import random as _stdrand
    _stdrand.seed(seed)

    import tensorflow as tf
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    import h5py
    from alphigo.agent.zero_agent import load_zero_agent
    from alphigo.agent.registry import create_agent
    from alphigo.simulate import simulate_game

    ext_value_model = None
    value_mode = 'network'
    if value_model_path:
        ext_value_model = tf.keras.models.load_model(value_model_path, compile=False)
        value_mode = 'external'

    with h5py.File(agent_path, 'r') as h5:
        az_agent = load_zero_agent(
            h5, value_mode=value_mode,
            external_value_model=ext_value_model)
    az_agent._num_simulations = int(sims_per_call)
    az_agent.set_temperature(0.0)
    az_label = (f'alphazero:agent_file={agent_path},'
                f'num_simulations={sims_per_call},temperature=0.0')

    for game_id in game_ids:
        # Build a fresh gnugo subprocess per game (the existing registry
        # builds the agent and tracks its GTP subprocess lifetime).
        opp_name, opp_kwargs = _parse_spec(opponent_spec)
        opp_agent = create_agent(opp_name, board_size=board_size, **opp_kwargs)
        try:
            # Alternate colours by game_id parity.
            from alphigo.alphigo_types import Player
            t0 = time.time()
            if game_id % 2 == 0:
                # AZ as black
                result = simulate_game(az_agent, opp_agent, board_size)
                az_won = (result['winner'] == Player.black)
            else:
                result = simulate_game(opp_agent, az_agent, board_size)
                az_won = (result['winner'] == Player.white)
            elapsed = time.time() - t0
            winner_str = ('B' if result['winner'] == Player.black else
                          'W' if result['winner'] == Player.white else 'draw')
            az_color = 'B' if game_id % 2 == 0 else 'W'
            score = result.get('score')
            conn.send((game_id, az_won, elapsed, result['move_log'],
                       winner_str, result['reason'], result['num_moves'],
                       az_color, board_size, score))
            score_part = f' score={score}' if score else ''
            print(f'  [w{worker_id} g{game_id}] '
                  f'winner={winner_str} reason={result["reason"]} '
                  f'moves={result["num_moves"]} AZ {"won" if az_won else "lost"}'
                  f'{score_part}  '
                  f'({elapsed:.0f}s)', flush=True)
        finally:
            # Clean up the gnugo subprocess if the agent owns one.
            close = getattr(opp_agent, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    conn.send(None)  # sentinel: all games done


def _point_to_sgf(point, board_size):
    """Convert a Point(row, col) to SGF coordinate pair (e.g. 'dp')."""
    col_char = chr(ord('a') + point.col - 1)
    row_char = chr(ord('a') + board_size - point.row)
    return col_char + row_char


def _write_sgf(path, move_log, board_size, winner_str, reason,
               az_color, sims):
    """Write a game record as SGF."""
    from alphigo.alphigo_types import Player
    lines = [f'(;GM[1]FF[4]SZ[{board_size}]KM[7.5]']
    lines.append(f'PB[{"AlphiGo" if az_color == "B" else "GnuGo"}]')
    lines.append(f'PW[{"AlphiGo" if az_color == "W" else "GnuGo"}]')
    lines.append(f'RE[{winner_str}+{"R" if reason == "resign" else "?"}]')
    lines.append(f'C[sims={sims} az={az_color}]')
    for player, move in move_log:
        color = 'B' if player == Player.black else 'W'
        if move.is_pass:
            lines.append(f';{color}[]')
        elif move.is_resign:
            break
        else:
            lines.append(f';{color}[{_point_to_sgf(move.point, board_size)}]')
    lines.append(')')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _parse_spec(spec):
    if ':' not in spec:
        return spec, {}
    name, params_str = spec.split(':', 1)
    kwargs = {}
    for pair in params_str.split(','):
        if '=' not in pair:
            continue
        k, v = pair.split('=', 1)
        kwargs[k.strip()] = v.strip()
    return name, kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True,
                        help='Path to AZ agent .h5 file')
    parser.add_argument('--opponent', default='gnugo:level=5')
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--num-simulations', type=int, required=True)
    parser.add_argument('--num-games', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-csv', default=None,
                        help='Append a sims-sweep-compatible row to this CSV')
    parser.add_argument('--value-model', default=None,
                        help='Path to a separate value network .keras/.h5 '
                             'file. When set, uses value_mode="external" '
                             'with this model for leaf evaluation.')
    parser.add_argument('--sgf-dir', default=None,
                        help='Directory to save SGF game records.')
    args = parser.parse_args()

    if not os.path.exists(args.agent):
        print(f'ERROR: agent not found: {args.agent}', file=sys.stderr)
        return 1

    # Distribute game ids across workers (round-robin to keep colour
    # parity balanced per worker).
    workers_assigns = [[] for _ in range(args.num_workers)]
    for g in range(args.num_games):
        workers_assigns[g % args.num_workers].append(g)

    print(f'=== bench_az_vs_gnugo_mp ===', flush=True)
    print(f'  agent:        {args.agent}', flush=True)
    print(f'  opponent:     {args.opponent}', flush=True)
    print(f'  board:        {args.board_size}x{args.board_size}', flush=True)
    print(f'  sims:         {args.num_simulations}', flush=True)
    print(f'  games:        {args.num_games}  (workers={args.num_workers})',
          flush=True)
    for i, ids in enumerate(workers_assigns):
        print(f'    worker {i}: {len(ids)} games  ids={ids}', flush=True)

    ctx = multiprocessing.get_context('spawn')

    t0 = time.time()
    procs = []
    parent_conns = []
    for i, ids in enumerate(workers_assigns):
        if not ids:
            continue
        seed = (i + 1) * 7919 + args.num_simulations
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=_worker,
            args=(i, child_conn, args.agent, args.opponent, args.board_size,
                  args.num_simulations, ids, seed, args.value_model),
        )
        p.start()
        procs.append((i, p))
        parent_conns.append(parent_conn)

    if args.sgf_dir:
        os.makedirs(args.sgf_dir, exist_ok=True)

    # Drain results incrementally. Workers send one
    # (game_id, won, sec, move_log, winner_str, reason, num_moves,
    #  az_color, board_size, score) tuple per game, then sentinel None.
    # On EOFError (worker crashed), keep whatever has arrived and continue.
    all_results = []
    failed_workers = []
    for (i, p), conn in zip(procs, parent_conns):
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                failed_workers.append(i)
                print(f'  WARN: worker {i} died mid-batch '
                      f'(EOFError on recv)', flush=True)
                break
            if msg is None:
                break
            all_results.append(msg)
            if args.sgf_dir:
                (gid, won, sec, move_log, winner_str, reason,
                 n_moves, az_color, bsz, *rest) = msg
                sgf_path = os.path.join(
                    args.sgf_dir,
                    f'sims{args.num_simulations}_g{gid:03d}_'
                    f'{"win" if won else "loss"}.sgf')
                _write_sgf(sgf_path, move_log, bsz, winner_str,
                           reason, az_color, args.num_simulations)
        p.join()
        if p.exitcode != 0:
            print(f'  WARN: worker {i} exited with code {p.exitcode}',
                  flush=True)

    elapsed = time.time() - t0
    az_wins = sum(1 for r in all_results if r[1])
    n = len(all_results)
    win_pct = 100.0 * az_wins / n if n else 0.0
    print(f'\n=== Results ===', flush=True)
    print(f'  AZ wins: {az_wins}/{n}  ({win_pct:.1f}%)', flush=True)
    print(f'  wall:    {elapsed:.0f}s', flush=True)

    scores = [r[9] for r in all_results if len(r) > 9 and r[9]]
    if scores:
        print(f'  scores:  {", ".join(scores)}', flush=True)

    # Wilson CI for compat with sims_sweep.sh output format.
    from alphigo.benchmark import wilson_ci
    lo, hi = wilson_ci(az_wins, n) if n else (0.0, 1.0)

    if args.out_csv:
        write_header = not os.path.exists(args.out_csv) or \
            os.path.getsize(args.out_csv) == 0
        with open(args.out_csv, 'a') as f:
            if write_header:
                f.write('sims,n_games,wins,win_pct,wilson_ci_low,'
                        'wilson_ci_high,wall_seconds\n')
            f.write(f'{args.num_simulations},{n},{az_wins},{win_pct:.1f},'
                    f'{lo*100:.1f},{hi*100:.1f},{int(elapsed)}\n')
        print(f'  CSV row written to {args.out_csv}', flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
