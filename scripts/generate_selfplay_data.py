"""Generate self-play training data with value + ownership labels.

Plays games using the current policy (+ optional external value net) via
MCTS, records every position with:
  - encoded board state (alphazero encoder, 17 planes)
  - score-margin value target: tanh(margin / 20) from mover's perspective
  - ownership map: per-intersection +1/-1/0 from final board state

Output: HDF5 chunks in the ZeroReplayBuffer format, extended with an
'ownership_targets' dataset.

Usage:
    python scripts/generate_selfplay_data.py \\
        --agent runs/.../r0_joint_smargin_b19.h5 \\
        --value-model runs/.../value_standalone_margin_b19.h5 \\
        --num-games 500 --num-workers 4 --sims 25 \\
        --out-dir data/selfplay_b19
"""
import argparse
import multiprocessing
import os
import sys
import time

import numpy as np


def _worker(worker_id, conn, agent_path, value_model_path,
            board_size, sims, game_ids, seed, score_margin_scale):
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
    from alphigo.alphigo_board import GameState, area_score, ownership_map
    from alphigo.alphigo_types import Player, Point
    from alphigo.encoders.base import get_encoder_by_name

    ext_value_model = None
    value_mode = 'network'
    if value_model_path:
        ext_value_model = tf.keras.models.load_model(
            value_model_path, compile=False)
        value_mode = 'external'

    with h5py.File(agent_path, 'r') as h5:
        az_agent = load_zero_agent(
            h5, value_mode=value_mode,
            external_value_model=ext_value_model)
    az_agent._num_simulations = int(sims)

    encoder = get_encoder_by_name('alphazero', board_size)

    for game_id in game_ids:
        t0 = time.time()
        game = GameState.new_game(board_size)
        positions = []
        movers = []
        ply = 0

        # Self-play: same agent plays both sides
        while not game.is_over() and ply < 600:
            # Temperature: exploratory for first 30 moves, then argmax
            az_agent.set_temperature(1.0 if ply < 30 else 0.0)
            encoded = encoder.encode(game)
            move = az_agent.select_move(game)

            if not move.is_pass and not move.is_resign:
                positions.append(encoded)
                movers.append(game.next_player)

            game = game.apply_move(move)
            ply += 1

        if not positions:
            conn.send(('skip', game_id, 0))
            continue

        # Score the final position
        b_score, w_score = area_score(game.board, komi=7.5)
        margin = b_score - w_score  # positive = black wins
        if margin > 0:
            winner = Player.black
        elif margin < 0:
            winner = Player.white
        else:
            winner = None

        # Ownership map from final board state
        omap = ownership_map(game.board)  # (board_size, board_size)
        omap_flat = omap.flatten()  # (board_size^2,)

        # Build training arrays
        n = len(positions)
        encoded_states = np.stack(positions)  # (N, H, W, 17)
        value_targets = np.zeros(n, dtype=np.float32)
        ownership_targets = np.zeros((n, board_size * board_size),
                                     dtype=np.float32)

        for i, mover in enumerate(movers):
            # Score margin from mover's perspective
            m = margin if mover == Player.black else -margin
            value_targets[i] = np.tanh(m / score_margin_scale)
            # Ownership from mover's perspective
            if mover == Player.black:
                ownership_targets[i] = omap_flat
            else:
                ownership_targets[i] = -omap_flat

        elapsed = time.time() - t0
        conn.send(('game', game_id, n, elapsed,
                    encoded_states, value_targets, ownership_targets))
        winner_str = 'B' if winner == Player.black else 'W' if winner == Player.white else 'draw'
        print(f'  [w{worker_id} g{game_id}] '
              f'{ply} moves, winner={winner_str} margin={margin:.1f} '
              f'positions={n} ({elapsed:.0f}s)', flush=True)

    conn.send(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True,
                        help='Path to AZ agent .h5 file')
    parser.add_argument('--value-model', default=None,
                        help='Path to separate value network')
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--sims', type=int, default=25)
    parser.add_argument('--num-games', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for HDF5 chunks')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Games per HDF5 chunk')
    parser.add_argument('--score-margin-scale', type=float, default=20.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Distribute games round-robin
    worker_games = [[] for _ in range(args.num_workers)]
    for g in range(args.num_games):
        worker_games[g % args.num_workers].append(g)

    print(f'=== generate_selfplay_data ===', flush=True)
    print(f'  agent:    {args.agent}', flush=True)
    print(f'  value:    {args.value_model}', flush=True)
    print(f'  sims:     {args.sims}', flush=True)
    print(f'  games:    {args.num_games} (workers={args.num_workers})', flush=True)
    print(f'  out:      {args.out_dir}', flush=True)

    ctx = multiprocessing.get_context('spawn')
    t0 = time.time()
    procs = []
    parent_conns = []

    for i, ids in enumerate(worker_games):
        if not ids:
            continue
        seed = args.seed + (i + 1) * 7919
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=_worker,
            args=(i, child_conn, args.agent, args.value_model,
                  args.board_size, args.sims, ids, seed,
                  args.score_margin_scale),
        )
        p.start()
        procs.append((i, p))
        parent_conns.append(parent_conn)

    # Collect results and write chunks
    import h5py
    all_states, all_values, all_ownership = [], [], []
    total_positions = 0
    total_games = 0
    chunk_idx = 0

    def flush_chunk():
        nonlocal chunk_idx, all_states, all_values, all_ownership, total_positions
        if not all_states:
            return
        states = np.concatenate(all_states)
        values = np.concatenate(all_values)
        owns = np.concatenate(all_ownership)

        # Shuffle within chunk
        perm = np.random.permutation(len(states))
        states, values, owns = states[perm], values[perm], owns[perm]

        path = os.path.join(args.out_dir,
                            f'selfplay_b{args.board_size}_{chunk_idx:04d}.h5')
        with h5py.File(path, 'w') as h5:
            grp = h5.create_group('zero_experience')
            grp.create_dataset('encoded_states', data=states)
            grp.create_dataset('policy_targets',
                               data=np.zeros((len(states),
                                              args.board_size**2),
                                             dtype=np.float32))
            grp.create_dataset('value_targets', data=values)
            grp.create_dataset('ownership_targets', data=owns)
            h5.attrs['encoder'] = 'alphazero'
            h5.attrs['board_size'] = args.board_size
            h5.attrs['num_positions'] = len(states)
            h5.attrs['chunk_idx'] = chunk_idx
        total_positions += len(states)
        print(f'  chunk {chunk_idx}: {len(states)} positions → {path}',
              flush=True)
        chunk_idx += 1
        all_states, all_values, all_ownership = [], [], []

    games_in_chunk = 0
    for (i, p), conn in zip(procs, parent_conns):
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                print(f'  WARN: worker {i} died', flush=True)
                break
            if msg is None:
                break
            if msg[0] == 'skip':
                continue
            _, game_id, n, elapsed, states, values, owns = msg
            all_states.append(states)
            all_values.append(values)
            all_ownership.append(owns)
            total_games += 1
            games_in_chunk += 1
            if games_in_chunk >= args.chunk_size:
                flush_chunk()
                games_in_chunk = 0
        p.join()

    flush_chunk()  # remaining games

    elapsed = time.time() - t0
    print(f'\n=== Done ===', flush=True)
    print(f'  {total_games} games, {total_positions} positions, '
          f'{chunk_idx} chunks', flush=True)
    print(f'  wall: {elapsed:.0f}s ({elapsed/3600:.1f}h)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
