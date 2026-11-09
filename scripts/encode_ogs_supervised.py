#!/usr/bin/env python3
"""Encode OGS 9x9 games into HDF5 training data for the 3-head network.

Filters the OGS 2021 JSONL dump for 9x9 games from strong players,
replays each game through the AlphaZero encoder, and writes HDF5 chunks
compatible with train_distill_b19.py.

Targets: one-hot policy (human move), +/-1 value (from mover's perspective),
and per-point ownership from the final board state.

Usage:
    python scripts/encode_ogs_supervised.py \
        --input data/ogs2021/ogs_games_2013_to_2021-08.json.gz \
        --out-dir data/ogs_supervised_b9 \
        --min-rank 25 --max-games 50000 --workers 6
"""
import argparse
import gzip
import json
import multiprocessing
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphigo.alphigo_board import GameState, Move, ownership_map
from alphigo.alphigo_types import Player, Point
from alphigo.encoders.alphazero import AlphaZeroEncoder


def encode_game(game_data, encoder, board_size):
    """Replay one OGS game and produce per-position training data."""
    moves_raw = game_data['moves']
    winner_id = game_data.get('winner')
    black_info = game_data.get('players', {}).get('black', {})
    white_info = game_data.get('players', {}).get('white', {})
    black_id = black_info.get('id')
    white_id = white_info.get('id')

    if winner_id is None or (winner_id != black_id and winner_id != white_id):
        return None
    winner = Player.black if winner_id == black_id else Player.white

    game = GameState.new_game(board_size)
    encoded_states = []
    policy_targets = []
    movers = []

    for move_data in moves_raw:
        col, row = int(move_data[0]), int(move_data[1])

        if col < 0 or row < 0 or col >= board_size or row >= board_size:
            game = game.apply_move(Move.pass_turn())
            continue

        point = Point(row=row + 1, col=col + 1)
        move = Move.play_move(point)

        if not game.is_valid_move(move):
            return None

        encoded = encoder.encode(game)

        policy = np.zeros(board_size * board_size, dtype=np.float32)
        policy[board_size * row + col] = 1.0

        encoded_states.append(encoded)
        policy_targets.append(policy)
        movers.append(game.next_player)

        game = game.apply_move(move)

    if not encoded_states:
        return None

    omap = ownership_map(game.board).flatten()

    n = len(encoded_states)
    states = np.stack(encoded_states)
    policies = np.stack(policy_targets)
    values = np.zeros(n, dtype=np.float32)
    ownerships = np.zeros((n, board_size * board_size), dtype=np.float32)

    for i, mover in enumerate(movers):
        values[i] = 1.0 if mover == winner else -1.0
        ownerships[i] = omap if mover == Player.black else -omap

    return states, policies, values, ownerships


# ── Multiprocessing worker ───────────────────────────────────────────

_worker_encoder = None
_worker_bs = None


def _init_worker(board_size):
    global _worker_encoder, _worker_bs
    _worker_bs = board_size
    _worker_encoder = AlphaZeroEncoder((board_size, board_size))


def _encode_worker(game_data):
    return encode_game(game_data, _worker_encoder, _worker_bs)


# ── Chunk writer ─────────────────────────────────────────────────────

def write_chunk(out_dir, board_size, chunk_idx, states, policies,
                values, ownerships):
    path = os.path.join(
        out_dir, f'ogs_supervised_b{board_size}_{chunk_idx:04d}.h5')
    with h5py.File(path, 'w') as h5:
        grp = h5.create_group('zero_experience')
        grp.create_dataset('encoded_states', data=states,
                           compression='gzip', compression_opts=1)
        grp.create_dataset('policy_targets', data=policies,
                           compression='gzip', compression_opts=1)
        grp.create_dataset('value_targets', data=values)
        grp.create_dataset('ownership_targets', data=ownerships,
                           compression='gzip', compression_opts=1)
        h5.attrs['encoder'] = 'alphazero'
        h5.attrs['board_size'] = board_size
        h5.attrs['num_positions'] = len(states)
        h5.attrs['chunk_idx'] = chunk_idx
        h5.attrs['source'] = 'ogs_supervised'
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='OGS JSONL gzip file')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--board-size', type=int, default=9)
    p.add_argument('--min-rank', type=int, default=25,
                   help='Min rank for BOTH players (25=5k, 20=10k, 30=1d)')
    p.add_argument('--min-moves', type=int, default=20)
    p.add_argument('--chunk-size', type=int, default=50000,
                   help='Positions per HDF5 chunk')
    p.add_argument('--max-games', type=int, default=0,
                   help='Stop after encoding N games (0=all)')
    p.add_argument('--workers', type=int, default=1,
                   help='Parallel encoding workers (1=single-process)')
    p.add_argument('--exclude-timeout', action='store_true', default=True,
                   help='Skip timeout games (default: yes)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bs = args.board_size

    # ── Pass 1: filter games ─────────────────────────────────────────
    print(f'Scanning {args.input} for {bs}x{bs} games '
          f'(min_rank={args.min_rank}, min_moves={args.min_moves}) ...')

    filtered_games = []
    scan_count = 0
    skip_stats = {'size': 0, 'rank': 0, 'moves': 0, 'timeout': 0}
    t0 = time.time()

    with gzip.open(args.input, 'rt') as f:
        for line in f:
            scan_count += 1
            d = json.loads(line)

            if scan_count % 1000000 == 0:
                elapsed = time.time() - t0
                print(f'  scanned {scan_count:,} lines, '
                      f'{len(filtered_games):,} matched ({elapsed:.0f}s)',
                      flush=True)

            if d.get('width') != bs or d.get('height') != bs:
                skip_stats['size'] += 1
                continue
            if args.exclude_timeout and d.get('outcome') == 'Timeout':
                skip_stats['timeout'] += 1
                continue
            moves = d.get('moves', [])
            if len(moves) < args.min_moves:
                skip_stats['moves'] += 1
                continue

            players = d.get('players', {})
            ranks_ok = True
            for color in ('black', 'white'):
                rank = players.get(color, {}).get('rank')
                if rank is None or int(rank) < args.min_rank:
                    ranks_ok = False
                    break
            if not ranks_ok:
                skip_stats['rank'] += 1
                continue

            filtered_games.append(d)

            if args.max_games and len(filtered_games) >= args.max_games:
                break

            if len(filtered_games) % 5000 == 0:
                elapsed = time.time() - t0
                print(f'  filtered {len(filtered_games):,} games '
                      f'from {scan_count:,} scanned ({elapsed:.0f}s)',
                      flush=True)

    elapsed = time.time() - t0
    print(f'Pass 1 done: {len(filtered_games):,} games from '
          f'{scan_count:,} scanned ({elapsed:.0f}s)')
    print(f'  skipped: {skip_stats}')

    if not filtered_games:
        print('No games matched filters. Exiting.')
        return

    # ── Pass 2: encode games ─────────────────────────────────────────
    print(f'\nEncoding {len(filtered_games):,} games '
          f'(workers={args.workers}) ...')

    buf_states = []
    buf_policies = []
    buf_values = []
    buf_ownerships = []
    buf_positions = 0
    chunk_idx = 0
    total_positions = 0
    total_encoded = 0
    errors = 0
    t0 = time.time()

    def flush_chunk():
        nonlocal buf_states, buf_policies, buf_values, buf_ownerships
        nonlocal buf_positions, chunk_idx, total_positions
        if not buf_states:
            return
        s = np.concatenate(buf_states)
        p = np.concatenate(buf_policies)
        v = np.concatenate(buf_values)
        o = np.concatenate(buf_ownerships)
        path = write_chunk(args.out_dir, bs, chunk_idx, s, p, v, o)
        print(f'  chunk {chunk_idx}: {len(s):,} positions -> {path}')
        total_positions += len(s)
        chunk_idx += 1
        buf_states, buf_policies, buf_values, buf_ownerships = [], [], [], []
        buf_positions = 0

    if args.workers > 1:
        pool = multiprocessing.Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(bs,))
        results = pool.imap_unordered(_encode_worker, filtered_games,
                                      chunksize=50)
    else:
        _init_worker(bs)
        results = map(_encode_worker, filtered_games)

    for result in results:
        if result is None:
            errors += 1
            continue

        states, policies, values, ownerships = result
        buf_states.append(states)
        buf_policies.append(policies)
        buf_values.append(values)
        buf_ownerships.append(ownerships)
        buf_positions += len(states)
        total_encoded += 1

        if buf_positions >= args.chunk_size:
            flush_chunk()

        if total_encoded % 500 == 0:
            elapsed = time.time() - t0
            rate = total_encoded / max(elapsed, 0.001)
            pos_so_far = total_positions + buf_positions
            print(f'  {total_encoded:,} games ({rate:.0f}/s), '
                  f'{pos_so_far:,} positions, '
                  f'{errors} errors ({elapsed:.0f}s)')

    flush_chunk()

    if args.workers > 1:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    print(f'\nDone: {total_encoded:,} games -> {total_positions:,} positions '
          f'in {chunk_idx} chunks ({elapsed:.0f}s)')
    print(f'  errors/invalid: {errors}')
    print(f'  avg positions/game: {total_positions / max(total_encoded, 1):.1f}')
    print(f'  output: {args.out_dir}/')


if __name__ == '__main__':
    main()
