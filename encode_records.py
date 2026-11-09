#!/usr/bin/env python
"""Replay binary game records through the Python engine and encode positions
with arbitrary encoders to produce NPY train/test files.

The binary records file is produced by generate_mcst_games(.exe) and contains
the raw move sequences for every game. This script replays each game through
the full Python GameState, so history-aware encoders (e.g. alphazero with 8
previous frames) work correctly.

Uses the same shuffle seed and split ratio as merge_batches.py so all encoders
— C++ and Python — share the exact same train/test partition.

Binary format (all little-endian):
    int32: total number of games
    For each game:
        int16: number of moves
        int16[num_moves]: move indices
            play move = row * board_size + col
            pass      = -1

Usage:
    python encode_records.py generated_games/mcst_9x9/mcts_game_records.bin \\
        --encoders nineplane,agplane,alphazero --board-size 9 \\
        --out-dir generated_games/mcst_9x9 --tag mcts --seed 42 --split 0.8
"""

import argparse
import os
import struct
import sys

import numpy as np


def read_game_records(path):
    """Read binary game records file. Returns list of lists of int16 move indices."""
    with open(path, 'rb') as f:
        total_games = struct.unpack('<i', f.read(4))[0]
        games = []
        for _ in range(total_games):
            num_moves = struct.unpack('<h', f.read(2))[0]
            if num_moves > 0:
                moves = list(struct.unpack(f'<{num_moves}h', f.read(2 * num_moves)))
            else:
                moves = []
            games.append(moves)
    return games


def replay_and_encode(games, board_size, encoder_names):
    """Replay games through Python engine, encode each play-move position.

    Returns dict mapping encoder_name -> ndarray of shape (total_positions, H, W, planes),
    and a labels array of shape (total_positions, board_size^2).
    """
    from alphigo.alphigo_board import GameState, Move
    from alphigo.alphigo_types import Point
    from alphigo.encoders.base import get_encoder_by_name

    encoders = {name: get_encoder_by_name(name, board_size) for name in encoder_names}
    num_classes = board_size * board_size

    # Pre-allocate lists for each encoder
    features = {name: [] for name in encoder_names}
    labels = []

    total_games = len(games)
    for gi, move_indices in enumerate(games):
        if gi % 500 == 0:
            print(f'  Replaying game {gi + 1}/{total_games}...', flush=True)

        game = GameState.new_game(board_size)
        for move_idx in move_indices:
            if move_idx == -1:
                # Pass
                move = Move.pass_turn()
            else:
                row = move_idx // board_size + 1  # Point is 1-indexed
                col = move_idx % board_size + 1
                point = Point(row, col)
                move = Move.play_move(point)

                # Encode current position before applying the move
                for name, enc in encoders.items():
                    features[name].append(enc.encode(game))

                # Label: one-hot for the move index
                label = np.zeros(num_classes, dtype=np.float32)
                label[move_idx] = 1.0
                labels.append(label)

            game = game.apply_move(move)

    # Stack into arrays
    feature_arrays = {}
    for name in encoder_names:
        if features[name]:
            feature_arrays[name] = np.array(features[name], dtype=np.float32)
        else:
            feature_arrays[name] = np.empty((0,), dtype=np.float32)

    label_array = np.array(labels, dtype=np.float32) if labels else np.empty((0, num_classes), dtype=np.float32)
    return feature_arrays, label_array


def main():
    parser = argparse.ArgumentParser(
        description='Encode game records with Python encoders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('records', help='Path to game records .bin file')
    parser.add_argument('--encoders', required=True,
                        help='Comma-separated encoder names (e.g. nineplane,agplane,alphazero)')
    parser.add_argument('--board-size', type=int, required=True,
                        help='Board size (must match the records)')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for NPY files')
    parser.add_argument('--tag', default='mcts',
                        help='Filename prefix (default: mcts)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Shuffle seed (default: 42)')
    parser.add_argument('--split', type=float, default=0.8,
                        help='Train fraction (default: 0.8)')
    args = parser.parse_args()

    encoder_names = [e.strip() for e in args.encoders.split(',')]

    print(f'Reading game records from {args.records}...')
    games = read_game_records(args.records)
    print(f'  {len(games)} games loaded.')

    print(f'Replaying and encoding with: {encoder_names}')
    feature_arrays, label_array = replay_and_encode(games, args.board_size, encoder_names)
    total = len(label_array)
    print(f'  {total} positions encoded.')

    if total == 0:
        print('No positions to save. Exiting.')
        sys.exit(0)

    # Shuffle and split with same seed/ratio as merge_batches.py
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(total)
    cut = int(args.split * total)
    print(f'Shuffle seed={args.seed}  split={args.split}  train={cut}  test={total - cut}')

    os.makedirs(args.out_dir, exist_ok=True)

    for name, features in feature_arrays.items():
        features = features[idx]
        train_path = os.path.join(args.out_dir, f'{args.tag}_{name}_features_train.npy')
        test_path = os.path.join(args.out_dir, f'{args.tag}_{name}_features_test.npy')
        np.save(train_path, features[:cut])
        np.save(test_path, features[cut:])
        print(f'  [{name}] saved  train={cut}  test={total - cut}')

    # Labels are shared — only save if not already present (C++ pipeline saves them).
    # Check if labels already exist from merge_batches.py; if so, skip.
    labels_train_path = os.path.join(args.out_dir, f'{args.tag}_labels_train.npy')
    labels_test_path = os.path.join(args.out_dir, f'{args.tag}_labels_test.npy')
    if os.path.exists(labels_train_path) and os.path.exists(labels_test_path):
        print(f'  [labels] Already exist — skipping (shared from C++ pipeline).')
    else:
        label_array = label_array[idx]
        np.save(labels_train_path, label_array[:cut])
        np.save(labels_test_path, label_array[cut:])
        print(f'  [labels] saved  train={cut}  test={total - cut}')

    print('Done.')


if __name__ == '__main__':
    main()
