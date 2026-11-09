"""Convert C++ MCTS .npy batch files to HDF5 chunks for train_distill_b19.py.

Memory-efficient: loads one .npy batch at a time, shuffles within each,
writes HDF5 chunks. Training-time chunk shuffling provides cross-batch mixing.

Usage:
  python scripts/convert_npy_to_h5.py \
      --input-dir data/bootstrap --tag mcts --board-size 9 \
      --out-dir data/selfplay_bootstrap_b9 --chunk-size 5000
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--tag', default='mcts')
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--chunk-size', type=int, default=5000)
    args = parser.parse_args()

    pat = os.path.join(args.input_dir, f'{args.tag}_b*_alphazero_features.npy')
    feat_files = sorted(glob.glob(pat))
    if not feat_files:
        print(f'ERROR: no feature files match {pat}', file=sys.stderr)
        return 1

    rng = np.random.default_rng(42)

    # Shuffle batch processing order
    batch_order = rng.permutation(len(feat_files))

    os.makedirs(args.out_dir, exist_ok=True)
    b = args.board_size
    chunk_idx = 0
    total_positions = 0

    # Residual buffer for partial chunks between batches
    buf_X, buf_P, buf_V, buf_O = [], [], [], []
    buf_len = 0

    def flush_chunk(X, P, V, O):
        nonlocal chunk_idx
        path = os.path.join(args.out_dir, f'selfplay_b{b}_{chunk_idx:04d}.h5')
        with h5py.File(path, 'w') as h5:
            grp = h5.create_group('zero_experience')
            grp.create_dataset('encoded_states', data=X)
            grp.create_dataset('policy_targets', data=P)
            grp.create_dataset('value_targets', data=V)
            grp.create_dataset('ownership_targets', data=O)
        chunk_idx += 1
        if chunk_idx % 50 == 0 or chunk_idx == 1:
            print(f'  chunk {chunk_idx}: {path} ({len(X)} pos)')

    for bi, file_idx in enumerate(batch_order):
        fp = feat_files[file_idx]
        batch_id = os.path.basename(fp).split('_alphazero_features')[0]
        base = os.path.join(args.input_dir, batch_id)

        feats = np.load(fp).astype(np.float32)
        if feats.ndim == 4 and feats.shape[1] > feats.shape[2]:
            # NCHW → NHWC
            feats = feats.transpose(0, 2, 3, 1)
        policy = np.load(base + '_policy.npy').astype(np.float32)
        values = np.load(base + '_values.npy').astype(np.float32)
        ownership = np.load(base + '_ownership.npy').astype(np.float32)

        # Shuffle within this batch
        perm = rng.permutation(len(feats))
        feats, policy, values, ownership = feats[perm], policy[perm], values[perm], ownership[perm]
        total_positions += len(feats)

        if (bi + 1) % 10 == 0 or bi == 0:
            print(f'  batch {bi+1}/{len(feat_files)}: {len(feats)} pos, '
                  f'total so far: {total_positions}')

        # Add to buffer
        buf_X.append(feats)
        buf_P.append(policy)
        buf_V.append(values)
        buf_O.append(ownership)
        buf_len += len(feats)

        # Flush full chunks from buffer
        while buf_len >= args.chunk_size:
            cat_X = np.concatenate(buf_X)
            cat_P = np.concatenate(buf_P)
            cat_V = np.concatenate(buf_V)
            cat_O = np.concatenate(buf_O)

            flush_chunk(cat_X[:args.chunk_size], cat_P[:args.chunk_size],
                        cat_V[:args.chunk_size], cat_O[:args.chunk_size])

            remainder = len(cat_X) - args.chunk_size
            if remainder > 0:
                buf_X = [cat_X[args.chunk_size:]]
                buf_P = [cat_P[args.chunk_size:]]
                buf_V = [cat_V[args.chunk_size:]]
                buf_O = [cat_O[args.chunk_size:]]
            else:
                buf_X, buf_P, buf_V, buf_O = [], [], [], []
            buf_len = remainder

        # Free batch memory
        del feats, policy, values, ownership

    # Flush remaining buffer
    if buf_len > 0:
        cat_X = np.concatenate(buf_X)
        cat_P = np.concatenate(buf_P)
        cat_V = np.concatenate(buf_V)
        cat_O = np.concatenate(buf_O)
        flush_chunk(cat_X, cat_P, cat_V, cat_O)

    print(f'\nDone: {chunk_idx} chunks, {total_positions} positions in {args.out_dir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
