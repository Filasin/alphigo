"""Merge per-batch .npy files produced by generate_mcst_games into
final train/test splits, with a global shuffle.

Also recovers data from old-style binary checkpoint files
(mcts_ckpt_Y.bin / mcts_ckpt_*_X.bin) if present alongside the batch files.

Usage:
    python merge_batches.py data/mcst_9x9 --tag mcts --split 0.8
"""

import argparse
import glob
import os
import struct

import numpy as np

ENC_PLANES = {'oneplane': 1, 'sevenplane': 7, 'elevenplane': 11}


def _load_ckpt_bin(out_dir, tag):
    """Read old-style binary checkpoint files into numpy arrays.

    Returns (labels, features_dict) where features_dict maps encoder name
    to ndarray, or (None, {}) if no checkpoint is found.
    """
    meta_path = os.path.join(out_dir, f'{tag}_ckpt_meta.bin')
    y_path    = os.path.join(out_dir, f'{tag}_ckpt_Y.bin')
    if not os.path.exists(meta_path) or not os.path.exists(y_path):
        return None, {}

    with open(meta_path, 'rb') as f:
        meta = struct.unpack('<3q', f.read(24))  # 3 x int64 little-endian
    positions_done = meta[1]
    if positions_done == 0:
        return None, {}

    # Infer N (board_size^2) from label file size.
    label_bytes = os.path.getsize(y_path)
    N = label_bytes // (positions_done * 4)
    if N == 0 or label_bytes % (positions_done * 4) != 0:
        print(f"  [ckpt] Cannot infer board size from {y_path}, skipping.")
        return None, {}

    labels = np.fromfile(y_path, dtype='<f4').reshape(positions_done, N)
    print(f"  [ckpt] Loaded {positions_done} positions from binary checkpoint (N={N})")

    features = {}
    for enc, planes in ENC_PLANES.items():
        x_path = os.path.join(out_dir, f'{tag}_ckpt_{enc}_X.bin')
        if not os.path.exists(x_path):
            continue
        expected = positions_done * planes * N
        actual   = os.path.getsize(x_path) // 4
        if actual != expected:
            print(f"  [ckpt] Size mismatch for {enc}, skipping encoder.")
            continue
        board_size = int(N ** 0.5)
        features[enc] = np.fromfile(x_path, dtype='<f4').reshape(
            positions_done, planes, board_size, board_size)

    return labels, features


def merge(out_dir, tag='mcts', split=0.8, seed=42):
    label_files = sorted(glob.glob(os.path.join(out_dir, f'{tag}_b*_labels.npy')))

    # Load binary checkpoint data if present (old-format batches b0001/b0002).
    ckpt_labels, ckpt_features = _load_ckpt_bin(out_dir, tag)

    if not label_files and ckpt_labels is None:
        raise FileNotFoundError(
            f"No batch .npy files or binary checkpoint found in {out_dir} with tag '{tag}'")

    print(f"Found {len(label_files)} .npy batch file(s)" +
          (f" + binary checkpoint ({len(ckpt_labels)} positions)" if ckpt_labels is not None else ""))

    # Concatenate labels: binary checkpoint first (chronologically), then batches.
    parts = []
    if ckpt_labels is not None:
        parts.append(ckpt_labels)
    parts += [np.load(f) for f in label_files]
    labels = np.concatenate(parts)
    total  = len(labels)
    cut    = int(split * total)
    print(f"Total positions: {total}  train={cut}  test={total - cut}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(total)

    labels = labels[idx]
    np.save(os.path.join(out_dir, f'{tag}_labels_train.npy'), labels[:cut])
    np.save(os.path.join(out_dir, f'{tag}_labels_test.npy'),  labels[cut:])
    del labels

    # Process each encoder separately to keep peak RAM low.
    for enc in ('oneplane', 'sevenplane', 'elevenplane'):
        feat_files = sorted(glob.glob(
            os.path.join(out_dir, f'{tag}_b*_{enc}_features.npy')))
        has_ckpt = ckpt_features.get(enc) is not None
        if not feat_files and not has_ckpt:
            continue

        parts = []
        if has_ckpt:
            parts.append(ckpt_features[enc])
        parts += [np.load(f) for f in feat_files]
        features = np.concatenate(parts)
        features = features[idx]
        np.save(os.path.join(out_dir, f'{tag}_{enc}_features_train.npy'), features[:cut])
        np.save(os.path.join(out_dir, f'{tag}_{enc}_features_test.npy'),  features[cut:])
        print(f"  [{enc}] saved  train={cut}  test={total - cut}")
        del features

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge generate_mcst_games batch files')
    parser.add_argument('out_dir', help='Directory containing batch .npy files')
    parser.add_argument('--tag',   default='mcts',  help='Filename prefix tag (default: mcts)')
    parser.add_argument('--split', type=float, default=0.8, help='Train fraction (default: 0.8)')
    parser.add_argument('--seed',  type=int,   default=42,  help='Shuffle seed (default: 42)')
    args = parser.parse_args()
    merge(args.out_dir, args.tag, args.split, args.seed)
