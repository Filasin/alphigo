"""Train a standalone value network on chunked SGF data.

Builds a small residual value network (no policy head), optionally
initialises its conv trunk from an existing joint two-head checkpoint,
and trains on the same chunked SGF buffers used by
train_value_joint_b19.py.

Supports both score-margin (default, as stored in v2 chunks) and binary
(±1) value labels via --binary.

Architecture: entry conv → N residual blocks → GlobalAvgPool →
Dense(256, relu) → Dense(1, tanh).  Default N=3, 128 filters ≈ 940K
params.  Trains in ~2–3 h on RTX 3090 for 19×19.
"""
import argparse
import gc
import ctypes
import glob
import os
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--init-from',
        default='runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/'
                'checkpoints/r0_joint_smargin_b19.h5',
        help='Joint two-head checkpoint to copy trunk weights from. '
             'Pass "none" to train from random init.')
    parser.add_argument(
        '--train-glob',
        default='data/sgf_value_buffer_b19_v2_*.h5',
        help='Glob for training chunks.')
    parser.add_argument(
        '--val-chunk',
        default='data/sgf_value_buffer_b19_v2_0015.h5',
        help='Single chunk held out as validation.')
    parser.add_argument(
        '--dst',
        default='runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/'
                'checkpoints/value_standalone_b19.h5')
    parser.add_argument('--num-blocks', type=int, default=3)
    parser.add_argument('--filters', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument(
        '--binary', action='store_true',
        help='Binarise value targets: sign(v) → ±1. Turns score-margin '
             'labels into standard win/loss labels.')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Explicit RNG seed for reproducibility.')
    parser.add_argument(
        '--board-size', type=int, default=19)
    args = parser.parse_args()

    if args.seed is not None:
        import random as _stdrand
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        _stdrand.seed(args.seed)
        np.random.seed(args.seed)
        print(f'[seed] explicit seed set: {args.seed}', flush=True)

    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from alphigo.data.networks import build_value_network, copy_trunk_weights
    from alphigo import kerasutil
    from alphigo.rl.zero_experience import load_zero_buffer

    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)

    # ── Build value network ───────────────────────────────────────────
    from alphigo.encoders.base import get_encoder_by_name
    encoder = get_encoder_by_name('alphazero', args.board_size)
    input_shape = encoder.shape()

    value_model = build_value_network(
        input_shape, num_blocks=args.num_blocks, filters=args.filters)
    print(f'[init] value network: {value_model.count_params():,} params '
          f'({args.num_blocks} blocks, {args.filters} filters)')

    # ── Optionally copy trunk weights from joint model ────────────────
    if args.init_from and args.init_from.lower() != 'none':
        with h5py.File(args.init_from, 'r') as h5:
            source_model = kerasutil.load_model_from_hdf5_group(h5['model'])
        copy_trunk_weights(source_model, value_model,
                           num_blocks=args.num_blocks)
        del source_model
        gc.collect()

    # ── Locate chunk files ────────────────────────────────────────────
    train_paths = sorted(glob.glob(args.train_glob))
    val_abs = os.path.abspath(args.val_chunk)
    train_paths = [p for p in train_paths
                   if os.path.abspath(p) != val_abs]
    if not train_paths:
        print(f'ERROR: no train chunks match {args.train_glob}',
              file=sys.stderr)
        return 1
    if not os.path.exists(args.val_chunk):
        print(f'ERROR: val chunk not found: {args.val_chunk}',
              file=sys.stderr)
        return 1
    print(f'[data] {len(train_paths)} train chunks, 1 val chunk')
    label_mode = 'binary ±1' if args.binary else 'score-margin (as stored)'
    print(f'[data] value labels: {label_mode}')

    # ── Load val set ──────────────────────────────────────────────────
    with h5py.File(args.val_chunk, 'r') as h5:
        val_buf = load_zero_buffer(h5)
    val_x = val_buf.encoded_states.astype(np.float32)
    val_v = val_buf.value_targets.astype(np.float32).reshape(-1, 1)
    if args.binary:
        val_v = np.sign(val_v)
    print(f'[data] val set: {val_x.shape[0]:,} positions  '
          f'value range [{val_v.min():.2f}, {val_v.max():.2f}]')

    # ── Compile ───────────────────────────────────────────────────────
    value_model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss='mse',
        jit_compile=False,
    )

    # ── Memory management ─────────────────────────────────────────────
    try:
        libc = ctypes.CDLL("libc.so.6")
        def trim(): libc.malloc_trim(0)
    except OSError:
        def trim(): pass

    # ── Training loop ─────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    best_val_loss = float('inf')
    best_weights = None
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f'\n===== Epoch {epoch}/{args.epochs} =====', flush=True)
        chunk_order = rng.permutation(len(train_paths))
        for ci_idx, ci in enumerate(chunk_order):
            p = train_paths[ci]
            with h5py.File(p, 'r') as h5:
                buf = load_zero_buffer(h5)
            n = len(buf)
            x = buf.encoded_states.astype(np.float32)
            y_v = buf.value_targets.astype(np.float32).reshape(-1, 1)
            if args.binary:
                y_v = np.sign(y_v)
            del buf
            print(f'  chunk {ci_idx + 1}/{len(chunk_order)}: '
                  f'{os.path.basename(p)} ({n} pos)', flush=True)
            value_model.fit(
                x, y_v,
                batch_size=args.bs, epochs=1, shuffle=True, verbose=2,
            )
            del x, y_v
            gc.collect()
            trim()

        # End-of-epoch validation.
        val_loss = value_model.evaluate(
            val_x, val_v, verbose=0, batch_size=args.bs)
        print(f'  >>> epoch {epoch} val_loss={val_loss:.4f}', flush=True)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights = value_model.get_weights()
            epochs_since_improvement = 0
            print(f'  *** new best val_loss', flush=True)
            os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
            value_model.save(args.dst)
            print(f'  [persist] wrote best to {args.dst}', flush=True)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f'  EARLY STOP at epoch {epoch} '
                      f'(no improvement for {args.patience} epochs)',
                      flush=True)
                break

    if best_weights is not None:
        print(f'\n[restore] best weights: val_loss={best_val_loss:.4f}',
              flush=True)
        value_model.set_weights(best_weights)

    # ── Final save ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
    value_model.save(args.dst)
    print(f'[done] saved {args.dst}')

    final_loss = value_model.evaluate(val_x, val_v, verbose=0)
    print(f'[done] final val_loss={final_loss:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
