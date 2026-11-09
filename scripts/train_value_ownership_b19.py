"""Train a value+ownership network on self-play data.

Loads HDF5 chunks produced by generate_selfplay_data.py (which include
ownership_targets alongside the standard zero_experience fields).
Trains with combined loss: MSE(value) + lambda * MSE(ownership).

Architecture: entry conv → N residual blocks → {value head, ownership head}.
"""
import argparse
import gc
import ctypes
import glob
import os
import sys

import h5py
import numpy as np


def load_selfplay_chunk(path):
    """Load a self-play chunk with ownership targets."""
    with h5py.File(path, 'r') as h5:
        grp = h5['zero_experience']
        states = np.array(grp['encoded_states'])
        values = np.array(grp['value_targets'])
        if 'ownership_targets' in grp:
            ownership = np.array(grp['ownership_targets'])
        else:
            ownership = None
    return states, values, ownership


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--init-from',
        default='runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/'
                'checkpoints/r0_joint_smargin_b19.h5',
        help='Joint two-head checkpoint to copy trunk weights from.')
    parser.add_argument(
        '--train-glob', required=True,
        help='Glob for self-play training chunks.')
    parser.add_argument(
        '--val-glob', default=None,
        help='Glob for validation chunks. If not set, holds out last chunk.')
    parser.add_argument(
        '--dst', required=True,
        help='Output path for trained model.')
    parser.add_argument('--num-blocks', type=int, default=3)
    parser.add_argument('--filters', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--ownership-weight', type=float, default=0.5,
                        help='Weight for ownership loss (value loss = 1.0).')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--board-size', type=int, default=19)
    args = parser.parse_args()

    import random as _stdrand
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    _stdrand.seed(args.seed)
    np.random.seed(args.seed)
    print(f'[seed] {args.seed}', flush=True)

    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from alphigo.data.networks import (
        build_value_ownership_network, copy_trunk_weights)
    from alphigo import kerasutil

    tf.keras.utils.set_random_seed(args.seed)

    # ── Build model ──────────────────────────────────────────────────
    from alphigo.encoders.base import get_encoder_by_name
    encoder = get_encoder_by_name('alphazero', args.board_size)
    input_shape = encoder.shape()

    model = build_value_ownership_network(
        input_shape, board_size=args.board_size,
        num_blocks=args.num_blocks, filters=args.filters)
    print(f'[init] value+ownership network: {model.count_params():,} params')

    # ── Copy trunk weights ───────────────────────────────────────────
    if args.init_from and args.init_from.lower() != 'none':
        with h5py.File(args.init_from, 'r') as h5:
            source_model = kerasutil.load_model_from_hdf5_group(h5['model'])
        copy_trunk_weights(source_model, model, num_blocks=args.num_blocks)
        del source_model
        gc.collect()

    # ── Locate chunks ────────────────────────────────────────────────
    all_paths = sorted(glob.glob(args.train_glob))
    if not all_paths:
        print(f'ERROR: no chunks match {args.train_glob}', file=sys.stderr)
        return 1

    if args.val_glob:
        val_paths = sorted(glob.glob(args.val_glob))
        train_paths = [p for p in all_paths
                       if p not in set(val_paths)]
    else:
        val_paths = [all_paths[-1]]
        train_paths = all_paths[:-1]

    if not train_paths:
        print('ERROR: no training chunks after val split', file=sys.stderr)
        return 1

    print(f'[data] {len(train_paths)} train, {len(val_paths)} val chunks')

    # ── Load val set ─────────────────────────────────────────────────
    val_states, val_values, val_owns = [], [], []
    for p in val_paths:
        s, v, o = load_selfplay_chunk(p)
        val_states.append(s)
        val_values.append(v)
        if o is not None:
            val_owns.append(o)
    val_x = np.concatenate(val_states).astype(np.float32)
    val_v = np.concatenate(val_values).astype(np.float32).reshape(-1, 1)
    has_ownership = len(val_owns) > 0
    if has_ownership:
        val_o = np.concatenate(val_owns).astype(np.float32)
    else:
        val_o = np.zeros((len(val_x), args.board_size**2), dtype=np.float32)
    print(f'[data] val: {len(val_x):,} positions, '
          f'ownership={has_ownership}')

    # ── Compile ──────────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss={
            'value_output': 'mse',
            'ownership_output': 'mse',
        },
        loss_weights={
            'value_output': 1.0,
            'ownership_output': args.ownership_weight,
        },
        jit_compile=False,
    )

    # ── Memory management ────────────────────────────────────────────
    try:
        libc = ctypes.CDLL("libc.so.6")
        def trim(): libc.malloc_trim(0)
    except OSError:
        def trim(): pass

    # ── Training loop ────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    best_val_loss = float('inf')
    best_weights = None
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f'\n===== Epoch {epoch}/{args.epochs} =====', flush=True)
        chunk_order = rng.permutation(len(train_paths))

        for ci_idx, ci in enumerate(chunk_order):
            p = train_paths[ci]
            x, y_v, y_o = load_selfplay_chunk(p)
            x = x.astype(np.float32)
            y_v = y_v.astype(np.float32).reshape(-1, 1)
            if y_o is None:
                y_o = np.zeros((len(x), args.board_size**2),
                               dtype=np.float32)
            else:
                y_o = y_o.astype(np.float32)

            print(f'  chunk {ci_idx + 1}/{len(chunk_order)}: '
                  f'{os.path.basename(p)} ({len(x)} pos)', flush=True)
            model.fit(
                x,
                {'value_output': y_v, 'ownership_output': y_o},
                batch_size=args.bs, epochs=1, shuffle=True, verbose=2,
            )
            del x, y_v, y_o
            gc.collect()
            trim()

        # Validation
        val_results = model.evaluate(
            val_x,
            {'value_output': val_v, 'ownership_output': val_o},
            verbose=0, batch_size=args.bs, return_dict=True)
        val_loss = val_results['loss']
        val_v_loss = val_results.get('value_output_loss', '?')
        val_o_loss = val_results.get('ownership_output_loss', '?')
        print(f'  >>> epoch {epoch} total={val_loss:.4f} '
              f'value={val_v_loss:.4f} ownership={val_o_loss:.4f}',
              flush=True)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            epochs_since_improvement = 0
            print(f'  *** new best', flush=True)
            os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
            model.save(args.dst)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f'  EARLY STOP at epoch {epoch}', flush=True)
                break

    if best_weights is not None:
        model.set_weights(best_weights)

    os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
    model.save(args.dst)
    print(f'[done] saved {args.dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
