"""Joint policy + value training, streaming all chunks from disk.

Loads a two-head AlphaZero-style checkpoint and continues training on a
chunked SGF buffer with the full AlphaZero supervised loss:

    L = CE(policy_target, π) + alpha * MSE(value_target, v)

Unlike train_value_head_only.py, this script:
  - does NOT freeze the trunk/policy — all layers are trainable
  - does NOT load the full corpus into RAM — chunks are streamed via
    tf.data.Dataset.from_generator with a large in-memory shuffle buffer
  - holds out one full chunk as the validation set (cleaner than the
    Keras --validation-split slice, which would mix chunks)

Designed for the 19x19 corpus (~2M positions, ~53 GB on disk). One
training epoch streams every chunk through the GPU once; expected wall
~80-120 min/epoch on RTX 3090 depending on I/O.

Saves the resulting checkpoint in the same .h5 format as ZeroAgent.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src',
        default='runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/'
                'checkpoints/round_000.h5',
        help='Source two-head checkpoint to initialise from.')
    parser.add_argument(
        '--train-glob',
        default='data/sgf_value_buffer_b19_v2_*.h5',
        help='Glob for training chunks (excluding val chunk).')
    parser.add_argument(
        '--val-chunk',
        default='data/sgf_value_buffer_b19_v2_0009.h5',
        help='Single chunk held out as validation. Must NOT be matched by '
             '--train-glob (will be filtered out if it is).')
    parser.add_argument(
        '--dst',
        default='runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/'
                'checkpoints/r0_joint_smargin_b19.h5')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Weight on the value MSE loss. Policy CE weight is 1.0. '
             'Default 0.5 gives roughly equal gradient norms in practice '
             'on Go-scale data (CE on 361 classes vs MSE on 1 scalar).')
    parser.add_argument(
        '--shuffle-buffer', type=int, default=50000,
        help='Size of the tf.data.Dataset.shuffle buffer (in samples). '
             'Larger = better shuffle, more RAM. 50k samples × ~25 KB '
             '= ~1.2 GB; fine on a 32 GB system.')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Explicit RNG seed. When set, numpy, python, and TF random '
             'state are all seeded for reproducibility. The chunk and '
             'mini-batch ordering will be deterministic given this seed.')
    args = parser.parse_args()

    if args.seed is not None:
        import random as _stdrand
        import os as _os
        _os.environ['PYTHONHASHSEED'] = str(args.seed)
        _stdrand.seed(args.seed)
        np.random.seed(args.seed)
        print(f'[seed] explicit seed set: {args.seed}', flush=True)

    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from alphigo import kerasutil
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)

    # ── Load source checkpoint ─────────────────────────────────────────
    with h5py.File(args.src, 'r') as h5:
        model = kerasutil.load_model_from_hdf5_group(h5['model'])
        encoder_name = h5['encoder'].attrs['name']
        if not isinstance(encoder_name, str):
            encoder_name = encoder_name.decode('ascii')
        bw = int(h5['encoder'].attrs['board_width'])
        bh = int(h5['encoder'].attrs['board_height'])

    # Unfreeze everything — joint training.
    for layer in model.layers:
        layer.trainable = True
    print(f'[init] loaded {args.src}')
    print(f'[init] encoder={encoder_name} board={bw}x{bh}  '
          f'trainable_params={model.count_params():,}')

    # ── Locate chunk files ─────────────────────────────────────────────
    train_paths = sorted(glob.glob(args.train_glob))
    # Filter out the val chunk if the glob matches it.
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
    print(f'[data] {len(train_paths)} train chunks, 1 val chunk:')
    for p in train_paths:
        print(f'   train: {p}')
    print(f'   val:   {args.val_chunk}')

    # ── Load val set (full, into RAM — one chunk fits) ─────────────────
    from alphigo.rl.zero_experience import load_zero_buffer
    with h5py.File(args.val_chunk, 'r') as h5:
        val_buf = load_zero_buffer(h5)
    val_x = val_buf.encoded_states.astype(np.float32)
    val_p = val_buf.policy_targets.astype(np.float32)
    val_v = val_buf.value_targets.astype(np.float32).reshape(-1, 1)
    n_val = val_x.shape[0]
    print(f'[data] val set: {n_val:,} positions  '
          f'value range [{val_v.min():.2f}, {val_v.max():.2f}]')
    # Sanity: if the val chunk was encoded with policy-target=none, the
    # policy targets are zeros and val_policy_loss won't be informative.
    if float(val_p.sum()) == 0.0:
        print('[warn] val chunk has all-zero policy targets — re-encode '
              'with --policy-target=one-hot for meaningful policy loss')

    # ── Compile ─────────────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss={'policy_output': 'categorical_crossentropy',
              'value_output': 'mse'},
        loss_weights={'policy_output': 1.0, 'value_output': args.alpha},
        jit_compile=False,
    )

    # ── Custom training loop: one chunk at a time ──────────────────────
    # Streaming via tf.data.from_generator OOMs on WSL's 32 GB cap due
    # to memory accumulation across chunk rotations (libc heap
    # fragmentation, h5py file handle retention). Processing one chunk
    # at a time with explicit gc/malloc_trim between chunks guarantees
    # only one chunk-worth (~5 GB) of training data is in RAM.
    import gc
    import ctypes
    try:
        libc = ctypes.CDLL("libc.so.6")
        def trim(): libc.malloc_trim(0)
    except OSError:
        def trim(): pass

    rng = np.random.default_rng(args.seed)
    best_val_value_loss = float('inf')
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
            y_p = buf.policy_targets.astype(np.float32)
            y_v = buf.value_targets.astype(np.float32).reshape(-1, 1)
            del buf
            print(f'  chunk {ci_idx + 1}/{len(chunk_order)}: '
                  f'{os.path.basename(p)} ({n} pos)', flush=True)
            model.fit(
                x, {'policy_output': y_p, 'value_output': y_v},
                batch_size=args.bs, epochs=1, shuffle=True, verbose=2,
            )
            del x, y_p, y_v
            gc.collect()
            trim()

        # End-of-epoch validation.
        val_metrics = model.evaluate(
            val_x, {'policy_output': val_p, 'value_output': val_v},
            verbose=0, batch_size=args.bs, return_dict=True,
        )
        v_loss = val_metrics['value_output_loss']
        p_loss = val_metrics['policy_output_loss']
        print(f'  >>> epoch {epoch} val_value_loss={v_loss:.4f}  '
              f'val_policy_loss={p_loss:.4f}', flush=True)

        if v_loss < best_val_value_loss - 1e-5:
            best_val_value_loss = v_loss
            best_weights = model.get_weights()
            epochs_since_improvement = 0
            print(f'  *** new best val_value_loss', flush=True)
            # Persist best-so-far to disk immediately. If training crashes
            # later (CUDA event errors, etc.), we still have the best
            # checkpoint reachable. The pipeline's skip-if-exists check
            # will leave this alone on resume.
            os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
            with h5py.File(args.dst, 'w') as h5:
                h5.create_group('encoder')
                h5['encoder'].attrs['name'] = encoder_name
                h5['encoder'].attrs['board_width'] = bw
                h5['encoder'].attrs['board_height'] = bh
                h5.create_group('model')
                kerasutil.save_model_to_hdf5_group(model, h5['model'])
                h5.create_group('agent')
                h5['agent'].attrs['num_simulations'] = 50
                h5['agent'].attrs['c_puct'] = 1.5
            print(f'  [persist] wrote best to {args.dst}', flush=True)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f'  EARLY STOP at epoch {epoch} '
                      f'(no improvement for {args.patience} epochs)',
                      flush=True)
                break

    if best_weights is not None:
        print(f'\n[restore] best weights: '
              f'val_value_loss={best_val_value_loss:.4f}', flush=True)
        model.set_weights(best_weights)

    # ── Save ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
    with h5py.File(args.dst, 'w') as h5:
        h5.create_group('encoder')
        h5['encoder'].attrs['name'] = encoder_name
        h5['encoder'].attrs['board_width'] = bw
        h5['encoder'].attrs['board_height'] = bh
        h5.create_group('model')
        kerasutil.save_model_to_hdf5_group(model, h5['model'])
        h5.create_group('agent')
        h5['agent'].attrs['num_simulations'] = 50
        h5['agent'].attrs['c_puct'] = 1.5
    print(f'[done] saved {args.dst}')

    # Final eval.
    final = model.evaluate(val_x, {'policy_output': val_p,
                                   'value_output': val_v},
                            verbose=0, return_dict=True)
    print(f'[done] final val metrics: {final}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
