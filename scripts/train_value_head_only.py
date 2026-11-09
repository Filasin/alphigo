"""Value-head-only training: freeze trunk+policy, train value MSE on
the existing self-play buffer.

Diagnosis from earlier eval: bootstrap r0 has SL-quality policy
(6/30 = 20% vs gnugo:level=1) but a random value head; 2 rounds of
AZ at lr=1e-4 degraded the policy (r2 pure-policy: 2/30 = 7%) while
the value head only crept from MSE 1.19 to 1.13 — too undertrained
for MCTS to be useful (r2 with MCTS: 0/30).

This script:
  1. Loads round_000 (= SL trunk+policy + random value head)
  2. Freezes every layer except the value-head branch
  3. Concatenates experience from rounds 1–N (default 7)
  4. Trains for many epochs on value MSE only
  5. Saves a new checkpoint suitable for MCTS eval

Time: ~5–10 min on RTX 3090 (no self-play, no MCTS, pure supervised).
"""
import argparse
import os
import sys
import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        default='runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed2/checkpoints/round_000.h5',
                        help='Source two-head checkpoint (round_000 recommended: strongest policy).')
    parser.add_argument('--exp-glob',
                        default='runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed2/experience/round_*.h5',
                        help='Glob of experience .h5 files to use.')
    parser.add_argument('--dst',
                        default='runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed2/checkpoints/r0_value_trained.h5',
                        help='Destination checkpoint path.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Adam lr (value head only, so OK to be higher than AZ-full).')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='If set, subsample this many positions uniformly at random from '
                             'the combined buffer. Keeps RAM bounded when combining many large '
                             'chunks (e.g. --max-samples 600000 across 10 x 200k chunks).')
    args = parser.parse_args()

    import glob
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from alphigo import kerasutil

    # Load source model.
    with h5py.File(args.src, 'r') as h5:
        model = kerasutil.load_model_from_hdf5_group(h5['model'])
        encoder_name = h5['encoder'].attrs['name']
        if not isinstance(encoder_name, str):
            encoder_name = encoder_name.decode('ascii')
        bw = int(h5['encoder'].attrs['board_width'])
        bh = int(h5['encoder'].attrs['board_height'])

    # Find the value-head branch by graph traversal from `value_output`
    # back to the trunk: collect layers in the value subgraph minus
    # layers also feeding the policy subgraph (= trunk). Layer names
    # don't follow a `value_*` convention so name-based filtering fails.
    def collect_subgraph(output_name):
        visited = set()
        queue = [model.get_layer(output_name)]
        while queue:
            layer = queue.pop()
            if layer.name in visited:
                continue
            visited.add(layer.name)
            # Keras 3: use layer.input tensors and trace via _keras_history.
            inputs = layer.input
            if not isinstance(inputs, list):
                inputs = [inputs]
            for t in inputs:
                # Tensor's keras_history points back to the producing layer.
                kh = getattr(t, '_keras_history', None)
                if kh is None:
                    continue
                prev = kh.operation if hasattr(kh, 'operation') else kh[0]
                queue.append(prev)
        return visited
    value_sub = collect_subgraph('value_output')
    policy_sub = collect_subgraph('policy_output')
    value_only = value_sub - policy_sub  # value branch (no trunk, no policy)

    value_layer_count = 0
    frozen = 0
    for layer in model.layers:
        if layer.name in value_only:
            layer.trainable = True
            value_layer_count += 1
        else:
            layer.trainable = False
            frozen += 1
    trainable_params = sum(
        layer.count_params() for layer in model.layers
        if layer.name in value_only
    )
    print(f'[freeze] trainable value-branch layers: {value_layer_count} '
          f'({trainable_params} params), frozen layers: {frozen}')
    print(f'[freeze] value branch: {sorted(value_only)}')

    # Load + concat experience.
    paths = sorted(glob.glob(args.exp_glob))
    if not paths:
        print(f'ERROR: no experience files match {args.exp_glob}', file=sys.stderr)
        return 1
    print(f'[data] {len(paths)} experience files:')
    for p in paths:
        print(f'   {p}')

    from alphigo.rl.zero_experience import load_zero_buffer, combine_zero_buffers
    rng_load = np.random.default_rng(seed=99999)

    if args.max_samples is not None:
        # Memory-bounded loading: load one chunk at a time, sample
        # proportionally, then free the full chunk. This keeps peak RAM at
        # max(1 chunk) + accumulated samples rather than sum(all chunks).
        sample_per_chunk = max(1, args.max_samples // len(paths))
        print(f'[data] --max-samples {args.max_samples}: '
              f'sampling ~{sample_per_chunk} from each of {len(paths)} chunks')
        sampled_states, sampled_policy, sampled_value = [], [], []
        for p in paths:
            with h5py.File(p, 'r') as h5:
                chunk = load_zero_buffer(h5)
            n = len(chunk)
            k = min(sample_per_chunk, n)
            idx = rng_load.choice(n, size=k, replace=False)
            sampled_states.append(chunk.encoded_states[idx])
            sampled_policy.append(chunk.policy_targets[idx])
            sampled_value.append(chunk.value_targets[idx])
            print(f'   sampled {k:,}/{n:,} from {p.split("/")[-1]}')
            del chunk
        from alphigo.rl.zero_experience import ZeroReplayBuffer
        buf = ZeroReplayBuffer(
            encoded_states=np.concatenate(sampled_states, axis=0),
            policy_targets=np.concatenate(sampled_policy, axis=0),
            value_targets=np.concatenate(sampled_value, axis=0),
        )
        del sampled_states, sampled_policy, sampled_value
    else:
        bufs = []
        for p in paths:
            with h5py.File(p, 'r') as h5:
                bufs.append(load_zero_buffer(h5))
        buf = combine_zero_buffers(bufs)
        del bufs

    print(f'[data] combined {len(buf)} samples '
          f'(value range: min={buf.value_targets.min():.2f} '
          f'max={buf.value_targets.max():.2f} '
          f'mean={buf.value_targets.mean():+.3f})')

    # Shuffle before training. Keras's validation_split takes the LAST
    # 10% of the array unshuffled, so without this the val set is the
    # last ~100 KGS games (chunks are SGF-iteration-ordered) — a
    # distribution-shifted slice when the corpus spans years/eras.
    # 19x19 SGF run without shuffle hit val_loss 1.23 (worse than
    # predict-0) at epoch 1, classic anti-learning from train/val mismatch.
    rng = np.random.default_rng(seed=12345)
    perm = rng.permutation(len(buf))
    shuffled_states = buf.encoded_states[perm]
    shuffled_policy = buf.policy_targets[perm]
    shuffled_value = buf.value_targets[perm]
    print(f'[shuffle] permuted {len(buf)} samples with seed 12345')

    # Compile with value MSE only. loss_weights={policy: 0, value: 1}
    # neutralises the policy head; the optimiser still computes policy
    # gradients but they're zero-weighted, and policy weights are frozen
    # anyway so they don't change.
    output_names = []
    for o in model.outputs:
        kh = getattr(o, '_keras_history', None)
        if kh is not None:
            layer = kh.operation if hasattr(kh, 'operation') else kh[0]
            output_names.append(layer.name)
        else:
            output_names.append(o.name.split('/')[0])
    loss_dict = {'value_output': 'mse'}
    weight_dict = {'value_output': 1.0}
    for name in output_names:
        if name != 'value_output':
            if name == 'policy_output':
                loss_dict[name] = 'categorical_crossentropy'
            else:
                loss_dict[name] = 'mse'
            weight_dict[name] = 0.0
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=loss_dict,
        loss_weights=weight_dict,
        jit_compile=False,
    )

    # Train value head with early stopping: val_value_loss bottoms out
    # around epoch 3–5 in observed runs, then overfits hard. Restore best.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_value_output_loss',
            patience=4,
            mode='min',
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    target_dict = {
        'policy_output': shuffled_policy,
        'value_output': shuffled_value.reshape(-1, 1),
    }
    for name in output_names:
        if name not in target_dict:
            out_shape = model.get_layer(name).output.shape
            if len(out_shape) == 2:
                target_dict[name] = np.zeros(
                    (len(shuffled_states), out_shape[1]), dtype=np.float32)
            else:
                target_dict[name] = np.zeros(
                    (len(shuffled_states), 1), dtype=np.float32)
    history = model.fit(
        shuffled_states,
        target_dict,
        batch_size=args.bs,
        epochs=args.epochs,
        verbose=2,
        validation_split=0.1,
        callbacks=callbacks,
    )

    # Quick sanity check: value head outputs on a few inputs.
    sample = buf.encoded_states[:5]
    out = model(sample, training=False)
    if isinstance(out, (list, tuple)):
        pred_value = out[1].numpy().flatten()
    else:
        pred_value = out.numpy().flatten()
    true_value = buf.value_targets[:5]
    print(f'\n[sanity] first 5 samples:')
    print(f'   pred: {pred_value}')
    print(f'   true: {true_value}')

    # Save as AZ-style checkpoint (model + encoder + agent groups).
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    with h5py.File(args.dst, 'w') as h5:
        h5.create_group('encoder')
        h5['encoder'].attrs['name'] = encoder_name
        h5['encoder'].attrs['board_width'] = bw
        h5['encoder'].attrs['board_height'] = bh
        h5.create_group('model')
        kerasutil.save_model_to_hdf5_group(model, h5['model'])
        h5.create_group('agent')
        # Inherit from source for sims/c_puct defaults; AZ eval can override.
        with h5py.File(args.src, 'r') as src_h5:
            if 'agent' in src_h5:
                for k in src_h5['agent'].attrs:
                    h5['agent'].attrs[k] = src_h5['agent'].attrs[k]
            else:
                h5['agent'].attrs['num_simulations'] = 25
                h5['agent'].attrs['c_puct'] = 1.5
    print(f'\n[saved] {args.dst}')

    final = history.history
    print(f'[final] value_loss: '
          f'train={final["value_output_loss"][-1]:.4f}  '
          f'val={final.get("val_value_output_loss", [float("nan")])[-1]:.4f}')


if __name__ == '__main__':
    sys.exit(main())
