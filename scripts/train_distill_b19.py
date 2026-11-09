"""Train a 2-head or 3-head network on self-play data.

Policy distillation: the policy head learns from MCTS visit-count
distributions (soft targets) rather than human expert moves.  Combined
with a value head, this completes the full AlphaZero self-improvement loop
in a single fine-tuning pass.

Architecture: entry conv -> N residual blocks -> {policy, value[, ownership]}.
Default is 2-head (--heads 2); pass --heads 3 for legacy 3-head mode.
"""
import argparse
import gc
import ctypes
import glob
import os
import sys

import h5py
import numpy as np


class OneCycleLR:
    """1cycle LR schedule (Smith 2018) applied per-batch within an epoch.

    Phases: warmup 40% (min_lr -> base_lr), decay 50% (base_lr -> min_lr),
    cooldown 10% (min_lr -> final_lr). min_lr = base/25, final_lr = base/250.

    Used as a Keras callback — call make_callback() to get a tf.keras.Callback
    that adjusts LR on each batch. The step counter persists across multiple
    model.fit() calls within one epoch; call reset() between epochs.
    """
    def __init__(self, base_lr, total_steps):
        self.base_lr = base_lr
        self.total_steps = max(total_steps, 1)
        self.min_lr = base_lr / 25.0
        self.final_lr = base_lr / 250.0
        self.step = 0
        self.warmup_end = int(0.4 * self.total_steps)
        self.decay_end = self.warmup_end + int(0.5 * self.total_steps)

    def get_lr(self):
        s = self.step
        if s < self.warmup_end:
            t = s / max(self.warmup_end, 1)
            return self.min_lr + (self.base_lr - self.min_lr) * t
        if s < self.decay_end:
            t = (s - self.warmup_end) / max(self.decay_end - self.warmup_end, 1)
            return self.base_lr - (self.base_lr - self.min_lr) * t
        cooldown_len = max(self.total_steps - self.decay_end, 1)
        t = min((s - self.decay_end) / cooldown_len, 1.0)
        return self.min_lr - (self.min_lr - self.final_lr) * t

    def reset(self):
        self.step = 0


def _count_positions(path):
    with h5py.File(path, 'r') as h5:
        return h5['zero_experience']['encoded_states'].shape[0]


def augment_d4(states, policy, ownership, board_size):
    """Apply random D4 symmetry (4 rotations x 2 reflections) to Go data.

    States: (N, H, W, C), policy: (N, H*W+1), ownership: (N, H*W).
    Value is unchanged. Pass probability (last element) is rotation-invariant.
    """
    n = len(states)
    num_points = board_size * board_size
    pass_probs = policy[:, num_points:].copy()
    board_policy = policy[:, :num_points]

    transforms = np.random.randint(0, 8, size=n)
    pol_2d = board_policy.reshape(n, board_size, board_size)
    own_2d = ownership.reshape(n, board_size, board_size)

    for t in range(8):
        mask = transforms == t
        if not mask.any() or t == 0:
            continue
        k = t % 4
        flip = t >= 4
        if flip:
            states[mask] = np.flip(states[mask], axis=2)
            pol_2d[mask] = np.flip(pol_2d[mask], axis=2)
            own_2d[mask] = np.flip(own_2d[mask], axis=2)
        if k > 0:
            states[mask] = np.rot90(states[mask], k=k, axes=(1, 2))
            pol_2d[mask] = np.rot90(pol_2d[mask], k=k, axes=(1, 2))
            own_2d[mask] = np.rot90(own_2d[mask], k=k, axes=(1, 2))

    augmented_policy = np.concatenate(
        [pol_2d.reshape(n, -1), pass_probs], axis=1)
    return states, augmented_policy, own_2d.reshape(n, -1)


def margin_to_scoredist(margins, score_min=-30, score_max=30):
    """Convert raw signed margins to one-hot score distribution targets."""
    n_bins = score_max - score_min + 1
    bins = np.clip(np.round(margins).astype(int), score_min, score_max) - score_min
    targets = np.zeros((len(margins), n_bins), dtype=np.float32)
    targets[np.arange(len(margins)), bins] = 1.0
    return targets


def load_selfplay_chunk(path, board_size=19, num_classes=None, q_blend=0.0):
    """Load a self-play chunk with policy, value, ownership, and margin targets.

    q_blend: fraction of root Q-value to blend into value targets.
    0.0 = pure game outcome (z), 1.0 = pure root Q-value (q).
    """
    if num_classes is None:
        num_classes = board_size ** 2 + 1
    with h5py.File(path, 'r') as h5:
        grp = h5['zero_experience']
        states = np.array(grp['encoded_states'])
        values = np.array(grp['value_targets'])
        root_q = np.array(grp['root_q_values']) if 'root_q_values' in grp else None
        ownership = np.array(grp['ownership_targets']) if 'ownership_targets' in grp else None
        policy = np.array(grp['policy_targets']) if 'policy_targets' in grp else None
        margins = np.array(grp['margin_targets']) if 'margin_targets' in grp else None
    if root_q is not None and q_blend > 0.0:
        values = (1.0 - q_blend) * values + q_blend * root_q
    if ownership is None:
        ownership = np.zeros((len(states), board_size**2), dtype=np.float32)
    if margins is None:
        margins = np.zeros(len(states), dtype=np.float32)
    if policy is None:
        policy = np.ones((len(states), num_classes), dtype=np.float32) / num_classes
    elif policy.shape[1] < num_classes:
        policy = np.pad(policy, ((0, 0), (0, num_classes - policy.shape[1])),
                        constant_values=0.0)
    elif policy.shape[1] > num_classes:
        policy = policy[:, :num_classes]
        row_sums = policy.sum(axis=1, keepdims=True)
        policy = np.where(row_sums > 1e-8, policy / row_sums,
                          np.ones_like(policy) / num_classes)
    return states, policy, values, ownership, margins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--init-from',
        default='runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/'
                'checkpoints/r0_joint_smargin_b19.h5',
        help='Joint two-head checkpoint to copy trunk weights from.')
    parser.add_argument('--train-glob', required=True, nargs='+',
                        help='One or more glob patterns for training chunks.')
    parser.add_argument('--val-glob', default=None)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--num-blocks', type=int, default=3)
    parser.add_argument('--filters', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--mega-chunks', type=int, default=20,
                        help='Number of HDF5 chunks to load at once.')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--clipnorm', type=float, default=0.0,
                        help='Gradient clipping (0=disabled). Use 1.0 for SGD.')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='Max gradient steps per epoch (0=unlimited). '
                             'Samples a random subset from each mega-chunk.')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--policy-weight', type=float, default=1.0,
                        help='Weight for policy loss (value = 1.0).')
    parser.add_argument('--value-weight', type=float, default=1.0,
                        help='Weight for value loss.')
    parser.add_argument('--ownership-weight', type=float, default=0.5,
                        help='Weight for ownership loss.')
    parser.add_argument('--q-blend', type=float, default=0.0,
                        help='Blend root Q into value target: 0=pure z, 1=pure q.')
    parser.add_argument('--full-init', action='store_true',
                        help='Copy all weights from --init-from (not just trunk).')
    parser.add_argument('--augment', action='store_true',
                        help='Apply random D4 symmetry augmentation (8x effective data).')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--heads', type=int, default=2, choices=[2, 3, 4],
                        help='Number of output heads: 2 (policy+value), 3 (+ownership), 4 (+score dist).')
    parser.add_argument('--score-weight', type=float, default=1.0,
                        help='Weight for score distribution loss (heads=4 only).')
    parser.add_argument('--value-fc', type=int, default=0,
                        help='Value head FC width (0=auto: 64 for 9x9, 256 for 19x19).')
    parser.add_argument('--bn-momentum', type=float, default=0.99,
                        help='BatchNormalization momentum (default 0.99, try 0.95).')
    parser.add_argument('--one-cycle', action='store_true',
                        help='Use 1cycle LR schedule per epoch (--lr becomes peak LR).')
    parser.add_argument('--encoder', default='alphazero',
                        help='Encoder name (alphazero=17 planes, '
                             'alphazero_komi=18 planes).')
    args = parser.parse_args()

    # LR hold: if lr_hold.txt sits next to --dst (format: "UNTIL_ROUND LR"),
    # rounds before UNTIL_ROUND train at least at LR. Lets the campaign delay
    # a scheduled decay without touching the running loop script (which
    # computes --lr from a round table parsed at launch). Inert unless the
    # file exists; delete the file to restore the schedule.
    try:
        import re as _re
        _m = _re.search(r'round_(\d+)', os.path.basename(args.dst or ''))
        _hold = os.path.join(os.path.dirname(args.dst) or '.', 'lr_hold.txt')
        if _m and os.path.exists(_hold):
            with open(_hold) as _hf:
                _until, _lr = _hf.read().split()[:2]
            if int(_m.group(1)) < int(_until) and args.lr < float(_lr):
                print(f'[lr-hold] round {int(_m.group(1))} < {_until}: '
                      f'lr {args.lr} -> {_lr}', flush=True)
                args.lr = float(_lr)
    except Exception as _e:
        print(f'[lr-hold] ignored ({_e})', flush=True)

    import random as _stdrand
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    _stdrand.seed(args.seed)
    np.random.seed(args.seed)
    print(f'[seed] {args.seed}', flush=True)

    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam, SGD
    from alphigo.data.networks import (
        build_alphazero_two_head,
        build_policy_value_ownership_network,
        build_policy_scoredist_ownership_network,
        copy_trunk_weights)
    from alphigo import kerasutil

    tf.keras.utils.set_random_seed(args.seed)

    # ── Build model ──────────────────────────────────────────────────
    from alphigo.encoders.base import get_encoder_by_name
    encoder = get_encoder_by_name(args.encoder, args.board_size)
    input_shape = encoder.shape()
    num_classes = args.board_size ** 2 + 1
    if args.init_from and args.init_from.lower() != 'none' and args.full_init:
        with h5py.File(args.init_from, 'r') as h5:
            src_model = kerasutil.load_model_from_hdf5_group(h5['model'])
            src_policy_width = src_model.output_shape[0][-1]
            del src_model
        if src_policy_width != num_classes:
            print(f'[init] source model policy_width={src_policy_width}, '
                  f'adjusting num_classes from {num_classes}')
            num_classes = src_policy_width
    value_fc = args.value_fc if args.value_fc > 0 else None

    if args.heads == 2:
        model = build_alphazero_two_head(
            input_shape, num_classes,
            num_blocks=args.num_blocks, filters=args.filters,
            value_fc=value_fc, bn_momentum=args.bn_momentum)
        print(f'[init] policy+value network: '
              f'{model.count_params():,} params')
    elif args.heads == 4:
        model = build_policy_scoredist_ownership_network(
            input_shape, num_classes=num_classes,
            num_blocks=args.num_blocks, filters=args.filters,
            value_fc=value_fc, bn_momentum=args.bn_momentum)
        print(f'[init] policy+value+scoredist+ownership network: '
              f'{model.count_params():,} params')
    else:
        model = build_policy_value_ownership_network(
            input_shape, num_classes=num_classes,
            num_blocks=args.num_blocks, filters=args.filters,
            value_fc=value_fc, bn_momentum=args.bn_momentum)
        print(f'[init] policy+value+ownership network: '
              f'{model.count_params():,} params')

    # ── Copy trunk weights ───────────────────────────────────────────
    if args.init_from and args.init_from.lower() != 'none':
        with h5py.File(args.init_from, 'r') as h5:
            source_model = kerasutil.load_model_from_hdf5_group(h5['model'])
        if args.full_init:
            src_w = source_model.get_weights()
            dst_w = model.get_weights()
            if len(src_w) == len(dst_w):
                model.set_weights(src_w)
                print(f'[init] full-init: copied all {len(src_w)} weight arrays from source')
            else:
                n = min(len(src_w), len(dst_w))
                matched = 0
                for i in range(n):
                    if src_w[i].shape == dst_w[i].shape:
                        dst_w[i] = src_w[i]
                        matched += 1
                    else:
                        break
                model.set_weights(dst_w)
                print(f'[init] partial-init: copied {matched}/{len(dst_w)} weight arrays '
                      f'(source has {len(src_w)}, new heads randomly initialised)')
        else:
            copy_trunk_weights(source_model, model, num_blocks=args.num_blocks)
        del source_model
        gc.collect()

    # ── Locate chunks ────────────────────────────────────────────────
    all_paths = sorted(set(p for g in args.train_glob for p in glob.glob(g)))
    if not all_paths:
        print(f'ERROR: no chunks match {args.train_glob}', file=sys.stderr)
        return 1

    if args.val_glob:
        val_paths = sorted(glob.glob(args.val_glob))
        train_paths = [p for p in all_paths if p not in set(val_paths)]
    else:
        val_paths = [all_paths[-1]]
        train_paths = all_paths[:-1]

    if not train_paths:
        print('ERROR: no training chunks after val split', file=sys.stderr)
        return 1

    print(f'[data] {len(train_paths)} train, {len(val_paths)} val chunks')

    # ── Load val set ─────────────────────────────────────────────────
    val_states, val_pol, val_val, val_own, val_margins = [], [], [], [], []
    for p in val_paths:
        s, pol, v, o, mg = load_selfplay_chunk(p, args.board_size, num_classes, q_blend=args.q_blend)
        val_states.append(s)
        val_pol.append(pol)
        val_val.append(v)
        val_own.append(o)
        val_margins.append(mg)
    val_x = np.concatenate(val_states).astype(np.float32)
    val_p = np.concatenate(val_pol).astype(np.float32)
    val_v = np.concatenate(val_val).astype(np.float32).reshape(-1, 1)
    val_o = np.concatenate(val_own).astype(np.float32)
    val_mg = np.concatenate(val_margins).astype(np.float32)
    val_sd = margin_to_scoredist(val_mg) if args.heads == 4 else None
    print(f'[data] val: {len(val_x):,} positions')

    # ── Compile ──────────────────────────────────────────────────────
    clip_kw = {'clipnorm': args.clipnorm} if args.clipnorm > 0 else {}
    opt = (SGD(learning_rate=args.lr, momentum=args.momentum,
               weight_decay=args.weight_decay, **clip_kw)
           if args.optimizer == 'sgd'
           else Adam(learning_rate=args.lr,
                     weight_decay=args.weight_decay, **clip_kw))

    if args.heads == 2:
        model.compile(
            optimizer=opt,
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mse',
            },
            loss_weights={
                'policy_output': args.policy_weight,
                'value_output': args.value_weight,
            },
            jit_compile=False,
        )
    elif args.heads == 3:
        model.compile(
            optimizer=opt,
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mse',
                'ownership_output': 'mse',
            },
            loss_weights={
                'policy_output': args.policy_weight,
                'value_output': args.value_weight,
                'ownership_output': args.ownership_weight,
            },
            jit_compile=False,
        )
    else:
        model.compile(
            optimizer=opt,
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mse',
                'score_output': 'categorical_crossentropy',
                'ownership_output': 'mse',
            },
            loss_weights={
                'policy_output': args.policy_weight,
                'value_output': args.value_weight,
                'score_output': args.score_weight,
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

    # ── 1cycle LR setup ─────────────────────────────────────────────
    one_cycle_cb = None
    if args.one_cycle:
        print(f'[1cycle] scanning chunk sizes...', flush=True)
        chunk_sizes = [_count_positions(p) for p in train_paths]
        total_positions = sum(chunk_sizes)
        total_steps = (total_positions + args.bs - 1) // args.bs
        if args.max_steps > 0:
            total_steps = min(total_steps, args.max_steps)
        one_cycle_sched = OneCycleLR(args.lr, total_steps)

        class _OneCycleCallback(tf.keras.callbacks.Callback):
            def __init__(self, sched):
                super().__init__()
                self.sched = sched
            def on_train_batch_begin(self, batch, logs=None):
                lr = self.sched.get_lr()
                self.model.optimizer.learning_rate.assign(lr)
                self.sched.step += 1

        one_cycle_cb = _OneCycleCallback(one_cycle_sched)
        print(f'[1cycle] base_lr={args.lr:.6f} min_lr={one_cycle_sched.min_lr:.6f} '
              f'final_lr={one_cycle_sched.final_lr:.6f} total_steps={total_steps}',
              flush=True)

    # ── Training loop ────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    best_val_loss = float('inf')
    best_weights = None
    epochs_since_improvement = 0

    mega_chunk_size = args.mega_chunks
    fit_callbacks = [one_cycle_cb] if one_cycle_cb else []

    for epoch in range(1, args.epochs + 1):
        print(f'\n===== Epoch {epoch}/{args.epochs} =====', flush=True)
        chunk_order = rng.permutation(len(train_paths))
        epoch_steps = 0
        if one_cycle_cb:
            one_cycle_cb.sched.reset()

        for start in range(0, len(chunk_order), mega_chunk_size):
            batch_indices = chunk_order[start:start + mega_chunk_size]
            xs, pols, vals, owns, mgs = [], [], [], [], []
            for ci in batch_indices:
                p = train_paths[ci]
                x, y_pol, y_v, y_o, y_mg = load_selfplay_chunk(p, args.board_size, num_classes, q_blend=args.q_blend)
                xs.append(x.astype(np.float32))
                pols.append(y_pol.astype(np.float32))
                vals.append(y_v.astype(np.float32).reshape(-1, 1))
                owns.append(y_o.astype(np.float32))
                mgs.append(y_mg.astype(np.float32))

            x = np.concatenate(xs); del xs
            y_pol = np.concatenate(pols); del pols
            y_v = np.concatenate(vals); del vals
            y_o = np.concatenate(owns); del owns
            y_mg = np.concatenate(mgs); del mgs

            if args.augment:
                x, y_pol, y_o = augment_d4(x, y_pol, y_o, args.board_size)

            if args.max_steps > 0:
                remaining = args.max_steps - epoch_steps
                if remaining <= 0:
                    del x, y_pol, y_v, y_o, y_mg
                    break
                max_samples = remaining * args.bs
                if len(x) > max_samples:
                    idx = rng.choice(len(x), max_samples, replace=False)
                    x, y_pol, y_v, y_o, y_mg = x[idx], y_pol[idx], y_v[idx], y_o[idx], y_mg[idx]

            chunk_lo = start + 1
            chunk_hi = min(start + mega_chunk_size, len(chunk_order))
            n_steps = (len(x) + args.bs - 1) // args.bs
            print(f'  chunks {chunk_lo}-{chunk_hi}/{len(chunk_order)}: '
                  f'{len(x)} pos ({n_steps} steps)'
                  f'{" (augmented)" if args.augment else ""}', flush=True)
            targets = {'policy_output': y_pol, 'value_output': y_v}
            if args.heads >= 3:
                targets['ownership_output'] = y_o
            if args.heads == 4:
                targets['score_output'] = margin_to_scoredist(y_mg)
            model.fit(x, targets,
                      batch_size=args.bs, epochs=1, shuffle=True, verbose=2,
                      callbacks=fit_callbacks)
            epoch_steps += n_steps
            del x, y_pol, y_v, y_o, y_mg
            gc.collect()
            trim()

        val_targets = {'policy_output': val_p, 'value_output': val_v}
        if args.heads >= 3:
            val_targets['ownership_output'] = val_o
        if args.heads == 4:
            val_targets['score_output'] = val_sd
        val_results = model.evaluate(
            val_x, val_targets,
            verbose=0, batch_size=args.bs, return_dict=True)
        val_loss = val_results['loss']
        val_p_loss = val_results.get('policy_output_loss', '?')
        val_v_loss = val_results.get('value_output_loss', '?')
        if args.heads == 4:
            val_o_loss = val_results.get('ownership_output_loss', '?')
            val_s_loss = val_results.get('score_output_loss', '?')
            print(f'  >>> epoch {epoch} total={val_loss:.4f} '
                  f'policy={val_p_loss:.4f} value={val_v_loss:.4f} '
                  f'score={val_s_loss:.4f} ownership={val_o_loss:.4f}', flush=True)
        elif args.heads == 3:
            val_o_loss = val_results.get('ownership_output_loss', '?')
            print(f'  >>> epoch {epoch} total={val_loss:.4f} '
                  f'policy={val_p_loss:.4f} value={val_v_loss:.4f} '
                  f'ownership={val_o_loss:.4f}', flush=True)
        else:
            print(f'  >>> epoch {epoch} total={val_loss:.4f} '
                  f'policy={val_p_loss:.4f} value={val_v_loss:.4f}',
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

    agent_dst = args.dst.replace('.h5', '_agent.h5')
    with h5py.File(agent_dst, 'w') as h5:
        h5.create_group('encoder')
        h5['encoder'].attrs['name'] = args.encoder
        h5['encoder'].attrs['board_width'] = args.board_size
        h5['encoder'].attrs['board_height'] = args.board_size
        h5.create_group('model')
        kerasutil.save_model_to_hdf5_group(model, h5['model'])
        h5.create_group('agent')
        h5['agent'].attrs['num_simulations'] = 200
        h5['agent'].attrs['c_puct'] = 1.5
    print(f'[done] saved agent format: {agent_dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
