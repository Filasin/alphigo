#!/usr/bin/env python3
"""Policy cross-entropy + value sign-accuracy on a fixed probe chunk.

Measurement-only diagnostic for the self-play loop: the probe chunk is
OGS human data the loop never trains on. Prints one parseable line:

    PROBE policy_ce=<float> value_acc=<float> n=<int>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', required=True,
                   help='ZeroAgent-format .h5 checkpoint')
    p.add_argument('--probe-chunk',
                   default='data/ogs_supervised_b9_full/ogs_supervised_b9_0290.h5')
    p.add_argument('--limit', type=int, default=10000)
    p.add_argument('--bs', type=int, default=1024)
    args = p.parse_args()

    import tensorflow as tf
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    from alphigo import kerasutil
    with h5py.File(args.agent, 'r') as h5:
        model = kerasutil.load_model_from_hdf5_group(h5['model'])

    with h5py.File(args.probe_chunk, 'r') as f:
        g = f['zero_experience']
        n = min(args.limit, g['encoded_states'].shape[0])
        states = g['encoded_states'][:n]
        pol_targets = g['policy_targets'][:n]
        val_targets = g['value_targets'][:n]

    # Komi-aware models (18 planes) vs legacy 17-plane chunks: synthesize
    # the komi plane for standard komi 7.5, mover-perspective (+ favors mover).
    model_planes = model.input_shape[-1]
    if states.shape[-1] == 17 and model_planes == 18:
        black_to_move = states[:, 0, 0, 16]  # 1 if black to move
        komi_val = np.where(black_to_move > 0, -0.75, 0.75).astype(np.float32)
        komi_plane = np.broadcast_to(
            komi_val[:, None, None, None],
            states.shape[:3] + (1,))
        states = np.concatenate([states, komi_plane], axis=-1)

    ce_sum = 0.0
    correct = 0
    nonzero = 0
    for i in range(0, n, args.bs):
        out = model(states[i:i + args.bs], training=False)
        pol = out[0].numpy()
        val = out[1].numpy()[:, 0]
        t = pol_targets[i:i + args.bs]
        if pol.shape[1] > t.shape[1]:
            pol = pol[:, :t.shape[1]]
        ce_sum += float(-(t * np.log(np.maximum(pol, 1e-12))).sum())
        vt = val_targets[i:i + args.bs]
        mask = vt != 0
        nonzero += int(mask.sum())
        correct += int((np.sign(val[mask]) == np.sign(vt[mask])).sum())

    policy_ce = ce_sum / n
    value_acc = correct / max(1, nonzero)
    print(f'PROBE policy_ce={policy_ce:.4f} value_acc={value_acc:.4f} n={n}')


if __name__ == '__main__':
    main()
