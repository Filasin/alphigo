"""Compare value predictions between two models on the same positions.

Diagnostic tool: checks whether the self-play-trained value net produces
reasonable outputs vs the original professional-data value net.
"""
import argparse
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-model', required=True)
    parser.add_argument('--sp-model', required=True)
    parser.add_argument('--data', required=True,
                        help='HDF5 chunk with zero_experience group')
    parser.add_argument('--num-positions', type=int, default=200)
    args = parser.parse_args()

    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    orig = tf.keras.models.load_model(args.orig_model, compile=False)
    sp = tf.keras.models.load_model(args.sp_model, compile=False)

    print(f'Original model: {orig.name}')
    print(f'  outputs: {[(o.name, o.shape) for o in orig.outputs]}')
    print(f'Self-play model: {sp.name}')
    print(f'  outputs: {[(o.name, o.shape) for o in sp.outputs]}')

    with h5py.File(args.data, 'r') as h5:
        grp = h5['zero_experience']
        n = min(args.num_positions, grp['encoded_states'].shape[0])
        states = np.array(grp['encoded_states'][:n], dtype=np.float32)
        true_v = np.array(grp['value_targets'][:n])

    orig_out = orig(states, training=False)
    sp_out = sp(states, training=False)

    if isinstance(orig_out, (list, tuple)):
        ov = orig_out[0].numpy().flatten()
    else:
        ov = orig_out.numpy().flatten()

    if isinstance(sp_out, (list, tuple)):
        sv = sp_out[0].numpy().flatten()
    else:
        sv = sp_out.numpy().flatten()

    print(f'\n=== Value predictions on {n} self-play positions ===')
    print(f'Original:  mean={ov.mean():+.4f} std={ov.std():.4f} '
          f'range=[{ov.min():.4f}, {ov.max():.4f}]')
    print(f'Self-play: mean={sv.mean():+.4f} std={sv.std():.4f} '
          f'range=[{sv.min():.4f}, {sv.max():.4f}]')
    print(f'True:      mean={true_v.mean():+.4f} std={true_v.std():.4f}')

    print(f'\n=== Correlations ===')
    print(f'Corr(orig,  true): {np.corrcoef(ov, true_v)[0, 1]:+.4f}')
    print(f'Corr(sp,    true): {np.corrcoef(sv, true_v)[0, 1]:+.4f}')
    print(f'Corr(orig,  sp):   {np.corrcoef(ov, sv)[0, 1]:+.4f}')

    print(f'\n=== Errors ===')
    print(f'MAE(orig,  true): {np.abs(ov - true_v).mean():.4f}')
    print(f'MAE(sp,    true): {np.abs(sv - true_v).mean():.4f}')

    print(f'\n=== Prediction distribution ===')
    for name, vals in [('orig', ov), ('sp', sv)]:
        near_zero = (np.abs(vals) < 0.1).mean()
        near_one = (np.abs(vals) > 0.9).mean()
        print(f'{name:8s}: |v|<0.1: {near_zero:.1%}  |v|>0.9: {near_one:.1%}')

    print(f'\n=== Sign agreement ===')
    orig_agrees = (np.sign(ov) == np.sign(true_v)).mean()
    sp_agrees = (np.sign(sv) == np.sign(true_v)).mean()
    both_agree = ((np.sign(ov) == np.sign(true_v)) &
                  (np.sign(sv) == np.sign(true_v))).mean()
    print(f'Orig sign matches true:  {orig_agrees:.1%}')
    print(f'SP sign matches true:    {sp_agrees:.1%}')
    print(f'Both match true:         {both_agree:.1%}')

    print(f'\n=== Sample predictions (first 20) ===')
    print(f'{"idx":>4s} {"true":>8s} {"orig":>8s} {"sp":>8s} {"agree?":>8s}')
    for i in range(min(20, n)):
        agree = ('YES' if np.sign(ov[i]) == np.sign(sv[i]) == np.sign(true_v[i])
                 else 'no')
        print(f'{i:4d} {true_v[i]:+8.4f} {ov[i]:+8.4f} {sv[i]:+8.4f} '
              f'{agree:>8s}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
