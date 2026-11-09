"""Create a randomly initialized agent checkpoint for bootstrapping self-play."""
import argparse
import os
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst', required=True, help='Output agent .h5 path')
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--num-blocks', type=int, default=3)
    parser.add_argument('--filters', type=int, default=64)
    parser.add_argument('--sims', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    import tensorflow as tf
    from alphigo.data.networks import build_policy_value_ownership_network
    from alphigo.encoders.base import get_encoder_by_name
    from alphigo import kerasutil

    tf.keras.utils.set_random_seed(args.seed)

    encoder = get_encoder_by_name('alphazero', args.board_size)
    input_shape = encoder.shape()

    model = build_policy_value_ownership_network(
        input_shape, board_size=args.board_size,
        num_blocks=args.num_blocks, filters=args.filters)
    print(f'[init] random {args.board_size}x{args.board_size} network: '
          f'{model.count_params():,} params '
          f'({args.num_blocks} blocks, {args.filters} filters)')

    os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
    with h5py.File(args.dst, 'w') as h5:
        h5.create_group('encoder')
        h5['encoder'].attrs['name'] = 'alphazero'
        h5['encoder'].attrs['board_width'] = args.board_size
        h5['encoder'].attrs['board_height'] = args.board_size
        h5.create_group('model')
        kerasutil.save_model_to_hdf5_group(model, h5['model'])
        h5.create_group('agent')
        h5['agent'].attrs['num_simulations'] = args.sims
        h5['agent'].attrs['c_puct'] = 1.5
    print(f'[done] saved {args.dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
