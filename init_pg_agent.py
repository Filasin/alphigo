#!/usr/bin/env python
"""Initialise a fresh PolicyAgent with random weights and save to HDF5.

This bootstraps the reinforcement-learning cycle: the untrained agent
plays essentially at random, providing the starting point for self-play
data collection and policy gradient training.

Usage:
    python init_pg_agent.py --agent-out agents/pg_agent.h5 \
                            --board-size 9 \
                            --encoder oneplane \
                            --network small
"""

import argparse

import h5py

from alphigo.agent.policy_agent import PolicyAgent
from alphigo.data.networks import NETWORK_MAP
from alphigo.encoders.base import get_encoder_by_name


def main():
    parser = argparse.ArgumentParser(
        description='Create a fresh (untrained) PolicyAgent.')
    parser.add_argument('--agent-out', required=True,
                        help='Output HDF5 file for the new agent')
    parser.add_argument('--board-size', type=int, default=9,
                        help='Board size (default: 9)')
    parser.add_argument('--encoder', default='oneplane',
                        help='Board encoder name (default: oneplane)')
    parser.add_argument('--network', default='small',
                        choices=list(NETWORK_MAP.keys()),
                        help='Network architecture (default: small)')
    args = parser.parse_args()

    encoder = get_encoder_by_name(args.encoder, args.board_size)
    input_shape = encoder.shape()
    num_classes = args.board_size * args.board_size

    model = NETWORK_MAP[args.network](input_shape, num_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
    )

    new_agent = PolicyAgent(model, encoder)

    print('Encoder:  %s (%d planes)' % (args.encoder, encoder.num_planes))
    print('Network:  %s' % args.network)
    print('Board:    %dx%d (%d output classes)' %
          (args.board_size, args.board_size, num_classes))
    print('Params:   %d' % model.count_params())

    with h5py.File(args.agent_out, 'w') as outf:
        new_agent.serialize(outf)
    print('Saved to  %s' % args.agent_out)


if __name__ == '__main__':
    main()
