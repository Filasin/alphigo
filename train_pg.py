#!/usr/bin/env python
"""Train a PolicyAgent from self-play experience data (Ch. 10).

Usage:
    python train_pg.py --learning-agent agents/pg_agent.h5 \
                       --agent-out agents/pg_agent_updated.h5 \
                       --lr 0.0001 --bs 512 \
                       experience/exp_0001.h5 experience/exp_0002.h5
"""

import argparse

import h5py

from alphigo.agent.policy_agent import load_policy_agent
from alphigo.rl import load_experience


def main():
    parser = argparse.ArgumentParser(
        description='Train a policy gradient agent from experience data.')
    parser.add_argument('--learning-agent', required=True,
                        help='HDF5 file containing the PolicyAgent to train')
    parser.add_argument('--agent-out', required=True,
                        help='Output HDF5 file for the updated agent')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for SGD (default: 0.0001)')
    parser.add_argument('--clipnorm', type=float, default=1.0,
                        help='Gradient clip norm (default: 1.0)')
    parser.add_argument('--bs', type=int, default=512,
                        help='Mini-batch size (default: 512)')
    parser.add_argument('experience', nargs='+',
                        help='One or more HDF5 experience files')
    args = parser.parse_args()

    print('Loading agent from %s ...' % args.learning_agent)
    with h5py.File(args.learning_agent, 'r') as h5:
        learning_agent = load_policy_agent(h5)

    for exp_filename in args.experience:
        print('Training on %s ...' % exp_filename)
        with h5py.File(exp_filename, 'r') as h5:
            exp_buffer = load_experience(h5)
        learning_agent.train(
            exp_buffer,
            lr=args.lr,
            clipnorm=args.clipnorm,
            batch_size=args.bs,
        )

    print('Saving updated agent to %s ...' % args.agent_out)
    with h5py.File(args.agent_out, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)
    print('Done.')


if __name__ == '__main__':
    main()
