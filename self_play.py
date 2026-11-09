#!/usr/bin/env python
"""Generate self-play experience data for policy gradient training (Ch. 9).

Usage:
    # Initialise a fresh agent first (see init_pg_agent.py), then:
    python self_play.py --learning-agent agents/pg_agent.h5 \
                        --experience-out experience/exp_0001.h5 \
                        --num-games 500 --board-size 9 --epsilon 0.0
"""

import argparse
import datetime

import h5py

from alphigo.agent.policy_agent import load_policy_agent
from alphigo.rl import experience_simulation


def main():
    parser = argparse.ArgumentParser(
        description='Generate self-play experience for policy gradient RL.')
    parser.add_argument('--learning-agent', required=True,
                        help='HDF5 file containing a PolicyAgent')
    parser.add_argument('--num-games', '-n', type=int, default=100,
                        help='Number of self-play games (default: 100)')
    parser.add_argument('--experience-out', required=True,
                        help='Output HDF5 file for experience data')
    parser.add_argument('--board-size', type=int, default=9,
                        help='Board size (default: 9)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Exploration rate — probability of a '
                             'random move (default: 0.0)')
    args = parser.parse_args()

    print('Loading agent from %s ...' % args.learning_agent)
    with h5py.File(args.learning_agent, 'r') as h5:
        agent1 = load_policy_agent(h5)
    with h5py.File(args.learning_agent, 'r') as h5:
        agent2 = load_policy_agent(h5)
    agent1.set_epsilon(args.epsilon)
    agent2.set_epsilon(args.epsilon)

    print('Starting self-play: %d games on %dx%d board' %
          (args.num_games, args.board_size, args.board_size))
    print('Started at %s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))

    experience = experience_simulation(
        args.num_games, agent1, agent2, args.board_size)

    print('Saving %d decisions to %s' %
          (len(experience.rewards), args.experience_out))
    with h5py.File(args.experience_out, 'w') as experience_outf:
        experience.serialize(experience_outf)
    print('Done.')


if __name__ == '__main__':
    main()
