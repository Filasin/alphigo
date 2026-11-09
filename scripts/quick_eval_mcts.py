#!/usr/bin/env python3
"""Quick MCTS evaluation of an agent vs GnuGo."""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py

# GPU memory growth so multiple eval processes can share the GPU
# (the loop runs plain and pass-hack GnuGo evals concurrently).
import tensorflow as tf
try:
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

from alphigo.agent.zero_agent import load_zero_agent
from alphigo.agent.registry import create_agent
from alphigo.simulate import simulate_game
from alphigo.alphigo_types import Player


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', required=True)
    p.add_argument('--sims', type=int, default=25)
    p.add_argument('--games', type=int, default=20)
    p.add_argument('--board-size', type=int, default=9)
    p.add_argument('--opponent', default='gnugo:level=1')
    p.add_argument('--value-mode', default='network',
                   choices=['network', 'zero', 'rollout', 'ownership', 'external'],
                   help='Value source for MCTS leaf evaluation')
    p.add_argument('--value-model', default=None,
                   help='Path to standalone value model (.h5) for value-mode=external')
    p.add_argument('--search', default='puct', choices=['puct', 'gumbel'])
    p.add_argument('--gumbel-m', type=int, default=16)
    p.add_argument('--pass-hack', action='store_true',
                   help='Wrap agent with PassHackAgent (pass when opponent '
                        'passes and TT or value head says ahead)')
    args = p.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    ext_value_model = None
    if args.value_model:
        ext_value_model = tf.keras.models.load_model(
            args.value_model, compile=False)
        if args.value_mode == 'network':
            args.value_mode = 'external'

    with h5py.File(args.agent, 'r') as h5:
        if args.search == 'gumbel':
            from alphigo.agent.gumbel_zero_agent import load_gumbel_agent
            agent = load_gumbel_agent(h5, num_simulations=args.sims,
                                      m=args.gumbel_m,
                                      value_mode=args.value_mode,
                                      external_value_model=ext_value_model)
        else:
            agent = load_zero_agent(h5, num_simulations=args.sims,
                                    c_puct=1.5,
                                    value_mode=args.value_mode,
                                    external_value_model=ext_value_model)

    if args.pass_hack:
        from alphigo.agent.base import PassHackAgent
        agent = PassHackAgent(agent)

    opp_parts = args.opponent.split(':')
    opp_name = opp_parts[0]
    opp_kwargs = {}
    if len(opp_parts) > 1:
        for kv in opp_parts[1].split(','):
            k, v = kv.split('=')
            try:
                v = int(v)
            except ValueError:
                pass
            opp_kwargs[k] = v
    gnugo = create_agent(opp_name, **opp_kwargs)

    wins = 0
    for g in range(args.games):
        agent_color = Player.black if g % 2 == 0 else Player.white
        if agent_color == Player.black:
            result = simulate_game(agent, gnugo, board_size=args.board_size)
        else:
            result = simulate_game(gnugo, agent, board_size=args.board_size)
        w = 'WIN' if result['winner'] == agent_color else 'LOSE'
        if result['winner'] == agent_color:
            wins += 1
        print(f'Game {g+1}/{args.games}: {w} '
              f'({result["num_moves"]} moves)', flush=True)

    pct = 100 * wins / args.games
    print(f'\nResult: {wins}/{args.games} = {pct:.0f}% '
          f'vs {args.opponent} at sims={args.sims} search={args.search}')


if __name__ == '__main__':
    main()
