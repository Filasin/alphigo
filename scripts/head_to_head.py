"""Play N games between two ZeroAgent checkpoints and report results."""
import argparse
import sys
import time
from multiprocessing import Process, Queue

import h5py
import numpy as np


def _worker(worker_id, game_ids, agent_a_path, agent_b_path, board_size,
            sims, search, gumbel_m, temperature, seed, result_queue):
    """Play assigned games and push results to queue."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    np.random.seed(seed + worker_id * 1000)

    import tensorflow as tf

    from alphigo.simulate import simulate_game
    from alphigo.alphigo_board import area_score
    from alphigo.alphigo_types import Player

    if search == 'gumbel':
        from alphigo.agent.gumbel_zero_agent import load_gumbel_agent
        with h5py.File(agent_a_path, 'r') as h5:
            agent_a = load_gumbel_agent(h5, num_simulations=sims, m=gumbel_m)
        with h5py.File(agent_b_path, 'r') as h5:
            agent_b = load_gumbel_agent(h5, num_simulations=sims, m=gumbel_m)
    else:
        from alphigo.agent.zero_agent import load_zero_agent
        with h5py.File(agent_a_path, 'r') as h5:
            agent_a = load_zero_agent(h5)
        agent_a._num_simulations = sims
        agent_a.set_temperature(temperature)
        with h5py.File(agent_b_path, 'r') as h5:
            agent_b = load_zero_agent(h5)
        agent_b._num_simulations = sims
        agent_b.set_temperature(temperature)

    for i in game_ids:
        if i % 2 == 0:
            result = simulate_game(agent_a, agent_b, board_size)
            a_color = Player.black
        else:
            result = simulate_game(agent_b, agent_a, board_size)
            a_color = Player.white

        winner = result['winner']
        final = result.get('final_state', None)
        if final is not None:
            b_score, w_score = area_score(final.board, 7.5)
            margin = b_score - w_score
        else:
            margin = 0.0

        a_margin = margin if a_color == Player.black else -margin

        if winner == a_color:
            who = 'A'
        elif winner is not None:
            who = 'B'
        else:
            who = 'draw'

        result_queue.put((i, who, a_margin, result['reason'],
                          result['num_moves']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-a', required=True, help='Agent A checkpoint')
    parser.add_argument('--agent-b', required=True, help='Agent B checkpoint')
    parser.add_argument('--num-games', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--sims', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Evaluation temperature (>0 breaks determinism)')
    parser.add_argument('--search', default='puct',
                        choices=['puct', 'gumbel'])
    parser.add_argument('--gumbel-m', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    n_workers = min(args.num_workers, args.num_games)

    if n_workers <= 1:
        _run_sequential(args)
        return

    game_ids = list(range(args.num_games))
    chunks = [[] for _ in range(n_workers)]
    for i, gid in enumerate(game_ids):
        chunks[i % n_workers].append(gid)

    result_queue = Queue()
    t0 = time.time()

    procs = []
    for w in range(n_workers):
        p = Process(target=_worker, args=(
            w, chunks[w], args.agent_a, args.agent_b,
            args.board_size, args.sims, args.search, args.gumbel_m,
            args.temperature, args.seed, result_queue))
        p.start()
        procs.append(p)

    results = []
    for _ in range(args.num_games):
        results.append(result_queue.get())

    for p in procs:
        p.join()

    results.sort(key=lambda x: x[0])

    a_wins, b_wins, draws = 0, 0, 0
    a_margins = []
    for i, who, a_margin, reason, num_moves in results:
        a_margins.append(a_margin)
        if who == 'A':
            a_wins += 1
        elif who == 'B':
            b_wins += 1
        else:
            draws += 1
        print(f'  game {i}: {who} wins ({reason}, '
              f'{num_moves} moves, margin={a_margin:+.1f})', flush=True)

    elapsed = time.time() - t0
    total = a_wins + b_wins + draws
    mean_margin = np.mean(a_margins)
    print(f'\n=== Results ({args.num_games} games, {elapsed:.0f}s, '
          f'{n_workers} workers) ===')
    print(f'  Agent A: {args.agent_a}')
    print(f'  Agent B: {args.agent_b}')
    print(f'  A wins: {a_wins} ({100*a_wins/total:.0f}%)')
    print(f'  B wins: {b_wins} ({100*b_wins/total:.0f}%)')
    print(f'  Draws:  {draws}')
    print(f'  A mean margin: {mean_margin:+.1f}')


def _run_sequential(args):
    """Original sequential path for --num-workers 1."""
    np.random.seed(args.seed)

    import tensorflow as tf
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    from alphigo.agent.zero_agent import load_zero_agent
    from alphigo.simulate import simulate_game
    from alphigo.alphigo_board import area_score

    if args.search == 'gumbel':
        from alphigo.agent.gumbel_zero_agent import load_gumbel_agent
        with h5py.File(args.agent_a, 'r') as h5:
            agent_a = load_gumbel_agent(h5, num_simulations=args.sims,
                                        m=args.gumbel_m)
        with h5py.File(args.agent_b, 'r') as h5:
            agent_b = load_gumbel_agent(h5, num_simulations=args.sims,
                                        m=args.gumbel_m)
    else:
        with h5py.File(args.agent_a, 'r') as h5:
            agent_a = load_zero_agent(h5)
        agent_a._num_simulations = args.sims
        agent_a.set_temperature(args.temperature)
        with h5py.File(args.agent_b, 'r') as h5:
            agent_b = load_zero_agent(h5)
        agent_b._num_simulations = args.sims
        agent_b.set_temperature(args.temperature)

    from alphigo.alphigo_types import Player

    a_wins, b_wins, draws = 0, 0, 0
    a_margins = []
    t0 = time.time()

    for i in range(args.num_games):
        if i % 2 == 0:
            result = simulate_game(agent_a, agent_b, args.board_size)
            a_color, b_color = Player.black, Player.white
        else:
            result = simulate_game(agent_b, agent_a, args.board_size)
            a_color, b_color = Player.white, Player.black

        winner = result['winner']
        final = result.get('final_state', None)
        if final is not None:
            b_score, w_score = area_score(final.board, 7.5)
            margin = b_score - w_score
        else:
            margin = 0.0

        a_margin = margin if a_color == Player.black else -margin
        a_margins.append(a_margin)

        if winner == a_color:
            a_wins += 1
            who = 'A'
        elif winner == b_color:
            b_wins += 1
            who = 'B'
        else:
            draws += 1
            who = 'draw'

        print(f'  game {i}: {who} wins ({result["reason"]}, '
              f'{result["num_moves"]} moves, margin={a_margin:+.1f})',
              flush=True)

    elapsed = time.time() - t0
    total = a_wins + b_wins + draws
    mean_margin = np.mean(a_margins)
    print(f'\n=== Results ({args.num_games} games, {elapsed:.0f}s) ===')
    print(f'  Agent A: {args.agent_a}')
    print(f'  Agent B: {args.agent_b}')
    print(f'  Temperature: {args.temperature}')
    print(f'  A wins: {a_wins} ({100*a_wins/total:.0f}%)')
    print(f'  B wins: {b_wins} ({100*b_wins/total:.0f}%)')
    print(f'  Draws:  {draws}')
    print(f'  A mean margin: {mean_margin:+.1f}')


if __name__ == '__main__':
    main()
