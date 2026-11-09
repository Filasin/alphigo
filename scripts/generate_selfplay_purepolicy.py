"""Generate self-play games using a pure policy (sims=1) for value training.

Uses batched_self_play_zero() with num_simulations=1 — no MCTS tree, just
one forward pass per move. Temperature=1.0 for the first 10 moves (game
diversity), then argmax.

Output: ZeroReplayBuffer HDF5 files (one combined + chunked splits).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent',
                        default='runs/ogs_supervised_b9_5b96f/model_agent.h5',
                        help='Agent checkpoint to use as pure policy.')
    parser.add_argument('--num-games', type=int, default=20000)
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--out-dir', default='data/selfplay_value_b9')
    parser.add_argument('--chunk-size', type=int, default=100000,
                        help='Positions per chunk for standalone trainer.')
    parser.add_argument('--temperature-moves', type=int, default=10,
                        help='Number of opening moves with T=1.0.')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import h5py
    import numpy as np
    from alphigo.agent.zero_agent import load_zero_agent
    from alphigo.rl.simulate import batched_self_play_zero

    print(f'Loading agent: {args.agent}')
    with h5py.File(args.agent, 'r') as h5:
        agent = load_zero_agent(h5, num_simulations=1, value_mode='zero')

    print(f'Generating {args.num_games} self-play games on {args.board_size}x{args.board_size}...')
    print(f'Temperature schedule: T=1.0 for first {args.temperature_moves} moves, then T=0.0')
    t0 = time.time()

    buf = batched_self_play_zero(
        agent,
        board_size=args.board_size,
        num_games=args.num_games,
        num_simulations=1,
        temperature_moves=args.temperature_moves,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f'\nGenerated {len(buf)} positions from {args.num_games} games '
          f'in {elapsed:.0f}s ({elapsed / args.num_games:.2f}s/game)')
    print(f'Value distribution: mean={buf.value_targets.mean():+.3f} '
          f'std={buf.value_targets.std():.3f}')
    n_pos = len(buf.value_targets[buf.value_targets > 0])
    n_neg = len(buf.value_targets[buf.value_targets < 0])
    print(f'  +1 (winner): {n_pos}  -1 (loser): {n_neg}')

    os.makedirs(args.out_dir, exist_ok=True)

    combined_path = os.path.join(args.out_dir, 'selfplay_20k.h5')
    with h5py.File(combined_path, 'w') as h5:
        buf.serialize(h5)
    print(f'Saved combined buffer: {combined_path} ({len(buf)} positions)')

    n = len(buf)
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n)
    chunk_idx = 0
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        idx = perm[start:end]
        from alphigo.rl.zero_experience import ZeroReplayBuffer
        chunk = ZeroReplayBuffer(
            encoded_states=buf.encoded_states[idx],
            policy_targets=buf.policy_targets[idx],
            value_targets=buf.value_targets[idx],
        )
        chunk_path = os.path.join(args.out_dir, f'chunk_{chunk_idx:04d}.h5')
        with h5py.File(chunk_path, 'w') as h5:
            chunk.serialize(h5)
        print(f'Saved chunk: {chunk_path} ({len(chunk)} positions)')
        chunk_idx += 1

    print(f'\nDone. {chunk_idx} chunks in {args.out_dir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
