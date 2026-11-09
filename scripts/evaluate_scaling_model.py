"""Evaluate a trained model for the data scaling experiment.

Measures:
  - Val loss (policy, value, ownership) on held-out chunk
  - Win rate vs random bot (pure policy, no MCTS)
  - Win rate vs GnuGo level=1 (pure policy)
  - Policy entropy on first-move positions (empty board)
  - Atari response accuracy on probe positions

Output format (one line, parseable by the shell runner):
  RESULT:positions,policy_loss,value_loss,ownership_loss,total_loss,vs_random_pct,vs_gnugo_pct,first_move_entropy

Usage:
  python scripts/evaluate_scaling_model.py \
      --agent runs/scaling_experiment_b9/model_g10000_agent.h5 \
      --board-size 9 --eval-games 50 --eval-sims 200 --seed 123
"""
import argparse
import sys
import os
import numpy as np


def compute_policy_entropy(model, encoder, board_size):
    """Compute policy entropy on an empty board (first-move position)."""
    from alphigo.alphigo_board import Board
    from alphigo.alphigo_types import Player
    from alphigo.alphigo_board import GameState

    game = GameState.new_game(board_size)
    encoded = encoder.encode(game)
    x = np.expand_dims(encoded, axis=0).astype(np.float32)

    preds = model.predict(x, verbose=0)
    if isinstance(preds, list):
        policy = preds[0][0]
    else:
        policy = preds[0]

    policy = np.clip(policy, 1e-10, 1.0)
    policy = policy / policy.sum()
    entropy = -np.sum(policy * np.log(policy))
    return entropy


def evaluate_vs_random(agent, board_size, num_games, seed):
    """Play games against a random bot, return win percentage."""
    from alphigo.agent.naive import RandomBot
    from alphigo.alphigo_board import GameState
    from alphigo.alphigo_types import Player

    rng = np.random.default_rng(seed)
    random_bot = RandomBot()
    wins = 0

    for g in range(num_games):
        game = GameState.new_game(board_size)
        agents = {Player.black: agent, Player.white: random_bot}
        if g % 2 == 1:
            agents = {Player.black: random_bot, Player.white: agent}
        agent_color = Player.black if g % 2 == 0 else Player.white

        moves = 0
        max_moves = board_size * board_size * 3
        while not game.is_over() and moves < max_moves:
            current_agent = agents[game.next_player]
            move = current_agent.select_move(game)
            game = game.apply_move(move)
            moves += 1

        if game.is_over():
            from alphigo.alphigo_board import area_score
            black_score, white_score = area_score(game.board)
            winner = Player.black if black_score > white_score else Player.white
            if winner == agent_color:
                wins += 1

    return wins / num_games * 100


def evaluate_vs_gnugo(agent, board_size, num_games, seed, level=1):
    """Play games against GnuGo at a given level, return win percentage."""
    try:
        from alphigo.agent.registry import create_agent
        gnugo = create_agent('gnugo', level=level)
    except Exception as e:
        print(f"  [warn] GnuGo level={level} not available: {e}", file=sys.stderr)
        return -1.0

    from alphigo.simulate import simulate_game
    from alphigo.alphigo_types import Player

    wins = 0
    for g in range(num_games):
        agent_color = Player.black if g % 2 == 0 else Player.white
        if agent_color == Player.black:
            result = simulate_game(agent, gnugo, board_size=board_size)
        else:
            result = simulate_game(gnugo, agent, board_size=board_size)

        if result['winner'] == agent_color:
            wins += 1

    if hasattr(gnugo, 'cleanup'):
        gnugo.cleanup()

    return wins / num_games * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True)
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--eval-games', type=int, default=50)
    parser.add_argument('--eval-sims', type=int, default=200)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--skip-gnugo', action='store_true')
    parser.add_argument('--val-dir', default=None,
                        help='Directory with HDF5 chunks for val loss. '
                             'Default: data/selfplay_bootstrap_b{board_size}')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    import h5py
    from alphigo import kerasutil
    from alphigo.encoders.base import get_encoder_by_name

    # ── Load agent ──────────────────────────────────────────────────
    print(f"  Loading agent: {args.agent}", file=sys.stderr)
    with h5py.File(args.agent, 'r') as h5:
        model = kerasutil.load_model_from_hdf5_group(h5['model'])
        enc_name = h5['encoder'].attrs['name']

    encoder = get_encoder_by_name(enc_name, args.board_size)
    N = args.board_size ** 2

    model.compile(
        optimizer='adam',
        loss={
            'policy_output': 'categorical_crossentropy',
            'value_output': 'mse',
            'ownership_output': 'mse',
        },
    )

    # ── Policy entropy on empty board ───────────────────────────────
    print("  Computing first-move entropy...", file=sys.stderr)
    entropy = compute_policy_entropy(model, encoder, args.board_size)
    print(f"  First-move entropy: {entropy:.3f}", file=sys.stderr)

    # ── Val loss on held-out data ───────────────────────────────────
    print("  Computing val loss...", file=sys.stderr)
    val_dir = args.val_dir or f"data/selfplay_bootstrap_b{args.board_size}"
    import glob
    val_chunks = sorted(glob.glob(f"{val_dir}/selfplay_b{args.board_size}_*.h5"))
    if len(val_chunks) > 2:
        val_chunk = val_chunks[-1]  # last chunk as held-out
    else:
        val_chunk = None

    policy_loss = -1.0
    value_loss = -1.0
    ownership_loss = -1.0
    total_loss = -1.0

    if val_chunk:
        from scripts.train_distill_b19 import load_selfplay_chunk
        states, pol, val, own = load_selfplay_chunk(val_chunk, args.board_size)
        states = states.astype(np.float32)
        pol = pol.astype(np.float32)
        val_t = val.astype(np.float32).reshape(-1, 1)
        own = own.astype(np.float32)

        results = model.evaluate(
            states,
            {'policy_output': pol, 'value_output': val_t, 'ownership_output': own},
            verbose=0, batch_size=256, return_dict=True)
        total_loss = results.get('loss', -1)
        policy_loss = results.get('policy_output_loss', -1)
        value_loss = results.get('value_output_loss', -1)
        ownership_loss = results.get('ownership_output_loss', -1)
        print(f"  Val loss: total={total_loss:.4f} policy={policy_loss:.4f} "
              f"value={value_loss:.4f} ownership={ownership_loss:.4f}", file=sys.stderr)

    # ── Win rate vs random ──────────────────────────────────────────
    print(f"  Playing {args.eval_games} games vs random...", file=sys.stderr)
    from alphigo.agent.deep_learning import DeepLearningAgent
    dl_agent = DeepLearningAgent(model, encoder)
    vs_random = evaluate_vs_random(dl_agent, args.board_size, args.eval_games, args.seed)
    print(f"  vs random: {vs_random:.1f}%", file=sys.stderr)

    # ── Win rate vs GnuGo ──────────────────────────────────────────
    vs_gnugo1 = -1.0
    vs_gnugo5 = -1.0
    if not args.skip_gnugo:
        print(f"  Playing {args.eval_games} games vs GnuGo level=1...", file=sys.stderr)
        vs_gnugo1 = evaluate_vs_gnugo(dl_agent, args.board_size, args.eval_games, args.seed, level=1)
        if vs_gnugo1 >= 0:
            print(f"  vs GnuGo L1: {vs_gnugo1:.1f}%", file=sys.stderr)

        print(f"  Playing {args.eval_games} games vs GnuGo level=5...", file=sys.stderr)
        vs_gnugo5 = evaluate_vs_gnugo(dl_agent, args.board_size, args.eval_games, args.seed, level=5)
        if vs_gnugo5 >= 0:
            print(f"  vs GnuGo L5: {vs_gnugo5:.1f}%", file=sys.stderr)

    # ── Output result line ──────────────────────────────────────────
    result = (f"{policy_loss:.4f},{value_loss:.4f},{ownership_loss:.4f},"
              f"{total_loss:.4f},{vs_random:.1f},{vs_gnugo1:.1f},{vs_gnugo5:.1f},{entropy:.4f}")
    print(f"RESULT:{result}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
