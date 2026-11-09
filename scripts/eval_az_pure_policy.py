"""Eval an AZ two-head checkpoint as pure-policy (argmax on legal moves,
no MCTS), to isolate policy strength from MCTS+value-head contribution.

Uses run_benchmark for color-swapped scoring against any registry opponent.
Defaults: bootstrap r2 vs gnugo:level=1 at N=30. Mirrors the layout of
scripts/eval_az_r9_vs_gnugo.sh's underlying benchmark.

Why this script exists: the `dl:` registry uses DeepLearningAgent which
calls `model(x).numpy()` and crashes on two-head models (gets a list).
The fix below is a 2-line wrapper that takes the first output.
"""
import argparse
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',
                        default='runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed2/checkpoints/round_002.h5')
    parser.add_argument('--opp', default='gnugo:level=1')
    parser.add_argument('-n', '--num-games', type=int, default=30)
    parser.add_argument('--board-size', type=int, default=9)
    args = parser.parse_args()

    import h5py
    from alphigo import kerasutil
    from alphigo.agent.deep_learning import DeepLearningAgent
    from alphigo.agent.registry import create_agent
    from alphigo.benchmark import run_benchmark
    from alphigo.encoders.base import get_encoder_by_name

    # Load model + encoder from the AZ-style h5 (which has 'model' and
    # 'encoder' groups).
    with h5py.File(args.ckpt, 'r') as h5:
        model = kerasutil.load_model_from_hdf5_group(h5['model'])
        encoder_name = h5['encoder'].attrs['name']
        if not isinstance(encoder_name, str):
            encoder_name = encoder_name.decode('ascii')
        bw = int(h5['encoder'].attrs['board_width'])
        bh = int(h5['encoder'].attrs['board_height'])
    encoder = get_encoder_by_name(encoder_name, (bw, bh))

    # Build a wrapper that picks the policy head when the model returns
    # [policy, value]. DeepLearningAgent does argmax on legal moves.
    agent = DeepLearningAgent(model, encoder)
    original_predict = agent.predict
    def predict_policy_head(game_state):
        encoded = encoder.encode(game_state)
        import numpy as np
        x = np.array([encoded], dtype=np.float32)
        out = model(x, training=False)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.numpy()[0]
    agent.predict = predict_policy_head

    opp_name, opp_kwargs = args.opp.split(':', 1) if ':' in args.opp else (args.opp, '')
    opp_dict = {}
    if opp_kwargs:
        for pair in opp_kwargs.split(','):
            k, v = pair.split('=', 1)
            opp_dict[k.strip()] = v.strip()
    opp_agent = create_agent(opp_name, **opp_dict)

    label_self = f'PURE-POLICY {os.path.basename(args.ckpt)}'
    print(f'[eval] {label_self}  vs  {args.opp}  '
          f'({args.num_games} games, color-swapped, 9x9)')
    results = run_benchmark(
        agent, opp_agent,
        black_label=label_self,
        white_label=args.opp,
        board_size=args.board_size,
        num_games=args.num_games,
        max_moves=2000,
        swap_colors=True,
        verbose=True,
    )
    s = results.summary_dict()
    print(f'\n[result] wins {s["a_wins"]}/{s["total_games"]} '
          f'({s["a_win_pct"]}%)  CI {s["a_win_pct_ci"]}')


if __name__ == '__main__':
    sys.exit(main())
