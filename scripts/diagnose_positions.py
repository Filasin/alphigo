"""Position-level diagnostic: compare value heads on real game positions.

Plays a game between AZ and GnuGo, then evaluates every position with
all three value heads (joint, pro-separate, self-play). Shows where they
agree, where they diverge, and optionally re-runs MCTS at divergent
positions to see how value guidance changes the chosen move.

Usage (WSL):
    python scripts/diagnose_positions.py \
        --agent runs/.../r0_joint_smargin_b19.h5 \
        --value-pro runs/.../value_standalone_margin_b19.h5 \
        --value-sp runs/.../value_selfplay_ownership_b19.h5 \
        --opponent gnugo:level=5 \
        --board-size 19 \
        --sims 25 \
        --probe-sims 200
"""
import argparse
import os
import sys
import time

import numpy as np


COLS = 'ABCDEFGHIJKLMNOPQRST'


def board_to_str(board):
    from alphigo.alphigo_types import Player, Point
    lines = []
    for row in range(board.num_rows, 0, -1):
        pad = " " if row <= 9 else ""
        cells = []
        for col in range(1, board.num_cols + 1):
            stone = board.get_player_on_point(Point(row=row, col=col))
            if stone == Player.black:
                cells.append(' x ')
            elif stone == Player.white:
                cells.append(' o ')
            else:
                cells.append(' . ')
        lines.append(f'{pad}{row} {"".join(cells)}')
    lines.append('    ' + '  '.join(COLS[:board.num_cols]))
    return '\n'.join(lines)


def move_to_str(move):
    if move.is_pass:
        return 'pass'
    if move.is_resign:
        return 'resign'
    return f'{COLS[move.point.col - 1]}{move.point.row}'


def top_policy_moves(priors, encoder, n=5):
    top_idx = np.argsort(priors)[::-1][:n]
    moves = []
    for idx in top_idx:
        if idx >= encoder.num_points():
            moves.append(('pass', float(priors[idx])))
        else:
            pt = encoder.decode_point_index(idx)
            moves.append((f'{COLS[pt.col - 1]}{pt.row}', float(priors[idx])))
    return moves


def predict_value(model, predict_fn, x):
    out = predict_fn(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return float(out.numpy()[0, 0])


def main():
    parser = argparse.ArgumentParser(
        description='Position-level value head diagnostic')
    parser.add_argument('--agent', required=True,
                        help='Path to AZ joint agent .h5 file')
    parser.add_argument('--value-pro', required=True,
                        help='Pro-data standalone value model')
    parser.add_argument('--value-sp', required=True,
                        help='Self-play standalone value model')
    parser.add_argument('--opponent', default='gnugo:level=5')
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--sims', type=int, default=25,
                        help='MCTS sims for the diagnostic game')
    parser.add_argument('--az-color', choices=['black', 'white'],
                        default='black')
    parser.add_argument('--probe-sims', type=int, default=0,
                        help='Re-run MCTS at divergent positions with '
                             'this many sims per value head (0 = skip)')
    parser.add_argument('--num-probes', type=int, default=5,
                        help='Max positions to probe with MCTS')
    parser.add_argument('--top-positions', type=int, default=8,
                        help='Number of divergent positions to display')
    args = parser.parse_args()

    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    import h5py
    from alphigo.agent.zero_agent import load_zero_agent
    from alphigo.agent.registry import create_agent
    from alphigo.alphigo_board import GameState
    from alphigo.alphigo_types import Player

    # ── Load models ──────────────────────────────────────────────────
    print('Loading models...', flush=True)

    pro_model = tf.keras.models.load_model(args.value_pro, compile=False)
    print(f'  pro value:  {args.value_pro}')
    print(f'    outputs: {[(o.name, o.shape) for o in pro_model.outputs]}')

    sp_model = tf.keras.models.load_model(args.value_sp, compile=False)
    print(f'  sp value:   {args.value_sp}')
    print(f'    outputs: {[(o.name, o.shape) for o in sp_model.outputs]}')

    with h5py.File(args.agent, 'r') as h5:
        agent = load_zero_agent(h5, value_mode='network')
    agent._num_simulations = args.sims
    agent.set_temperature(0.0)
    print(f'  agent:      {args.agent} (sims={args.sims})')

    # ── Opponent ─────────────────────────────────────────────────────
    if ':' in args.opponent:
        opp_name, params_str = args.opponent.split(':', 1)
        opp_kwargs = {}
        for pair in params_str.split(','):
            if '=' in pair:
                k, v = pair.split('=', 1)
                opp_kwargs[k.strip()] = v.strip()
    else:
        opp_name, opp_kwargs = args.opponent, {}
    opp = create_agent(opp_name, board_size=args.board_size, **opp_kwargs)
    print(f'  opponent:   {args.opponent}')

    az_is_black = (args.az_color == 'black')
    if az_is_black:
        black, white = agent, opp
    else:
        black, white = opp, agent

    # ── Phase 1: Play game, record positions ─────────────────────────
    print(f'\n=== Playing diagnostic game (AZ={args.az_color}, '
          f'sims={args.sims}) ===', flush=True)
    game = GameState.new_game(args.board_size)
    bots = {Player.black: black, Player.white: white}

    positions = [game]
    move_log = []

    while not game.is_over() and len(move_log) < 500:
        move = bots[game.next_player].select_move(game)
        move_log.append((game.next_player, move))
        game = game.apply_move(move)
        positions.append(game)
        if len(move_log) % 20 == 0:
            print(f'  move {len(move_log)}...', flush=True)

    if game.last_move and game.last_move.is_resign:
        reason = "resign"
    else:
        reason = "pass"
    winner = game.winner()
    winner_str = ('Black' if winner == Player.black else
                  'White' if winner == Player.white else 'Draw')
    az_won = ((winner == Player.black and az_is_black) or
              (winner == Player.white and not az_is_black))
    print(f'  Done: {len(move_log)} moves, winner={winner_str} '
          f'({reason}), AZ {"WON" if az_won else "LOST"}', flush=True)

    close_fn = getattr(opp, 'close', None)
    if callable(close_fn):
        close_fn()

    # ── Phase 2: Evaluate all positions ──────────────────────────────
    print(f'\n=== Evaluating {len(positions)} positions ===', flush=True)

    encoder = agent._encoder
    joint_model = agent._model
    input_shape = (1,) + tuple(encoder.shape())
    input_sig = [tf.TensorSpec(shape=input_shape, dtype=tf.float32)]

    @tf.function(input_signature=input_sig, reduce_retracing=True)
    def predict_joint(x):
        return joint_model(x, training=False)

    @tf.function(input_signature=input_sig, reduce_retracing=True)
    def predict_pro(x):
        return pro_model(x, training=False)

    @tf.function(input_signature=input_sig, reduce_retracing=True)
    def predict_sp(x):
        return sp_model(x, training=False)

    results = []
    for i, state in enumerate(positions):
        encoded = encoder.encode(state)
        x = np.array([encoded], dtype=np.float32)

        policy_out, value_out = predict_joint(x)
        policy_priors = policy_out.numpy()[0]
        v_joint = float(value_out.numpy()[0, 0])
        v_pro = predict_value(pro_model, predict_pro, x)
        v_sp = predict_value(sp_model, predict_sp, x)

        is_az_turn = ((state.next_player == Player.black and az_is_black) or
                      (state.next_player == Player.white and not az_is_black))

        results.append({
            'move_num': i,
            'v_joint': v_joint,
            'v_pro': v_pro,
            'v_sp': v_sp,
            'is_az_turn': is_az_turn,
            'current_player': state.next_player,
            'top_moves': top_policy_moves(policy_priors, encoder, n=5),
            'state': state,
        })
        if (i + 1) % 50 == 0:
            print(f'  evaluated {i + 1}/{len(positions)}', flush=True)

    # ── Phase 3: Value trajectory ────────────────────────────────────
    print(f'\n{"=" * 80}')
    print(f'VALUE TRAJECTORY  (current-player perspective)')
    print(f'  AZ plays {"Black" if az_is_black else "White"}, '
          f'outcome={winner_str} ({reason}), '
          f'AZ {"WON" if az_won else "LOST"}')
    print(f'{"=" * 80}')
    print(f'{"move":>5s} {"turn":>4s} {"joint":>7s} {"pro":>7s} '
          f'{"sp":>7s} {"j-p":>6s} {"j-s":>6s}  {"played":<8s}')
    print('-' * 80)

    for r in results:
        i = r['move_num']
        turn = 'AZ' if r['is_az_turn'] else 'opp'
        played = ''
        if 0 < i <= len(move_log):
            played = move_to_str(move_log[i - 1][1])
        jp = r['v_joint'] - r['v_pro']
        js = r['v_joint'] - r['v_sp']
        print(f'{i:5d} {turn:>4s} {r["v_joint"]:+7.3f} {r["v_pro"]:+7.3f} '
              f'{r["v_sp"]:+7.3f} {jp:+6.3f} {js:+6.3f}  {played:<8s}')

    # ── Phase 4: Divergent positions ─────────────────────────────────
    for r in results:
        vals = [r['v_joint'], r['v_pro'], r['v_sp']]
        r['divergence'] = max(vals) - min(vals)

    skip = max(5, len(move_log) // 20)
    interesting = [r for r in results
                   if skip <= r['move_num'] <= len(move_log) - 3]
    interesting.sort(key=lambda r: r['divergence'], reverse=True)
    top_n = interesting[:args.top_positions]
    top_n.sort(key=lambda r: r['move_num'])

    print(f'\n{"=" * 80}')
    print(f'TOP {len(top_n)} MOST DIVERGENT POSITIONS')
    print(f'{"=" * 80}')

    for r in top_n:
        i = r['move_num']
        cp = 'Black' if r['current_player'] == Player.black else 'White'
        turn = 'AZ' if r['is_az_turn'] else 'Opponent'
        print(f'\n--- Move {i} ({turn} to play as {cp}) ---')
        print(board_to_str(r['state'].board))
        print(f'\n  Value estimates (current player perspective):')
        print(f'    Joint:     {r["v_joint"]:+.4f}')
        print(f'    Pro:       {r["v_pro"]:+.4f}')
        print(f'    Self-play: {r["v_sp"]:+.4f}')
        print(f'    Spread:    {r["divergence"]:.4f}')
        print(f'\n  Policy top-5:')
        for mv_str, prob in r['top_moves']:
            bar = '#' * int(prob * 50)
            print(f'    {mv_str:>4s} {prob:6.3f} {bar}')
        if 0 < i <= len(move_log):
            print(f'\n  Played: {move_to_str(move_log[i - 1][1])}')

    # ── Phase 5 (optional): MCTS probe ──────────────────────────────
    if args.probe_sims > 0 and top_n:
        print(f'\n{"=" * 80}')
        print(f'MCTS PROBE  (sims={args.probe_sims} per position per head)')
        print(f'{"=" * 80}')

        with h5py.File(args.agent, 'r') as h5:
            a_joint = load_zero_agent(h5, value_mode='network')
            a_pro = load_zero_agent(
                h5, value_mode='external', external_value_model=pro_model)
            a_sp = load_zero_agent(
                h5, value_mode='external', external_value_model=sp_model)

        for a in (a_joint, a_pro, a_sp):
            a._num_simulations = args.probe_sims
            a.set_temperature(0.0)

        probes = top_n[:args.num_probes]
        for r in probes:
            i = r['move_num']
            turn = 'AZ' if r['is_az_turn'] else 'Opp'
            print(f'\n  Move {i} ({turn}, values: j={r["v_joint"]:+.3f} '
                  f'p={r["v_pro"]:+.3f} s={r["v_sp"]:+.3f}):')
            for label, a in [('Joint', a_joint),
                             ('Pro', a_pro),
                             ('Self-play', a_sp)]:
                t0 = time.time()
                mv = a.select_move(r['state'])
                sec = time.time() - t0
                print(f'    {label:10s} -> {move_to_str(mv):>5s}  ({sec:.1f}s)')

    # ── Summary ──────────────────────────────────────────────────────
    v_joints = np.array([r['v_joint'] for r in results])
    v_pros = np.array([r['v_pro'] for r in results])
    v_sps = np.array([r['v_sp'] for r in results])
    divs = np.array([r['divergence'] for r in results])

    print(f'\n{"=" * 80}')
    print(f'SUMMARY')
    print(f'{"=" * 80}')
    print(f'  Game: {len(move_log)} moves, {winner_str} ({reason}), '
          f'AZ {"WON" if az_won else "LOST"}')
    print(f'\n  {"":12s} {"mean":>8s} {"std":>8s} {"min":>8s} {"max":>8s}')
    for label, vals in [('Joint', v_joints), ('Pro', v_pros),
                        ('Self-play', v_sps)]:
        print(f'  {label:12s} {vals.mean():+8.4f} {vals.std():8.4f} '
              f'{vals.min():+8.4f} {vals.max():+8.4f}')

    print(f'\n  Divergence (spread among 3 heads):')
    print(f'    mean={divs.mean():.4f}  max={divs.max():.4f}  '
          f'>0.3: {(divs > 0.3).sum()}  >0.5: {(divs > 0.5).sum()} positions')

    print(f'\n  Correlations:')
    print(f'    Joint vs Pro:  {np.corrcoef(v_joints, v_pros)[0,1]:+.4f}')
    print(f'    Joint vs SP:   {np.corrcoef(v_joints, v_sps)[0,1]:+.4f}')
    print(f'    Pro vs SP:     {np.corrcoef(v_pros, v_sps)[0,1]:+.4f}')

    az_j = np.array([r['v_joint'] * (1 if r['is_az_turn'] else -1)
                     for r in results])
    az_p = np.array([r['v_pro'] * (1 if r['is_az_turn'] else -1)
                     for r in results])
    az_s = np.array([r['v_sp'] * (1 if r['is_az_turn'] else -1)
                     for r in results])

    print(f'\n  AZ-perspective mean value:')
    print(f'    Joint:     {az_j.mean():+.4f}')
    print(f'    Pro:       {az_p.mean():+.4f}')
    print(f'    Self-play: {az_s.mean():+.4f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
