"""Encode 9x9 SGF games into a ZeroReplayBuffer suitable for value-head
supervised training.

For each game in --src-dir:
  - parse RE[...] to determine winner (B or W); skip games without a
    clear winner (draws, empty RE, unparseable)
  - replay moves; at every position, record:
      encoded_state = alphazero-encoded board
      policy_target = zeros (consumed at loss_weight=0 during training)
      value_target  = +1 if mover_at_this_position == winner else -1

Output is one ZeroReplayBuffer .h5 file that train_value_head_only.py
can ingest unchanged via --exp-glob.

Designed for the 8,606 SGF files in Downloads/go9/. Expected output:
~500k–800k positions depending on average game length.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py


def parse_result(re_str):
    """Return 'b' / 'w' / None from an SGF RE tag like 'B+3.5' or 'W+R'."""
    if not re_str:
        return None
    s = re_str.strip().lower()
    if s.startswith('b+'):
        return 'b'
    if s.startswith('w+'):
        return 'w'
    return None  # draw, void, unparseable


def parse_score_margin(re_str):
    """Return float score margin (always positive) from RE tag, or None.

    Examples:
      'B+3.5'   -> 3.5
      'W+12.0'  -> 12.0
      'B+R'     -> None (resignation; caller treats as ±inf -> ±1)
      'B+T'     -> None (timeout; caller treats as ±inf -> ±1)
      'B+F'     -> None (forfeit; caller treats as ±inf -> ±1)
      'W+'      -> None
    """
    if not re_str:
        return None
    s = re_str.strip().lower()
    if len(s) < 3 or s[1] != '+':
        return None
    suffix = s[2:]
    try:
        return float(suffix)
    except ValueError:
        return None


def encode_game(sgf_bytes, encoder, board_size=9,
                score_margin_scale=None, want_policy_targets=False):
    """Replay one SGF and return (states, values, [policy_targets]).

    Returns (None, None) (or (None, None, None) if want_policy_targets)
    if the game has no clear winner or fails to parse.

    When ``score_margin_scale`` is None (default), value = ±1 from
    winner-of-position-to-move perspective (backwards compatible).
    When set, value = sign × tanh(margin / score_margin_scale); games
    won by resignation/timeout/forfeit (no numeric margin in RE) map
    to ±1 since their effective margin is "infinite."

    When ``want_policy_targets`` is True, the third return value is a
    list of one-hot policy targets (length num_points) indexed by the
    encoder; positions where the played move was a pass are SKIPPED
    entirely (state, value, policy_target are all dropped for that
    ply). Pass moves in pro games are rare and mid-game passes don't
    have a usable one-hot policy label.
    """
    from sgfmill.sgf import Sgf_game
    from alphigo.alphigo_board import GameState, Move
    from alphigo.alphigo_types import Player, Point

    try:
        game = Sgf_game.from_bytes(sgf_bytes)
    except Exception:
        return (None, None, None) if want_policy_targets else (None, None)

    try:
        if game.get_size() != board_size:
            return (None, None, None) if want_policy_targets else (None, None)
    except Exception:
        # Malformed SZ tag (e.g. multiple values).
        return (None, None, None) if want_policy_targets else (None, None)
    root = game.get_root()
    re_str = None
    if root.has_property('RE'):
        try:
            re_str = root.get('RE')
        except Exception:
            # sgfmill raises ValueError on RE tags with multiple values etc.
            return (None, None, None) if want_policy_targets else (None, None)
    winner = parse_result(re_str)
    if winner is None:
        return (None, None, None) if want_policy_targets else (None, None)
    winner_player = Player.black if winner == 'b' else Player.white

    # Compute the magnitude of the value label.
    if score_margin_scale is None:
        win_value = 1.0
    else:
        margin = parse_score_margin(re_str)
        if margin is None:
            # Resignation / timeout / forfeit — treat as decisive (effective
            # margin is "infinite", tanh saturates to 1).
            win_value = 1.0
        else:
            win_value = math.tanh(margin / score_margin_scale)

    state = GameState.new_game(board_size)
    states = []
    values = []
    policy_targets = [] if want_policy_targets else None
    num_points = encoder.num_points()

    for node in game.get_main_sequence()[1:]:  # skip root
        move_pair = node.get_move()
        if move_pair is None:
            continue
        color, coords = move_pair

        # If the played move is a pass and we're recording policy targets,
        # skip this ply (no usable one-hot policy label).
        if want_policy_targets and coords is None:
            # Still need to advance the game state.
            move = Move.pass_turn()
            if not state.is_valid_move(move):
                return None, None, None
            state = state.apply_move(move)
            continue

        # Record the encoded state *before* applying this move (the to-play
        # side is about to move, and we want value from their perspective).
        states.append(encoder.encode(state))
        mover = state.next_player
        values.append(win_value if mover == winner_player else -win_value)

        if coords is None:
            move = Move.pass_turn()
        else:
            row, col = coords
            # sgfmill row 0 = bottom; alphigo row 1 = bottom.
            point = Point(row=row + 1, col=col + 1)
            move = Move.play_move(point)
            if want_policy_targets:
                idx = (point.row - 1) * board_size + (point.col - 1)
                onehot = np.zeros(num_points, dtype=np.float32)
                onehot[idx] = 1.0
                policy_targets.append(onehot)
        # Defensive: skip move if illegal (shouldn't happen with SGFs but be safe).
        if not state.is_valid_move(move):
            return (None, None, None) if want_policy_targets else (None, None)
        state = state.apply_move(move)

    if not states:
        return (None, None, None) if want_policy_targets else (None, None)
    if want_policy_targets:
        return states, values, policy_targets
    return states, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir',
                        default='data/go9')
    parser.add_argument('--out',
                        default='data/sgf_value_buffer_b9.h5')
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--encoder', default='alphazero')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Cap on games to encode (None = all).')
    parser.add_argument('--chunk-games', type=int, default=0,
                        help='If >0, save a separate .h5 every this many '
                             'games (suffix _0001.h5 etc). Keeps RAM '
                             'bounded for large corpora — 19x19 at 24KB/'
                             'position × 250 moves × 1000 games ≈ 6GB.')
    parser.add_argument('--score-margin-scale', type=float, default=None,
                        metavar='SCALE',
                        help='If set, write value = sign × tanh(margin/SCALE) '
                             'from RE[B+M.M] tags instead of ±1. Resignation/'
                             'timeout/forfeit (no numeric margin) saturate to '
                             '±1. Suggested SCALE for 19x19: 20.')
    parser.add_argument('--policy-target', choices=('none', 'one-hot'),
                        default='none',
                        help='Policy target encoding. "none" (default, '
                             'backwards-compatible) writes zeros. "one-hot" '
                             'writes a one-hot at the played move\'s index '
                             'and SKIPS pass positions (rare in pro games).')
    args = parser.parse_args()

    from alphigo.encoders.base import get_encoder_by_name
    from alphigo.rl.zero_experience import ZeroReplayBuffer

    encoder = get_encoder_by_name(args.encoder, args.board_size)

    sgf_paths = []
    for root, _, files in os.walk(args.src_dir):
        for f in files:
            if f.endswith('.sgf'):
                sgf_paths.append(os.path.join(root, f))
    sgf_paths.sort()
    if args.max_games:
        sgf_paths = sgf_paths[:args.max_games]
    print(f'[scan] {len(sgf_paths)} SGFs under {args.src_dir}')

    all_states, all_values = [], []
    all_policy = [] if args.policy_target == 'one-hot' else None
    skipped_parse = 0
    skipped_size = 0
    skipped_re = 0
    skipped_illegal = 0
    chunk_idx = 0
    games_since_flush = 0
    out_base, out_ext = os.path.splitext(args.out)
    t0 = time.perf_counter()
    last_print = t0

    def flush_chunk(states_list, values_list, policy_list, idx):
        if not states_list:
            return None
        states_arr = np.stack(states_list).astype(np.float32)
        values_arr = np.asarray(values_list, dtype=np.float32)
        n = states_arr.shape[0]
        if policy_list is None:
            policy_arr = np.zeros((n, encoder.num_points()), dtype=np.float32)
        else:
            policy_arr = np.stack(policy_list).astype(np.float32)
            assert policy_arr.shape[0] == n, (
                f'policy/state count mismatch: {policy_arr.shape[0]} vs {n}')
        from alphigo.rl.zero_experience import ZeroReplayBuffer
        buf = ZeroReplayBuffer(
            encoded_states=states_arr,
            policy_targets=policy_arr,
            value_targets=values_arr,
        )
        path = (f'{out_base}_{idx:04d}{out_ext}'
                if args.chunk_games > 0 else args.out)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with h5py.File(path, 'w') as h5:
            buf.serialize(h5)
            h5.attrs['encoder'] = args.encoder
            h5.attrs['board_size'] = args.board_size
            h5.attrs['chunk_idx'] = int(idx)
            h5.attrs['num_positions'] = int(n)
        size_mb = (states_arr.nbytes + values_arr.nbytes
                   + policy_arr.nbytes) / 1e6
        print(f'  [chunk {idx}] saved {path} ({n} positions, '
              f'{size_mb:.1f} MB)', flush=True)
        return path

    want_policy = (args.policy_target == 'one-hot')
    for i, p in enumerate(sgf_paths):
        try:
            with open(p, 'rb') as f:
                sgf_bytes = f.read()
        except OSError:
            skipped_parse += 1
            continue
        if want_policy:
            states, values, policy = encode_game(
                sgf_bytes, encoder, args.board_size,
                score_margin_scale=args.score_margin_scale,
                want_policy_targets=True)
        else:
            states, values = encode_game(
                sgf_bytes, encoder, args.board_size,
                score_margin_scale=args.score_margin_scale,
                want_policy_targets=False)
            policy = None
        if states is None:
            # Distinguish failure modes for diagnostics. Re-parse to bucket.
            from sgfmill.sgf import Sgf_game
            try:
                g = Sgf_game.from_bytes(sgf_bytes)
                if g.get_size() != args.board_size:
                    skipped_size += 1
                else:
                    re_str = (g.get_root().get('RE')
                              if g.get_root().has_property('RE') else None)
                    if parse_result(re_str) is None:
                        skipped_re += 1
                    else:
                        skipped_illegal += 1
            except Exception:
                skipped_parse += 1
            continue
        all_states.extend(states)
        all_values.extend(values)
        if want_policy and policy is not None:
            all_policy.extend(policy)
        games_since_flush += 1

        now = time.perf_counter()
        if now - last_print > 5 or i == len(sgf_paths) - 1:
            rate = (i + 1) / (now - t0)
            print(f'  [{i + 1:>5}/{len(sgf_paths)}]  '
                  f'{len(all_states):>7} positions  '
                  f'({rate:.1f} games/s)  '
                  f'skipped: parse={skipped_parse} '
                  f'size={skipped_size} re={skipped_re} '
                  f'illegal={skipped_illegal}',
                  flush=True)
            last_print = now

        if args.chunk_games > 0 and games_since_flush >= args.chunk_games:
            flush_chunk(all_states, all_values, all_policy, chunk_idx)
            all_states = []
            all_values = []
            if all_policy is not None:
                all_policy = []
            chunk_idx += 1
            games_since_flush = 0

    print(f'\n[done] total positions (in last chunk + flushed): '
          f'~{len(all_states)} (current buffer)')
    print(f'  current-buffer value balance: '
          f'+1: {sum(1 for v in all_values if v > 0)}  '
          f'-1: {sum(1 for v in all_values if v < 0)}')

    # Always flush the last (possibly partial) chunk if any data remains.
    if all_states:
        flush_chunk(all_states, all_values, all_policy, chunk_idx)
    elif chunk_idx == 0:
        print('ERROR: no states produced. Check encoder name and SGF dir.',
              file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
