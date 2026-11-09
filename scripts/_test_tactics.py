#!/usr/bin/env python3
"""Test model tactical awareness on basic Go puzzles (9x9).

Sets up known positions, runs model search, checks if the correct move is found.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from alphigo.alphigo_board import Board, GameState, Move
from alphigo.alphigo_types import Player, Point
from alphigo.agent.zero_agent import load_zero_agent
from alphigo.agent.gumbel_zero_agent import GumbelZeroAgent
import h5py

def place_stones(board_size, black_points, white_points):
    """Create a GameState with stones placed at given points.

    Places stones by replaying moves alternately, skipping when needed
    to get the right color at each position.
    """
    game = GameState.new_game(board_size)
    all_moves = []
    bi, wi = 0, 0
    while bi < len(black_points) or wi < len(white_points):
        if game.next_player == Player.black:
            if bi < len(black_points):
                all_moves.append(('B', black_points[bi]))
                bi += 1
            else:
                all_moves.append(('B', None))  # pass
        else:
            if wi < len(white_points):
                all_moves.append(('W', white_points[wi]))
                wi += 1
            else:
                all_moves.append(('W', None))  # pass

    for color, pt in all_moves:
        if pt is None:
            game = game.apply_move(Move.pass_turn())
        else:
            move = Move.play_move(pt)
            if game.board.is_on_grid(pt) and game.is_valid_move(move):
                game = game.apply_move(move)
            else:
                game = game.apply_move(Move.pass_turn())
    return game


def setup_position(board_size, black_str, white_str, next_player=Player.black):
    """Parse position strings like 'a1 b2 c3' into a GameState."""
    def parse_points(s):
        if not s.strip():
            return []
        pts = []
        for token in s.strip().split():
            col = ord(token[0]) - ord('a') + 1
            row = int(token[1:])
            pts.append(Point(row=row, col=col))
        return pts

    black_pts = parse_points(black_str)
    white_pts = parse_points(white_str)

    board = Board(board_size, board_size)
    for pt in black_pts:
        board = board._place_stone(Player.black, pt)
    for pt in white_pts:
        board = board._place_stone(Player.white, pt)

    game = GameState(board, next_player, None, None)
    return game


def print_board(game):
    """Simple text board display."""
    board = game.board
    size = board.num_rows
    print('  ', end='')
    for c in range(1, size + 1):
        print(f' {chr(ord("a") + c - 1)}', end='')
    print()
    for r in range(size, 0, -1):
        print(f'{r:2d}', end='')
        for c in range(1, size + 1):
            go_string = board._grid.get(Point(r, c))
            if go_string is None:
                print(' .', end='')
            elif go_string.color == Player.black:
                print(' X', end='')
            else:
                print(' O', end='')
        print(f' {r}')
    print('  ', end='')
    for c in range(1, size + 1):
        print(f' {chr(ord("a") + c - 1)}', end='')
    print()


def point_to_str(pt):
    if pt is None:
        return 'pass'
    return f'{chr(ord("a") + pt.col - 1)}{pt.row}'


def run_puzzle(agent, game, correct_moves, description, sims=64):
    """Run agent on a position and check against correct moves."""
    agent._num_simulations = sims
    agent.set_temperature(0.0)  # greedy for testing

    result = agent.search(game)
    move = result['move']
    value = result['value']

    chosen_pt = move.point if not move.is_pass else None
    chosen_str = point_to_str(chosen_pt)
    correct_strs = [point_to_str(p) for p in correct_moves]

    is_correct = chosen_pt in correct_moves or (chosen_pt is None and None in correct_moves)

    status = 'PASS' if is_correct else 'FAIL'
    print(f'\n{"="*60}')
    print(f'Puzzle: {description}')
    print(f'Next to play: {"Black" if game.next_player == Player.black else "White"}')
    print_board(game)
    print(f'Correct: {", ".join(correct_strs)}')
    print(f'Model chose: {chosen_str} (value={value:.3f})')
    print(f'Result: {status}')
    print(f'{"="*60}')
    return is_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True)
    parser.add_argument('--sims', type=int, default=64)
    args = parser.parse_args()

    print(f'Loading agent: {args.agent}')
    with h5py.File(args.agent, 'r') as h5:
        agent = load_zero_agent(h5)
    if hasattr(agent, '_gumbel_m'):
        print(f'Gumbel agent: m={agent._gumbel_m}, sims={agent._num_simulations}')

    puzzles = []
    BS = 9

    # ── 1. Capture: black group in atari, white to capture ──────────
    # White stone at e5 with 1 liberty at e6. Black surrounds.
    # Black to play e6 to capture.
    game = GameState.new_game(BS)
    moves_seq = [
        # Build a position where white e5 is in atari
        (Player.black, 'd5'), (Player.white, 'e5'),
        (Player.black, 'f5'), (Player.white, 'a1'),
        (Player.black, 'e4'), (Player.white, 'a2'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Now black plays e6 to capture white e5
    puzzles.append((game, [Point(row=6, col=5)],
                    '1. Capture: take white stone in atari (e6 captures e5)'))

    # ── 2. Escape atari: black stone at e5 has 1 liberty ────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'd5'),
        (Player.black, 'a9'), (Player.white, 'f5'),
        (Player.black, 'a8'), (Player.white, 'e4'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Black e5 has 1 liberty at e6. Black should play e6 to escape.
    puzzles.append((game, [Point(row=6, col=5)],
                    '2. Escape atari: extend black stone at e5 (play e6)'))

    # ── 3. Capture two stones ──────────────────────────────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'd5'), (Player.white, 'e5'),
        (Player.black, 'f5'), (Player.white, 'e4'),
        (Player.black, 'e3'), (Player.white, 'a1'),
        (Player.black, 'd4'), (Player.white, 'a2'),
        (Player.black, 'f4'), (Player.white, 'a3'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White e5+e4 have 1 shared liberty at e6. Black plays e6 to capture both.
    # Wait, let me recalculate. e5 neighbors: d5(B), f5(B), e4(W), e6.
    # e4 neighbors: d4(B), f4(B), e3(B), e5(W).
    # So the white group {e5,e4} has liberties: e6 only. Black plays e6.
    puzzles.append((game, [Point(row=6, col=5)],
                    '3. Capture group: take 2 white stones with 1 liberty (e6)'))

    # ── 4. Don't fill own eye ──────────────────────────────────────
    # Black has a safe group with an eye. Should not fill it.
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'a1'), (Player.white, 'a3'),
        (Player.black, 'b1'), (Player.white, 'b3'),
        (Player.black, 'c1'), (Player.white, 'c3'),
        (Player.black, 'a2'), (Player.white, 'd3'),
        (Player.black, 'c2'), (Player.white, 'd2'),
        (Player.black, 'b4'), (Player.white, 'd1'),  # dummy white
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # b2 is an eye for black. Black should NOT play b2.
    # We test that the model does NOT choose b2.
    # Any move except b2 is acceptable.
    puzzles.append((game, [None],  # placeholder - we'll handle specially
                    '4. Eye awareness: do NOT fill own eye at b2'))

    # ── 5. Connect to save: two black groups, connect them ─────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'd5'),
        (Player.black, 'e7'), (Player.white, 'f5'),
        (Player.black, 'a9'), (Player.white, 'e4'),
        (Player.black, 'a8'), (Player.white, 'e8'),
        (Player.black, 'a7'), (Player.white, 'f7'),
        (Player.black, 'a6'), (Player.white, 'd7'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # e5 has liberty e6. e7 has liberties e6 (shared path).
    # Black should play e6 to connect the two groups.
    puzzles.append((game, [Point(row=6, col=5)],
                    '5. Connect: link two black groups via e6'))

    # ── 6. Simple ladder (3 moves ahead) ───────────────────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'e6'),
        (Player.black, 'd6'), (Player.white, 'a1'),
        (Player.black, 'f6'), (Player.white, 'a2'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White e6 is in atari (liberties: e7 only, since d6,f6,e5 are black).
    # Black should play e7 to capture.
    puzzles.append((game, [Point(row=7, col=5)],
                    '6. Ladder start: capture white e6 by playing e7'))

    # ── 7. Corner capture: take white stone in corner ────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'a2'), (Player.white, 'a1'),
        (Player.black, 'b1'), (Player.white, 'i9'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White a1 has no liberties except... a2(B), b1(B). Wait — a1 neighbors
    # are a2 and b1, both black. But white was placed there, so it must have
    # had a liberty when placed. Let me redo: place white first.
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'b1'), (Player.white, 'a1'),
        (Player.black, 'i9'), (Player.white, 'i8'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White a1 neighbors: a2 (empty), b1 (black). So 1 liberty at a2.
    # Black plays a2 to capture.
    puzzles.append((game, [Point(row=2, col=1)],
                    '7. Corner capture: take white a1 by playing a2'))

    # ── 8. Avoid self-atari: don't play into 1-liberty spot ────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'e4'),
        (Player.black, 'i9'), (Player.white, 'd5'),
        (Player.black, 'i8'), (Player.white, 'f5'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Black e5 has 1 liberty at e6. Playing elsewhere is fine.
    # The model should NOT play a random move that's self-atari.
    # We just check it doesn't pass — it should play somewhere useful.
    # Actually, let's make this: black should extend at e6 to save e5.
    puzzles.append((game, [Point(row=6, col=5)],
                    '8. Defend: save black e5 by extending to e6'))

    # ── 9. Edge capture: white on edge with 1 liberty ──────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'a2'), (Player.white, 'a3'),
        (Player.black, 'b3'), (Player.white, 'i9'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White a3 neighbors: a2(B), b3(B), a4(empty). 1 liberty at a4.
    # Black plays a4 to capture.
    puzzles.append((game, [Point(row=4, col=1)],
                    '9. Edge capture: take white a3 by playing a4'))

    # ── 10. Don't self-capture: avoid playing into surrounded point ─
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'i9'), (Player.white, 'e4'),
        (Player.black, 'i8'), (Player.white, 'd5'),
        (Player.black, 'i7'), (Player.white, 'f5'),
        (Player.black, 'i6'), (Player.white, 'e6'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # e5 is surrounded by white on all 4 sides. Playing e5 is suicide.
    # Black should play anywhere except e5.
    puzzles.append((game, [None],  # placeholder - special handling
                    '10. Avoid suicide: do NOT play e5 (surrounded by white)'))

    # ── 11. Capture 3 stones: bigger group capture ─────────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'd5'), (Player.white, 'e5'),
        (Player.black, 'f5'), (Player.white, 'e6'),
        (Player.black, 'd6'), (Player.white, 'e7'),
        (Player.black, 'f6'), (Player.white, 'a1'),
        (Player.black, 'e3'), (Player.white, 'a2'),
        (Player.black, 'd4'), (Player.white, 'a3'),
        (Player.black, 'f4'), (Player.white, 'a4'),
        (Player.black, 'e4'), (Player.white, 'a5'),
        (Player.black, 'f7'), (Player.white, 'a6'),
        (Player.black, 'd7'), (Player.white, 'a7'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White group: e5, e6, e7. Surrounded by black on all sides.
    # Liberties: e8 only. Black plays e8 to capture 3 stones.
    puzzles.append((game, [Point(row=8, col=5)],
                    '11. Capture 3-stone group: play e8'))

    # ── 12. Tenuki penalty: respond to atari before playing elsewhere ─
    # White just played to put black in atari. Black must respond.
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'd5'),
        (Player.black, 'e6'), (Player.white, 'f5'),
        (Player.black, 'i1'), (Player.white, 'e4'),
        (Player.black, 'i2'), (Player.white, 'f6'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Black e5 has liberties: none? d5(W), f5(W), e4(W), e6(B).
    # Actually e5-e6 group. e5 neighbors: d5(W), f5(W), e4(W), e6(B).
    # e6 neighbors: d6(empty), f6(W), e5(B), e7(empty).
    # Group {e5,e6} liberties: d6, e7. Has 2 liberties.
    # Not in atari. Let me make a real atari scenario.
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'd5'),
        (Player.black, 'i1'), (Player.white, 'f5'),
        (Player.black, 'i2'), (Player.white, 'e6'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Black e5 neighbors: d5(W), f5(W), e6(W), e4(empty).
    # 1 liberty at e4. Black should save by playing e4.
    puzzles.append((game, [Point(row=4, col=5)],
                    '12. Respond to atari: save e5 by extending to e4'))

    # ── 13. Capture on second line: edge awareness ─────────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'c2'), (Player.white, 'c1'),
        (Player.black, 'b1'), (Player.white, 'i9'),
        (Player.black, 'd1'), (Player.white, 'i8'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White c1 neighbors: b1(B), d1(B), c2(B). 0 liberties?
    # Wait, b1 and d1 are black, c2 is black. c1 has no liberties — it
    # would have been captured. Let me fix: place in order that keeps liberties.
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'c2'), (Player.white, 'c1'),
        (Player.black, 'd1'), (Player.white, 'i9'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White c1 neighbors: c2(B), d1(B), b1(empty). 1 liberty at b1.
    # Black plays b1 to capture.
    puzzles.append((game, [Point(row=1, col=2)],
                    '13. First-line capture: take white c1 by playing b1'))

    # ── 14. Play in center vs edge: prefer center early ────────────
    # Empty board, black to play. Center (e5 region) is stronger than corner.
    game = GameState.new_game(BS)
    # Fresh game, black to play. Good first moves: c3-g7 area, especially
    # e5, d5, f5, e4, e6, d4, f6, d6, f4 (star points / center).
    center_points = []
    for r in range(3, 8):
        for c in range(3, 8):
            center_points.append(Point(row=r, col=c))
    puzzles.append((game, center_points,
                    '14. Opening sense: play in center area (c3-g7)'))

    # ── 15. Capture vs extend: take the capture ────────────────────
    # Both a capture opportunity and an extension exist.
    # Capture is worth more (removes opponent stones).
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'c5'), (Player.white, 'd5'),
        (Player.black, 'e5'), (Player.white, 'i9'),
        (Player.black, 'd4'), (Player.white, 'i8'),
        (Player.black, 'd6'), (Player.white, 'i7'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White d5 is fully surrounded. Wait — d5 neighbors: c5(B), e5(B),
    # d4(B), d6(B). All 4 occupied by black. White d5 has 0 liberties —
    # it was captured when d6 was played. Let me redo.
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'c5'), (Player.white, 'd5'),
        (Player.black, 'e5'), (Player.white, 'i9'),
        (Player.black, 'd4'), (Player.white, 'i8'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White d5 neighbors: c5(B), e5(B), d4(B), d6(empty). 1 liberty at d6.
    # Black plays d6 to capture.
    puzzles.append((game, [Point(row=6, col=4)],
                    '15. Obvious capture: take white d5 by playing d6'))

    # ── 16. Double atari: fork two groups ──────────────────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'd4'),
        (Player.black, 'e3'), (Player.white, 'f4'),
        (Player.black, 'i1'), (Player.white, 'c5'),
        (Player.black, 'i2'), (Player.white, 'g5'),
        (Player.black, 'i3'), (Player.white, 'e6'),
        (Player.black, 'd5'), (Player.white, 'i9'),
        (Player.black, 'f5'), (Player.white, 'i8'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Black e5,d5,f5 and e3. White has d4 (neighbors: d5(B),d3,c4,e4)
    # and f4 (neighbors: f5(B),f3,g4,e4).
    # e4 would put both d4 and f4 in atari (double atari).
    # d4 liberties: c4, d3, e4. f4 liberties: e4, g4, f3.
    # Playing e4 removes a liberty from both. d4 would have c4,d3.
    # f4 would have g4,f3. Not atari. This needs more surrounding.
    # Let me simplify: just test if black can see e4 as a good move.
    puzzles.append((game, [Point(row=4, col=5)],
                    '16. Central pressure: play e4 to squeeze white'))

    # ── 17. Atari from the side: white stone on second line ──────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e2'), (Player.white, 'e1'),
        (Player.black, 'd1'), (Player.white, 'i9'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White e1 neighbors: e2(B), d1(B), f1(empty). 1 liberty at f1.
    puzzles.append((game, [Point(row=1, col=6)],
                    '17. Bottom-edge capture: take white e1 by playing f1'))

    # ── 18. Escape atari on edge: black on first line ──────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e1'), (Player.white, 'd1'),
        (Player.black, 'i9'), (Player.white, 'e2'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Black e1 neighbors: d1(W), e2(W), f1(empty). 1 liberty at f1.
    # Black must play f1 to survive.
    puzzles.append((game, [Point(row=1, col=6)],
                    '18. Edge escape: save black e1 by extending to f1'))

    # ── 19. Recapture: take back after opponent captures ───────────
    # White just captured a black stone at e5. Black recaptures.
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'd5'), (Player.white, 'e6'),
        (Player.black, 'f5'), (Player.white, 'e4'),
        (Player.black, 'e5'), (Player.white, 'a1'),
        # Now surround white stone that will appear at e5
        (Player.black, 'i1'), (Player.white, 'e5'),
        # White captures black e5? No — black e5 has liberties via d5,f5.
        # Let me think more carefully.
    ]
    # This is tricky to set up without ko. Skip complex recapture.
    # Instead: simple "take the last liberty" from a different angle.
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'f5'), (Player.white, 'e5'),
        (Player.black, 'e6'), (Player.white, 'a1'),
        (Player.black, 'e4'), (Player.white, 'a2'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White e5 neighbors: f5(B), e6(B), e4(B), d5(empty). 1 liberty at d5.
    puzzles.append((game, [Point(row=5, col=4)],
                    '19. Capture from left: take white e5 by playing d5'))

    # ── 20. Counter-atari: put opponent in atari instead of running ─
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'd5'),
        (Player.black, 'e4'), (Player.white, 'f5'),
        (Player.black, 'f4'), (Player.white, 'e6'),
        (Player.black, 'i9'), (Player.white, 'd4'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Black {e5,e4} group. e5: d5(W), f5(W), e6(W), e4(B).
    # e4: d4(W), f4(B), e3(empty), e5(B).
    # Group liberties: e3 only. In atari!
    # But d4 is a lone white stone. d4 neighbors: d5(W), d3(empty),
    # c4(empty), e4(B). d4 has liberties d3, c4 — 2 libs. Not in atari.
    # Black should play e3 to escape (extend). Or capture something.
    # Actually, e3 just extends. Let's accept e3 as correct.
    puzzles.append((game, [Point(row=3, col=5)],
                    '20. Escape under pressure: extend at e3 to save group'))

    # ── 21. Two-liberty group: reduce then capture ─────────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'd5'), (Player.white, 'e5'),
        (Player.black, 'f5'), (Player.white, 'a1'),
        (Player.black, 'e4'), (Player.white, 'a2'),
        (Player.black, 'e7'), (Player.white, 'a3'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White e5: d5(B), f5(B), e4(B), e6(empty). 1 liberty at e6.
    # Black plays e6 to capture.
    puzzles.append((game, [Point(row=6, col=5)],
                    '21. Capture with approach: take white e5 at e6'))

    # ── 22. Save a bigger group: 2-stone black in atari ────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'd5'),
        (Player.black, 'e4'), (Player.white, 'f5'),
        (Player.black, 'i9'), (Player.white, 'f4'),
        (Player.black, 'i8'), (Player.white, 'e3'),
        (Player.black, 'i7'), (Player.white, 'e6'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Black {e5,e4}: e5 neighbors d5(W),f5(W),e6(W),e4(B).
    # e4: d4(empty),f4(W),e3(W),e5(B). Group liberties: d4 only. Atari!
    # Black plays d4 to escape.
    puzzles.append((game, [Point(row=4, col=4)],
                    '22. Save 2-stone group: extend to d4'))

    # ── 23. Approach from star point: good shape ───────────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'c3'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # Second move for black. Should approach white or take another big point.
    # Good moves: g7, g3, c7 (star points), or approach c3 with d3/c5/d5.
    good_2nd = []
    for r in range(3, 8):
        for c in range(3, 8):
            if not (r == 5 and c == 5):  # already played
                good_2nd.append(Point(row=r, col=c))
    puzzles.append((game, good_2nd,
                    '23. Second move: play in center-ish area (c3-g7)'))

    # ── 24. Don't pass early: game just started ────────────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'e5'), (Player.white, 'd5'),
        (Player.black, 'f5'), (Player.white, 'e6'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # 4 moves in. Board is mostly empty. Model should NOT pass.
    puzzles.append((game, [None],
                    '24. No early pass: board is mostly empty'))

    # ── 25. Capture in corner: white trapped ───────────────────────
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'b1'), (Player.white, 'a1'),
        (Player.black, 'a2'), (Player.white, 'i9'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White a1: b1(B), a2(B). 0 liberties? No — when a2 was played,
    # a1 had a2 as a liberty. After a2 is placed by black, a1 has
    # no liberties and is captured automatically. So white a1 is gone.
    # Need different setup: place black after white has liberty.
    game = GameState.new_game(BS)
    moves_seq = [
        (Player.black, 'a2'), (Player.white, 'a1'),
        (Player.black, 'i9'), (Player.white, 'i8'),
    ]
    for player, ms in moves_seq:
        col = ord(ms[0]) - ord('a') + 1
        row = int(ms[1:])
        while game.next_player != player:
            game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.play_move(Point(row=row, col=col)))
    # White a1: a2(B), b1(empty). 1 liberty at b1. Black plays b1.
    puzzles.append((game, [Point(row=1, col=2)],
                    '25. Corner kill: capture white a1 by playing b1'))

    # ── Run all puzzles ────────────────────────────────────────────
    passed = 0
    total = len(puzzles)
    results = []

    for i, (game, correct, desc) in enumerate(puzzles):
        num = int(desc.split('.')[0])
        agent._num_simulations = args.sims
        agent.set_temperature(0.0)
        result = agent.search(game)
        move = result['move']
        value = result['value']
        chosen_pt = move.point if not move.is_pass else None
        chosen_str = point_to_str(chosen_pt)

        if '4. Eye' in desc:
            bad_move = Point(row=2, col=2)
            is_correct = chosen_pt != bad_move
            correct_str = 'anything except b2'
        elif '10. Avoid suicide' in desc:
            bad_move = Point(row=5, col=5)
            is_correct = chosen_pt != bad_move
            correct_str = 'anything except e5'
        elif '24. No early pass' in desc:
            is_correct = not move.is_pass
            correct_str = 'any move (not pass)'
        else:
            is_correct = chosen_pt in correct or (chosen_pt is None and None in correct)
            correct_str = ', '.join(point_to_str(p) for p in correct)

        if is_correct:
            passed += 1
        results.append((num, is_correct))

        status = 'PASS' if is_correct else 'FAIL'
        print(f'\n{"="*60}')
        print(f'Puzzle: {desc}')
        print(f'Next to play: {"Black" if game.next_player == Player.black else "White"}')
        print_board(game)
        print(f'Correct: {correct_str}')
        print(f'Model chose: {chosen_str} (value={value:.3f})')
        print(f'Result: {status}')
        print(f'{"="*60}')

    # ── Category breakdown ────────────────────────────────────────
    categories = {
        'capture_center': [1, 3, 6, 11, 15, 16, 19, 21],
        'capture_edge':   [7, 9, 13, 17, 25],
        'defense':        [2, 5, 8, 12, 18, 20, 22],
        'avoidance':      [4, 10, 24],
        'opening':        [14, 23],
    }
    result_map = {num: ok for num, ok in results}

    print(f'\n{"#"*60}')
    print(f'RESULTS: {passed}/{total} puzzles passed')
    print(f'{"#"*60}')
    print(f'\nCategory breakdown:')
    for cat, nums in categories.items():
        cat_results = [(n, result_map.get(n)) for n in nums if n in result_map]
        cat_pass = sum(1 for _, ok in cat_results if ok)
        cat_total = len(cat_results)
        details = ' '.join(f'{n}:{"ok" if ok else "X"}' for n, ok in cat_results)
        print(f'  {cat:16s} {cat_pass}/{cat_total}  [{details}]')
    print(f'\nPUZZLE_SCORE={passed}/{total}')


if __name__ == '__main__':
    main()
