from sgfmill import sgf
from alphigo.alphigo_board import GameState, Move
from alphigo.alphigo_types import Point
from alphigo.utils import print_board

sgf_content = "(;GM[1]FF[4]SZ[9];B[ee];W[ef];B[ff];W[df];B[fe];W[fc];B[ec];W[gd];B[fb])"
sgf_game = sgf.Sgf_game.from_string(sgf_content)
game_state = GameState.new_game(9)

for item in sgf_game.get_main_sequence():
    color, move_tuple = item.get_move()
    if color is not None and move_tuple is not None:
        row, col = move_tuple
        point = Point(row + 1, col + 1)
        move = Move.play_move(point)
        game_state = game_state.apply_move(move)
        print_board(game_state.board)
