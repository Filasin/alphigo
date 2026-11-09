"""Board logic and rules validation.

Test cases adapted from macfergus/baduk (MIT license) and extended with
additional edge cases for capture, ko, superko, liberty tracking, string
merging, and scoring.
"""
import unittest

from alphigo.alphigo_board import Board, GameState, GoString, Move
from alphigo.alphigo_types import Player, Point


def place_stones(board, color, points):
    for r, c in points:
        board.place_stone(color, Point(row=r, col=c))


class TestPlaceStone(unittest.TestCase):
    def test_place_and_read(self):
        board = Board(5, 5)
        self.assertIsNone(board.get_player_on_point(Point(2, 3)))
        board.place_stone(Player.black, Point(3, 3))
        self.assertIsNone(board.get_player_on_point(Point(2, 3)))
        self.assertEqual(Player.black, board.get_player_on_point(Point(3, 3)))

    def test_occupied_after_place(self):
        board = Board(9, 9)
        board.place_stone(Player.white, Point(5, 5))
        self.assertEqual(Player.white, board.get_player_on_point(Point(5, 5)))


class TestLiberties(unittest.TestCase):
    def test_center_stone_four_liberties(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(3, 3))
        s = board.get_go_string(Point(3, 3))
        self.assertEqual(4, s.num_liberties)

    def test_corner_stone_two_liberties(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(1, 1))
        s = board.get_go_string(Point(1, 1))
        self.assertEqual(2, s.num_liberties)

    def test_edge_stone_three_liberties(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(1, 3))
        s = board.get_go_string(Point(1, 3))
        self.assertEqual(3, s.num_liberties)

    def test_remove_liberties(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(3, 3))
        board.place_stone(Player.white, Point(2, 2))
        s = board.get_go_string(Point(2, 2))
        self.assertEqual(4, s.num_liberties)
        board.place_stone(Player.black, Point(3, 2))
        s = board.get_go_string(Point(2, 2))
        self.assertEqual(3, s.num_liberties)

    def test_empty_triangle_liberties(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(1, 1))
        board.place_stone(Player.black, Point(1, 2))
        board.place_stone(Player.black, Point(2, 2))
        board.place_stone(Player.white, Point(2, 1))
        s = board.get_go_string(Point(1, 1))
        self.assertEqual(3, s.num_liberties)


class TestStringMerging(unittest.TestCase):
    def test_merge_two_stones(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(3, 3))
        board.place_stone(Player.black, Point(3, 4))
        s = board.get_go_string(Point(3, 3))
        self.assertIs(s, board.get_go_string(Point(3, 4)))
        self.assertEqual(2, len(s.stones))
        self.assertEqual(6, s.num_liberties)

    def test_merge_three_stones_L(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(1, 1))
        board.place_stone(Player.black, Point(1, 2))
        board.place_stone(Player.black, Point(2, 1))
        s = board.get_go_string(Point(1, 1))
        self.assertEqual(3, len(s.stones))

    def test_bridge_merge(self):
        """Two separate strings joined by a bridge stone."""
        board = Board(9, 9)
        board.place_stone(Player.black, Point(3, 3))
        board.place_stone(Player.black, Point(3, 5))
        s1 = board.get_go_string(Point(3, 3))
        s2 = board.get_go_string(Point(3, 5))
        self.assertIsNot(s1, s2)
        board.place_stone(Player.black, Point(3, 4))
        s = board.get_go_string(Point(3, 3))
        self.assertIs(s, board.get_go_string(Point(3, 5)))
        self.assertEqual(3, len(s.stones))


class TestCapture(unittest.TestCase):
    def test_capture_single_stone(self):
        board = Board(19, 19)
        board.place_stone(Player.black, Point(2, 2))
        board.place_stone(Player.white, Point(1, 2))
        board.place_stone(Player.white, Point(2, 1))
        board.place_stone(Player.white, Point(2, 3))
        self.assertEqual(Player.black, board.get_player_on_point(Point(2, 2)))
        board.place_stone(Player.white, Point(3, 2))
        self.assertIsNone(board.get_player_on_point(Point(2, 2)))

    def test_capture_two_stones(self):
        board = Board(19, 19)
        place_stones(board, Player.black, [(2, 2), (2, 3)])
        place_stones(board, Player.white, [(1, 2), (1, 3), (3, 2), (3, 3)])
        self.assertIsNotNone(board.get_player_on_point(Point(2, 2)))
        board.place_stone(Player.white, Point(2, 1))
        board.place_stone(Player.white, Point(2, 4))
        self.assertIsNone(board.get_player_on_point(Point(2, 2)))
        self.assertIsNone(board.get_player_on_point(Point(2, 3)))

    def test_corner_capture(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(1, 1))
        board.place_stone(Player.white, Point(1, 2))
        board.place_stone(Player.white, Point(2, 1))
        self.assertIsNone(board.get_player_on_point(Point(1, 1)))

    def test_capture_restores_liberties(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(3, 3))
        board.place_stone(Player.white, Point(3, 2))
        board.place_stone(Player.white, Point(2, 3))
        board.place_stone(Player.white, Point(4, 3))
        s = board.get_go_string(Point(3, 3))
        self.assertEqual(1, s.num_liberties)
        board.place_stone(Player.black, Point(2, 2))
        board.place_stone(Player.black, Point(3, 1))
        board.place_stone(Player.black, Point(4, 2))
        board.place_stone(Player.black, Point(3, 4))
        self.assertIsNone(board.get_player_on_point(Point(3, 2)))
        s = board.get_go_string(Point(3, 3))
        self.assertGreater(s.num_liberties, 1)

    def test_capture_is_not_suicide(self):
        board = Board(19, 19)
        board.place_stone(Player.black, Point(1, 1))
        board.place_stone(Player.black, Point(2, 2))
        board.place_stone(Player.black, Point(1, 3))
        board.place_stone(Player.white, Point(2, 1))
        board.place_stone(Player.white, Point(1, 2))
        self.assertIsNone(board.get_player_on_point(Point(1, 1)))
        self.assertEqual(Player.white, board.get_player_on_point(Point(2, 1)))
        self.assertEqual(Player.white, board.get_player_on_point(Point(1, 2)))


class TestSelfCapture(unittest.TestCase):
    def test_self_capture_detected(self):
        board = Board(5, 5)
        board.place_stone(Player.white, Point(1, 2))
        board.place_stone(Player.white, Point(2, 1))
        self.assertTrue(board.is_self_capture(Player.black, Point(1, 1)))

    def test_not_self_capture_with_liberty(self):
        board = Board(5, 5)
        board.place_stone(Player.white, Point(1, 2))
        self.assertFalse(board.is_self_capture(Player.black, Point(1, 1)))

    def test_not_self_capture_with_capture(self):
        """Playing into a spot with no liberties is legal if it captures."""
        board = Board(5, 5)
        board.place_stone(Player.black, Point(1, 1))
        board.place_stone(Player.black, Point(2, 2))
        board.place_stone(Player.black, Point(1, 3))
        board.place_stone(Player.white, Point(2, 1))
        self.assertFalse(board.is_self_capture(Player.white, Point(1, 2)))

    def test_not_self_capture_connecting_to_group(self):
        board = Board(5, 5)
        board.place_stone(Player.black, Point(3, 3))
        board.place_stone(Player.white, Point(2, 2))
        board.place_stone(Player.white, Point(1, 1))
        self.assertFalse(board.is_self_capture(Player.white, Point(1, 2)))


class TestKo(unittest.TestCase):
    def _setup_ko(self):
        """Set up a standard ko position and return game state.

            .wb.
            wb.b
            .wb.
        """
        game = GameState.new_game(19)
        game = game.apply_move(Move.play_move(Point(1, 3)))
        game = game.apply_move(Move.play_move(Point(1, 2)))
        game = game.apply_move(Move.play_move(Point(2, 2)))
        game = game.apply_move(Move.play_move(Point(2, 1)))
        game = game.apply_move(Move.play_move(Point(3, 3)))
        game = game.apply_move(Move.play_move(Point(3, 2)))
        game = game.apply_move(Move.play_move(Point(2, 4)))
        return game

    def test_ko_prevents_immediate_recapture(self):
        game = self._setup_ko()
        game = game.apply_move(Move.play_move(Point(2, 3)))
        self.assertTrue(
            game.does_move_violate_ko(game.next_player,
                                      Move.play_move(Point(2, 2))))
        self.assertFalse(game.is_valid_move(Move.play_move(Point(2, 2))))

    def test_ko_allows_recapture_after_threat(self):
        game = self._setup_ko()
        game = game.apply_move(Move.play_move(Point(2, 3)))
        game = game.apply_move(Move.play_move(Point(19, 19)))
        game = game.apply_move(Move.play_move(Point(18, 18)))
        self.assertFalse(
            game.does_move_violate_ko(game.next_player,
                                      Move.play_move(Point(2, 2))))
        self.assertTrue(game.is_valid_move(Move.play_move(Point(2, 2))))


class TestGameState(unittest.TestCase):
    def test_new_game_starts_black(self):
        game = GameState.new_game(9)
        self.assertEqual(Player.black, game.next_player)

    def test_alternates_players(self):
        game = GameState.new_game(9)
        game = game.apply_move(Move.play_move(Point(5, 5)))
        self.assertEqual(Player.white, game.next_player)
        game = game.apply_move(Move.play_move(Point(3, 3)))
        self.assertEqual(Player.black, game.next_player)

    def test_cant_play_occupied(self):
        game = GameState.new_game(9)
        game = game.apply_move(Move.play_move(Point(5, 5)))
        self.assertFalse(game.is_valid_move(Move.play_move(Point(5, 5))))

    def test_pass_is_valid(self):
        game = GameState.new_game(9)
        self.assertTrue(game.is_valid_move(Move.pass_turn()))

    def test_resign_is_valid(self):
        game = GameState.new_game(9)
        self.assertTrue(game.is_valid_move(Move.resign()))

    def test_game_over_double_pass(self):
        game = GameState.new_game(9)
        game = game.apply_move(Move.pass_turn())
        self.assertFalse(game.is_over())
        game = game.apply_move(Move.pass_turn())
        self.assertTrue(game.is_over())

    def test_game_over_resign(self):
        game = GameState.new_game(9)
        game = game.apply_move(Move.resign())
        self.assertTrue(game.is_over())

    def test_last_move(self):
        game = GameState.new_game(9)
        self.assertIsNone(game.last_move)
        m = Move.play_move(Point(5, 5))
        game = game.apply_move(m)
        self.assertEqual(m.point, game.last_move.point)


class TestZobristHash(unittest.TestCase):
    def test_empty_boards_same_hash(self):
        b1 = Board(9, 9)
        b2 = Board(9, 9)
        self.assertEqual(b1.zobrist_hash(), b2.zobrist_hash())

    def test_different_positions_different_hash(self):
        b1 = Board(9, 9)
        b1.place_stone(Player.black, Point(3, 3))
        b2 = Board(9, 9)
        b2.place_stone(Player.black, Point(5, 5))
        self.assertNotEqual(b1.zobrist_hash(), b2.zobrist_hash())

    def test_same_position_same_hash(self):
        b1 = Board(9, 9)
        b1.place_stone(Player.black, Point(3, 3))
        b1.place_stone(Player.white, Point(5, 5))
        b2 = Board(9, 9)
        b2.place_stone(Player.black, Point(3, 3))
        b2.place_stone(Player.white, Point(5, 5))
        self.assertEqual(b1.zobrist_hash(), b2.zobrist_hash())

    def test_hash_changes_on_capture(self):
        b = Board(5, 5)
        b.place_stone(Player.black, Point(1, 1))
        h1 = b.zobrist_hash()
        b.place_stone(Player.white, Point(1, 2))
        b.place_stone(Player.white, Point(2, 1))
        h2 = b.zobrist_hash()
        self.assertNotEqual(h1, h2)
        self.assertIsNone(b.get_player_on_point(Point(1, 1)))


class TestBoardCopy(unittest.TestCase):
    def test_copy_independence(self):
        b = Board(5, 5)
        b.place_stone(Player.black, Point(3, 3))
        b2 = b.copy()
        b2.place_stone(Player.white, Point(1, 1))
        self.assertIsNone(b.get_player_on_point(Point(1, 1)))
        self.assertEqual(Player.white, b2.get_player_on_point(Point(1, 1)))

    def test_copy_preserves_hash(self):
        b = Board(9, 9)
        b.place_stone(Player.black, Point(5, 5))
        b2 = b.copy()
        self.assertEqual(b.zobrist_hash(), b2.zobrist_hash())


class TestGoString(unittest.TestCase):
    def test_stones_immutable(self):
        s = GoString(Player.black, [Point(1, 1)], [Point(1, 2)])
        self.assertIsInstance(s.stones, frozenset)

    def test_without_liberty(self):
        s = GoString(Player.black, [Point(1, 1)],
                     [Point(1, 2), Point(2, 1)])
        s2 = s.without_liberty(Point(1, 2))
        self.assertEqual(2, s.num_liberties)
        self.assertEqual(1, s2.num_liberties)

    def test_with_liberty(self):
        s = GoString(Player.black, [Point(1, 1)], [Point(1, 2)])
        s2 = s.with_liberty(Point(2, 1))
        self.assertEqual(1, s.num_liberties)
        self.assertEqual(2, s2.num_liberties)

    def test_merge_with(self):
        s1 = GoString(Player.black, [Point(1, 1)],
                      [Point(1, 2), Point(2, 1)])
        s2 = GoString(Player.black, [Point(1, 2)],
                      [Point(1, 1), Point(1, 3), Point(2, 2)])
        merged = s1.merge_with(s2)
        self.assertEqual(2, len(merged.stones))
        self.assertEqual(3, merged.num_liberties)
        self.assertNotIn(Point(1, 1), merged.liberties)
        self.assertNotIn(Point(1, 2), merged.liberties)


class TestFullGame(unittest.TestCase):
    """Play through a short 5x5 game to exercise the full stack."""

    def test_5x5_game(self):
        game = GameState.new_game(5)
        moves = [
            (3, 3), (3, 2),
            (2, 3), (4, 3),
            (3, 4), (2, 2),
            (4, 4), (2, 4),
            (4, 2), (1, 3),
        ]
        for r, c in moves:
            m = Move.play_move(Point(row=r, col=c))
            self.assertTrue(game.is_valid_move(m),
                            f"Move at ({r},{c}) should be valid")
            game = game.apply_move(m)
        self.assertFalse(game.is_over())

    def test_legal_moves_count_opening(self):
        game = GameState.new_game(9)
        legal = game.legal_moves()
        self.assertEqual(81 + 2, len(legal))


class TestLargeBoard(unittest.TestCase):
    def test_19x19_center(self):
        board = Board(19, 19)
        board.place_stone(Player.black, Point(10, 10))
        s = board.get_go_string(Point(10, 10))
        self.assertEqual(4, s.num_liberties)

    def test_19x19_capture(self):
        board = Board(19, 19)
        board.place_stone(Player.black, Point(10, 10))
        place_stones(board, Player.white,
                     [(9, 10), (11, 10), (10, 9), (10, 11)])
        self.assertIsNone(board.get_player_on_point(Point(10, 10)))


if __name__ == '__main__':
    unittest.main()
