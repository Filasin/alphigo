import sys

import pygame

from alphigo.alphigo_types import Point, Player


class GuiParams:
    STONE_SIZE = 30
    WHITE = (255, 255, 255)
    WHITE_STONE = (248, 248, 248)  # slight gray so it's visible on cream and has depth
    BLACK = (0, 0, 0)
    BOARD_COLOR = (220, 179, 92)  # light wood/cream so white stones are visible


class GuiBoard:

    def __init__(self, board_size, stone_size=None, panel_width=0):
        pygame.init()
        self.num_rows = board_size
        self.num_cols = board_size
        self.ss = stone_size or GuiParams.STONE_SIZE
        self.height = board_size * self.ss
        self.width = board_size * self.ss
        self.panel_width = panel_width
        self.screen = pygame.display.set_mode(
            (self.width + panel_width, self.height))
        pygame.display.set_caption("AlphiGo")

    def draw(self, game_state, flip=True):
        """Draw the current game state. After applying a move (especially bot moves),
        call draw() then pygame.event.pump() so the updated board is visible on screen."""
        # board background (cream so white stones show up)
        self.screen.fill(GuiParams.BOARD_COLOR,
                         pygame.Rect(0, 0, self.width, self.height))

        # drawing lines (y flipped so row 9 at top, matching terminal)
        for row in range(self.num_rows):
            cy = int((self.num_rows - 1 - row) * self.ss + self.ss / 2)
            pygame.draw.line(self.screen, GuiParams.BLACK,
                             (int(self.ss / 2), cy),
                             (int(self.num_cols * self.ss - self.ss / 2), cy),
                             )
        for col in range(self.num_cols):
            pygame.draw.line(self.screen, GuiParams.BLACK,
                             (int(col * self.ss + self.ss / 2), int(self.ss / 2)),
                             (int(col * self.ss + self.ss / 2), int(self.num_rows * self.ss - self.ss / 2)),
                             )

        # draw stars on board

        stars_pos_large = [[3, 3], [3, 9], [3, 15],
                           [9, 3], [9, 9], [9, 15],
                           [15, 3], [15, 9], [15, 15]]
        stars_pos_medium = [[3, 3], [3, 9],
                            [6, 6],
                            [9, 3], [9, 9]]
        stars_pos_small = [[2, 2], [2, 6],
                           [4, 4],
                           [6, 2], [6, 6]]

        stars_pos = []
        if self.num_rows == 19:
            stars_pos = stars_pos_large
        elif self.num_rows == 13:
            stars_pos = stars_pos_medium
        elif self.num_rows == 9:
            stars_pos = stars_pos_small

        for pos in stars_pos:
            # pos is [col, row] 0-indexed; flip y so row 9 at top
            cx = int(pos[0] * self.ss + self.ss / 2)
            cy = int((self.num_rows - 1 - pos[1]) * self.ss + self.ss / 2)
            pygame.draw.circle(self.screen, GuiParams.BLACK, (cx, cy),
                               max(3, int(self.ss * 0.11)))
        # drawing stones (x = col, y = row; flip y so row 9 at top, matching terminal)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                color = game_state.board.get_player_on_point(Point(row=row + 1, col=col + 1))
                if color is None:
                    continue
                cx = int(col * self.ss + self.ss / 2)
                cy = int((self.num_rows - 1 - row) * self.ss + self.ss / 2)
                if color == Player.white:
                    r_outer = int(self.ss / 2 - 2)
                    r_inner = int(self.ss / 2 - 4)
                    pygame.draw.circle(self.screen, GuiParams.BLACK, (cx, cy), r_outer, 2)
                    pygame.draw.circle(self.screen, GuiParams.WHITE_STONE, (cx, cy), r_inner)
                elif color == Player.black:
                    pygame.draw.circle(self.screen, GuiParams.BLACK, (cx, cy),
                                       int(self.ss / 2 - 3))
        if flip:
            pygame.display.flip()
