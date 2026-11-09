"""Pygame overlay rendering for the Agent.analyze top-K + compare modes."""

import pygame

from gui.gui_board import GuiParams


# Default tints (RGBA) for live overlay and compare mode.
TINT_LIVE = (80, 200, 110, 210)        # green — side-to-move
                                       # (red is territory now)
TINT_COMPARE_A = (60, 130, 220, 220)   # blue
TINT_COMPARE_B = (250, 170, 60, 220)   # amber


def _point_to_pixel(point, num_rows, ss):
    """Game Point (row 1..N, col 1..N) -> pygame (cx, cy) center."""
    cx = int((point.col - 1) * ss + ss / 2)
    cy = int((num_rows - point.row) * ss + ss / 2)
    return cx, cy


def _draw_marker(surface, cx, cy, ss, rank, score, tint, ring_offset=0):
    """Draw a small filled circle with rank number + score% label."""
    radius = int(ss / 2 - 4) + ring_offset
    # Filled translucent circle.
    s = pygame.Surface((ss, ss), pygame.SRCALPHA)
    pygame.draw.circle(s, tint, (int(ss / 2), int(ss / 2)), radius)
    surface.blit(s, (cx - int(ss / 2), cy - int(ss / 2)))
    # Outline + rank text.
    pygame.draw.circle(surface, GuiParams.BLACK, (cx, cy), radius, 1)

    font = pygame.font.SysFont(None, max(12, int(ss * 0.5)))
    label = font.render(str(rank), True, GuiParams.BLACK)
    rect = label.get_rect(center=(cx, cy))
    surface.blit(label, rect)

    # Score percentage just below the marker - or above it near the bottom
    # edge, so the label never falls off the board.
    pct_font = pygame.font.SysFont(None, max(11, int(ss * 0.35)))
    pct_label = pct_font.render(f"{score:.0f}%", True, GuiParams.BLACK)
    offset = int(ss * 0.55)
    if cy + offset + pct_label.get_height() > surface.get_height():
        offset = -offset
    pct_rect = pct_label.get_rect(center=(cx, cy + offset))
    surface.blit(pct_label, pct_rect)


def draw_overlay(gui_board, result, tint=TINT_LIVE, flip=True):
    """Draw a single AnalysisResult on top of the rendered board."""
    if result is None or not result.top_moves:
        return
    ss = gui_board.ss
    for rank, (point, score) in enumerate(result.top_moves, start=1):
        cx, cy = _point_to_pixel(point, gui_board.num_rows, ss)
        _draw_marker(gui_board.screen, cx, cy, ss, rank, score, tint)
    if flip:
        pygame.display.flip()


def draw_compare(gui_board, result_a, result_b,
                 tints=(TINT_COMPARE_A, TINT_COMPARE_B)):
    """Draw two AnalysisResults; concentric rings on same-point overlap."""
    if result_a is None and result_b is None:
        return
    ss = gui_board.ss
    points_a = {p: (i, s) for i, (p, s) in
                enumerate(result_a.top_moves if result_a else [], start=1)}
    points_b = {p: (i, s) for i, (p, s) in
                enumerate(result_b.top_moves if result_b else [], start=1)}

    overlap = set(points_a) & set(points_b)
    only_a = set(points_a) - overlap
    only_b = set(points_b) - overlap

    for point in only_a:
        rank, score = points_a[point]
        cx, cy = _point_to_pixel(point, gui_board.num_rows, ss)
        _draw_marker(gui_board.screen, cx, cy, ss, rank, score, tints[0])
    for point in only_b:
        rank, score = points_b[point]
        cx, cy = _point_to_pixel(point, gui_board.num_rows, ss)
        _draw_marker(gui_board.screen, cx, cy, ss, rank, score, tints[1])
    for point in overlap:
        ra, sa = points_a[point]
        rb, sb = points_b[point]
        cx, cy = _point_to_pixel(point, gui_board.num_rows, ss)
        # Inner ring = A, outer ring = B (same point: stack labels).
        _draw_marker(gui_board.screen, cx, cy, ss, ra, sa, tints[0],
                     ring_offset=-3)
        _draw_marker(gui_board.screen, cx, cy, ss, rb, sb, tints[1],
                     ring_offset=+3)
    pygame.display.flip()


# Territory / ownership view ------------------------------------------------

OWN_BLACK = (0, 70, 235)      # blue  - predicted Black territory
OWN_WHITE = (215, 45, 45)     # red   - predicted White territory


def draw_territory(gui_board, result, side_to_move, game_state=None,
                   flip=False):
    """Shade each intersection by the network's ownership prediction.

    ``result.ownership`` is from the perspective of the player to move, so
    it is converted to absolute colours here: blue where Black is predicted
    to own the point, red where White is. Opacity tracks confidence, so
    contested areas stay almost clear.

    Empty points get a translucent square. Occupied points get a small
    solid dot instead, so the stone stays readable - and a stone carrying
    the opposite colour's dot is one the network expects to die.
    """
    from alphigo.alphigo_types import Player, Point

    if result is None or result.ownership is None:
        return
    own = result.ownership
    n = gui_board.num_rows
    if len(own) < n * n:
        return

    ss = gui_board.ss
    sign = 1.0 if side_to_move == Player.black else -1.0
    layer = pygame.Surface((n * ss, n * ss), pygame.SRCALPHA)

    board = game_state.board if game_state is not None else None

    for row in range(n):
        for col in range(n):
            # ownership is stored row-major from row 1 upward
            v = float(own[row * n + col]) * sign   # +1 = Black owns
            strength = min(1.0, abs(v))
            if strength < 0.06:
                continue
            color = OWN_BLACK if v > 0 else OWN_WHITE
            cx = int(col * ss + ss / 2)
            cy = int((n - 1 - row) * ss + ss / 2)

            occupied = (board is not None and board.get_player_on_point(
                Point(row=row + 1, col=col + 1)) is not None)
            if occupied:
                # Solid dot centred on the stone: visible on black and white.
                radius = max(2, int(ss * 0.11 * (0.5 + strength)))
                pygame.draw.circle(layer, color + (255,), (cx, cy), radius)
            else:
                size = int(ss * 0.58)
                rect = pygame.Rect(0, 0, size, size)
                rect.center = (cx, cy)
                # Blue needs more opacity than red to register against the
                # warm board colour.
                peak = 150 if v > 0 else 115
                pygame.draw.rect(layer, color + (int(strength * peak),),
                                 rect, border_radius=3)

    gui_board.screen.blit(layer, (0, 0))
    if flip:
        pygame.display.flip()
