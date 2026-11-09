"""Shared GUI game loop and human move input for AlphiGo scripts."""

import threading

import pygame

from alphigo import alphigo_board, alphigo_types
from alphigo.alphigo_board import Move, Point
from alphigo.utils import print_board, print_move

from gui.gui_board import GuiBoard, GuiParams


def get_human_move(game, gui_board, on_key=None):
    """Block until the human picks a move via mouse (click=play, middle=resign, right=pass).
    Returns the Move, or None if the user closed the window.

    ``on_key`` is an optional callable that receives each pygame KEYDOWN event
    so the runner can handle global hotkeys (H/C) without losing them to the
    mouse-wait loop.
    """
    gui_quit = False
    is_valid_move = False
    move = None
    board_size = game.board.num_rows

    while not is_valid_move and not gui_quit:
        x, y = -1, -1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gui_quit = True
                break
            if event.type == pygame.KEYDOWN and on_key is not None:
                on_key(event)
                continue
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                # Use the board's actual stone size, not the class default -
                # the runner scales the board up and adds a side panel.
                ss = getattr(gui_board, 'ss', GuiParams.STONE_SIZE)
                if pos[0] >= gui_board.num_cols * ss:
                    continue  # click landed in the info panel, not the board
                x = int(round((pos[0] - int(ss / 2)) / ss))
                y = int(round((pos[1] - int(ss / 2)) / ss))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                is_valid_move = True
                move = Move(is_resign=True)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                is_valid_move = True
                move = Move(is_pass=True)

        if gui_quit:
            return None
        if x != -1 and y != -1:
            point = Point(row=board_size - y, col=x + 1)
            if game.is_valid_move(Move.play_move(point)):
                is_valid_move = True
                move = Move.play_move(point)

    return move


def _draw_thinking(gui_board, elapsed):
    """Draw a 'Thinking...' indicator on the board while the bot computes."""
    font = pygame.font.SysFont(None, max(18, int(gui_board.ss * 0.6)))
    text = f"Thinking... ({elapsed:.1f}s)"
    label = font.render(text, True, (200, 50, 50))
    rect = label.get_rect(center=(gui_board.width // 2, gui_board.height - 14))
    # Semi-transparent background for readability.
    bg = pygame.Surface((rect.width + 12, rect.height + 6), pygame.SRCALPHA)
    bg.fill((255, 255, 255, 180))
    gui_board.screen.blit(bg, (rect.x - 6, rect.y - 3))
    gui_board.screen.blit(label, rect)
    pygame.display.flip()


def _print_search_info(bot, move, game):
    """Print search statistics after the bot picks a move.

    Extracts win-rate from the bot's last search result (stored on the
    agent as ``_last_search``) so we avoid running a second expensive
    MCTS search just for the display.

    Returns an AnalysisResult (or None) for the title-bar display.
    """
    from alphigo.agent.analysis import AnalysisResult

    # Pull last search root from the agent if available (ZeroAgent /
    # GumbelZeroAgent store it after search()).
    root = getattr(bot, '_last_root', None)
    if root is None:
        return None

    total_visits = sum(c.N for c in root.children) if root.children else 0
    winrate = (float(root.Q) + 1.0) / 2.0
    winrate = max(0.0, min(1.0, winrate))

    # Build top-K from visit counts (no extra search needed).
    ranked = sorted(root.children, key=lambda c: c.N, reverse=True)
    top = []
    for child in ranked[:5]:
        mv = child.move
        if mv.is_pass or mv.is_resign or mv.point is None:
            continue
        frac = float(child.N) / total_visits * 100.0 if total_visits else 0
        top.append((mv.point, frac))

    # Print to console.
    move_winrates = []
    parts = [f"WR {winrate * 100:.0f}%"]
    if top:
        top_strs = []
        for i, (pt, score) in enumerate(top[:5], 1):
            col_letter = chr(ord('A') + pt.col - 1)
            if col_letter >= 'I':
                col_letter = chr(ord(col_letter) + 1)  # skip 'I' in Go
            child_q = ranked[i - 1].Q if i - 1 < len(ranked) else 0
            # A child's Q belongs to the player to move *at the child* - the
            # opponent - so flip it to keep every win rate on screen on the
            # same side as the root WR printed above.
            child_wr = 1.0 - (float(child_q) + 1.0) / 2.0
            child_wr = max(0.0, min(1.0, child_wr))
            move_winrates.append(child_wr)
            top_strs.append(
                f"  {i}. {col_letter}{pt.row}  visits {score:.0f}%  "
                f"WR {child_wr * 100:.0f}%")
        parts.append("Top moves:\n" + "\n".join(top_strs))
    print("  " + "\n  ".join(parts))

    ownership, score_estimate = None, None
    if hasattr(bot, 'forward_heads'):
        try:
            ownership, score_estimate = bot.forward_heads(game)
        except Exception as exc:
            print(f"  [heads] unavailable: {exc}")

    # `game` is the position the bot just searched, i.e. before its move
    # was applied - so the side to move here is who all these numbers are
    # about, even though the caller renders them one ply later.
    return AnalysisResult(top_moves=top, position_winrate=winrate,
                          move_winrates=move_winrates,
                          ownership=ownership,
                          score_estimate=score_estimate,
                          perspective=game.next_player)


def get_bot_move_async(bot, game, _gui_board, on_key=None):
    """Get move from a bot in a background thread so the GUI stays responsive (e.g. for MCST).
    Returns the Move, or pass if the window was closed before the bot finished."""
    import time
    result = []

    def task():
        result.append(bot.select_move(game))

    thread = threading.Thread(target=task, daemon=True)
    thread.start()
    gui_quit = False
    t0 = time.time()
    while thread.is_alive() and not gui_quit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gui_quit = True
            elif event.type == pygame.KEYDOWN and on_key is not None:
                on_key(event)
        # Show thinking indicator with elapsed time.
        if _gui_board is not None:
            _draw_thinking(_gui_board, time.time() - t0)
        pygame.event.pump()
        thread.join(timeout=0.05)

    move = result[0] if result else Move.pass_turn()
    # Print search info to console.
    if result and not gui_quit:
        _print_search_info(bot, move, game)
    return move


# ── Info panel ────────────────────────────────────────────────────────

PANEL_BG = (245, 242, 236)
PANEL_FG = (25, 25, 25)
PANEL_MUTED = (115, 112, 105)
PANEL_RULE = (205, 200, 190)
BAR_BLACK = (35, 35, 35)
BAR_WHITE = (235, 232, 226)

CONTROLS = [
    ("left click", "play a stone"),
    ("right click", "PASS"),
    ("middle click", "resign"),
    ("H", "move hints on/off"),
    ("T", "territory view on/off"),
    ("C", "compare two agents"),
]


def _col_letter(col):
    """Board column index (1-based) -> letter, skipping 'I' as Go does."""
    letter = chr(ord('A') + col - 1)
    return chr(ord(letter) + 1) if letter >= 'I' else letter


def _draw_panel(gui_board, game, analysis, state, agent_name):
    """Draw the right-hand information panel."""
    if not gui_board.panel_width:
        return
    screen = gui_board.screen
    x0 = gui_board.width
    w = gui_board.panel_width
    pygame.draw.rect(screen, PANEL_BG, pygame.Rect(x0, 0, w, gui_board.height))
    pygame.draw.line(screen, PANEL_RULE, (x0, 0), (x0, gui_board.height))

    pad = 14
    x = x0 + pad
    y = pad
    f_head = pygame.font.SysFont(None, 26)
    f_body = pygame.font.SysFont(None, 21)
    f_small = pygame.font.SysFont(None, 18)

    def line(text, font=f_body, color=PANEL_FG, dy=None):
        nonlocal y
        screen.blit(font.render(text, True, color), (x, y))
        y += dy if dy is not None else font.get_height() + 2

    def rule(gap=8):
        nonlocal y
        y += gap
        pygame.draw.line(screen, PANEL_RULE, (x, y), (x0 + w - pad, y))
        y += gap

    to_play = game.next_player.name.capitalize()
    line(f"{to_play} to play", f_head)
    line(f"move {state['move_num']}", f_small, PANEL_MUTED)
    if agent_name:
        line(f"AlphiGo plays {agent_name}", f_small, PANEL_MUTED)
    rule()

    # Every number below belongs to whoever was to move at the position the
    # search actually analysed - normally the previous ply, not `game`.
    if analysis is not None and analysis.perspective is not None:
        subject = analysis.perspective.name.capitalize()
    else:
        subject = to_play
    other = "White" if subject == "Black" else "Black"

    # ── win rate ──────────────────────────────────────────────────
    if analysis is not None and analysis.position_winrate is not None:
        wr = analysis.position_winrate
        line(f"Win rate for {subject}", f_small, PANEL_MUTED)
        bar_w = w - 2 * pad
        bar_h = 18
        bar = pygame.Rect(x, y, bar_w, bar_h)
        pygame.draw.rect(screen, BAR_WHITE, bar)
        pygame.draw.rect(screen, BAR_BLACK,
                         pygame.Rect(x, y, int(bar_w * wr), bar_h))
        pygame.draw.rect(screen, PANEL_RULE, bar, 1)
        label = f_body.render(f"{wr * 100:.0f}%", True, PANEL_FG)
        screen.blit(label, (x + bar_w - label.get_width() - 4,
                            y + bar_h + 3))
        y += bar_h + label.get_height() + 8

        if analysis.score_estimate is not None:
            margin = analysis.score_estimate
            leader = subject if margin >= 0 else other
            line(f"Score: {leader} +{abs(margin):.1f}", f_body)
    else:
        line("Win rate: press H", f_small, PANEL_MUTED)
    rule()

    # ── top moves ─────────────────────────────────────────────────
    line("AlphiGo's candidates", f_small, PANEL_MUTED)
    if analysis is not None and analysis.top_moves:
        wrs = analysis.move_winrates or []
        for i, (pt, visits) in enumerate(analysis.top_moves[:5]):
            name = f"{_col_letter(pt.col)}{pt.row}"
            wr_txt = f"{wrs[i] * 100:4.0f}%" if i < len(wrs) else "   -"
            line(f" {i + 1}. {name:<4} {visits:4.0f}% {wr_txt}", f_body)
        line("    move  visits  win", f_small, PANEL_MUTED)
    else:
        line(" press H to show", f_small, PANEL_MUTED)
    rule()

    # ── territory legend ──────────────────────────────────────────
    if state.get('territory_on'):
        from gui import overlay as _ov
        line("Predicted territory", f_small, PANEL_MUTED)
        sw = 13
        for color, who in ((_ov.OWN_BLACK, "Black"), (_ov.OWN_WHITE, "White")):
            pygame.draw.rect(screen, color, pygame.Rect(x, y + 2, sw, sw))
            screen.blit(f_small.render(f"  {who}", True, PANEL_FG),
                        (x + sw + 4, y))
            y += sw + 4
        y += 4

    # ── controls, pinned to the bottom ────────────────────────────
    ctrl_h = (len(CONTROLS) + 1) * (f_small.get_height() + 3) + 10
    y = max(y, gui_board.height - ctrl_h - pad)
    pygame.draw.line(screen, PANEL_RULE, (x, y), (x0 + w - pad, y))
    y += 8
    line("Controls", f_small, PANEL_MUTED)
    for key, what in CONTROLS:
        screen.blit(f_small.render(key, True, PANEL_FG), (x, y))
        screen.blit(f_small.render(what, True, PANEL_MUTED), (x + 96, y))
        y += f_small.get_height() + 3


def _draw_game_over(gui_board, game, winner, state, agent_name):
    """Draw the final position with a result banner, and wait for input."""
    from alphigo.alphigo_board import GameState  # noqa: F401
    _render(gui_board, game, state.get('overlay_on', False),
            state.get('analysis'), {}, state=state, agent_name=agent_name)

    side = winner.name.capitalize()
    try:
        black_pts, white_pts = _final_points(game)
        # komi is 7.5, so scores are fractional - do not round them away
        detail = f"Black {black_pts:g}  -  White {white_pts:g}  (komi included)"
    except Exception:
        detail = ""

    w, h = gui_board.width, gui_board.height
    band_h = 96
    band = pygame.Surface((w, band_h), pygame.SRCALPHA)
    band.fill((250, 250, 248, 238))
    gui_board.screen.blit(band, (0, h // 2 - band_h // 2))
    pygame.draw.rect(gui_board.screen, (60, 60, 60),
                     pygame.Rect(0, h // 2 - band_h // 2, w, band_h), 2)

    f_big = pygame.font.SysFont(None, 44)
    f_small = pygame.font.SysFont(None, 22)
    title = f_big.render(f"{side} wins", True, (20, 20, 20))
    gui_board.screen.blit(
        title, title.get_rect(center=(w // 2, h // 2 - 20)))
    if detail:
        sub = f_small.render(detail, True, (70, 70, 70))
        gui_board.screen.blit(sub, sub.get_rect(center=(w // 2, h // 2 + 10)))
    hint = f_small.render("click or press any key to close", True, (120, 120, 120))
    gui_board.screen.blit(hint, hint.get_rect(center=(w // 2, h // 2 + 34)))
    pygame.display.flip()


def _final_points(game):
    """Area score for both sides (Chinese rules), for the result banner.

    Scores the board as-is: no dead-stone detection, so the figure can
    differ from a GTP engine's official count when dead stones remain.
    """
    from alphigo.alphigo_board import area_score
    return area_score(game.board)


def _format_title(side_to_play, overlay_on, analysis, move_num=0):
    base = "AlphiGo"
    bits = [base]
    if move_num > 0:
        bits.append(f"move {move_num}")
    bits.append(f"to play: {side_to_play.name}")
    if analysis is not None and analysis.position_winrate is not None:
        wr = analysis.position_winrate * 100
        # Name the side the win rate belongs to: it is the side that moved
        # last, not the side to play now.
        subject = (analysis.perspective.name
                   if analysis.perspective is not None else side_to_play.name)
        bits.append(f"WR {wr:.0f}% ({subject})")
    if overlay_on and analysis is not None and analysis.pass_score is not None:
        bits.append(f"PASS {analysis.pass_score:.0f}%")
    return "  ".join(bits)


def _render(gui_board, game, overlay_on, analysis,
            compare_state, state=None, agent_name=None):
    """Draw board + (optional) territory, overlay or compare result + panel."""
    from gui import overlay as _overlay
    gui_board.draw(game_state=game, flip=False)
    if state is not None and state.get('territory_on'):
        # The ownership map is signed relative to whoever was to move at the
        # analysed position, which is usually the previous ply - using
        # game.next_player here inverts every colour.
        side = game.next_player
        if analysis is not None and analysis.perspective is not None:
            side = analysis.perspective
        _overlay.draw_territory(gui_board, analysis, side, game_state=game)
    if compare_state and compare_state.get('active'):
        _overlay.draw_compare(gui_board,
                              compare_state.get('result_a'),
                              compare_state.get('result_b'))
    elif overlay_on and analysis is not None:
        _overlay.draw_overlay(gui_board, analysis, flip=False)
    if state is not None:
        _draw_panel(gui_board, game, analysis, state, agent_name)
    pygame.display.flip()


def _enter_compare_mode(game, gui_board):
    """Terminal picker → returns (label_a, agent_a, label_b, agent_b) or None."""
    from gui.discovery import list_all_for_picker
    from alphigo.agent.registry import create_agent

    # Flush one frame and pump events so the pygame window doesn't go
    # stale (black on X11/WSL) while input() blocks the main thread.
    pygame.display.flip()
    pygame.event.pump()

    items = list_all_for_picker()
    if len(items) < 2:
        print("\n[compare] Need at least 2 agents to compare. "
              "Found:", len(items))
        return None
    print("\n=== Compare mode: pick two agents ===")
    for i, (label, _spec, _meta) in enumerate(items):
        print(f"  {i:2d}  {label}")
    print("(pygame window will not refresh until you answer)")
    try:
        raw = input("indices (e.g. '0 1', or empty to cancel): ").strip().split()
        if not raw:
            print("[compare] cancelled")
            return None
        i_a, i_b = int(raw[0]), int(raw[1])
    except (ValueError, IndexError, EOFError):
        print("[compare] invalid input, aborting")
        return None
    if not (0 <= i_a < len(items)) or not (0 <= i_b < len(items)):
        print("[compare] indices out of range, aborting")
        return None

    def _build(spec):
        if ':' in spec:
            name, kvs = spec.split(':', 1)
            kwargs = dict(kv.split('=', 1) for kv in kvs.split(','))
        else:
            name, kwargs = spec, {}
        return create_agent(name, **kwargs)

    label_a, spec_a, _ = items[i_a]
    label_b, spec_b, _ = items[i_b]
    try:
        agent_a = _build(spec_a)
        agent_b = _build(spec_b)
    except Exception as exc:
        print(f"[compare] failed to load agents: {exc}")
        return None

    # Board-size sanity check (only if encoders are exposed).
    bs = game.board.num_rows
    for label, ag in ((label_a, agent_a), (label_b, agent_b)):
        enc = getattr(ag, 'encoder', None) or getattr(ag, '_encoder', None)
        if enc is not None and enc.board_width != bs:
            print(f"[compare] board-size mismatch: {label} expects "
                  f"{enc.board_width}x{enc.board_height}, current game is "
                  f"{bs}x{bs}; aborting compare.")
            return None
    return label_a, agent_a, label_b, agent_b


def run_gui_game(
    board_size,
    get_black_move,
    get_white_move,
    *,
    clear_console_each_turn=False,
    delay_after_draw=0,
    black_agent=None,
    white_agent=None,
    show_panel=True,
    stone_size=None,
    panel_width=None,
    show_result=True,
):
    """Run one GUI game until it ends or the user quits.

    get_black_move and get_white_move are callables (game, gui_board) -> Move | None.
    Return None from a move getter to quit.

    Optional ``black_agent`` / ``white_agent`` Agent instances enable the
    H/C overlay keys: H toggles a top-K hint overlay for the side-to-move
    bot, C enters compare mode (terminal picker for two arbitrary agents).
    Backwards-compatible: when both agents are None the overlay is off
    and no extra work runs.

    Returns the winner (Player) or None if the user quit.
    """
    import os

    game = alphigo_board.GameState.new_game(board_size)
    # A bigger board and a side panel: the defaults (30 px stones, no panel)
    # are unreadable on a projector. Callers can still override both.
    if stone_size is None:
        stone_size = 62 if board_size <= 9 else (44 if board_size <= 13 else 32)
    if panel_width is None:
        panel_width = 250 if show_panel else 0
    gui_board = GuiBoard(board_size=board_size, stone_size=stone_size,
                         panel_width=panel_width)
    gui_quit = False

    # state['analysis'] is recomputed once per turn (in the main loop, never
    # from on_key) so we never run analyze() concurrently with the bot's
    # select_move() on the same TF model.
    state = {'overlay_on': False, 'analysis': None, 'move_num': 0,
             'territory_on': False}
    compare_state = {}

    def _side_agent():
        return (black_agent if game.next_player == alphigo_types.Player.black
                else white_agent)

    def _agent_side_name():
        if black_agent is not None and white_agent is None:
            return "Black"
        if white_agent is not None and black_agent is None:
            return "White"
        return None

    def _redraw():
        _render(gui_board, game, state['overlay_on'],
                state['analysis'], compare_state, state=state,
                agent_name=_agent_side_name())
        pygame.display.set_caption(_format_title(
            game.next_player, state['overlay_on'], state['analysis'],
            move_num=state['move_num']))

    def _recompute_analysis():
        ag = _side_agent()
        if state['overlay_on'] and ag is not None:
            try:
                state['analysis'] = ag.analyze(game)
            except Exception as exc:
                print(f"[overlay] analyze failed: {exc}")
                state['analysis'] = None
        elif not state['overlay_on']:
            # Keep last analysis from bot move for title bar WR, but
            # don't run a new expensive search just for the overlay.
            pass

    def on_key(event):
        if event.key == pygame.K_t:
            state['territory_on'] = not state['territory_on']
            _redraw()
            return
        if event.key == pygame.K_h:
            state['overlay_on'] = not state['overlay_on']
            # Don't call analyze() here — it could collide with the bot's
            # concurrent select_move().  Use whatever analysis was last
            # computed at the top of this turn.  If None, just show the
            # plain board with overlay flag flipped.
            _redraw()
        elif event.key == pygame.K_c:
            if compare_state.get('active'):
                compare_state.clear()
                print("[compare] exited compare mode.")
            else:
                picked = _enter_compare_mode(game, gui_board)
                if picked is not None:
                    la, aa, lb, ab = picked
                    print(f"[compare] analysing current position "
                          f"(side to play: {game.next_player.name}): "
                          f"{la} vs {lb}…")
                    compare_state['active'] = True
                    compare_state['result_a'] = aa.analyze(game)
                    compare_state['result_b'] = ab.analyze(game)
            # Always force a clean redraw + event pump after returning
            # from input(): the pygame window may have been black while
            # the terminal had focus.
            _redraw()
            pygame.event.pump()

    while not game.is_over() and not gui_quit:
        if delay_after_draw:
            import time
            time.sleep(delay_after_draw)

        _recompute_analysis()
        _redraw()

        if gui_quit:
            break
        if clear_console_each_turn:
            os.system("cls" if os.name == "nt" else "clear")
        print_board(game.board)

        # If we're in compare mode (paused), block here in a quiet
        # wait-loop — don't return to the top of the outer loop where
        # _recompute_analysis / print_board would spin and cause blinking.
        if compare_state.get('active'):
            pygame.display.set_caption(
                "AlphiGo — compare mode (press C to exit)")
            clock = pygame.time.Clock()
            while compare_state.get('active') and not gui_quit:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        gui_quit = True
                    elif event.type == pygame.KEYDOWN:
                        on_key(event)
                pygame.event.pump()
                clock.tick(30)
            if gui_quit:
                break
            # On exit, redraw plain board (no compare overlay) before
            # falling through to the move getter.
            _redraw()

        try:
            if game.next_player == alphigo_types.Player.black:
                move = get_black_move(game, gui_board, on_key=on_key)
            else:
                move = get_white_move(game, gui_board, on_key=on_key)
        except TypeError:
            # Backwards-compat: callers that didn't expect on_key.
            if game.next_player == alphigo_types.Player.black:
                move = get_black_move(game, gui_board)
            else:
                move = get_white_move(game, gui_board)

        if move is None:
            gui_quit = True
            break
        # Update analysis from last bot search (free, no extra compute).
        current_agent = _side_agent()
        if current_agent is not None:
            last_analysis = _print_search_info(current_agent, move, game)
            if last_analysis is not None:
                state['analysis'] = last_analysis
        print_move(game.next_player, move)
        state['move_num'] += 1
        game = game.apply_move(move)
        _render(gui_board, game, state['overlay_on'],
                state['analysis'], compare_state, state=state,
                agent_name=_agent_side_name())
        pygame.display.set_caption(_format_title(
            game.next_player, state['overlay_on'], state['analysis'],
            move_num=state['move_num']))
        pygame.event.pump()

    winner = game.winner() if not gui_quit else None

    # Hold the final position on screen. Two passes end a game instantly,
    # so without this the window just vanishes the moment someone passes.
    if winner is not None and not gui_quit and show_result:
        _draw_game_over(gui_board, game, winner, state, _agent_side_name())
        clock = pygame.time.Clock()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN,
                                  pygame.MOUSEBUTTONDOWN):
                    waiting = False
            pygame.event.pump()
            clock.tick(30)

    pygame.quit()
    return winner
