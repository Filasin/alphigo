"""GTP agent wrapper: play against any GTP-compatible engine.

Wraps an external program (KataGo, GNU Go, Pachi, etc.) as an alphigo Agent
so it can be used with the benchmark and play systems.

Usage via CLI:
    python run.py benchmark --black mcts:num_rounds=500 --white katago:model=weights.bin.gz -n 10
    python run.py benchmark --black dl:agent_file=bot.h5 --white gnugo -n 5

Usage programmatically:
    agent = GtpAgent(['katago', 'gtp', '-model', 'weights.bin.gz',
                       '-override-config', 'maxVisits=100'])
"""

import atexit
import logging
import platform
import shutil
import subprocess
from pathlib import Path

from alphigo.agent.base import Agent
from alphigo.alphigo_board import Move
from alphigo.alphigo_types import Player, Point

logger = logging.getLogger(__name__)

# GTP standard: columns A-T, skipping I to avoid confusion with J
GTP_COLS = 'ABCDEFGHJKLMNOPQRST'

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GTP_AGENTS_DIR = _PROJECT_ROOT / 'gtp_agents'


def _point_to_gtp(point):
    return GTP_COLS[point.col - 1] + str(point.row)


def _gtp_to_point(gtp_coord):
    col = GTP_COLS.index(gtp_coord[0].upper()) + 1
    row = int(gtp_coord[1:])
    return Point(row=row, col=col)


def _player_to_gtp(player):
    return 'black' if player == Player.black else 'white'


_KNOWN_LOCATIONS = {
    'gnugo': ['gnugo-3.8/gnugo-3.8/gnugo.exe',
              'gnugo-3.8/gnugo-3.8/gnugo',
              'gnugo/gnugo.exe',
              'gnugo/gnugo'],
    'pachi': ['pachi-12.88-win64-avx/Pachi-12.88/pachi.exe',
              'pachi-12.88-linux-amd64-avx/Pachi-12.88/pachi',
              'pachi/pachi.exe',
              'pachi/pachi'],
    'katago': ['katago/katago.exe',
               'katago/katago'],
    'fuego':  ['fuego/fuego.exe',
               'fuego/fuego'],
    'leela':  ['leela/leela.exe',
               'leela/leela'],
}


def find_binary(name):
    """Locate a GTP engine binary: first in gtp_agents/, then on PATH."""
    # Check gtp_agents/ directory for known locations
    for rel in _KNOWN_LOCATIONS.get(name, []):
        candidate = GTP_AGENTS_DIR / rel
        if candidate.is_file():
            logger.info("Found %s at %s", name, candidate)
            return str(candidate)

    # Fallback: search PATH
    path = shutil.which(name)
    if path is not None:
        return path

    os_name = platform.system()
    hint = f"Place it in {GTP_AGENTS_DIR} or add it to your PATH."
    if name == 'katago':
        hint = (f"Download from https://github.com/lightvector/KataGo/releases "
                f"and place it in {GTP_AGENTS_DIR} or add it to your PATH.")
    elif name == 'gnugo' and os_name == 'Linux':
        hint = f"Install with: sudo apt install gnugo"
    raise FileNotFoundError(f"Could not find '{name}'. {hint}")


class GtpAgent(Agent):
    """Wraps any GTP-speaking program as an alphigo Agent.

    The external process is kept alive across games (for benchmark reuse)
    and cleaned up automatically on interpreter exit.
    """

    def __init__(self, cmd, cwd=None):
        super().__init__()
        self._cmd = cmd
        self._cwd = cwd
        self._cmd_display = ' '.join(cmd)
        self._board_size = None
        self._synced_moves = 0
        self._process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self.close)

    def restart(self):
        """Kill the current process and spawn a fresh one."""
        self.close()
        self._board_size = None
        self._synced_moves = 0
        self._process = subprocess.Popen(
            self._cmd,
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _send(self, text):
        self._process.stdin.write((text + '\n').encode('utf-8'))
        self._process.stdin.flush()

    def _read_response(self):
        while True:
            raw = self._process.stdout.readline()
            if not raw:
                raise RuntimeError(
                    f'GTP process exited unexpectedly ({self._cmd_display})')
            line = raw.decode('utf-8').strip()
            if line.startswith('='):
                return line[1:].strip()
            if line.startswith('?'):
                raise RuntimeError(f'GTP error: {line[1:].strip()}')

    def _command(self, cmd):
        self._send(cmd)
        return self._read_response()

    def _ensure_board(self, board_size):
        """Send boardsize + clear_board when the size changes or on first use."""
        if board_size != self._board_size:
            self._command(f'boardsize {board_size}')
            self._command('clear_board')
            self._command('komi 7.5')
            self._board_size = board_size
            self._synced_moves = 0

    def select_move(self, game_state):
        board_size = game_state.board.num_rows
        self._ensure_board(board_size)

        history = game_state.move_history()

        # Detect game reset (new game with fewer moves than we've synced)
        if len(history) < self._synced_moves:
            self._command('clear_board')
            self._synced_moves = 0

        # Send any moves the engine hasn't seen yet
        for player, move in history[self._synced_moves:]:
            color = _player_to_gtp(player)
            if move.is_pass:
                self._command(f'play {color} pass')
            elif move.is_resign:
                self._command(f'play {color} resign')
            else:
                self._command(f'play {color} {_point_to_gtp(move.point)}')
        self._synced_moves = len(history)

        # Ask the engine to generate a move
        color = _player_to_gtp(game_state.next_player)
        response = self._command(f'genmove {color}')
        self._synced_moves += 1

        if response.lower() == 'pass':
            return Move.pass_turn()
        if response.lower() == 'resign':
            return Move.resign()
        return Move.play_move(_gtp_to_point(response))

    def final_score(self):
        """Ask the GTP engine for the final score (e.g. 'W+10.5' or 'B+3.5').
        The engine performs its own life/death reading, so dead stones are handled."""
        return self._command('final_score')

    def close(self):
        if self._process is not None and self._process.poll() is None:
            try:
                self._send('quit')
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
                self._process.wait()
        self._process = None

    def __del__(self):
        self.close()
