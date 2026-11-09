"""Zobrist hashing for superko detection.

HASH_CODE maps (Point, Player) → random 63-bit integer.
EMPTY_BOARD is the hash of an empty board.

The seed 2016 is fixed so hash values are identical across runs and
environments. Do not change the seed — it would invalidate any
checkpoints or cached data that embeds board hashes.
"""
import numpy as np

from alphigo.alphigo_types import Player, Point

_rng = np.random.RandomState(2016)

HASH_CODE: dict = {
    (Point(row, col), color): int(_rng.randint(0, 2**63, dtype=np.int64))
    for row in range(1, 20)
    for col in range(1, 20)
    for color in (Player.black, Player.white)
}

EMPTY_BOARD: int = int(_rng.randint(0, 2**63, dtype=np.int64))
