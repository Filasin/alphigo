"""Run identity — the 5-tuple (method, encoder, network, lr, board_size).

Per CONTEXT.md, a Run is uniquely identified by these five axes and lives at
runs/{method}_{encoder}_{network}_lr{lr}_b{board_size}/. This module owns
slug formatting, path resolution, and on-disk completion checks.
"""
from dataclasses import dataclass
from pathlib import Path

RUNS_ROOT = Path('runs')


def _format_lr(lr: float) -> str:
    """Format a learning rate as '1e-4', '2.5e-4', '1e-1', etc.

    Always uses scientific notation, drops a trailing '.0' mantissa.
    """
    s = f'{lr:.1e}'                  # e.g. '1.0e-04'
    mantissa, exponent = s.split('e')
    mantissa = mantissa.rstrip('0').rstrip('.') or '0'
    exp_int = int(exponent)          # strips leading zeros and sign-pad
    return f'{mantissa}e{exp_int}'


@dataclass(frozen=True)
class Run:
    method: str
    encoder: str
    network: str
    lr: float
    board_size: int
    # seed=0 is the legacy "no seed" form: slug/jsonl_key stay identical
    # to pre-seed runs so existing runs/ dirs and JSONL entries resolve
    # unchanged. seed >= 1 appends `_seed{N}` to slug and `__seed{N}` to
    # jsonl_key, isolating per-seed checkpoints and results.
    seed: int = 0

    @property
    def _seed_suffix(self) -> str:
        return '' if self.seed == 0 else f'_seed{self.seed}'

    @property
    def slug(self) -> str:
        return (f'{self.method}_{self.encoder}_{self.network}_'
                f'lr{_format_lr(self.lr)}_b{self.board_size}'
                f'{self._seed_suffix}')

    @property
    def dir(self) -> Path:
        return RUNS_ROOT / self.slug

    def checkpoint_path(self, round_num: int) -> Path:
        return self.dir / 'checkpoints' / f'round_{round_num:03d}.h5'

    def experience_path(self, round_num: int) -> Path:
        return self.dir / 'experience' / f'round_{round_num:03d}.h5'

    @property
    def config_path(self) -> Path:
        return self.dir / 'config.json'

    @property
    def results_path(self) -> Path:
        return self.dir / 'results.json'

    def jsonl_key(self, round_num: int) -> str:
        seed_part = '' if self.seed == 0 else f'__seed{self.seed}'
        return (f'{self.method}__{self.encoder}__{self.network}__'
                f'lr{_format_lr(self.lr)}__b{self.board_size}'
                f'{seed_part}__round_{round_num:03d}')

    def latest_round(self) -> int:
        """Highest round number with a saved checkpoint, or 0 if none."""
        ckpt_dir = self.dir / 'checkpoints'
        if not ckpt_dir.is_dir():
            return 0
        rounds = []
        for f in ckpt_dir.glob('round_*.h5'):
            stem = f.stem  # 'round_NNN'
            try:
                rounds.append(int(stem.split('_')[1]))
            except (IndexError, ValueError):
                continue
        return max(rounds) if rounds else 0
