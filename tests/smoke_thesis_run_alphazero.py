"""Smoke test for thesis_run.py --training-method alphazero (issue 0006)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean():
    runs_root = REPO_ROOT / 'runs'
    if runs_root.exists():
        for d in runs_root.glob('alphazero_alphazero_alphazero_*'):
            shutil.rmtree(d, ignore_errors=True)
    jsonl = REPO_ROOT / 'results' / 'experiments.jsonl'
    if jsonl.exists():
        kept = []
        with open(jsonl, 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if (e.get('training_method') == 'alphazero' and
                            e.get('board_size') == 9 and
                            e.get('encoder') == 'alphazero'):
                        continue
                except Exception:
                    pass
                kept.append(line)
        with open(jsonl, 'w') as f:
            f.writelines(kept)


def test_thesis_run_alphazero_smoke():
    _clean()
    cmd = [
        sys.executable, 'thesis_run.py',
        '--board-size', '9',
        '--training-method', 'alphazero',
        '--encoders', 'alphazero',
        '--networks', 'alphazero',
        '--phase', 'train',
        '--rl-rounds', '1',
        '--rl-games', '10',
        '--rl-mcts-simulations', '3',
        '--eval-games', '4',
        '--batch-size', '32',
        '--eval-opponents',  # require an arg, give a dummy that's allowed
        'random',
    ]
    print('  [cmd]', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=1800)
    if proc.returncode != 0:
        print('STDOUT:', proc.stdout[-2000:])
        print('STDERR:', proc.stderr[-2000:])
        raise AssertionError(f'thesis_run.py exited {proc.returncode}')

    ckpts = list((REPO_ROOT / 'runs').glob(
        'alphazero_alphazero_alphazero_*/checkpoints/round_001.h5'))
    assert ckpts, 'no AZ round_001.h5 produced via thesis_run'
    print('  [ok]', ckpts[0])


def test_thesis_run_alphazero_skips_incompatible_combos():
    """alphazero method + non-alphazero encoder is logged as skipped, not crashed."""
    _clean()
    cmd = [
        sys.executable, 'thesis_run.py',
        '--board-size', '9',
        '--training-method', 'alphazero',
        '--encoders', 'oneplane',  # incompatible
        '--networks', 'small',
        '--phase', 'train',
        '--rl-rounds', '1',
        '--rl-games', '5',
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, \
        f'expected clean skip, got exit {proc.returncode}\nstderr={proc.stderr[-1000:]}'
    assert '[skip]' in proc.stdout or 'skip' in proc.stdout.lower(), \
        'expected a [skip] log line for incompatible combo'
    print('  [ok] incompatible combo skipped, not crashed')


if __name__ == '__main__':
    test_thesis_run_alphazero_skips_incompatible_combos()
    test_thesis_run_alphazero_smoke()
    print('tests/smoke_thesis_run_alphazero.py: passed')
