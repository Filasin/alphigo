"""End-to-end smoke test: train_rl.py --method alphazero (issue 0004)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_run_dirs():
    """Remove any prior alphazero smoke artifacts so the test is hermetic."""
    runs_root = REPO_ROOT / 'runs'
    if runs_root.exists():
        for d in runs_root.glob('alphazero_alphazero_alphazero_*'):
            shutil.rmtree(d, ignore_errors=True)
    # Strip any stale jsonl entries for this exact key prefix.
    jsonl = REPO_ROOT / 'results' / 'experiments.jsonl'
    if jsonl.exists():
        kept = []
        with open(jsonl, 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get('training_method') == 'alphazero' and \
                       e.get('board_size') == 9 and \
                       e.get('encoder') == 'alphazero':
                        continue
                except Exception:
                    pass
                kept.append(line)
        with open(jsonl, 'w') as f:
            f.writelines(kept)


def test_train_rl_alphazero_one_round():
    _clean_run_dirs()

    # Run one round, 10 games, 3 simulations.
    cmd = [
        sys.executable, 'train_rl.py',
        '--board-size', '9',
        '--encoder', 'alphazero',
        '--network', 'alphazero',
        '--method', 'alphazero',
        '--rounds', '1',
        '--games', '10',
        '--mcts-simulations', '3',
        '--eval-games', '4',
        '--bs', '32',
    ]
    print('  [cmd]', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=1800)
    if proc.returncode != 0:
        print('STDOUT:', proc.stdout[-2000:])
        print('STDERR:', proc.stderr[-2000:])
        raise AssertionError(f'train_rl.py exited {proc.returncode}')

    # Check checkpoint exists at the canonical Run path.
    ckpts = list((REPO_ROOT / 'runs').glob(
        'alphazero_alphazero_alphazero_*/checkpoints/round_001.h5'))
    assert ckpts, f'no round_001.h5 checkpoint found under runs/. STDOUT tail:\n{proc.stdout[-1500:]}'
    print('  [ok] checkpoint at', ckpts[0])

    # Loadable via load_zero_agent + select_move returns legal move.
    import h5py
    from alphigo.agent.registry import create_agent
    from alphigo.alphigo_board import GameState
    agent = create_agent('alphazero', agent_file=str(ckpts[0]),
                         num_simulations=3)
    gs = GameState.new_game(9)
    move = agent.select_move(gs)
    assert move.is_pass or move.is_resign or gs.is_valid_move(move)
    print('  [ok] checkpoint round-trips through registry and plays')

    # JSONL has the entry.
    jsonl = REPO_ROOT / 'results' / 'experiments.jsonl'
    assert jsonl.exists(), 'experiments.jsonl was not written'
    found = None
    with open(jsonl, 'r') as f:
        for line in f:
            e = json.loads(line)
            if e.get('training_method') == 'alphazero' and \
               e.get('rl_round') == 1 and \
               e.get('encoder') == 'alphazero':
                found = e
    assert found is not None, 'no alphazero entry in experiments.jsonl'
    assert 0.0 <= found['vs_ref_win_rate'] <= 1.0
    assert 'round_001' in found['key']
    print('  [ok] experiments.jsonl entry: key=', found['key'])


if __name__ == '__main__':
    test_train_rl_alphazero_one_round()
    print('tests/smoke_zero_training.py: passed')
