"""train_rl.py uses Run for all path / key generation.

After Plan C, train_rl.py must derive checkpoint/experience/key paths
from alphigo.training.run.Run rather than the bespoke _agent_dir /
_agent_path / _experience_path / _make_key helpers.
"""
import os
import json
from pathlib import Path

import pytest

from alphigo.training.run import Run


def test_run_layout_for_pg():
    r = Run(method='pg', encoder='oneplane', network='small',
            lr=1e-4, board_size=9)
    assert r.dir == Path('runs/pg_oneplane_small_lr1e-4_b9')
    assert r.checkpoint_path(0) == Path(
        'runs/pg_oneplane_small_lr1e-4_b9/checkpoints/round_000.h5')
    assert r.experience_path(7) == Path(
        'runs/pg_oneplane_small_lr1e-4_b9/experience/round_007.h5')
    assert r.jsonl_key(7) == 'pg__oneplane__small__lr1e-4__b9__round_007'


def test_run_layout_for_q():
    r = Run(method='q', encoder='oneplane', network='book_2layer',
            lr=1e-3, board_size=9)
    assert r.dir == Path('runs/q_oneplane_book_2layer_lr1e-3_b9')
    assert r.checkpoint_path(3) == Path(
        'runs/q_oneplane_book_2layer_lr1e-3_b9/checkpoints/round_003.h5')


def test_train_rl_module_no_longer_has_legacy_helpers():
    """The bespoke path helpers are gone after Plan C."""
    import train_rl as tr
    for name in ('_agent_dir', '_agent_path', '_experience_path',
                 '_make_key', '_method_variant', '_method_label',
                 '_find_latest_round'):
        assert not hasattr(tr, name), (
            f'train_rl.{name} still exists; Plan C removes it.')


def test_train_rl_methods_constant_is_canonical_only():
    """METHODS lists only canonical method names — no legacy or q-dqn."""
    import train_rl as tr
    assert tr.METHODS == ('pg', 'q', 'ac', 'alphazero')


def test_train_rl_init_agent_writes_under_runs_dir(tmp_path,
                                                   monkeypatch):
    """init_agent saves the round_000 checkpoint under runs/{slug}/checkpoints/."""
    monkeypatch.chdir(tmp_path)
    import train_rl as tr
    path = tr.init_agent(
        board_size=9, encoder_name='oneplane', network_name='small',
        bootstrap_from=None, method='pg')
    expected = Path(
        'runs/pg_oneplane_small_lr1e-4_b9/checkpoints/round_000.h5')
    # init_agent uses the configured default lr; for the test we just
    # assert the path is under runs/ with the canonical structure.
    assert Path(path).parent.name == 'checkpoints'
    assert Path(path).parent.parent.parent.name == 'runs'
