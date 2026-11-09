"""Run identity (5-tuple) and slug formatting."""
import pytest
from dataclasses import FrozenInstanceError
from alphigo.training.run import Run


def test_run_holds_five_axes():
    r = Run(method='pg', encoder='oneplane', network='small',
            lr=1e-4, board_size=9)
    assert r.method == 'pg'
    assert r.encoder == 'oneplane'
    assert r.network == 'small'
    assert r.lr == 1e-4
    assert r.board_size == 9


def test_run_is_frozen():
    r = Run(method='pg', encoder='oneplane', network='small',
            lr=1e-4, board_size=9)
    with pytest.raises(FrozenInstanceError):
        r.method = 'q'  # type: ignore[misc]


def test_run_slug_uses_canonical_format():
    r = Run(method='pg', encoder='oneplane', network='small',
            lr=1e-4, board_size=9)
    assert r.slug == 'pg_oneplane_small_lr1e-4_b9'


def test_run_slug_handles_round_lr():
    r = Run(method='q', encoder='sevenplane', network='large',
            lr=0.001, board_size=9)
    assert r.slug == 'q_sevenplane_large_lr1e-3_b9'


def test_run_slug_handles_lr_with_mantissa():
    r = Run(method='ac', encoder='oneplane', network='small',
            lr=2.5e-4, board_size=9)
    assert r.slug == 'ac_oneplane_small_lr2.5e-4_b9'


from pathlib import Path


def test_run_dir_uses_runs_root():
    r = Run('pg', 'oneplane', 'small', 1e-4, 9)
    assert r.dir == Path('runs/pg_oneplane_small_lr1e-4_b9')


def test_run_checkpoint_path_is_zero_padded():
    r = Run('pg', 'oneplane', 'small', 1e-4, 9)
    p = r.checkpoint_path(round_num=5)
    assert p == Path('runs/pg_oneplane_small_lr1e-4_b9/checkpoints/round_005.h5')


def test_run_experience_path_is_zero_padded():
    r = Run('q', 'oneplane', 'small', 1e-3, 9)
    p = r.experience_path(round_num=12)
    assert p == Path('runs/q_oneplane_small_lr1e-3_b9/experience/round_012.h5')


def test_run_config_and_results_paths():
    r = Run('pg', 'oneplane', 'small', 1e-4, 9)
    assert r.config_path == Path('runs/pg_oneplane_small_lr1e-4_b9/config.json')
    assert r.results_path == Path('runs/pg_oneplane_small_lr1e-4_b9/results.json')


def test_run_jsonl_key_includes_all_axes_plus_round():
    r = Run('pg', 'oneplane', 'small', 1e-4, 9)
    assert r.jsonl_key(round_num=3) == \
        'pg__oneplane__small__lr1e-4__b9__round_003'


def test_latest_round_is_zero_when_no_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = Run('pg', 'oneplane', 'small', 1e-4, 9)
    assert r.latest_round() == 0


def test_latest_round_returns_highest_round_on_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = Run('pg', 'oneplane', 'small', 1e-4, 9)
    ckpt_dir = r.dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / 'round_000.h5').touch()
    (ckpt_dir / 'round_005.h5').touch()
    (ckpt_dir / 'round_012.h5').touch()
    assert r.latest_round() == 12


def test_latest_round_ignores_non_round_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = Run('pg', 'oneplane', 'small', 1e-4, 9)
    ckpt_dir = r.dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / 'round_003.h5').touch()
    (ckpt_dir / 'best.h5').touch()           # not a round
    (ckpt_dir / 'round_xx.h5').touch()       # malformed
    assert r.latest_round() == 3
