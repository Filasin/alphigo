"""Run.seed: isolates checkpoints per seed; seed=0 stays backwards-compat."""


def test_run_seed_default_is_zero_and_slug_unchanged():
    from alphigo.training.run import Run
    r = Run(method='pg', encoder='oneplane', network='small',
            lr=1e-4, board_size=9)
    # Default seed must be 0 so existing runs/ dirs continue to resolve.
    assert r.seed == 0
    assert r.slug == 'pg_oneplane_small_lr1e-4_b9'
    assert r.jsonl_key(3) == 'pg__oneplane__small__lr1e-4__b9__round_003'


def test_run_seed_nonzero_appends_to_slug_and_key():
    from alphigo.training.run import Run
    r1 = Run(method='alphazero', encoder='alphazero', network='alphazero',
             lr=1e-4, board_size=9, seed=1)
    r2 = Run(method='alphazero', encoder='alphazero', network='alphazero',
             lr=1e-4, board_size=9, seed=2)
    assert r1.slug == 'alphazero_alphazero_alphazero_lr1e-4_b9_seed1'
    assert r2.slug == 'alphazero_alphazero_alphazero_lr1e-4_b9_seed2'
    assert r1.dir != r2.dir
    assert 'seed1' in r1.jsonl_key(0)
    assert 'seed2' in r2.jsonl_key(0)
    # Different seeds produce different keys for the same round number.
    assert r1.jsonl_key(5) != r2.jsonl_key(5)


def test_init_agent_writes_seed_isolated_dirs(tmp_path, monkeypatch):
    """Two seeds → two independent checkpoint dirs under runs/."""
    monkeypatch.chdir(tmp_path)
    from alphigo.training.run import Run
    import train_rl as tr

    r1 = Run(method='pg', encoder='oneplane', network='small',
             lr=1e-4, board_size=9, seed=1)
    r2 = Run(method='pg', encoder='oneplane', network='small',
             lr=1e-4, board_size=9, seed=2)

    p1 = tr.init_agent(r1)
    p2 = tr.init_agent(r2)
    from pathlib import Path
    p1, p2 = Path(p1), Path(p2)
    assert p1 != p2
    assert p1.exists() and p2.exists()
    assert 'seed1' in str(p1)
    assert 'seed2' in str(p2)


def test_run_seed_zero_key_does_not_collide_with_seed1():
    from alphigo.training.run import Run
    r0 = Run(method='pg', encoder='oneplane', network='small',
             lr=1e-4, board_size=9, seed=0)
    r1 = Run(method='pg', encoder='oneplane', network='small',
             lr=1e-4, board_size=9, seed=1)
    assert r0.jsonl_key(0) != r1.jsonl_key(0)
    # seed=0 form is the legacy form (no suffix).
    assert 'seed' not in r0.slug
