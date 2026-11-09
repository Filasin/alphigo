"""REFERENCE_ELOS canonical baseline-Elo registry + evaluate_external helper."""


def test_reference_elos_importable():
    from alphigo.training.baseline import REFERENCE_ELOS  # noqa: F401


def test_reference_elos_has_canonical_keys():
    from alphigo.training.baseline import REFERENCE_ELOS
    expected = {
        'gnugo': 1200,
        'gnugo:level=1': 600,
        'gnugo:level=3': 900,
        'gnugo:level=5': 1050,
        'gnugo:level=10': 1200,
        'fuego:max_games=1000': 1600,
        'fuego:max_games=5000': 1800,
        'pachi:playouts=500': 1200,
        'pachi:playouts=1000': 1400,
        'pachi:playouts=5000': 1700,
    }
    for k, v in expected.items():
        assert k in REFERENCE_ELOS, f'missing baseline {k!r}'
        assert REFERENCE_ELOS[k] == v, (
            f'baseline {k!r}: expected {v}, got {REFERENCE_ELOS[k]}')


def test_reference_elos_values_are_positive_ints():
    from alphigo.training.baseline import REFERENCE_ELOS
    for k, v in REFERENCE_ELOS.items():
        assert isinstance(v, int), f'{k!r} value {v!r} is not int'
        assert v > 0, f'{k!r} value {v} must be positive'


def test_evaluate_external_importable():
    from alphigo.training.baseline import evaluate_external  # noqa: F401


def test_evaluate_external_delegates_with_reference_elos(monkeypatch):
    """evaluate_external must inject REFERENCE_ELOS and reshape rows."""
    from alphigo.training import baseline

    captured = {}

    def fake_evaluate_checkpoints(checkpoint_paths, opponent_specs,
                                  num_games, board_size, reference_elos,
                                  verbose=True):
        captured['checkpoint_paths'] = list(checkpoint_paths)
        captured['opponent_specs'] = list(opponent_specs)
        captured['num_games'] = num_games
        captured['board_size'] = board_size
        captured['reference_elos'] = reference_elos
        rows = [
            {'opponent': 'gnugo', 'win_pct': 60.0,
             'elo_diff': 70, 'abs_elo': 1270},
            {'opponent': 'pachi:playouts=1000', 'win_pct': 25.0,
             'elo_diff': -190, 'abs_elo': 1210},
        ]
        return rows, list(checkpoint_paths)

    import eval_checkpoints
    monkeypatch.setattr(eval_checkpoints, 'evaluate_checkpoints',
                        fake_evaluate_checkpoints)

    out = baseline.evaluate_external(
        agent_path='rl_agents/9x9_oneplane_small/round_003.h5',
        board_size=9,
        eval_opponents=['gnugo', 'pachi:playouts=1000'],
        num_games=20,
    )

    assert captured['checkpoint_paths'] == [
        'rl_agents/9x9_oneplane_small/round_003.h5']
    assert captured['opponent_specs'] == ['gnugo', 'pachi:playouts=1000']
    assert captured['num_games'] == 20
    assert captured['board_size'] == 9
    assert captured['reference_elos'] is baseline.REFERENCE_ELOS

    assert out == {
        'gnugo': {'win_rate': 0.6, 'elo_delta': 70, 'abs_elo': 1270},
        'pachi:playouts=1000': {
            'win_rate': 0.25, 'elo_delta': -190, 'abs_elo': 1210},
    }
