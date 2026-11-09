"""PolicyGradientMethod end-to-end smoke: build → save → load → self-play
→ train → evaluate. Uses real Keras with a tiny network."""
from pathlib import Path
import numpy as np
import pytest

from alphigo.encoders.base import get_encoder_by_name


def test_pg_method_is_registered_under_canonical_name():
    from alphigo.training.method.policy_gradient import PolicyGradientMethod
    m = PolicyGradientMethod()
    assert m.name == 'pg'
    assert m.default_epsilon == 0.0
    assert m.allows_bootstrap_from_sl is True


def test_pg_method_round_trip(tmp_path):
    pytest.importorskip('tensorflow')
    from alphigo.training.method.policy_gradient import PolicyGradientMethod
    np.random.seed(0)

    method = PolicyGradientMethod()
    encoder = get_encoder_by_name('oneplane', 9)
    agent = method.build_agent(encoder, network_name='small', board_size=9)
    assert agent is not None

    save_path = tmp_path / 'pg.h5'
    import h5py
    with h5py.File(save_path, 'w') as h5:
        agent.serialize(h5)

    loaded = method.load_agent(save_path)
    assert loaded is not None


def test_pg_method_self_play_and_train(tmp_path):
    pytest.importorskip('tensorflow')
    from alphigo.training.method.policy_gradient import PolicyGradientMethod
    np.random.seed(0)

    method = PolicyGradientMethod()
    encoder = get_encoder_by_name('oneplane', 9)
    learner = method.build_agent(encoder, 'small', 9)
    reference = method.build_agent(encoder, 'small', 9)

    exp = method.self_play(learner, reference, board_size=9,
                           num_games=2, epsilon=0.0, batch_size=2)
    assert exp.states.shape[0] > 0
    assert exp.rewards.shape[0] == exp.states.shape[0]

    method.train(learner, exp, lr=1e-3, clipnorm=1.0, batch_size=8)


def test_pg_method_evaluate_returns_win_rate_in_unit_interval(tmp_path):
    pytest.importorskip('tensorflow')
    from alphigo.training.method.policy_gradient import PolicyGradientMethod
    np.random.seed(0)

    method = PolicyGradientMethod()
    encoder = get_encoder_by_name('oneplane', 9)
    a = method.build_agent(encoder, 'small', 9)
    b = method.build_agent(encoder, 'small', 9)
    win_rate, summary = method.evaluate(a, b, board_size=9, num_games=2)
    assert 0.0 <= win_rate <= 1.0
    assert summary['total_games'] == 2
    assert 'a_wins' in summary
