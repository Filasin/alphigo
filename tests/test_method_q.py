"""QMethod (book Ch. 11 two-input formulation) end-to-end smoke."""
import numpy as np
import pytest
from alphigo.encoders.base import get_encoder_by_name


def test_q_method_metadata():
    from alphigo.training.method.q import QMethod
    m = QMethod()
    assert m.name == 'q'
    assert m.default_epsilon == 0.1
    assert m.allows_bootstrap_from_sl is False


def test_q_method_self_play_and_train(tmp_path):
    pytest.importorskip('tensorflow')
    from alphigo.training.method.q import QMethod
    np.random.seed(0)
    method = QMethod()
    encoder = get_encoder_by_name('oneplane', 9)
    learner = method.build_agent(encoder, 'small', 9)
    reference = method.build_agent(encoder, 'small', 9)
    exp = method.self_play(learner, reference, 9, num_games=2,
                           epsilon=0.5, batch_size=2)
    assert exp.states.shape[0] > 0
    method.train(learner, exp, lr=1e-3, clipnorm=1.0, batch_size=8)


def test_q_method_evaluate(tmp_path):
    pytest.importorskip('tensorflow')
    from alphigo.training.method.q import QMethod
    np.random.seed(0)
    method = QMethod()
    encoder = get_encoder_by_name('oneplane', 9)
    a = method.build_agent(encoder, 'small', 9)
    b = method.build_agent(encoder, 'small', 9)
    rate, summary = method.evaluate(a, b, 9, num_games=2)
    assert 0.0 <= rate <= 1.0
    assert summary['total_games'] == 2
