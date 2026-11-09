"""ActorCriticMethod (book Ch. 12) end-to-end smoke."""
import numpy as np
import pytest
from alphigo.encoders.base import get_encoder_by_name


def test_ac_method_metadata():
    from alphigo.training.method.actor_critic import ActorCriticMethod
    m = ActorCriticMethod()
    assert m.name == 'ac'
    assert m.default_epsilon == 0.0
    assert m.allows_bootstrap_from_sl is False


def test_ac_method_self_play_records_advantages():
    pytest.importorskip('tensorflow')
    from alphigo.training.method.actor_critic import ActorCriticMethod
    np.random.seed(0)
    method = ActorCriticMethod()
    encoder = get_encoder_by_name('oneplane', 9)
    learner = method.build_agent(encoder, 'small', 9)
    reference = method.build_agent(encoder, 'small', 9)
    exp = method.self_play(learner, reference, 9, num_games=2,
                           epsilon=0.0, batch_size=2)
    assert exp.advantages is not None
    assert exp.advantages.shape == exp.rewards.shape
    method.train(learner, exp, lr=1e-3, clipnorm=1.0, batch_size=8)
