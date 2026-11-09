"""Method registry: get_method(name) returns the canonical Method instance."""
import pytest


def test_get_method_returns_pg():
    from alphigo.training.method import get_method
    from alphigo.training.method.policy_gradient import PolicyGradientMethod
    m = get_method('pg')
    assert isinstance(m, PolicyGradientMethod)


def test_get_method_returns_q():
    from alphigo.training.method import get_method
    from alphigo.training.method.q import QMethod
    assert isinstance(get_method('q'), QMethod)


def test_get_method_returns_ac():
    from alphigo.training.method import get_method
    from alphigo.training.method.actor_critic import ActorCriticMethod
    assert isinstance(get_method('ac'), ActorCriticMethod)


def test_get_method_returns_alphazero():
    from alphigo.training.method import get_method
    from alphigo.training.method.alphazero import AlphaZeroMethod
    assert isinstance(get_method('alphazero'), AlphaZeroMethod)


def test_get_method_unknown_name_raises():
    from alphigo.training.method import get_method
    with pytest.raises(ValueError, match='Unknown method'):
        get_method('not-a-method')
