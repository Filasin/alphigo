"""AlphaZeroMethod is now end-to-end (issues 0001–0004)."""


def test_alphazero_method_metadata():
    from alphigo.training.method.alphazero import AlphaZeroMethod
    m = AlphaZeroMethod()
    assert m.name == 'alphazero'
    # Issue 0005 enabled SL → AZ warm-start.
    assert m.allows_bootstrap_from_sl is True
    assert m.default_epsilon == 0.0


def test_alphazero_method_kwargs_via_get_method():
    from alphigo.training.method import get_method
    m = get_method('alphazero', num_simulations=7, c_puct=2.0,
                   temperature_moves=4)
    assert m.num_simulations == 7
    assert m.c_puct == 2.0
    assert m.temperature_moves == 4


def test_alphazero_build_agent_returns_zero_agent():
    from alphigo.encoders.alphazero import AlphaZeroEncoder
    from alphigo.training.method.alphazero import AlphaZeroMethod
    from alphigo.agent.zero_agent import ZeroAgent
    encoder = AlphaZeroEncoder((9, 9))
    m = AlphaZeroMethod(num_simulations=3)
    agent = m.build_agent(encoder, 'alphazero', 9)
    assert isinstance(agent, ZeroAgent)
    assert agent._num_simulations == 3


def test_alphazero_build_agent_rejects_wrong_network():
    import pytest
    from alphigo.encoders.alphazero import AlphaZeroEncoder
    from alphigo.training.method.alphazero import AlphaZeroMethod
    encoder = AlphaZeroEncoder((9, 9))
    m = AlphaZeroMethod()
    with pytest.raises(ValueError, match="network='alphazero'"):
        m.build_agent(encoder, 'small', 9)
