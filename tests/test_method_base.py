"""The Method protocol exists and exposes the canonical interface."""
from alphigo.training.method.base import Method


def test_method_protocol_has_required_attrs():
    # name, default_epsilon, allows_bootstrap_from_sl as class attrs
    # build_agent, load_agent, self_play, train, evaluate as methods
    required = {'name', 'default_epsilon', 'allows_bootstrap_from_sl',
                'build_agent', 'load_agent', 'self_play', 'train', 'evaluate'}
    members = set(dir(Method))
    missing = required - members
    assert not missing, f'Method protocol missing: {missing}'
