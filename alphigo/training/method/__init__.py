"""Method registry — canonical names per CONTEXT.md.

A Method (canonical: pg | q | ac | alphazero) owns build_agent, load_agent,
self_play, train, evaluate for one learning algorithm.
"""
from alphigo.training.method.actor_critic import ActorCriticMethod
from alphigo.training.method.alphazero import AlphaZeroMethod
from alphigo.training.method.policy_gradient import PolicyGradientMethod
from alphigo.training.method.q import QMethod


_REGISTRY = {
    'pg': PolicyGradientMethod,
    'q': QMethod,
    'ac': ActorCriticMethod,
    'alphazero': AlphaZeroMethod,
}


def get_method(name: str, **kwargs):
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f'Unknown method {name!r}. Canonical: pg, q, ac, alphazero.')
    if cls is AlphaZeroMethod:
        return cls(**kwargs)
    return cls()
