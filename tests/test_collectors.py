"""Per-Method experience collectors. Each owns its opaque Experience type."""
import numpy as np
from alphigo.training.collectors import PolicyGradientCollector


def test_pg_collector_records_per_episode_and_assigns_reward():
    coll = PolicyGradientCollector()
    coll.begin_episode(game_idx=0)
    coll.record_step(0, state_tensor=np.zeros((1, 9, 9)), action=4)
    coll.record_step(0, state_tensor=np.zeros((1, 9, 9)), action=12)
    coll.complete_episode(0, reward=+1.0)

    coll.begin_episode(game_idx=1)
    coll.record_step(1, state_tensor=np.zeros((1, 9, 9)), action=7)
    coll.complete_episode(1, reward=-1.0)

    exp = coll.to_experience()
    assert exp.states.shape[0] == 3
    assert exp.actions.tolist() == [4, 12, 7]
    assert exp.rewards.tolist() == [1.0, 1.0, -1.0]


from alphigo.training.collectors import QCollector


def test_q_collector_assigns_reward_per_episode():
    """Q-learning uses the same (state, action, reward) shape as PG today.
    The collector exists so QMethod.train can interpret the buffer with
    Q semantics (e.g., target = reward), even though the in-memory shape
    matches PG's."""
    coll = QCollector()
    coll.begin_episode(0)
    coll.record_step(0, state_tensor=np.zeros((1, 9, 9)), action=2)
    coll.record_step(0, state_tensor=np.zeros((1, 9, 9)), action=5)
    coll.complete_episode(0, reward=+1.0)
    exp = coll.to_experience()
    assert exp.actions.tolist() == [2, 5]
    assert exp.rewards.tolist() == [1.0, 1.0]


import pytest
from alphigo.training.collectors import ActorCriticCollector


def test_ac_collector_records_value_estimate_and_computes_advantage():
    coll = ActorCriticCollector()
    coll.begin_episode(0)
    coll.record_step(0, state_tensor=np.zeros((1, 9, 9)), action=3,
                     estimated_value=0.2)
    coll.record_step(0, state_tensor=np.zeros((1, 9, 9)), action=8,
                     estimated_value=0.5)
    coll.complete_episode(0, reward=+1.0)
    exp = coll.to_experience()
    # advantage = reward - estimated_value
    assert exp.advantages.tolist() == [pytest.approx(0.8), pytest.approx(0.5)]
