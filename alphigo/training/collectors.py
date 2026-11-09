"""Per-Method experience collectors implementing the simulator.Collector
protocol. Each Method's collector owns its own opaque Experience type
(per the Q5 (j) decision in the architecture grilling)."""
from collections import defaultdict
import numpy as np

from alphigo.rl.experience import ExperienceBuffer


class PolicyGradientCollector:
    """Records (state, action) per step; reward broadcast to all steps of
    an episode. Output: ExperienceBuffer (the existing PG/Q shared format)."""

    def __init__(self):
        self._per_episode_states = defaultdict(list)
        self._per_episode_actions = defaultdict(list)
        self._completed_states = []
        self._completed_actions = []
        self._completed_rewards = []

    def begin_episode(self, game_idx: int) -> None:
        self._per_episode_states[game_idx] = []
        self._per_episode_actions[game_idx] = []

    def record_step(self, game_idx: int, state_tensor, action: int) -> None:
        self._per_episode_states[game_idx].append(state_tensor)
        self._per_episode_actions[game_idx].append(int(action))

    def complete_episode(self, game_idx: int, reward: float) -> None:
        states = self._per_episode_states.pop(game_idx, [])
        actions = self._per_episode_actions.pop(game_idx, [])
        self._completed_states.extend(states)
        self._completed_actions.extend(actions)
        self._completed_rewards.extend([reward] * len(states))

    def to_experience(self) -> ExperienceBuffer:
        if not self._completed_states:
            return ExperienceBuffer(
                states=np.empty((0,)), actions=np.empty((0,), dtype=np.int32),
                rewards=np.empty((0,), dtype=np.float32))
        return ExperienceBuffer(
            states=np.stack(self._completed_states),
            actions=np.array(self._completed_actions, dtype=np.int32),
            rewards=np.array(self._completed_rewards, dtype=np.float32),
        )


class QCollector(PolicyGradientCollector):
    """Q-learning collector — same on-disk shape as PG (state, action,
    reward); separate type to keep Method experience opaque per the
    architecture decision. QMethod.train interprets `rewards` as Q targets,
    not policy-gradient signals."""
    pass


class ActorCriticCollector:
    """Records (state, action, value_estimate) per step; computes
    advantage = reward - value_estimate at episode end. Produces an
    ExperienceBuffer with `advantages` populated."""

    def __init__(self):
        self._per_episode = defaultdict(list)   # gi -> list[(s, a, v)]
        self._completed_states = []
        self._completed_actions = []
        self._completed_rewards = []
        self._completed_advantages = []

    def begin_episode(self, game_idx: int) -> None:
        self._per_episode[game_idx] = []

    def record_step(self, game_idx: int, state_tensor, action: int,
                    estimated_value: float = 0.0) -> None:
        self._per_episode[game_idx].append(
            (state_tensor, int(action), float(estimated_value)))

    def complete_episode(self, game_idx: int, reward: float) -> None:
        steps = self._per_episode.pop(game_idx, [])
        for state, action, value in steps:
            self._completed_states.append(state)
            self._completed_actions.append(action)
            self._completed_rewards.append(reward)
            self._completed_advantages.append(reward - value)

    def to_experience(self) -> ExperienceBuffer:
        if not self._completed_states:
            return ExperienceBuffer(
                states=np.empty((0,)),
                actions=np.empty((0,), dtype=np.int32),
                rewards=np.empty((0,), dtype=np.float32),
                advantages=np.empty((0,), dtype=np.float32))
        return ExperienceBuffer(
            states=np.stack(self._completed_states),
            actions=np.array(self._completed_actions, dtype=np.int32),
            rewards=np.array(self._completed_rewards, dtype=np.float32),
            advantages=np.array(self._completed_advantages,
                                dtype=np.float32),
        )
