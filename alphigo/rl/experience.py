"""Experience collection and storage for reinforcement learning.

ExperienceCollector tracks per-episode state/action decisions and assigns
rewards at episode end.  ExperienceBuffer is a simple container of NumPy
arrays with HDF5 serialization.

Optionally tracks per-decision estimated state values and computes
``advantage = reward - estimated_value`` at episode end (actor-critic).  Buffers without advantages remain fully supported —
``advantages`` defaults to ``None`` and the HDF5 dataset is only written
when present, so legacy Q/PG checkpoints load unchanged.
"""

import numpy as np

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer',
    'combine_experience',
    'load_experience',
]


class ExperienceCollector:
    """Accumulates (state, action) pairs during self-play episodes.

    Usage (policy gradient / Q):
        collector.begin_episode()
        # ... agent plays moves, calling record_decision(state, action) ...
        collector.complete_episode(reward=+1)   # or -1

    Usage (actor-critic):
        collector.begin_episode()
        # ... record_decision(state, action, estimated_value=v) per move ...
        collector.complete_episode(reward=+1)
        # advantages = reward - estimated_value computed per decision.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self._has_advantages = False
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def record_decision(self, state, action, estimated_value=None):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        if estimated_value is not None:
            self._has_advantages = True
            self._current_episode_estimated_values.append(
                float(estimated_value))
        elif self._has_advantages:
            self._current_episode_estimated_values.append(0.0)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward] * num_states
        if self._has_advantages:
            for v in self._current_episode_estimated_values:
                self.advantages.append(reward - v)
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []


class ExperienceBuffer:
    """Container for states, actions, rewards (and optionally advantages)."""

    def __init__(self, states, actions, rewards, advantages=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)
        if self.advantages is not None:
            h5file['experience'].create_dataset('advantages',
                                                data=self.advantages)


def combine_experience(collectors):
    """Merge multiple ExperienceCollectors into a single ExperienceBuffer.

    Advantages are preserved when *all* non-empty collectors have them;
    otherwise the resulting buffer's ``advantages`` is None.
    """
    non_empty = [c for c in collectors if len(c.states) > 0]
    if not non_empty:
        return ExperienceBuffer(
            np.empty((0,)), np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32))
    combined_states = np.concatenate([np.array(c.states) for c in non_empty])
    combined_actions = np.concatenate([np.array(c.actions) for c in non_empty])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in non_empty])
    if all(c._has_advantages for c in non_empty):
        combined_advantages = np.concatenate(
            [np.array(c.advantages) for c in non_empty])
    else:
        combined_advantages = None
    return ExperienceBuffer(combined_states, combined_actions,
                            combined_rewards, combined_advantages)


def load_experience(h5file):
    """Restore an ExperienceBuffer from an HDF5 file."""
    grp = h5file['experience']
    advantages = np.array(grp['advantages']) if 'advantages' in grp else None
    return ExperienceBuffer(
        states=np.array(grp['states']),
        actions=np.array(grp['actions']),
        rewards=np.array(grp['rewards']),
        advantages=advantages,
    )
