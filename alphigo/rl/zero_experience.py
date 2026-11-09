"""AlphaZero replay buffer.

Distinct from ``ExperienceBuffer`` because AlphaZero's training target
is a per-position visit-count *distribution* (one float per board point),
not a scalar action index. Stored as three NumPy arrays alongside the
HDF5 conventions used by the rest of the codebase.
"""

import numpy as np


__all__ = [
    'ZeroReplayBuffer',
    'load_zero_buffer',
    'combine_zero_buffers',
]


class ZeroReplayBuffer:
    """Container for AlphaZero training triples.

    Fields
    ──────
    encoded_states : (N, H, W, planes) float32 — encoder output per decision
    policy_targets : (N, num_points) float32 — visit-count distribution,
                     each row sums to ~1.0
    value_targets  : (N,) float32 — terminal reward from the deciding
                     player's perspective (typically ±1, 0 for draws/cap)
    """

    def __init__(self, encoded_states, policy_targets, value_targets):
        self.encoded_states = np.asarray(encoded_states, dtype=np.float32)
        self.policy_targets = np.asarray(policy_targets, dtype=np.float32)
        self.value_targets = np.asarray(value_targets, dtype=np.float32)

    def __len__(self):
        return int(self.encoded_states.shape[0])

    # Aliases so train_rl.py can treat ZeroReplayBuffer and
    # ExperienceBuffer polymorphically for "num decisions" reporting.
    @property
    def states(self):
        return self.encoded_states

    @property
    def rewards(self):
        return self.value_targets

    def serialize(self, h5file):
        grp = h5file.create_group('zero_experience')
        grp.create_dataset('encoded_states', data=self.encoded_states)
        grp.create_dataset('policy_targets', data=self.policy_targets)
        grp.create_dataset('value_targets', data=self.value_targets)


def load_zero_buffer(h5file):
    grp = h5file['zero_experience']
    return ZeroReplayBuffer(
        encoded_states=np.array(grp['encoded_states']),
        policy_targets=np.array(grp['policy_targets']),
        value_targets=np.array(grp['value_targets']),
    )


def combine_zero_buffers(buffers):
    non_empty = [b for b in buffers if len(b) > 0]
    if not non_empty:
        return ZeroReplayBuffer(
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return ZeroReplayBuffer(
        encoded_states=np.concatenate([b.encoded_states for b in non_empty]),
        policy_targets=np.concatenate([b.policy_targets for b in non_empty]),
        value_targets=np.concatenate([b.value_targets for b in non_empty]),
    )
