"""Policy callables for batched self-play and evaluation.

A Policy is a pure function:
    policy(model_outputs, legal_mask) -> action_indices
where:
    model_outputs : np.ndarray (num_games, num_actions) — raw logits/Q-values
    legal_mask    : np.ndarray (num_games, num_actions) — bool, True = legal
    action_indices: np.ndarray (num_games,)            — int; -1 = pass
"""
from typing import Callable
import numpy as np

Policy = Callable[[np.ndarray, np.ndarray], np.ndarray]


def softmax_policy(temperature: float = 1.0) -> Policy:
    """Softmax-sample over legal moves with the given temperature."""
    def policy(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
        n_games = logits.shape[0]
        actions = np.full(n_games, -1, dtype=np.int32)
        for gi in range(n_games):
            mask = legal[gi]
            if not mask.any():
                continue                  # pass
            scaled = logits[gi] / max(temperature, 1e-6)
            scaled = scaled - scaled.max()
            exp = np.exp(scaled) * mask
            total = exp.sum()
            if total <= 0:
                # All legal moves underflowed — fall back to uniform.
                probs = mask.astype(np.float64) / mask.sum()
            else:
                probs = exp / total
            actions[gi] = np.random.choice(logits.shape[1], p=probs)
        return actions
    return policy


def epsilon_greedy_q_policy(epsilon: float = 0.0) -> Policy:
    """ε-greedy over Q-values. With prob ε pick uniformly over legal moves;
    otherwise pick argmax over legal Q-values."""
    def policy(q_values: np.ndarray, legal: np.ndarray) -> np.ndarray:
        n_games = q_values.shape[0]
        actions = np.full(n_games, -1, dtype=np.int32)
        for gi in range(n_games):
            mask = legal[gi]
            if not mask.any():
                continue                  # pass
            if np.random.random() < epsilon:
                legal_idx = np.flatnonzero(mask)
                actions[gi] = np.random.choice(legal_idx)
            else:
                masked_q = np.where(mask, q_values[gi], -np.inf)
                actions[gi] = int(np.argmax(masked_q))
        return actions
    return policy


def actor_critic_policy(epsilon: float = 0.0) -> Policy:
    """Sample from the policy head of an AC two-output model.

    Accepts model output as a tuple (policy_logits, value); ignores value.
    With probability epsilon picks uniformly over legal moves.
    """
    softmax = softmax_policy(temperature=1.0)

    def policy(model_output, legal: np.ndarray) -> np.ndarray:
        if isinstance(model_output, (tuple, list)):
            policy_logits, _value = model_output
        else:
            policy_logits = model_output
        if hasattr(policy_logits, 'numpy'):
            policy_logits = policy_logits.numpy()

        if epsilon > 0.0 and np.random.random() < epsilon:
            return epsilon_greedy_q_policy(epsilon=1.0)(
                np.zeros_like(policy_logits), legal)
        return softmax(policy_logits, legal)

    return policy


def actor_critic_scheduled_policy(open_temp: float = 1.0,
                                  late_temp: float = 0.05,
                                  schedule_plies: int = 16) -> Policy:
    """Actor-critic temperature schedule.

    T = open_temp for the side's first ``schedule_plies // 2`` calls
    (i.e. while ply count for the *whole* game is < schedule_plies),
    T = late_temp thereafter. Tracks per-side call count via closure;
    relies on run_batch calling each side's policy once per its own
    move with all batched games in lockstep.
    """
    open_softmax = softmax_policy(temperature=open_temp)
    late_softmax = softmax_policy(temperature=late_temp)
    side_call_threshold = schedule_plies // 2  # one side's calls in opening
    state = {'calls': 0}

    def policy(model_output, legal: np.ndarray) -> np.ndarray:
        if isinstance(model_output, (tuple, list)):
            policy_logits, _value = model_output
        else:
            policy_logits = model_output
        if hasattr(policy_logits, 'numpy'):
            policy_logits = policy_logits.numpy()
        sm = open_softmax if state['calls'] < side_call_threshold else late_softmax
        state['calls'] += 1
        return sm(policy_logits, legal)

    return policy
