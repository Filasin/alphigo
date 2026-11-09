"""Q-learning Method — two-input Q(s,a) -> scalar formulation.

Self-play uses ε-greedy over Q-values.
Training calls QAgent.train (existing implementation).

The Q network takes two inputs (board, action_one_hot). To plug into the
generic run_batch loop (which passes a single state tensor), we wrap the
model in `_QSingleInputAdapter` that, given a (n_games, ...) batch of
state tensors, expands each to (num_points, ...) one-hot-action pairs,
runs the underlying model, and returns a (n_games, num_points) Q matrix.
"""
from pathlib import Path
import h5py
import numpy as np

from alphigo.agent.q_agent import QAgent, load_q_agent
from alphigo.data.networks import build_q_two_input
from alphigo.training.collectors import QCollector
from alphigo.training.policies import epsilon_greedy_q_policy
from alphigo.training.simulator import run_batch


class _QSingleInputAdapter:
    """Adapter so a two-input Q(s, a) model fits run_batch's `model(x)` call."""

    def __init__(self, q_model, num_points: int):
        self._q_model = q_model
        self._num_points = num_points
        # Pre-build identity matrix for one-hot action vectors.
        self._eye = np.eye(num_points, dtype=np.float32)

    def __call__(self, state_tensors, training=False):
        n_games = state_tensors.shape[0]
        # Expand each state to num_points copies, pair with each one-hot action.
        states = np.repeat(state_tensors, self._num_points, axis=0)
        actions = np.tile(self._eye, (n_games, 1))
        q = self._q_model([states, actions], training=training)
        if hasattr(q, 'numpy'):
            q = q.numpy()
        return q.reshape(n_games, self._num_points)


class QMethod:
    name = 'q'
    default_epsilon = 0.1
    allows_bootstrap_from_sl = False

    def build_agent(self, encoder, network_name: str, board_size: int):
        model = build_q_two_input(encoder.shape(), encoder.num_points(),
                                  network_name=network_name)
        return QAgent(model, encoder)

    def load_agent(self, path: Path):
        with h5py.File(path, 'r') as h5:
            return load_q_agent(h5)

    def self_play(self, learner, reference, board_size, num_games,
                  epsilon, batch_size=64):
        collector = QCollector()
        num_points = learner._encoder.num_points()
        run_batch(
            model_a=_QSingleInputAdapter(learner._model, num_points),
            policy_a=epsilon_greedy_q_policy(epsilon),
            model_b=_QSingleInputAdapter(reference._model, num_points),
            policy_b=epsilon_greedy_q_policy(epsilon),
            encoder=learner._encoder, board_size=board_size,
            num_games=num_games,
            collector_a=collector,
        )
        return collector.to_experience()

    def train(self, agent, experience, lr, clipnorm, batch_size):
        agent.train(experience, lr=lr, clipnorm=clipnorm,
                    batch_size=batch_size)

    EVAL_EPSILON = 0.1

    def evaluate(self, new_agent, old_agent, board_size, num_games):
        # ε-greedy at EVAL_EPSILON with color swap. Matches the book's
        # play_train_eval_q.py, which uses a single (decaying) ε for both
        # self-play and eval. Stochastic eval gives continuous win rates
        # rather than the {0.0, 0.5, 1.0} bimodal pattern of pure argmax.
        num_points = new_agent._encoder.num_points()
        new_adapter = _QSingleInputAdapter(new_agent._model, num_points)
        old_adapter = _QSingleInputAdapter(old_agent._model, num_points)
        half = num_games // 2
        rest = num_games - half
        a_wins = 0
        total = 0
        eps = self.EVAL_EPSILON
        if half > 0:
            r1 = run_batch(
                model_a=new_adapter, policy_a=epsilon_greedy_q_policy(eps),
                model_b=old_adapter, policy_b=epsilon_greedy_q_policy(eps),
                encoder=new_agent._encoder, board_size=board_size,
                num_games=half,
            )
            a_wins += r1.a_wins
            total += r1.total
        if rest > 0:
            r2 = run_batch(
                model_a=old_adapter, policy_a=epsilon_greedy_q_policy(eps),
                model_b=new_adapter, policy_b=epsilon_greedy_q_policy(eps),
                encoder=new_agent._encoder, board_size=board_size,
                num_games=rest,
            )
            a_wins += r2.b_wins
            total += r2.total
        win_rate = a_wins / max(total, 1)
        return win_rate, {'a_wins': a_wins, 'total_games': total}
