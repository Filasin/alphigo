"""Policy gradient (REINFORCE) Method — book Chs. 9-10.

Self-play uses softmax sampling over the policy network output.
Training calls PolicyAgent.train (existing REINFORCE implementation).
"""
from pathlib import Path
import h5py

from alphigo.agent.policy_agent import PolicyAgent, load_policy_agent
from alphigo.data.networks import NETWORK_MAP
from alphigo.training.collectors import PolicyGradientCollector
from alphigo.training.policies import softmax_policy
from alphigo.training.simulator import run_batch


class PolicyGradientMethod:
    name = 'pg'
    default_epsilon = 0.0
    allows_bootstrap_from_sl = True

    def build_agent(self, encoder, network_name: str, board_size: int):
        input_shape = encoder.shape()
        num_classes = board_size * board_size
        model = NETWORK_MAP[network_name](input_shape, num_classes)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        return PolicyAgent(model, encoder)

    def load_agent(self, path: Path):
        with h5py.File(path, 'r') as h5:
            return load_policy_agent(h5)

    def self_play(self, learner, reference, board_size, num_games,
                  epsilon, batch_size=64):
        collector = PolicyGradientCollector()
        run_batch(
            model_a=learner._model, policy_a=softmax_policy(temperature=1.0),
            model_b=reference._model,
            policy_b=softmax_policy(temperature=1.0),
            encoder=learner._encoder, board_size=board_size,
            num_games=num_games,
            collector_a=collector,
        )
        return collector.to_experience()

    def train(self, agent, experience, lr, clipnorm, batch_size):
        agent.train(experience, lr=lr, clipnorm=clipnorm,
                    batch_size=batch_size)

    def evaluate(self, new_agent, old_agent, board_size, num_games):
        # Color-swap to neutralise first-move advantage. Matches Q/AC eval.
        half = num_games // 2
        rest = num_games - half
        a_wins = 0
        total = 0
        if half > 0:
            r1 = run_batch(
                model_a=new_agent._model,
                policy_a=softmax_policy(temperature=1.0),
                model_b=old_agent._model,
                policy_b=softmax_policy(temperature=1.0),
                encoder=new_agent._encoder, board_size=board_size,
                num_games=half,
            )
            a_wins += r1.a_wins
            total += r1.total
        if rest > 0:
            r2 = run_batch(
                model_a=old_agent._model,
                policy_a=softmax_policy(temperature=1.0),
                model_b=new_agent._model,
                policy_b=softmax_policy(temperature=1.0),
                encoder=new_agent._encoder, board_size=board_size,
                num_games=rest,
            )
            a_wins += r2.b_wins  # new played as B in this half
            total += r2.total
        win_rate = a_wins / max(total, 1)
        return win_rate, {'a_wins': a_wins, 'total_games': total}
