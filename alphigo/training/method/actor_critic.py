"""Actor-critic Method — two-output (policy + value) network."""
from pathlib import Path
import h5py

from alphigo.agent.ac_agent import ACAgent, load_ac_agent
from alphigo.data.networks import build_ac_two_output
from alphigo.training.collectors import ActorCriticCollector
from alphigo.training.policies import (
    actor_critic_policy, actor_critic_scheduled_policy)
from alphigo.training.simulator import run_batch


class ActorCriticMethod:
    name = 'ac'
    default_epsilon = 0.0
    allows_bootstrap_from_sl = False

    def build_agent(self, encoder, network_name: str, board_size: int):
        model = build_ac_two_output(encoder.shape(), encoder.num_points(),
                                    network_name=network_name)
        return ACAgent(model, encoder)

    def load_agent(self, path: Path):
        with h5py.File(path, 'r') as h5:
            return load_ac_agent(h5)

    def self_play(self, learner, reference, board_size, num_games,
                  epsilon, batch_size=64):
        collector = ActorCriticCollector()
        # AC collector needs the value-head estimate per step. The simulator
        # only exposes raw model output to the policy; we capture each
        # active-row's value into a dict keyed by *active-row index*, and
        # have the run_batch caller pass that same row index to record_step
        # via game_idx. (run_batch maps active rows -> game ids; the policy
        # sees consecutive active rows, the collector sees game ids.)
        # In practice: we cache the most recent value per game_idx by
        # walking active games in order during the policy call.
        last_values: dict[int, float] = {}
        # We need the active-game-id mapping that run_batch builds. Since
        # run_batch passes us only model_output and legal masks, and the
        # legal mask order matches active_indices, we cache values by
        # active-row index and translate via a side-channel set up below.
        active_ids_for_step: list[int] = []

        def _capture_active_ids(num_games_):
            # We can't intercept active_indices from inside the policy,
            # so instead we cache values by active-row index, and the
            # record_step adapter reverse-maps. See record_step_with_value.
            pass

        # Scheduled per-side: T=1 first 16 plies, T=0.05 after.
        # Instantiate once outside the closures so the per-side call counter
        # persists across run_batch's calls into the policy.
        scheduled_a = actor_critic_scheduled_policy()
        scheduled_b = actor_critic_scheduled_policy()
        _ = epsilon  # epsilon currently unused under the scheduled policy

        def policy_a(model_output, legal):
            _policy_logits, value_head = model_output
            if hasattr(value_head, 'numpy'):
                value_head = value_head.numpy()
            for ai in range(value_head.shape[0]):
                active_ids_for_step.append(float(value_head[ai, 0]))
            return scheduled_a(model_output, legal)

        def policy_b(model_output, legal):
            return scheduled_b(model_output, legal)

        # The collector adapter: pop one value per record_step call (in the
        # same order the policy saw them) so each (state, action) is paired
        # with the value head that produced it.
        original_record = collector.record_step

        def record_step_with_value(game_idx, state_tensor, action):
            v = active_ids_for_step.pop(0) if active_ids_for_step else 0.0
            original_record(game_idx, state_tensor, action,
                            estimated_value=v)
        collector.record_step = record_step_with_value  # type: ignore[assignment]

        run_batch(
            model_a=learner._model, policy_a=policy_a,
            model_b=reference._model, policy_b=policy_b,
            encoder=learner._encoder, board_size=board_size,
            num_games=num_games, collector_a=collector,
        )
        return collector.to_experience()

    def train(self, agent, experience, lr, clipnorm, batch_size):
        agent.train(experience, lr=lr, clipnorm=clipnorm,
                    batch_size=batch_size)

    def evaluate(self, new_agent, old_agent, board_size, num_games):
        # Schedule: T=1 for first 16 plies (random-ish opening
        # sampled from learned policy), T=0.05 thereafter (near-greedy
        # mid/endgame). Color-swap to neutralise first-move advantage.
        half = num_games // 2
        rest = num_games - half
        a_wins = 0
        total = 0
        if half > 0:
            r1 = run_batch(
                model_a=new_agent._model,
                policy_a=actor_critic_scheduled_policy(),
                model_b=old_agent._model,
                policy_b=actor_critic_scheduled_policy(),
                encoder=new_agent._encoder, board_size=board_size,
                num_games=half,
            )
            a_wins += r1.a_wins
            total += r1.total
        if rest > 0:
            r2 = run_batch(
                model_a=old_agent._model,
                policy_a=actor_critic_scheduled_policy(),
                model_b=new_agent._model,
                policy_b=actor_critic_scheduled_policy(),
                encoder=new_agent._encoder, board_size=board_size,
                num_games=rest,
            )
            a_wins += r2.b_wins
            total += r2.total
        win_rate = a_wins / max(total, 1)
        return win_rate, {'a_wins': a_wins, 'total_games': total}
