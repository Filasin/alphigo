"""run_batch: one batched-game loop, two policies, optional per-side collectors."""
import numpy as np
from alphigo.training.simulator import run_batch, BatchResult
from alphigo.training.policies import softmax_policy
from alphigo.encoders.base import get_encoder_by_name


def _random_logits_policy(rng):
    def policy(model_output, legal):
        # Ignore the model entirely; pick uniformly over legal moves.
        n_games, n_actions = legal.shape
        out = np.full(n_games, -1, dtype=np.int32)
        for gi in range(n_games):
            legal_idx = np.flatnonzero(legal[gi])
            if legal_idx.size:
                out[gi] = rng.choice(legal_idx)
        return out
    return policy


class _NullModel:
    """Stand-in model that returns zeros; the test policies ignore it."""
    def __call__(self, x, training=False):
        import numpy as _np
        return _np.zeros((x.shape[0], 81), dtype=_np.float32)


def test_run_batch_finishes_all_games():
    rng = np.random.default_rng(42)
    encoder = get_encoder_by_name('oneplane', 9)
    result = run_batch(
        model_a=_NullModel(), policy_a=_random_logits_policy(rng),
        model_b=_NullModel(), policy_b=_random_logits_policy(rng),
        encoder=encoder, board_size=9, num_games=4, max_moves=200,
    )
    assert isinstance(result, BatchResult)
    assert result.total == 4
    assert result.a_wins + result.b_wins + result.draws == 4


def test_run_batch_invokes_collector_for_recording_side():
    rng = np.random.default_rng(0)
    encoder = get_encoder_by_name('oneplane', 9)

    class _CountingCollector:
        def __init__(self):
            self.steps = 0
            self.episodes_started = 0
            self.episodes_completed = 0
        def begin_episode(self, game_idx):
            self.episodes_started += 1
        def record_step(self, game_idx, state_tensor, action):
            self.steps += 1
        def complete_episode(self, game_idx, reward):
            self.episodes_completed += 1

    coll = _CountingCollector()
    run_batch(
        model_a=_NullModel(), policy_a=_random_logits_policy(rng),
        model_b=_NullModel(), policy_b=_random_logits_policy(rng),
        encoder=encoder, board_size=9, num_games=2, max_moves=200,
        collector_a=coll,   # only A is recording
    )
    assert coll.episodes_started == 2
    assert coll.episodes_completed == 2
    assert coll.steps > 0
