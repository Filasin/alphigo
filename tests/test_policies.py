"""Policy callables: pure functions over (model_output, legal_mask)."""
import numpy as np
from alphigo.training.policies import softmax_policy


def test_softmax_policy_returns_legal_actions():
    np.random.seed(0)
    policy = softmax_policy(temperature=1.0)
    # 3 games, 9 actions each. Game 1 only has action 4 legal.
    logits = np.random.randn(3, 9).astype(np.float32)
    legal = np.zeros((3, 9), dtype=bool)
    legal[0, [0, 4, 8]] = True
    legal[1, 4] = True
    legal[2, :] = True
    actions = policy(logits, legal)
    assert actions.shape == (3,)
    for gi in range(3):
        assert legal[gi, actions[gi]], f'illegal action in game {gi}'


def test_softmax_policy_returns_pass_when_no_legal():
    policy = softmax_policy(temperature=1.0)
    logits = np.zeros((1, 9), dtype=np.float32)
    legal = np.zeros((1, 9), dtype=bool)
    actions = policy(logits, legal)
    assert actions[0] == -1   # -1 sentinel = pass


def test_softmax_policy_with_low_temperature_picks_argmax():
    policy = softmax_policy(temperature=0.01)
    logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
    legal = np.array([[True, True, True]])
    counts = np.zeros(3, dtype=int)
    for _ in range(20):
        counts[policy(logits, legal)[0]] += 1
    assert counts[1] >= 18   # action 1 has highest logit


from alphigo.training.policies import epsilon_greedy_q_policy


def test_epsilon_zero_picks_argmax_over_legal_q():
    policy = epsilon_greedy_q_policy(epsilon=0.0)
    q = np.array([[0.1, 0.9, 0.5, 0.99]])
    legal = np.array([[True, True, True, False]])    # action 3 illegal
    actions = policy(q, legal)
    assert actions[0] == 1                            # argmax over legal


def test_epsilon_one_is_uniform_over_legal():
    np.random.seed(0)
    policy = epsilon_greedy_q_policy(epsilon=1.0)
    q = np.zeros((1, 4))
    legal = np.array([[True, True, False, True]])    # 3 legal actions
    counts = np.zeros(4, dtype=int)
    for _ in range(3000):
        counts[policy(q, legal)[0]] += 1
    assert counts[2] == 0                             # never picks illegal
    # Each legal action ~1000 times; allow generous tolerance.
    for legal_idx in (0, 1, 3):
        assert 850 < counts[legal_idx] < 1150


from alphigo.training.policies import actor_critic_policy


def test_actor_critic_policy_samples_from_policy_head_only():
    """AC model emits (policy_head, value_head). The policy ignores the value
    head and samples from the policy head, masked to legal moves."""
    np.random.seed(0)
    policy = actor_critic_policy(epsilon=0.0)
    p_head = np.array([[10.0, 0.0, 0.0]])     # near-deterministic action 0
    v_head = np.array([[0.5]])
    legal = np.array([[True, True, True]])
    counts = np.zeros(3, dtype=int)
    for _ in range(20):
        actions = policy((p_head, v_head), legal)
        counts[actions[0]] += 1
    assert counts[0] >= 18     # action 0 dominates


def test_actor_critic_policy_respects_legality():
    np.random.seed(0)
    policy = actor_critic_policy(epsilon=0.0)
    p_head = np.array([[10.0, 0.0, 0.0]])
    v_head = np.array([[0.0]])
    legal = np.array([[False, True, True]])   # action 0 illegal
    for _ in range(20):
        action = policy((p_head, v_head), legal)[0]
        assert action != 0
        assert action in (1, 2)
