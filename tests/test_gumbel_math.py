"""Unit tests for alphigo/agent/gumbel_math.py (pure NumPy, CPU-only)."""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_schedule_16_32():
    from alphigo.agent.gumbel_math import sequential_halving_schedule
    sched = sequential_halving_schedule(16, 32)
    assert sched == [(16, 1), (8, 1), (4, 2)], f"got {sched}"
    total = sum(s * v for s, v in sched)
    # m=16 is a power of 2, so the budget is consumed exactly
    assert total == 32, f"schedule must use exactly the budget, used {total}"
    print('  [ok] schedule(16, 32) == [(16,1),(8,1),(4,2)], total 32')


def test_schedule_8_8():
    from alphigo.agent.gumbel_math import sequential_halving_schedule
    sched = sequential_halving_schedule(8, 8)
    assert sched == [(8, 1)], f"got {sched}"
    print('  [ok] schedule(8, 8) == [(8,1)]')


def test_schedule_2_16():
    from alphigo.agent.gumbel_math import sequential_halving_schedule
    sched = sequential_halving_schedule(2, 16)
    assert sched == [(2, 8)], f"got {sched}"
    print('  [ok] schedule(2, 16) == [(2,8)]')


def test_schedule_m1_and_degenerate():
    from alphigo.agent.gumbel_math import sequential_halving_schedule
    assert sequential_halving_schedule(1, 10) == [(1, 10)]
    assert sequential_halving_schedule(5, 0) == []
    assert sequential_halving_schedule(0, 10) == []
    # Budget smaller than candidate count: schedule may be empty
    # (caller is expected to clamp m <= n), must not crash or overspend.
    sched = sequential_halving_schedule(16, 8)
    total = sum(s * v for s, v in sched)
    assert total <= 8, f"overspent budget: {total}"
    print('  [ok] degenerate schedules are safe')


def test_schedule_never_overspends_property():
    from alphigo.agent.gumbel_math import sequential_halving_schedule
    rng = np.random.RandomState(7)
    for _ in range(200):
        m = int(rng.randint(1, 90))
        n = int(rng.randint(0, 500))
        sched = sequential_halving_schedule(m, n)
        total = sum(s * v for s, v in sched)
        assert total <= n, f"m={m} n={n} overspent: {total}"
        for s, v in sched:
            assert s >= 1 and v >= 1
    print('  [ok] 200 random (m, n) pairs never overspend')


def test_sigma_monotonic_and_scales():
    from alphigo.agent.gumbel_math import sigma_q
    assert sigma_q(0.5, 8) > sigma_q(0.1, 8)
    assert sigma_q(-0.5, 8) < 0
    # (c_visit + max_visits) * c_scale * q  with defaults 50, 1.0
    assert abs(sigma_q(0.5, 8) - (50 + 8) * 1.0 * 0.5) < 1e-9
    assert abs(sigma_q(0.5, 8, c_scale=0.1) - (50 + 8) * 0.1 * 0.5) < 1e-9
    print('  [ok] sigma_q monotonic, matches formula')


def test_mixed_value_hand_computed():
    from alphigo.agent.gumbel_math import mixed_value
    raw_value = 0.2
    priors = np.array([0.5, 0.3, 0.2])
    q = np.array([0.6, 0.0, 0.0])          # only child 0 visited
    visits = np.array([2.0, 0.0, 0.0])
    # weighted_q = (0.5*0.6)/0.5 = 0.6 ; v_mix = (0.2 + 2*0.6)/(1+2)
    expected = (0.2 + 2 * 0.6) / 3.0
    assert abs(mixed_value(raw_value, priors, q, visits) - expected) < 1e-9
    # No visited children -> raw value
    assert mixed_value(0.3, priors, q, np.zeros(3)) == 0.3
    print('  [ok] mixed_value matches hand computation')


def test_completed_q():
    from alphigo.agent.gumbel_math import completed_q_values, mixed_value
    priors = np.array([0.5, 0.3, 0.2])
    q = np.array([0.6, -0.2, 0.0])
    visits = np.array([2.0, 1.0, 0.0])
    cq = completed_q_values(q, visits, priors, raw_value=0.1)
    assert cq[0] == 0.6 and cq[1] == -0.2, "visited children keep their q"
    vm = mixed_value(0.1, priors, q, visits)
    assert abs(cq[2] - vm) < 1e-9, "unvisited children get v_mix"
    print('  [ok] completed_q_values: visited keep q, unvisited get v_mix')


def test_improved_policy_upweights_better_q():
    from alphigo.agent.gumbel_math import improved_policy
    logits = np.log(np.array([0.5, 0.5]))
    cq = np.array([0.5, -0.5])
    pi = improved_policy(logits, cq, max_visits=2)
    assert abs(pi.sum() - 1.0) < 1e-6
    assert pi[0] > 0.99, f"sigma multiplier (50+2) should dominate: {pi}"
    # Equal q -> follows the prior
    pi2 = improved_policy(np.log(np.array([0.8, 0.2])),
                          np.array([0.1, 0.1]), max_visits=2)
    assert abs(pi2[0] - 0.8) < 1e-6 and abs(pi2[1] - 0.2) < 1e-6
    print('  [ok] improved_policy normalizes and up-weights higher q')


if __name__ == '__main__':
    test_schedule_16_32()
    test_schedule_8_8()
    test_schedule_2_16()
    test_schedule_m1_and_degenerate()
    test_schedule_never_overspends_property()
    test_sigma_monotonic_and_scales()
    test_mixed_value_hand_computed()
    test_completed_q()
    test_improved_policy_upweights_better_q()
    print('tests/test_gumbel_math.py: all passed')
