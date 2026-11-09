"""Pure-math helpers for Gumbel AlphaZero search (Danihelka et al. 2022).

All functions are NumPy-only (no TF) so they are trivially unit-testable.
Perspective convention: every q passed in must already be from the
*choosing player's* perspective (callers negate child.Q as needed).
"""
import math

import numpy as np


def sequential_halving_schedule(m, n):
    """Allocate a budget of ``n`` simulations over ``m`` candidates.

    Returns a list of phases ``[(num_survivors, visits_per_survivor), ...]``.
    Each phase visits every surviving candidate equally; after a phase the
    caller halves the survivor set. Guarantees the total visit count never
    exceeds ``n``. May return [] when n < m (caller should clamp m <= n).

    Note: the budget is consumed exactly when ``m`` is a power of two;
    for other ``m`` the final phases may be truncated and part of the
    budget left unused (e.g. m=3, n=32 uses only 15). The production
    caller uses m=16, where consumption is exact.
    """
    if n <= 0 or m <= 0:
        return []
    if m == 1:
        return [(1, n)]
    num_phases = max(1, math.ceil(math.log2(m)))
    phases = []
    survivors = m
    used = 0
    while survivors >= 2 and used < n:
        per = max(1, n // (num_phases * survivors))
        if used + per * survivors > n:
            per = (n - used) // survivors
            if per <= 0:
                break
        phases.append((survivors, per))
        used += per * survivors
        survivors //= 2
    return phases


def sigma_q(q, max_visits, c_visit=50.0, c_scale=1.0):
    """Monotone transform of q used in Gumbel ranking and pi-prime.

    sigma(q) = (c_visit + max_visits) * c_scale * q

    Preserves the sign of q (negative q stays negative).
    """
    return (c_visit + float(max_visits)) * c_scale * q


def mixed_value(raw_value, priors, q_values, visit_counts):
    """v_mix: interpolation of the node's network value and the
    prior-weighted mean q of its *visited* children (paper eq. for
    completed Q-values). All q must share the caller's perspective.
    """
    visit_counts = np.asarray(visit_counts, dtype=np.float64)
    visited = visit_counts > 0
    if not visited.any():
        return float(raw_value)
    priors = np.asarray(priors, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    sum_n = float(visit_counts.sum())
    pi_visited = float(priors[visited].sum())
    if pi_visited <= 0.0:
        return float(raw_value)
    weighted_q = float((priors[visited] * q_values[visited]).sum()) / pi_visited
    return (float(raw_value) + sum_n * weighted_q) / (1.0 + sum_n)


def completed_q_values(q_values, visit_counts, priors, raw_value):
    """Per-child completed Q: own q if visited, v_mix otherwise."""
    q_values = np.asarray(q_values, dtype=np.float64)
    visit_counts = np.asarray(visit_counts, dtype=np.float64)
    vm = mixed_value(raw_value, priors, q_values, visit_counts)
    return np.where(visit_counts > 0, q_values, vm)


def improved_policy(logits, completed_q, max_visits,
                    c_visit=50.0, c_scale=1.0):
    """pi' = softmax(logits + sigma(completed_q)) — the Gumbel policy
    target and the basis of interior-node selection. No Gumbel noise here.
    """
    scores = np.asarray(logits, dtype=np.float64) + sigma_q(
        np.asarray(completed_q, dtype=np.float64), max_visits,
        c_visit, c_scale)
    scores = scores - scores.max()
    exp = np.exp(scores)
    return exp / exp.sum()
