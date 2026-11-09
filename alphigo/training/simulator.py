"""Generic batched self-play / evaluation loop.

run_batch runs `num_games` games in parallel between two policies.
Each side may supply a Collector (or None) to record experience.
The simulator itself is method-agnostic — Method-specific behaviour is
isolated in the Policy callables and the Collectors.
"""
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

from alphigo.training.batched_board import BatchedSimulator


@dataclass(frozen=True)
class BatchResult:
    a_wins: int
    b_wins: int
    draws: int
    total: int


class Collector(Protocol):
    def begin_episode(self, game_idx: int) -> None: ...
    def record_step(self, game_idx: int, state_tensor: np.ndarray,
                    action: int) -> None: ...
    def complete_episode(self, game_idx: int, reward: float) -> None: ...


def run_batch(
    *,
    model_a, policy_a,
    model_b, policy_b,
    encoder,
    board_size: int,
    num_games: int,
    max_moves: int = 200,
    collector_a: Optional[Collector] = None,
    collector_b: Optional[Collector] = None,
) -> BatchResult:
    """Run num_games to completion. Returns aggregate win counts."""
    sim = BatchedSimulator(encoder=encoder, board_size=board_size,
                           num_games=num_games)

    if collector_a is not None:
        for gi in range(num_games):
            collector_a.begin_episode(gi)
    if collector_b is not None:
        for gi in range(num_games):
            collector_b.begin_episode(gi)

    # Player A is black (moves first), player B is white.
    a_to_move = True
    for _step in range(max_moves * 2):
        if all(sim.is_finished(gi) for gi in range(num_games)):
            break
        active_indices = [gi for gi in range(num_games)
                          if not sim.is_finished(gi)]
        if not active_indices:
            break

        # Build batch of state tensors for the side to move.
        state_tensors = np.stack([sim.state_tensor(gi)
                                  for gi in active_indices])
        legal_masks = np.stack([sim.legal_mask(gi)
                                for gi in active_indices])

        model = model_a if a_to_move else model_b
        policy = policy_a if a_to_move else policy_b
        collector = collector_a if a_to_move else collector_b

        outputs = model(state_tensors, training=False)
        if hasattr(outputs, 'numpy'):
            outputs = outputs.numpy()
        actions = policy(outputs, legal_masks)

        # Apply per game.
        full_actions = np.full(num_games, -1, dtype=np.int32)
        for ai, gi in enumerate(active_indices):
            full_actions[gi] = actions[ai]
            if collector is not None:
                collector.record_step(
                    gi, state_tensors[ai], int(actions[ai]))
        sim.step(full_actions)
        a_to_move = not a_to_move

    a_wins = b_wins = draws = 0
    for gi in range(num_games):
        outcome = sim.outcome(gi)        # +1 = A wins, -1 = B wins, 0 = draw
        if outcome > 0:
            a_wins += 1
        elif outcome < 0:
            b_wins += 1
        else:
            draws += 1
        if collector_a is not None:
            collector_a.complete_episode(gi, reward=outcome)
        if collector_b is not None:
            collector_b.complete_episode(gi, reward=-outcome)

    return BatchResult(a_wins=a_wins, b_wins=b_wins,
                       draws=draws, total=num_games)
