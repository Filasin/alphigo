"""Method protocol — the canonical seam for a learning algorithm.

Per CONTEXT.md: a Method is the learning algorithm used to train an agent.
Canonical names: pg, q, ac, alphazero. (sl does not implement this protocol;
its training pipeline is in train.py.)
"""
from typing import Protocol, runtime_checkable, Tuple, Dict, Any
from pathlib import Path


@runtime_checkable
class Method(Protocol):
    name: str = ''
    default_epsilon: float = 0.0
    allows_bootstrap_from_sl: bool = False

    def build_agent(self, encoder, network_name: str,
                    board_size: int): ...

    def load_agent(self, path: Path): ...

    def self_play(self, learner, reference, board_size: int,
                  num_games: int, epsilon: float,
                  batch_size: int = 64) -> Any: ...

    def train(self, agent, experience: Any, lr: float,
              clipnorm: float, batch_size: int) -> None: ...

    def evaluate(self, new_agent, old_agent, board_size: int,
                 num_games: int) -> Tuple[float, Dict[str, int]]: ...
