"""Policy gradient reinforcement-learning agent.

PolicyAgent selects moves by sampling from a neural network policy.
It integrates with ExperienceCollector for self-play data collection
and supports training via the REINFORCE (Monte Carlo policy gradient)
algorithm.
"""

import numpy as np
from tensorflow.keras.optimizers import SGD

from alphigo.agent.base import Agent
from alphigo.agent.helpers import is_point_an_eye
from alphigo import alphigo_board
from alphigo import encoders
from alphigo import kerasutil


__all__ = [
    'PolicyAgent',
    'load_policy_agent',
]


def _clip_probs(original_probs):
    """Clip probabilities so none reach 0 or 1, then re-normalise."""
    min_p = 1e-5
    max_p = 1 - min_p
    clipped = np.clip(original_probs, min_p, max_p)
    return clipped / np.sum(clipped)


class PolicyAgent(Agent):
    """Deep-learning agent that learns through policy gradients.

    During self-play it samples moves stochastically from the network
    output (with optional epsilon-greedy exploration).  After collecting
    experience, call ``train()`` to update the policy using REINFORCE.
    """

    def __init__(self, model, encoder):
        Agent.__init__(self)
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._epsilon = 0.0

    # ── exploration control ───────────────────────────────────────────

    def set_epsilon(self, epsilon):
        """Set exploration rate (probability of choosing a random move)."""
        self._epsilon = epsilon

    # keep backward compat for any callers using the old name
    set_temperature = set_epsilon

    # ── experience collection ─────────────────────────────────────────

    def set_collector(self, collector):
        self._collector = collector

    # ── inference ─────────────────────────────────────────────────────

    def predict(self, game_state):
        encoded_state = self._encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        return self._model(input_tensor, training=False).numpy()[0]

    def analyze(self, game_state, k=5):
        from alphigo.agent.analysis import AnalysisResult
        from alphigo.agent._analyze_helpers import top_k_from_probs
        probs = self.predict(game_state)
        top, pass_score = top_k_from_probs(
            probs, game_state, self._encoder, k=k)
        return AnalysisResult(top_moves=top, position_winrate=None,
                              pass_score=pass_score)

    def select_move(self, game_state):
        num_moves = self._encoder.board_width * self._encoder.board_height

        board_tensor = self._encoder.encode(game_state)
        x = np.array([board_tensor])

        if np.random.random() < self._epsilon:
            # Explore: uniform random over all moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            # Exploit: follow the current policy.
            move_probs = self._model(x, training=False).numpy()[0]

        move_probs = _clip_probs(move_probs)

        # Sample moves according to the policy distribution.
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)

        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(alphigo_board.Move.play_move(point)) \
                    and not is_point_an_eye(game_state.board, point,
                                           game_state.next_player):
                if self._collector is not None:
                    self._collector.record_decision(
                        state=board_tensor,
                        action=point_idx,
                    )
                return alphigo_board.Move.play_move(point)

        return alphigo_board.Move.pass_turn()

    # ── serialization ─────────────────────────────────────────────────

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    # ── training (policy gradient / REINFORCE) ────────────

    def train(self, experience, lr=0.0001, clipnorm=1.0, batch_size=512):
        """Train the policy from an ExperienceBuffer using REINFORCE.

        For each move in the experience:
          - If the agent won  (reward = +1): push the policy toward that move.
          - If the agent lost (reward = -1): push the policy away from it.

        This works because cross-entropy loss with a -1 target reverses
        the gradient direction, effectively decreasing the probability of
        the chosen move.
        """
        self._model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(learning_rate=lr, clipnorm=clipnorm),
            jit_compile=False,
        )

        n = experience.states.shape[0]
        num_moves = self._encoder.board_width * self._encoder.board_height

        # Build target vectors: reward at the chosen action index.
        y = np.zeros((n, num_moves))
        y[np.arange(n), experience.actions] = experience.rewards

        self._model.fit(
            experience.states, y,
            batch_size=batch_size,
            epochs=1,
        )


def load_policy_agent(h5file):
    """Restore a PolicyAgent from an HDF5 file."""
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))
    return PolicyAgent(model, encoder)
