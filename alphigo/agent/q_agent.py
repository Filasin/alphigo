"""Q-learning reinforcement-learning agents.

One variant:
  - QAgent — book-faithful two-input Q(s, a) -> scalar model.

Trains on Monte-Carlo ±1 terminal rewards (no bootstrapping, no
replay, no target network — matches the book).  Move selection is
epsilon-greedy argmax over legal non-eye-filling moves.
"""

import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD

from alphigo.agent.base import Agent
from alphigo.agent.helpers import is_point_an_eye
from alphigo import alphigo_board
from alphigo import encoders
from alphigo import kerasutil


__all__ = [
    'QAgent',
    'load_q_agent',
]


# =====================================================================
# QAgent — book-style two-input model
# =====================================================================

class QAgent(Agent):
    """Q-learning agent with a two-input Q(s, a) -> scalar network.

    Matches the architecture from *Deep Learning and the Game of Go*:
    inputs are (board, action_plane), output is a scalar
    Q-value in [-1, +1] via tanh.  Move selection batches one copy of
    the state per legal action through a single forward pass.
    """

    def __init__(self, model, encoder):
        Agent.__init__(self)
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._epsilon = 0.0

    def set_epsilon(self, epsilon):
        self._epsilon = float(epsilon)

    set_temperature = set_epsilon  # backward-compat alias

    def set_collector(self, collector):
        self._collector = collector

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    def _legal_indices(self, game_state):
        """Return np.ndarray of legal non-eye move indices."""
        n = self._encoder.num_points()
        legal = []
        for idx in range(n):
            point = self._encoder.decode_point_index(idx)
            move = alphigo_board.Move.play_move(point)
            if game_state.is_valid_move(move) and not is_point_an_eye(
                    game_state.board, point, game_state.next_player):
                legal.append(idx)
        return np.array(legal, dtype=np.int64)

    def analyze(self, game_state, k=5):
        from alphigo.agent.analysis import AnalysisResult
        from alphigo.agent._analyze_helpers import softmax_normalise
        legal = self._legal_indices(game_state)
        if legal.size == 0:
            return AnalysisResult()
        board_tensor = self._encoder.encode(game_state)
        num_points = self._encoder.num_points()
        states = np.repeat(board_tensor[None, ...], legal.size, axis=0)
        action_vecs = np.zeros((legal.size, num_points), dtype=np.float32)
        action_vecs[np.arange(legal.size), legal] = 1.0
        q = self._model([states, action_vecs],
                        training=False).numpy()[:, 0]
        probs = softmax_normalise(q)  # softmax across legal moves at T=1
        order = np.argsort(-probs)[:k]
        top = []
        for i in order:
            point = self._encoder.decode_point_index(int(legal[i]))
            top.append((point, float(probs[i] * 100.0)))
        return AnalysisResult(top_moves=top, position_winrate=None,
                              pass_score=None)

    def select_move(self, game_state):
        legal = self._legal_indices(game_state)
        if legal.size == 0:
            return alphigo_board.Move.pass_turn()

        board_tensor = self._encoder.encode(game_state)
        num_points = self._encoder.num_points()

        if np.random.random() < self._epsilon:
            chosen = int(np.random.choice(legal))
        else:
            # Batch one (state, action_vector) pair per legal action.
            states = np.repeat(board_tensor[None, ...], legal.size, axis=0)
            action_vecs = np.zeros((legal.size, num_points), dtype=np.float32)
            action_vecs[np.arange(legal.size), legal] = 1.0
            q = self._model([states, action_vecs],
                            training=False).numpy()[:, 0]
            chosen = int(legal[int(np.argmax(q))])

        point = self._encoder.decode_point_index(chosen)
        if self._collector is not None:
            self._collector.record_decision(state=board_tensor,
                                            action=chosen)
        return alphigo_board.Move.play_move(point)

    def train(self, experience, lr=0.0001, clipnorm=1.0, batch_size=512):
        """Train Q(s, a) -> r using MSE.

        Builds per-sample one-hot action vectors (shape (n, num_points))
        from experience.actions; targets are experience.rewards shape
        (n,).  SGD, one epoch.
        """
        n = experience.states.shape[0]
        if n == 0:
            print('  [train] empty experience — skipping')
            return

        num_points = self._encoder.num_points()
        action_vecs = np.zeros((n, num_points), dtype=np.float32)
        action_vecs[np.arange(n), experience.actions] = 1.0

        self._model.compile(
            loss=MeanSquaredError(),
            optimizer=SGD(learning_rate=lr, clipnorm=clipnorm),
            jit_compile=False,
        )

        self._model.fit(
            [experience.states, action_vecs],
            experience.rewards.astype(np.float32),
            batch_size=batch_size, epochs=1,
        )


def load_q_agent(h5file):
    """Restore a QAgent (book-style two-input) from an HDF5 file."""
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))
    return QAgent(model, encoder)
