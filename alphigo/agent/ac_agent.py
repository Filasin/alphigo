"""Actor-critic reinforcement-learning agent.

ACAgent learns a single network with two outputs: a softmax policy
(actor) and a tanh state-value (critic).  At decision time it samples
from the policy distribution and records the critic's value estimate
alongside (state, action).  Training uses categorical cross-entropy on
the policy output (weighted by per-decision *advantage*) and MSE on the
value output (against ±1 terminal reward), summed via Keras
``loss_weights=[1.0, 0.5]``.
"""

import numpy as np
from tensorflow.keras.optimizers import SGD

from alphigo.agent.base import Agent
from alphigo.agent.helpers import is_point_an_eye
from alphigo import alphigo_board
from alphigo import encoders
from alphigo import kerasutil


__all__ = [
    'ACAgent',
    'load_ac_agent',
]


def _clip_probs(original_probs):
    min_p = 1e-5
    max_p = 1 - min_p
    clipped = np.clip(original_probs, min_p, max_p)
    return clipped / np.sum(clipped)


class ACAgent(Agent):
    """Actor-critic agent.

    The neural network has one input (the encoded board) and two outputs:
    a softmax over moves (the *actor*) and a tanh scalar in [-1, +1] (the
    *critic*).  At play time the value output is recorded into the
    experience buffer but is never used to pick a move — it shapes the
    training target for the policy via the advantage.
    """

    def __init__(self, model, encoder):
        Agent.__init__(self)
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._epsilon = 0.0

    # ── exploration control ──────────────────────────────────────────
    def set_epsilon(self, epsilon):
        self._epsilon = float(epsilon)

    set_temperature = set_epsilon  # backward-compat alias

    # ── experience collection ────────────────────────────────────────
    def set_collector(self, collector):
        self._collector = collector

    # ── inference ────────────────────────────────────────────────────
    def analyze(self, game_state, k=5):
        from alphigo.agent.analysis import AnalysisResult
        from alphigo.agent._analyze_helpers import top_k_from_probs
        x = np.array([self._encoder.encode(game_state)])
        outputs = self._model(x, training=False)
        move_probs = outputs[0].numpy()[0]
        value = float(outputs[1].numpy()[0][0])
        top, pass_score = top_k_from_probs(
            move_probs, game_state, self._encoder, k=k)
        winrate = (value + 1.0) / 2.0
        winrate = max(0.0, min(1.0, winrate))
        return AnalysisResult(top_moves=top, position_winrate=winrate,
                              pass_score=pass_score)

    def select_move(self, game_state):
        num_moves = self._encoder.board_width * self._encoder.board_height
        board_tensor = self._encoder.encode(game_state)
        x = np.array([board_tensor])

        # Two-output model: predict returns [policy_probs, value].
        if np.random.random() < self._epsilon:
            move_probs = np.ones(num_moves) / num_moves
            value_pred = self._model(x, training=False)
            estimated_value = float(value_pred[1].numpy()[0][0])
        else:
            outputs = self._model(x, training=False)
            move_probs = outputs[0].numpy()[0]
            estimated_value = float(outputs[1].numpy()[0][0])

        move_probs = _clip_probs(move_probs)
        candidates = np.arange(num_moves)
        ranked = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)

        for point_idx in ranked:
            point = self._encoder.decode_point_index(point_idx)
            mv = alphigo_board.Move.play_move(point)
            if game_state.is_valid_move(mv) and not is_point_an_eye(
                    game_state.board, point, game_state.next_player):
                if self._collector is not None:
                    self._collector.record_decision(
                        state=board_tensor,
                        action=int(point_idx),
                        estimated_value=estimated_value,
                    )
                return mv

        return alphigo_board.Move.pass_turn()

    # ── serialization ────────────────────────────────────────────────
    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    # ── training (actor-critic) ──────────────────────────
    def train(self, experience, lr=0.01, clipnorm=1.0, batch_size=512,
              policy_loss_weight=1.0, value_loss_weight=None):
        """Train the actor-critic network from an ExperienceBuffer.

        Policy target: a one-hot vector at the chosen action, scaled by
        the advantage of that decision (R - V_est).  Loss is categorical
        cross-entropy.

        Value target: the final ±1 reward of the episode.  Loss is MSE.

        Loss weights default to (1.0, 0.5) — the value
        loss is scaled down because its gradient magnitudes tend to
        dominate the policy gradients otherwise.  ``value_loss_weight``
        can also be overridden via the ``AC_VALUE_LOSS_WEIGHT`` env var
        for sensitivity sweeps without a CLI plumbing change.
        """
        if value_loss_weight is None:
            import os
            value_loss_weight = float(
                os.environ.get('AC_VALUE_LOSS_WEIGHT', 0.5))

        n = experience.states.shape[0]
        if n == 0:
            print('  [train] empty experience — skipping')
            return
        if experience.advantages is None:
            raise ValueError(
                "ACAgent.train requires experience.advantages "
                "(use record_decision(..., estimated_value=v))")

        num_moves = self._encoder.board_width * self._encoder.board_height
        policy_target = np.zeros((n, num_moves), dtype=np.float32)
        policy_target[np.arange(n), experience.actions.astype(np.int64)] = \
            experience.advantages.astype(np.float32)
        value_target = experience.rewards.astype(np.float32)

        self._model.compile(
            optimizer=SGD(learning_rate=lr, clipnorm=clipnorm),
            loss=['categorical_crossentropy', 'mse'],
            loss_weights=[policy_loss_weight, value_loss_weight],
            jit_compile=False,
        )
        self._model.fit(
            experience.states,
            [policy_target, value_target],
            batch_size=batch_size,
            epochs=1,
        )


def load_ac_agent(h5file):
    """Restore an ACAgent from an HDF5 file."""
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))
    return ACAgent(model, encoder)
