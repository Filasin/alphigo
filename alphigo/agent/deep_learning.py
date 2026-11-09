import numpy as np

from alphigo import alphigo_board
from alphigo import encoders
from alphigo import kerasutil
from alphigo.agent.base import Agent
from alphigo.agent.helpers import is_point_an_eye


class DeepLearningAgent(Agent):
    def __init__(self, model, encoder, temperature=0.0):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder
        self.temperature = temperature

    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        preds = self.model(input_tensor, training=False)
        if isinstance(preds, list):
            return preds[0].numpy()[0]
        return preds.numpy()[0]

    def analyze(self, game_state, k=5):
        from alphigo.agent.analysis import AnalysisResult
        from alphigo.agent._analyze_helpers import top_k_from_probs
        probs = self.predict(game_state)
        top, pass_score = top_k_from_probs(
            probs, game_state, self.encoder, k=k)
        return AnalysisResult(top_moves=top, position_winrate=None,
                              pass_score=pass_score)

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state)
        if self.temperature > 0:
            move_probs = move_probs ** (1.0 / self.temperature)
            eps = 1e-6
            move_probs = np.clip(move_probs, eps, 1 - eps)
            move_probs = move_probs / np.sum(move_probs)
            candidates = np.arange(num_moves)
            ranked_moves = np.random.choice(
                candidates, num_moves, replace=False, p=move_probs)
        else:
            ranked_moves = np.argsort(-move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(alphigo_board.Move.play_move(point)) and \
                    not is_point_an_eye(game_state.board, point,
                                        game_state.next_player):  # Starting from the top, find a valid move that doesn't reduce eye-space.
                return alphigo_board.Move.play_move(point)
        return alphigo_board.Move.pass_turn()  # If no legal and non-self-destructive moves are left, pass.

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])


def load_prediction_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))
    return DeepLearningAgent(model, encoder)
