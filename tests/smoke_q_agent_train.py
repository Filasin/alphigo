"""Smoke test: QAgent.train reduces MSE on scalar Q output."""
import numpy as np

from alphigo.agent.q_agent import QAgent
from alphigo.data.networks import build_q_two_input
from alphigo.encoders.base import get_encoder_by_name
from alphigo.rl import ExperienceBuffer


def _build_planes(actions, h, w):
    n = len(actions)
    planes = np.zeros((n, h, w, 1), dtype=np.float32)
    rows = actions // w
    cols = actions % w
    planes[np.arange(n), rows, cols, 0] = 1.0
    return planes


def test_train_reduces_mse():
    np.random.seed(0)
    board_size = 9
    encoder = get_encoder_by_name('oneplane', board_size)
    model = build_q_two_input(encoder.shape(), encoder.num_points(),
                              network_name='small')
    agent = QAgent(model, encoder)

    n = 16
    states = np.random.randn(n, 9, 9, 1).astype('float32')
    actions = np.random.randint(0, 81, size=n).astype('int32')
    rewards = np.ones(n, dtype='float32')
    exp = ExperienceBuffer(states=states, actions=actions, rewards=rewards)

    planes = _build_planes(actions, 9, 9)
    q_before = model([states, planes], training=False).numpy()[:, 0]
    err_before = np.mean((q_before - rewards) ** 2)

    agent.train(exp, lr=0.01, clipnorm=1.0, batch_size=16)

    q_after = model([states, planes], training=False).numpy()[:, 0]
    err_after = np.mean((q_after - rewards) ** 2)

    assert err_after < err_before, \
        f"expected training to reduce error; before={err_before:.4f} " \
        f"after={err_after:.4f}"
    print(f'  [ok] err_before={err_before:.4f} -> err_after={err_after:.4f}')


if __name__ == '__main__':
    test_train_reduces_mse()
    print('tests/smoke_q_agent_train.py: all passed')
