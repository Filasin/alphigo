# AlphiGo

AlphaZero and Gumbel AlphaZero for the game of Go, trained on a single consumer GPU.

This repository contains the code behind the master's thesis of Filip Grujić and the accompanying paper: the Go engine, supervised learning on human games, PUCT and Gumbel AlphaZero search, the self-play training loops, and the evaluation scripts.

## Contents

| Path | Contents |
|---|---|
| `alphigo/` | Core package: board and rules, encoders, networks, agents, search, training methods |
| `alphigo/agent/gumbel_zero_agent.py`, `gumbel_math.py`, `fast_gumbel_agent.py` | Gumbel AlphaZero search (root: Gumbel top-m sampling and sequential halving; interior nodes: deterministic selection) |
| `alphigo/agent/zero_agent.py` | AlphaZero agent with PUCT search |
| `alphigo/mcts/_cboard.pyx` | Optional Cython version of the hot search loop (`python setup_cfast.py build_ext --inplace`); without it the code falls back to pure Python |
| `scripts/` | Self-play generation, training, evaluation and experiment runners (see `scripts/README.md`) |
| `tests/` | Unit tests, including the search sign tests |
| `results/` | Per-round results of the self-play experiments and of the sign experiment |
| `model/round_019_agent.h5` | A trained 9×9 agent (8 residual blocks, 64 filters), used by the sign experiment |
| `generate_mcst_games.cpp` | Fast C++ Monte Carlo tree search game generator (bootstrap data for the first Gumbel experiments) |
| `gui/`, `run.py` | Interactive board and command line interface |

## Installation

Python 3.12 was used.

    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt

A CPU is enough for the tests and the sign experiment. The self-play experiments were run on Linux (WSL2) with one NVIDIA GPU, ONNX Runtime and TensorRT 10.16.1.11. The GPU packages are listed as comments in `requirements.txt`.

## Quick checks (CPU)

Run the tests (about half a minute):

    python -m pytest -q tests

`tests/test_puct_semantics.py` and `tests/test_gumbel_semantics.py` build a network with hand-set weights in which exactly one move wins, and check that the search concentrates its visits on that move.

Run the sign experiment: the included network plays with PUCT search against itself without search, once with the correct sign of Q in the selection rule and once with the sign inverted.

    python scripts/puct_inversion_exp.py . correct  100 40
    python scripts/puct_inversion_exp.py . inverted 100 40

Arguments: `<code_root> <correct|inverted> <simulations> <games> [checkpoint]`. One cell with 100 simulations and 40 games takes about 11 minutes on a desktop CPU. Results for 1, 8, 32, 100 and 400 simulations are in `results/sign_experiment.csv`.

## Reproducing the self-play experiments

Each Gumbel experiment has its own runner script in `scripts/`. The mapping to the thesis experiments and result files is in `scripts/README.md`. A run needs a GPU and several days of compute.

Paths are taken from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `ALPHIGO_PROJECT_DIR` | repository root | where the code lives |
| `ALPHIGO_PYTHON` | `python` | Python interpreter |
| `VIRTUAL_ENV` | `venv` | virtual environment (used to find the NVIDIA libraries) |
| `ALPHIGO_DATA_DIR` | `data` | self-play data |

## External Go engines

GNU Go, Pachi and KataGo are not included. They are used only as evaluation opponents. Download them from:

- GNU Go 3.8: https://www.gnu.org/software/gnugo/ (on Debian/Ubuntu: `apt install gnugo`)
- Pachi 12.88: https://github.com/pasky/pachi
- KataGo: https://github.com/lightvector/KataGo

`gtp_agents/SETUP.md` describes where the code looks for the binaries.

## Data

No game records are included.

- Professional 19×19 games (supervised learning, thesis chapters 4 and 5): the game collection of Andries Brouwer, https://homepages.cwi.nl/~aeb/go/games/
- Human 9×9 games from OGS (supervised baseline and the fixed evaluation set of 10,000 positions): the OGS archive 2013–2021, https://archive.org/details/ogs2021. `scripts/encode_ogs_supervised.py` keeps 9×9 games in which both players are ranked 5 kyu or stronger (OGS rank ≥ 25), without games lost on time and with at least 20 moves.

The Gumbel self-play experiments do not train on human games.

## License

MIT, see `LICENSE`. Parts of the code started from the companion code of *Deep Learning and the Game of Go* (Pumperla and Ferguson, 2019). `NOTICE` lists them.
