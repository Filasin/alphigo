"""Self-play simulation for reinforcement learning.

Provides ``simulate_game`` (one full game between two agents) and
``experience_simulation`` (N games with color swapping, returning a
combined ``ExperienceBuffer``).

The batched helpers that previously lived here (BatchedSimulator,
batched_self_play{,_q,_ac}, batched_eval{,_q,_ac}) were removed in
Plan C — train_rl.py now routes self-play through Method.self_play and
the new BatchedSimulator in alphigo.training.batched_board.
"""

import multiprocessing
import os
import tempfile

import numpy as np

from alphigo import rl
from alphigo.alphigo_board import GameState
from alphigo.alphigo_types import Player
from alphigo.rl.zero_experience import (
    ZeroReplayBuffer, combine_zero_buffers, load_zero_buffer)


MAX_MOVES_DEFAULT = 2000


# Default temperature schedule: T=1 for the first N moves of a 9x9 game
# (encourages exploration of the opening), T=0 thereafter.
DEFAULT_TEMPERATURE_MOVES_9 = 10


def simulate_game(black_player, white_player, board_size,
                  max_moves=MAX_MOVES_DEFAULT):
    """Play one full game between two agents.

    Returns the winner (Player.black or Player.white).  If *max_moves*
    is reached without the game ending naturally, area scoring is used.
    """
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    num_moves = 0
    while not game.is_over():
        if num_moves >= max_moves:
            break
        move = agents[game.next_player].select_move(game)
        game = game.apply_move(move)
        num_moves += 1

    return game.winner()


def experience_simulation(num_games, agent1, agent2, board_size,
                          max_moves=MAX_MOVES_DEFAULT, verbose=True):
    """Run *num_games* self-play games with color swapping.

    Both agents share the experience via separate collectors so that
    wins and losses are recorded from both perspectives.

    Returns a combined ``ExperienceBuffer``.
    """
    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()

    color1 = Player.black

    for i in range(num_games):
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            black_player, white_player = agent2, agent1

        winner = simulate_game(black_player, white_player, board_size,
                               max_moves)

        if winner == color1:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)

        color1 = color1.other

        if verbose and ((i + 1) % 50 == 0 or (i + 1) == num_games):
            print(f'  [self-play] {i + 1}/{num_games} games done')

    return rl.combine_experience([collector1, collector2])


# ── AlphaZero self-play (issue 0003) ──────────────────────────────────────

def batched_self_play_zero(agent, board_size, num_games, num_simulations=None,
                           temperature_moves=None, max_moves=MAX_MOVES_DEFAULT,
                           verbose=True):
    """Run AlphaZero self-play and return a ``ZeroReplayBuffer``.

    A single ZeroAgent plays both colors. For each move we record the
    encoded state from the current player's perspective, the visit-count
    policy target, and (after the game ends) the terminal value from
    that mover's perspective (+1 win / -1 loss / 0 max-moves draw).

    Temperature schedule: T=1 for the first ``temperature_moves`` plies
    of each game (encourages exploration), T=0 for the remainder. This
    is set on the agent per move via ``set_temperature``.
    """
    if num_simulations is not None:
        agent._num_simulations = int(num_simulations)
    if temperature_moves is None:
        temperature_moves = (DEFAULT_TEMPERATURE_MOVES_9
                             if board_size <= 9 else 30)
    temperature_moves = int(temperature_moves)

    states_all, policies_all, players_all = [], [], []
    rewards_all = []  # final reward written once we know the winner

    for g in range(num_games):
        game = GameState.new_game(board_size)
        game_states, game_policies, game_players = [], [], []
        ply = 0
        while not game.is_over() and ply < max_moves:
            agent.set_temperature(1.0 if ply < temperature_moves else 0.0)
            result = agent.search(game)
            policy_target = np.asarray(result['policy_target'],
                                       dtype=np.float32)
            # Skip recording forced-pass positions: no legal non-eye-filling
            # children means the visit-count distribution is all zeros and
            # carries no training signal.
            if policy_target.sum() > 0:
                game_states.append(agent._encoder.encode(game))
                game_policies.append(policy_target)
                game_players.append(game.next_player)
            game = game.apply_move(result['move'])
            ply += 1

        winner = game.winner() if game.is_over() else None

        for s, p, mover in zip(game_states, game_policies, game_players):
            states_all.append(s)
            policies_all.append(p)
            if winner is None:
                rewards_all.append(0.0)
            else:
                rewards_all.append(1.0 if mover == winner else -1.0)

        if verbose and ((g + 1) % 50 == 0 or (g + 1) == num_games):
            print(f'  [zero self-play] {g + 1}/{num_games} games done')

    if not states_all:
        return ZeroReplayBuffer(
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return ZeroReplayBuffer(
        encoded_states=np.stack(states_all).astype(np.float32),
        policy_targets=np.stack(policies_all).astype(np.float32),
        value_targets=np.asarray(rewards_all, dtype=np.float32),
    )


# ── Multiprocess AZ self-play (adapted from parallel_self_play in the
#    companion code of Pumperla and Ferguson, Deep Learning and the Game of Go,
#    https://github.com/maxpumperla/deep_learning_and_the_game_of_go;
#    pattern: N worker processes each play a slice of games independently,
#    write their ZeroReplayBuffer to a temp file, main combines them) ─────

def _az_self_play_worker(worker_id, agent_path, board_size, num_games,
                         num_simulations, temperature_moves, seed,
                         output_path):
    """Run AZ self-play in a worker process and write the buffer to disk.

    Critical: numpy seed is set BEFORE the agent/model is loaded so that
    move sampling diverges across workers. The agent itself is freshly
    loaded from ``agent_path`` so the parent process's TF state is not
    inherited (avoiding CUDA fork issues; this also lets ``spawn`` work).
    """
    # Per-worker RNG isolation. Without this, all workers sample the same
    # moves from the same policy and produce identical games.
    np.random.seed(seed)
    import random as _stdrand
    _stdrand.seed(seed)

    # Allow TF to grow GPU memory rather than grabbing all of it; otherwise
    # the first worker would OOM the rest.
    import tensorflow as tf
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    import h5py
    from alphigo.agent.zero_agent import load_zero_agent

    with h5py.File(agent_path, 'r') as h5:
        agent = load_zero_agent(h5)
    if num_simulations is not None:
        agent._num_simulations = int(num_simulations)

    buf = batched_self_play_zero(
        agent, board_size, num_games,
        num_simulations=num_simulations,
        temperature_moves=temperature_moves,
        verbose=False,
    )

    with h5py.File(output_path, 'w') as h5:
        buf.serialize(h5)
        # Stamp metadata for the diversity-verification script.
        h5.attrs['worker_id'] = int(worker_id)
        h5.attrs['seed'] = int(seed)
        h5.attrs['num_games'] = int(num_games)


def multiprocess_self_play_zero(agent_path, board_size, num_games,
                                num_workers, num_simulations=None,
                                temperature_moves=None, base_seed=0,
                                tmpdir=None):
    """Run AZ self-play across ``num_workers`` processes; return combined buffer.

    The learner is serialized to ``agent_path`` *before* this is called.
    Games are split as evenly as possible across workers; each worker
    gets a distinct numpy seed (``base_seed * 1000 + worker_id + 1``)
    so that even with identical model weights, sampled trajectories
    diverge. Buffers are written to temp h5 files and combined in the
    main process; temp files are deleted before return.
    """
    if num_workers <= 1:
        import h5py
        from alphigo.agent.zero_agent import load_zero_agent
        with h5py.File(agent_path, 'r') as h5:
            agent = load_zero_agent(h5)
        if num_simulations is not None:
            agent._num_simulations = int(num_simulations)
        return batched_self_play_zero(
            agent, board_size, num_games,
            num_simulations=num_simulations,
            temperature_moves=temperature_moves,
            verbose=False,
        )

    games_per_worker = [num_games // num_workers] * num_workers
    for i in range(num_games % num_workers):
        games_per_worker[i] += 1

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix='az_mp_selfplay_')
    output_paths = [os.path.join(tmpdir, f'worker_{i}.h5')
                    for i in range(num_workers)]

    # ``spawn`` (not ``fork``) so TF/CUDA state from the parent is not
    # inherited; required for clean GPU init in each worker.
    ctx = multiprocessing.get_context('spawn')
    workers = []
    for i in range(num_workers):
        if games_per_worker[i] == 0:
            continue
        # +1 so base_seed=0 still produces non-zero per-worker seeds; *1000
        # leaves headroom for the inner seeding logic.
        seed = base_seed * 1000 + i + 1
        p = ctx.Process(
            target=_az_self_play_worker,
            args=(i, agent_path, board_size, games_per_worker[i],
                  num_simulations, temperature_moves, seed,
                  output_paths[i]),
        )
        p.start()
        workers.append((i, p))

    failures = []
    for i, p in workers:
        p.join()
        if p.exitcode != 0:
            failures.append((i, p.exitcode))
    if failures:
        raise RuntimeError(
            f"AZ self-play workers failed: {failures}")

    import h5py
    buffers = []
    for path in output_paths:
        if os.path.exists(path):
            with h5py.File(path, 'r') as h5:
                buffers.append(load_zero_buffer(h5))
    for path in output_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    return combine_zero_buffers(buffers)


# ── Multiprocess AZ evaluation (analogue of self-play multiprocess path) ──

def _az_eval_worker(worker_id, new_path, old_path, board_size, num_games,
                    num_simulations, eval_temperature_plies, seed,
                    output_path):
    """Worker: play ``num_games`` head-to-head (new vs old) with alternating
    colors and the AGZ temperature schedule (T=1 first N plies, T=0 after).
    Writes JSON ``{a_wins, num_games}`` to ``output_path``.

    Forces CPU (``CUDA_VISIBLE_DEVICES=""`` before TF import) — AZ eval is
    batch-1 sims-bound; CPU is comparable to single-process GPU at batch=1
    and lets us run 4 eval workers without VRAM contention.
    """
    # CPU-only: must come before TF import.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    np.random.seed(seed)
    import random as _stdrand
    _stdrand.seed(seed)

    import tensorflow as tf
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    import json
    import h5py
    from alphigo.agent.zero_agent import load_zero_agent

    with h5py.File(new_path, 'r') as h5:
        new_agent = load_zero_agent(h5)
    with h5py.File(old_path, 'r') as h5:
        old_agent = load_zero_agent(h5)
    new_agent._num_simulations = int(num_simulations)
    old_agent._num_simulations = int(num_simulations)

    def play_one(black_player, white_player):
        game = GameState.new_game(board_size)
        agents = {Player.black: black_player, Player.white: white_player}
        ply = 0
        while not game.is_over() and ply < 400:
            t = 1.0 if ply < eval_temperature_plies else 0.0
            agents[game.next_player].set_temperature(t)
            move = agents[game.next_player].select_move(game)
            game = game.apply_move(move)
            ply += 1
        return game.winner()

    a_wins = 0
    for i in range(num_games):
        if i % 2 == 0:
            if play_one(new_agent, old_agent) == Player.black:
                a_wins += 1
        else:
            if play_one(old_agent, new_agent) == Player.white:
                a_wins += 1

    with open(output_path, 'w') as f:
        json.dump({'worker_id': int(worker_id),
                   'a_wins': int(a_wins),
                   'num_games': int(num_games)}, f)


def multiprocess_evaluate_zero(new_agent_path, old_agent_path, board_size,
                               num_games, num_workers, num_simulations,
                               eval_temperature_plies, base_seed=0,
                               tmpdir=None):
    """Run AZ head-to-head eval across N CPU workers; return (win_rate, summary).

    Games are split as evenly as possible across workers. Each worker plays
    its slice with the same color-alternation pattern as the serial path
    (i%2 == 0 → new is black). Per-worker numpy seed = base_seed*1000 + wid + 1.
    """
    if num_workers <= 1 or num_games <= 0:
        raise ValueError("multiprocess_evaluate_zero requires num_workers > 1 "
                         "and num_games > 0; use serial path otherwise.")

    games_per_worker = [num_games // num_workers] * num_workers
    for i in range(num_games % num_workers):
        games_per_worker[i] += 1

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix='az_mp_eval_')
    output_paths = [os.path.join(tmpdir, f'eval_{i}.json')
                    for i in range(num_workers)]

    ctx = multiprocessing.get_context('spawn')
    workers = []
    for i in range(num_workers):
        if games_per_worker[i] == 0:
            continue
        # AlphaZeroMethod.evaluate already passes a large, run-unique base_seed
        # (base_seed + 10_000*round + 5_000_000). Don't multiply it again —
        # that overflows numpy's uint32 seed cap (~4.29e9) for round >= 1.
        # `+ i + 1` differentiates workers within a round; consecutive eval
        # rounds differ by 10_000 in base_seed which is far larger than any
        # plausible num_workers.
        seed = (base_seed + i + 1) & 0xFFFFFFFF
        p = ctx.Process(
            target=_az_eval_worker,
            args=(i, new_agent_path, old_agent_path, board_size,
                  games_per_worker[i], num_simulations,
                  eval_temperature_plies, seed, output_paths[i]),
        )
        p.start()
        workers.append((i, p))

    failures = []
    for i, p in workers:
        p.join()
        if p.exitcode != 0:
            failures.append((i, p.exitcode))
    if failures:
        raise RuntimeError(f"AZ eval workers failed: {failures}")

    import json
    total_wins, total_games = 0, 0
    for path in output_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        total_wins += d['a_wins']
        total_games += d['num_games']
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    win_rate = total_wins / max(total_games, 1)
    return win_rate, {'a_wins': total_wins, 'total_games': total_games}
