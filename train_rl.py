#!/usr/bin/env python
"""train_rl.py — automated reinforcement-learning training loop (Ch. 9-10).

Iterates the RL cycle:  self-play → train → evaluate → log → repeat.

Each round:
  1. Load the current agent (or bootstrap from a supervised checkpoint).
  2. Self-play N games, collecting experience.
  3. Train the agent with policy gradient (REINFORCE).
  4. Evaluate the updated agent against the previous version.
  5. If improved (or no regression), promote the updated agent.
  6. Log results to experiments.jsonl for thesis_report integration.

Examples:
    # RL from scratch, 19x19 (default)
    python train_rl.py --encoder oneplane --network small --rounds 20

    # RL from scratch, 9x9
    python train_rl.py --board-size 9 --encoder oneplane --network small --rounds 20

    # Bootstrap from a supervised checkpoint
    python train_rl.py --bootstrap checkpoints/supervised_oneplane_small_adam_b19_model_epoch_15.h5 \
                       --rounds 20

    # Resume a previous run (auto-detects latest round)
    python train_rl.py --encoder oneplane --network small --resume

    # Evaluate against external opponents too
    python train_rl.py --encoder oneplane --network small --rounds 10 \
                       --eval-opponents gnugo pachi:playouts=500
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from alphigo.training.baseline import REFERENCE_ELOS, evaluate_external
from alphigo.training.run import Run


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

RESULTS_DIR = 'results'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'experiments.jsonl')

METHODS = ('pg', 'q', 'ac', 'alphazero')


# ---------------------------------------------------------------------------
# Deprecation shim for legacy --method names
# ---------------------------------------------------------------------------

_DEPRECATED_METHOD_ALIASES = {
    'policy_gradient': 'pg',
    'actor_critic': 'ac',
}

_REMOVED_METHODS = ('q-dqn', 'q_dqn')


def _resolve_method_arg(method):
    """Translate legacy --method values; raise on q-dqn."""
    if method in _REMOVED_METHODS:
        raise SystemExit(
            f"--method {method!r} was removed in Plan C. "
            f"The Q-DQN variant is no longer supported; use --method q "
            f"for the book-style two-input Q agent.")
    if method in _DEPRECATED_METHOD_ALIASES:
        canonical = _DEPRECATED_METHOD_ALIASES[method]
        print(f'[deprecation] --method {method!r} renamed to {canonical!r}; '
              f'continuing.', file=sys.stderr)
        return canonical
    return method


# ---------------------------------------------------------------------------
# JSONL helpers (shared with thesis_run.py)
# ---------------------------------------------------------------------------

def _load_completed_keys():
    keys = {}
    if not os.path.exists(RESULTS_FILE):
        return keys
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                keys[entry['key']] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def _append_result(result):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')


# ---------------------------------------------------------------------------
# Core RL steps
# ---------------------------------------------------------------------------

def init_agent(run=None, bootstrap_from=None, *, board_size=None,
               encoder_name='oneplane', network_name='small',
               method='pg', lr=1e-4, method_kwargs=None):
    """Create or bootstrap a fresh agent for ``run``, save as round_000.

    Accepts either a ``Run`` instance as the first argument, or the legacy
    keyword arguments (board_size, encoder_name, network_name, method) for
    backwards compatibility with callers that have not yet been updated.
    """
    from alphigo.encoders.base import get_encoder_by_name
    from alphigo.training.method import get_method

    # Legacy call-site compatibility: callers may pass board_size=, encoder_name=, etc.
    if not isinstance(run, Run):
        if board_size is None and run is not None:
            board_size = run  # positional int passed as first arg
        if board_size is None:
            raise ValueError('init_agent requires a Run or board_size=')
        run = Run(method=method, encoder=encoder_name, network=network_name,
                  lr=lr, board_size=board_size)

    agent_path = run.checkpoint_path(0)
    agent_path.parent.mkdir(parents=True, exist_ok=True)

    m = get_method(run.method, **(method_kwargs or {}))

    if bootstrap_from:
        if not m.allows_bootstrap_from_sl:
            raise ValueError(
                f'bootstrap is not supported for {run.method}')
        print(f'[init] Bootstrapping from {bootstrap_from}')
        if run.method == 'alphazero':
            encoder = get_encoder_by_name(run.encoder, run.board_size)
            agent = m.bootstrap_from_sl(bootstrap_from, encoder,
                                        run.board_size)
        else:
            # PG-only branch — keep existing PolicyAgent fallback for SL
            # checkpoints since SL checkpoints are DeepLearningAgent-formatted.
            from alphigo.agent.policy_agent import PolicyAgent, load_policy_agent
            with h5py.File(bootstrap_from, 'r') as h5:
                try:
                    agent = load_policy_agent(h5)
                except Exception:
                    from alphigo.agent.deep_learning import load_prediction_agent
                    pred_agent = load_prediction_agent(h5)
                    agent = PolicyAgent(pred_agent.model, pred_agent.encoder)
    else:
        print(f'[init] Creating fresh {run.method} agent: '
              f'{run.encoder}/{run.network} on '
              f'{run.board_size}x{run.board_size}')
        encoder = get_encoder_by_name(run.encoder, run.board_size)
        agent = m.build_agent(encoder, run.network, run.board_size)

    with h5py.File(agent_path, 'w') as outf:
        agent.serialize(outf)
    print(f'[init] Saved round_000 to {agent_path}')
    return str(agent_path)


def run_self_play(learner_path, reference_path, board_size, num_games,
                  epsilon, experience_path, method, batch_size=64,
                  method_kwargs=None):
    """Learner plays against reference, experience collected for learner only."""
    os.makedirs(os.path.dirname(experience_path), exist_ok=True)

    from alphigo.training.method import get_method
    m = get_method(method, **(method_kwargs or {}))
    learner = m.load_agent(learner_path)
    reference = m.load_agent(reference_path)
    experience = m.self_play(learner, reference, board_size, num_games,
                             epsilon, batch_size=batch_size)

    with h5py.File(experience_path, 'w') as outf:
        experience.serialize(outf)

    num_decisions = len(experience.rewards)
    print(f'  [self-play] {num_decisions} decisions saved to {experience_path}')
    return num_decisions


def train_from_experience(agent_path, experience_path, output_path,
                          lr, clipnorm, batch_size, method,
                          method_kwargs=None):
    """Load agent + experience, train one epoch, save updated agent."""
    from alphigo.training.method import get_method

    m = get_method(method, **(method_kwargs or {}))
    agent = m.load_agent(agent_path)
    if hasattr(m, 'load_experience'):
        exp = m.load_experience(experience_path)
    else:
        from alphigo.rl import load_experience
        with h5py.File(experience_path, 'r') as h5:
            exp = load_experience(h5)
    print(f'  [train] {exp.states.shape[0]} samples, '
          f'lr={lr}, clipnorm={clipnorm}, bs={batch_size}')
    m.train(agent, exp, lr=lr, clipnorm=clipnorm, batch_size=batch_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as outf:
        agent.serialize(outf)
    print(f'  [train] Saved updated agent to {output_path}')


def evaluate_agents(agent_path_new, agent_path_old, board_size, num_games,
                    method, method_kwargs=None):
    """Pit the new agent against the old one. Returns win rate of new agent."""
    from alphigo.training.method import get_method
    m = get_method(method, **(method_kwargs or {}))
    agent_new = m.load_agent(agent_path_new)
    agent_old = m.load_agent(agent_path_old)
    win_rate, summary = m.evaluate(agent_new, agent_old, board_size, num_games)
    print(f'  [eval] new vs best: {summary["a_wins"]}/{summary["total_games"]}'
          f' ({win_rate:.1%})')
    return win_rate, summary


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_rl_loop(
    run, num_rounds, games_per_round,
    clipnorm, batch_size, epsilon, eval_games, eval_opponents,
    bootstrap_from, min_win_rate, method_kwargs=None,
    early_stop_rounds=None, max_promotions=None, max_wall_hours=None,
):
    bootstrap_path = bootstrap_from
    """Run the full RL training loop (book-style, Ch. 10).

    All path / key generation is delegated to ``run`` (a ``Run`` instance).
    """
    # Determine starting round
    latest = run.latest_round()
    if latest > 0 and not bootstrap_from:
        print(f'[resume] Found existing run at round {latest}')
    else:
        init_agent(run, bootstrap_from, method_kwargs=method_kwargs)

    end_round = num_rounds

    # Reconstruct state from JSONL: find the reference (last promoted)
    # and the learning agent (latest round's checkpoint)
    completed_keys = _load_completed_keys()
    reference_agent = str(run.checkpoint_path(0))
    reference_round = 0
    learning_agent = reference_agent

    for r in range(1, end_round + 1):
        rk = run.jsonl_key(r)
        entry = completed_keys.get(rk)
        if not entry:
            break
        ckpt = entry.get('checkpoint_path', '')
        if ckpt and os.path.exists(ckpt):
            learning_agent = ckpt
            if entry.get('promoted', False):
                reference_agent = ckpt
                reference_round = r

    print(f'\n{"=" * 60}')
    print(f'RL training [{run.method}]: {run.encoder}/{run.network} '
          f'on {run.board_size}x{run.board_size}')
    print(f'Rounds 1–{end_round}  '
          f'({games_per_round} games/round, epsilon={epsilon})')
    print(f'lr={run.lr}  clipnorm={clipnorm}  batch_size={batch_size}')
    print(f'Promotion threshold: {min_win_rate:.0%}')
    print(f'Reference: round_{reference_round:03d}')
    if early_stop_rounds is not None:
        print(f'Early stop: abort after {early_stop_rounds} rounds without promotion')
    if max_promotions is not None:
        print(f'Stop after: {max_promotions} promotions')
    if max_wall_hours is not None:
        print(f'Wall-time cap: {max_wall_hours} h')
    if eval_opponents:
        print(f'External eval opponents: {eval_opponents}')
    print(f'Run dir: {run.dir}')
    print(f'{"=" * 60}\n')

    loop_t0 = time.time()
    promotions_this_session = 0

    rounds_since_promotion = 0
    if reference_round > 0:
        # `learning_agent`'s round = whatever we resumed at (== latest in the
        # JSONL); rounds since promotion = (last completed round) - reference_round.
        last_completed = 0
        for r in range(end_round, 0, -1):
            if run.jsonl_key(r) in completed_keys:
                last_completed = r
                break
        rounds_since_promotion = max(0, last_completed - reference_round)

    for round_num in range(1, end_round + 1):
        round_key = run.jsonl_key(round_num)

        if round_key in completed_keys:
            entry = completed_keys[round_key]
            ckpt = entry.get('checkpoint_path', '')
            if ckpt and os.path.exists(ckpt):
                print(f'[round {round_num}] SKIP (already in experiments.jsonl)')
                continue
            else:
                print(f'[round {round_num}] REDO (checkpoint missing: {ckpt})')

        round_start = time.time()
        new_agent = str(run.checkpoint_path(round_num))
        exp_path = str(run.experience_path(round_num))

        print(f'\n{"─" * 60}')
        print(f'  ROUND {round_num}/{end_round}  '
              f'(learner=round_{round_num - 1:03d}, '
              f'ref=round_{reference_round:03d})')
        print(f'{"─" * 60}')

        # Step 1: Self-play — learning agent vs reference agent
        print(f'\n  ── Self-play ({games_per_round} games) ──')
        t0 = time.time()
        num_decisions = run_self_play(
            learning_agent, reference_agent, run.board_size,
            games_per_round, epsilon, exp_path, method=run.method,
            method_kwargs=method_kwargs)
        self_play_time = time.time() - t0

        # Step 2: Train from the learning agent (always moves forward)
        print(f'\n  ── Training ──')
        t0 = time.time()
        train_from_experience(
            learning_agent, exp_path, new_agent, run.lr, clipnorm,
            batch_size, method=run.method, method_kwargs=method_kwargs)
        train_time = time.time() - t0

        learning_agent = new_agent

        # Step 3: Evaluate learning agent vs reference
        print(f'\n  ── Evaluation (learner vs ref, {eval_games} games) ──')
        t0 = time.time()
        win_rate, vs_summary = evaluate_agents(
            new_agent, reference_agent, run.board_size, eval_games,
            method=run.method, method_kwargs=method_kwargs)
        eval_time = time.time() - t0

        # Step 4: External evaluation (if any)
        external_eval = {}
        if eval_opponents:
            print(f'\n  ── External evaluation ──')
            try:
                external_eval = evaluate_external(
                    new_agent, run.board_size, eval_opponents, eval_games)
            except Exception as e:
                print(f'  [eval] External eval failed: {e}')

        # Step 5: Promote to reference if above threshold
        promoted = win_rate >= min_win_rate
        if promoted:
            reference_agent = new_agent
            reference_round = round_num
            rounds_since_promotion = 0
            promotions_this_session += 1
            print(f'  [OK] New reference! Win rate {win_rate:.1%} >= '
                  f'{min_win_rate:.1%}  '
                  f'(promotion {promotions_this_session}'
                  + (f'/{max_promotions}' if max_promotions else '')
                  + ')')
        else:
            rounds_since_promotion += 1
            print(f'  [--] Not promoted ({win_rate:.1%} < {min_win_rate:.1%}). '
                  f'Learner keeps going, ref stays round_{reference_round:03d}'
                  f'  ({rounds_since_promotion} rounds without promotion)')

        round_time = time.time() - round_start

        # Step 6: Log
        result = {
            'key': round_key,
            'board_size': run.board_size,
            'encoder': run.encoder,
            'network': run.network,
            'optimizer': f'sgd_lr{run.lr}',
            'training_method': run.method,
            'rl_round': round_num,
            'timestamp': datetime.now(timezone.utc).strftime(
                '%Y-%m-%dT%H:%M:%S'),
            'games_per_round': games_per_round,
            'num_decisions': num_decisions,
            'epsilon': epsilon,
            'lr': run.lr,
            'clipnorm': clipnorm,
            'batch_size': batch_size,
            'self_play_time_seconds': round(self_play_time, 1),
            'training_time_seconds': round(train_time, 1),
            'eval_time_seconds': round(eval_time, 1),
            'total_round_time_seconds': round(round_time, 1),
            'vs_ref_win_rate': round(win_rate, 4),
            'vs_ref_wins': vs_summary.get('a_wins', 0),
            'vs_ref_games': vs_summary.get('total_games', 0),
            'promoted': promoted,
            'reference_round': reference_round,
            'checkpoint_path': new_agent,
            'eval': external_eval,
            'bootstrap': bootstrap_path or None,
        }
        _append_result(result)

        print(f'\n  Round {round_num} done in {round_time:.0f}s  '
              f'(self-play {self_play_time:.0f}s, train {train_time:.0f}s, '
              f'eval {eval_time:.0f}s)')
        print(f'  vs ref: {win_rate:.1%}  promoted={promoted}')
        if external_eval:
            for opp, res in external_eval.items():
                print(f'  vs {opp}: {res["win_rate"]:.1%}')

        # Clean up GPU memory between rounds
        gc.collect()
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

        # Step 7: Stop-condition checks
        if (early_stop_rounds is not None
                and rounds_since_promotion >= early_stop_rounds):
            print(f'\n{"=" * 60}')
            print(f'EARLY STOP: {rounds_since_promotion} rounds without promotion '
                  f'(threshold {early_stop_rounds}) at round {round_num}.')
            print(f'Last promoted reference: round_{reference_round:03d}')
            print(f'{"=" * 60}')
            break

        if (max_promotions is not None
                and promotions_this_session >= max_promotions):
            print(f'\n{"=" * 60}')
            print(f'STOP: reached --max-promotions {max_promotions} '
                  f'at round {round_num}.')
            print(f'Last promoted reference: round_{reference_round:03d}')
            print(f'{"=" * 60}')
            break

        if max_wall_hours is not None:
            elapsed_h = (time.time() - loop_t0) / 3600.0
            if elapsed_h >= max_wall_hours:
                print(f'\n{"=" * 60}')
                print(f'STOP: wall-time cap reached '
                      f'({elapsed_h:.1f}h >= {max_wall_hours}h) '
                      f'at round {round_num}.')
                print(f'Last promoted reference: round_{reference_round:03d}')
                print(f'{"=" * 60}')
                break

    print(f'\n{"=" * 60}')
    print(f'RL training complete: {num_rounds} rounds')
    print(f'Latest agent: {run.checkpoint_path(end_round)}')
    print(f'Reference agent: {reference_agent}')
    print(f'{"=" * 60}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Automated RL training loop (policy gradient | Q | AC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Agent configuration
    parser.add_argument('--board-size', type=int, default=19,
                        help='Board size (default: 19)')
    parser.add_argument('--encoder', default='oneplane',
                        help='Board encoder (default: oneplane)')
    parser.add_argument('--network', default='small',
                        help='Network architecture (default: small)')
    parser.add_argument('--bootstrap', default=None, metavar='CHECKPOINT',
                        help='Bootstrap from a supervised checkpoint .h5 '
                             '(pg only)')
    parser.add_argument(
        '--method',
        # Accept legacy long names + q-dqn so the deprecation shim can
        # emit a clear error/translation. Validation happens after parse.
        default='pg',
        help='Training method: pg (REINFORCE) | q (book Ch. 11 two-input) | '
             'ac (book Ch. 12 actor-critic). Default: pg. Legacy values '
             'policy_gradient/actor_critic are translated; q-dqn is removed.')

    # RL loop parameters
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of RL rounds (default: 10)')
    parser.add_argument('--games', type=int, default=500,
                        help='Self-play games per round (default: 500)')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Exploration rate. Default: 0.0 for pg, '
                             '0.1 for q')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--clipnorm', type=float, default=1.0,
                        help='Gradient clip norm (default: 1.0)')
    parser.add_argument('--bs', type=int, default=512,
                        help='Mini-batch size (default: 512)')

    # Evaluation
    parser.add_argument('--eval-games', type=int, default=50)
    parser.add_argument('--eval-opponents', nargs='*', default=None)
    parser.add_argument('--min-win-rate', type=float, default=0.6)

    # AlphaZero-only knobs
    parser.add_argument('--mcts-simulations', type=int, default=50,
                        help='AlphaZero PUCT simulations per move '
                             '(default: 50; only used by --method alphazero)')
    parser.add_argument('--c-puct', type=float, default=1.5,
                        help='AlphaZero PUCT exploration constant '
                             '(default: 1.5; alphazero only)')
    parser.add_argument('--temperature-moves', type=int, default=None,
                        help='Plies of T=1 sampling at game start '
                             '(default: 10 for 9x9, 30 for 19x19; alphazero only)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='AlphaZero self-play worker processes '
                             '(default: 1; book parallel_self_play pattern). '
                             'Each worker reloads the learner from a temp '
                             '.h5 each round and plays a slice of games '
                             'with its own numpy RNG seed.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Run seed (default: 0 = legacy unseeded). '
                             'Non-zero values isolate per-seed checkpoints '
                             'in runs/{slug}_seed{N}/ for variance studies.')
    parser.add_argument('--early-stop-rounds', type=int, default=None,
                        metavar='N',
                        help='Abort the loop after N consecutive rounds '
                             'without a promotion (default: disabled). '
                             'Resumed runs count prior non-promoting rounds '
                             'toward the budget.')
    parser.add_argument('--max-promotions', type=int, default=None,
                        metavar='N',
                        help='Stop after N reference promotions this session '
                             '(default: disabled). Matches the book\'s '
                             'promote-then-repeat loop.')
    parser.add_argument('--max-wall-hours', type=float, default=None,
                        metavar='H',
                        help='Stop after H wall-clock hours of this session '
                             '(default: disabled). Checked at end of each '
                             'round.')

    args = parser.parse_args()

    method = _resolve_method_arg(args.method)
    if method not in METHODS:
        parser.error(
            f'--method {method!r} is not one of {METHODS}')

    if args.epsilon is None:
        args.epsilon = 0.1 if method == 'q' else 0.0

    if args.bootstrap and method not in ('pg', 'alphazero'):
        parser.error('--bootstrap is only supported for --method pg or alphazero')

    run = Run(method=method, encoder=args.encoder, network=args.network,
              lr=args.lr, board_size=args.board_size, seed=args.seed)

    if args.seed != 0:
        # Make seed runs actually diverge: numpy controls move sampling
        # in pg/q/ac/zero agents; tf controls weight init.
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import tensorflow as _tf
            _tf.random.set_seed(args.seed)
        except Exception:
            pass

    method_kwargs = {}
    if method == 'alphazero':
        method_kwargs['num_simulations'] = args.mcts_simulations
        method_kwargs['c_puct'] = args.c_puct
        if args.temperature_moves is not None:
            method_kwargs['temperature_moves'] = args.temperature_moves
        method_kwargs['num_workers'] = args.num_workers
        # Use --seed (the run-level seed) as the base for per-worker
        # seeding so independent runs get independent worker RNG streams.
        method_kwargs['base_seed'] = args.seed

    run_rl_loop(
        run=run,
        num_rounds=args.rounds,
        games_per_round=args.games,
        clipnorm=args.clipnorm,
        batch_size=args.bs,
        epsilon=args.epsilon,
        eval_games=args.eval_games,
        eval_opponents=args.eval_opponents or [],
        bootstrap_from=args.bootstrap,
        min_win_rate=args.min_win_rate,
        method_kwargs=method_kwargs,
        early_stop_rounds=args.early_stop_rounds,
        max_promotions=args.max_promotions,
        max_wall_hours=args.max_wall_hours,
    )


if __name__ == '__main__':
    main()
