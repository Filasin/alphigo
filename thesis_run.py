#!/usr/bin/env python
"""thesis_run.py — unified experiment runner for the AlphiGo thesis.

Handles the full pipeline: data generation, training, evaluation, and reporting.
Every experiment is identified by a unique key and results are stored append-safe
in results/experiments.jsonl. Completed runs are skipped unless --force is used.

Examples:
    # Full pipeline for 9x9
    python thesis_run.py --board-size 9 --phase all

    # Only train, skip eval
    python thesis_run.py --board-size 9 --phase train

    # Re-run a specific combo
    python thesis_run.py --board-size 9 --encoder sevenplane --network medium --force

    # Dry run — show what would run
    python thesis_run.py --board-size 9 --dry-run

    # Report only
    python thesis_run.py --report

    # RL training methods
    python thesis_run.py --board-size 9 --training-method pg --phase train
    python thesis_run.py --board-size 9 --training-method q --phase train \\
                         --rl-rounds 10 --rl-games 500
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from alphigo.training.baseline import REFERENCE_ELOS

# ---------------------------------------------------------------------------
# Defaults and constants
# ---------------------------------------------------------------------------

RESULTS_DIR = 'results'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'experiments.jsonl')
CHECKPOINT_DIR = 'checkpoints'

NPY_DATA_DIR_9 = 'generated_games/mcst_9x9'
SGF_DATA_DIR = 'data'

# Experiment matrices per board size
MATRIX_9 = {
    'encoders': ['oneplane', 'sevenplane', 'elevenplane', 'nineplane', 'agplane'],
    'networks': ['small', 'medium', 'large', 'small_orig', 'medium_orig', 'large_orig'],
    'special': [{'encoder': 'alphazero', 'network': 'alphazero'}],
}

MATRIX_19 = {
    'encoders': ['oneplane', 'sevenplane', 'elevenplane', 'agplane'],
    'networks': ['small', 'medium', 'large'],
    'special': [{'encoder': 'alphazero', 'network': 'alphazero'}],
}

# Encoders that require Python re-encoding from game records (no C++ support)
PYTHON_ONLY_ENCODERS = {'nineplane', 'agplane', 'alphazero'}

# Encoders produced by the C++ generator
CPP_ENCODERS = {'oneplane', 'sevenplane', 'elevenplane'}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_completed_keys():
    """Load all completed experiment keys from experiments.jsonl."""
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


def make_key(board_size, encoder, network, optimizer, training_method):
    return f'{board_size}x{board_size}__{encoder}__{network}__{optimizer}__{training_method}'


def append_result(result):
    """Append a result dict as one JSON line to experiments.jsonl."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(tag, encoder, network, optimizer, board_size, epoch):
    name = f'{tag}_{encoder}_{network}_{optimizer}_b{board_size}_model_epoch_{epoch}.h5'
    return os.path.join(CHECKPOINT_DIR, name)


def latest_checkpoint(tag, encoder, network, optimizer, board_size):
    """Find the highest-epoch checkpoint for this combo, or None."""
    pattern = f'{tag}_{encoder}_{network}_{optimizer}_b{board_size}_model_epoch_*.h5'
    matches = sorted(Path(CHECKPOINT_DIR).glob(pattern),
                     key=lambda p: int(p.stem.rsplit('_', 1)[-1]))
    return str(matches[-1]) if matches else None


# ---------------------------------------------------------------------------
# Build experiment combos
# ---------------------------------------------------------------------------

def build_combos(board_size, encoders_filter, networks_filter, optimizer,
                 training_method):
    """Build list of (encoder, network) tuples for the given board size."""
    matrix = MATRIX_9 if board_size == 9 else MATRIX_19

    encoders = encoders_filter if encoders_filter else matrix['encoders']
    networks = networks_filter if networks_filter else matrix['networks']

    combos = []

    for enc in encoders:
        for net in networks:
            combos.append((enc, net))

    # Add special combos (alphazero) if not filtered out
    for special in matrix.get('special', []):
        enc, net = special['encoder'], special['network']
        if encoders_filter and enc not in encoders_filter:
            continue
        if networks_filter and net not in networks_filter:
            continue
        if (enc, net) not in combos:
            combos.append((enc, net))

    return combos


# ---------------------------------------------------------------------------
# Data phase
# ---------------------------------------------------------------------------

def phase_data(board_size, tag='mcts'):
    """Ensure training data exists for the given board size."""
    if board_size == 9:
        _ensure_9x9_data(tag)
    else:
        _ensure_19x19_data()


def _ensure_9x9_data(tag='mcts'):
    """Check and generate 9x9 MCTS self-play data if needed."""
    data_dir = NPY_DATA_DIR_9
    os.makedirs(data_dir, exist_ok=True)

    labels_train = os.path.join(data_dir, f'{tag}_labels_train.npy')

    # Step 1: Check if C++ generation + merge is needed
    if not os.path.exists(labels_train):
        # Check for batch files that need merging
        batch_files = list(Path(data_dir).glob(f'{tag}_b*_labels.npy'))
        if not batch_files:
            # Need to run C++ generator
            print('[data] No MCTS data found. Running C++ generator...')
            print('[data] Looking for generate_mcst_games executable...')

            exe = None
            for candidate in ['generate_mcst_games.exe', './generate_mcst_games.exe',
                               'generate_mcst_games', './generate_mcst_games']:
                if os.path.exists(candidate):
                    exe = candidate
                    break

            if exe is None:
                print('[data] ERROR: generate_mcst_games executable not found.')
                print('[data] Compile it first:')
                print('  cl /O2 /std:c++17 /EHsc generate_mcst_games.cpp /Fe:generate_mcst_games.exe')
                print('  g++ -O3 -std=c++17 -pthread -o generate_mcst_games generate_mcst_games.cpp')
                sys.exit(1)

            cmd = [
                exe,
                '--output-dir', data_dir,
                '--games', '10000',
                '--board_size', '9',
                '--workers', '4',
                '--rounds', '500',
                '-e', 'oneplane,sevenplane,elevenplane',
                '--tag', tag,
            ]
            print(f'[data] {" ".join(cmd)}')
            subprocess.run(cmd, check=True)

        # Step 2: Merge batches
        print('[data] Merging batches...')
        sys.path.insert(0, 'generated_games')
        from generated_games.merge_batches import merge
        merge(data_dir, tag=tag, split=0.8, seed=42)

    # Step 3: Encode records for Python-only encoders
    records_file = os.path.join(data_dir, f'{tag}_game_records.bin')
    missing_python_encoders = []
    for enc in PYTHON_ONLY_ENCODERS:
        feat_train = os.path.join(data_dir, f'{tag}_{enc}_features_train.npy')
        if not os.path.exists(feat_train):
            missing_python_encoders.append(enc)

    if missing_python_encoders:
        if not os.path.exists(records_file):
            print(f'[data] WARNING: Game records file not found: {records_file}')
            print(f'[data] Python-only encoders ({", ".join(missing_python_encoders)}) '
                  f'cannot be encoded without game records.')
            print(f'[data] Re-generate data with the updated C++ generator.')
            return

        print(f'[data] Encoding records for: {", ".join(missing_python_encoders)}')
        from encode_records import read_game_records, replay_and_encode
        import numpy as np

        games = read_game_records(records_file)
        feature_arrays, label_array = replay_and_encode(games, 9, missing_python_encoders)
        total = len(label_array)

        if total > 0:
            rng = np.random.default_rng(42)
            idx = rng.permutation(total)
            cut = int(0.8 * total)

            for enc_name, features in feature_arrays.items():
                features = features[idx]
                np.save(os.path.join(data_dir, f'{tag}_{enc_name}_features_train.npy'),
                        features[:cut])
                np.save(os.path.join(data_dir, f'{tag}_{enc_name}_features_test.npy'),
                        features[cut:])
                print(f'  [{enc_name}] saved  train={cut}  test={total - cut}')

    print('[data] 9x9 data ready.')


def _ensure_19x19_data():
    """Ensure 19x19 SGF data is extracted (handled by GoDataProcessor)."""
    from alphigo.data.extract import extract_games
    extract_games()
    print('[data] 19x19 SGF data ready.')


def _resolve_train_games(num_games, seed):
    """Draw a fixed training game list once and save to a file.

    Returns the file path.  Reuses the file if it already exists with the
    correct count, so re-running the same --num-games is instant.
    """
    import random as _random
    from alphigo.data.extract import extract_games
    from alphigo.data.sampling import Sampler

    games_file = os.path.join(SGF_DATA_DIR, f'train_games_{num_games}.txt')
    if os.path.exists(games_file):
        with open(games_file) as f:
            games = [line.strip() for line in f if line.strip()]
        if len(games) == num_games:
            print(f'[data] Reusing training game list: {games_file} '
                  f'({len(games)} games)')
            return games_file

    extract_games(output_dir=SGF_DATA_DIR)

    # Test set is 10% of the *training* sample, not 10% of the full corpus.
    # Using the full corpus (93k games) would give ~9k validation games against
    # 15k training games — wasteful and disproportionate.
    num_test = max(100, int(num_games * 0.1))

    # Invalidate the cached test_samples.txt if it has a different count so
    # the Sampler regenerates it with the correct size.
    test_samples_path = os.path.join(SGF_DATA_DIR, 'test_samples.txt')
    if os.path.exists(test_samples_path):
        with open(test_samples_path) as _f:
            _cached = [l.strip() for l in _f if l.strip()]
        if len(_cached) != num_test:
            os.remove(test_samples_path)
            print(f'[data] Resizing test set: {len(_cached)} → {num_test} games '
                  f'(proportional to --num-games {num_games})')

    sampler = Sampler(data_dir=SGF_DATA_DIR, num_test_games=num_test, seed=seed)
    games = sampler.draw_training_samples(num_games)

    with open(games_file, 'w') as f:
        for g in games:
            f.write(g + '\n')
    print(f'[data] Saved training game list: {games_file} ({len(games)} games)')
    return games_file


# ---------------------------------------------------------------------------
# Train phase
# ---------------------------------------------------------------------------

def phase_train(board_size, combos, optimizer, training_method, epochs,
                patience, batch_size, force, tag='supervised', xla=True,
                seed=42, train_games_file=None):
    """Train all combos, skipping completed ones unless --force."""
    completed = load_completed_keys()
    results = []

    total = len(combos)
    for ci, (encoder, network) in enumerate(combos, 1):
        key = make_key(board_size, encoder, network, optimizer, training_method)

        if key in completed and not force:
            entry = completed[key]
            # Backfill new metric fields from stored history if missing
            if 'best_val_loss' not in entry and entry.get('history'):
                entry = _backfill_metrics(entry)
                _update_jsonl_entry(key, entry)
            ts = entry.get('timestamp', '?')
            print(f'  SKIP  [{ci}/{total}] {key}  (completed {ts})')
            results.append(entry)
            continue

        print(f'\n{"─" * 60}')
        print(f'  [{ci}/{total}]  encoder={encoder}  network={network}  '
              f'optimizer={optimizer}')
        print(f'  key: {key}')
        print(f'  epochs={epochs}  patience={patience}  batch_size={batch_size}')
        print(f'{"─" * 60}')
        result = _train_one(board_size, encoder, network, optimizer,
                            training_method, epochs, patience, batch_size,
                            tag, xla=xla, seed=seed,
                            train_games_file=train_games_file)
        if result:
            append_result(result)
            results.append(result)
            t = result['training_time_seconds']
            acc = result['final_val_accuracy']
            params = result['total_params']
            print(f'\n{"─" * 60}')
            print(f'  [{ci}/{total}] DONE  val_acc={acc:.4f}  '
                  f'params={params:,}  time={t:.0f}s')
            print(f'  checkpoint: {result["checkpoint_path"]}')
            print(f'{"─" * 60}')
        else:
            print(f'  [{ci}/{total}] FAIL  {key}')

    return results


def _train_one(board_size, encoder, network, optimizer, training_method,
               epochs, patience, batch_size, tag, xla=True, seed=42,
               train_games_file=None):
    """Train a single encoder/network combo. Returns result dict or None.

    Automatically resumes from the latest checkpoint if one exists on disk.
    """

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # The alphazero ResNet (12 residual blocks, 128 filters on 19×19) needs
    # ~722 MiB per layer for BatchNorm fp32 casts at large batch sizes.
    # Cap to 1024 to stay within 24 GB VRAM.
    if board_size >= 19 and network == 'alphazero' and batch_size > 1024:
        print(f'    [mem] Capping batch_size {batch_size} -> 1024 '
              f'(alphazero ResNet on {board_size}x{board_size})')
        batch_size = 1024

    # Check for an existing checkpoint to resume from
    resume_from = latest_checkpoint(tag, encoder, network, optimizer, board_size)

    # Import TrainingSession (avoids top-level TF import for --dry-run)
    from train import TrainingSession

    start_time = time.time()

    try:
        if board_size == 9 and training_method == 'supervised':
            data_dir = NPY_DATA_DIR_9
            data_tag = 'mcts'

            feat_train = os.path.join(data_dir, f'{data_tag}_{encoder}_features_train.npy')
            lab_train = os.path.join(data_dir, f'{data_tag}_labels_train.npy')
            feat_test = os.path.join(data_dir, f'{data_tag}_{encoder}_features_test.npy')
            lab_test = os.path.join(data_dir, f'{data_tag}_labels_test.npy')

            for p in [feat_train, lab_train, feat_test, lab_test]:
                if not os.path.exists(p):
                    print(f'    [MISS] {p}')
                    return None

            session = TrainingSession(
                encoder_name=encoder, network=network, optimizer=optimizer,
                board_size=board_size, batch_size=batch_size, epochs=epochs,
                checkpoint_dir=CHECKPOINT_DIR, tag=tag, patience=patience,
                xla=xla, seed=seed,
            )
            trained_model = session.run_npy(
                feat_train, lab_train, feat_test, lab_test,
                resume_from=resume_from,
            )

        elif board_size == 19 and training_method == 'supervised':
            session = TrainingSession(
                encoder_name=encoder, network=network, optimizer=optimizer,
                board_size=board_size, batch_size=batch_size, epochs=epochs,
                checkpoint_dir=CHECKPOINT_DIR, tag=tag, patience=patience,
                xla=xla, seed=seed,
            )
            trained_model = session.run_sgf(
                num_games=0, data_dir=SGF_DATA_DIR,
                train_games_file=train_games_file,
                resume_from=resume_from,
            )

        else:
            print(f'    Training method "{training_method}" for board_size={board_size} '
                  f'not yet implemented.')
            return None

    except Exception as e:
        print(f'    [ERROR] {e}')
        return None

    training_time = time.time() - start_time

    # Get param count from the trained model
    total_params = trained_model.count_params()

    # Collect training history (Keras sets model.history after fit())
    history = {}
    if hasattr(trained_model, 'history') and trained_model.history is not None:
        h = trained_model.history.history
        history = {k: [float(v) for v in vals] for k, vals in h.items()}

    epochs_trained = len(history.get('loss', list(range(epochs))))

    # Find the final checkpoint path
    ckpt = latest_checkpoint(tag, encoder, network, optimizer, board_size)

    # Extract final metrics from history (last element of each series)
    def _last(key):
        vals = history.get(key, [])
        return float(vals[-1]) if vals else None

    # Best val_loss and the epoch it occurred at (1-indexed)
    val_loss_series = history.get('val_loss', [])
    if val_loss_series:
        best_val_loss = float(min(val_loss_series))
        best_epoch = int(val_loss_series.index(min(val_loss_series))) + 1
    else:
        best_val_loss = None
        best_epoch = None

    result = {
        'key': make_key(board_size, encoder, network, optimizer, training_method),
        'board_size': board_size,
        'encoder': encoder,
        'network': network,
        'optimizer': optimizer,
        'training_method': training_method,
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'),
        'epochs_trained': epochs_trained,
        'total_params': total_params,
        'training_time_seconds': round(training_time, 1),
        'history': history,
        'final_train_loss': _last('loss'),
        'final_train_accuracy': _last('accuracy'),
        'final_val_loss': _last('val_loss'),
        'final_val_accuracy': _last('val_accuracy'),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'checkpoint_path': ckpt,
    }

    # Free model, TF session state, and cached XLA/cuDNN kernels so memory
    # doesn't accumulate across the 30+ combos in a full thesis run.
    # Without this, the last combo (alphazero ResNet) can OOM the WSL2 VM.
    del trained_model
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Eval phase
# ---------------------------------------------------------------------------

def phase_eval(board_size, eval_games, eval_opponents, force):
    """Evaluate all trained checkpoints against GTP opponents.

    Incrementally adds new opponents to existing eval data.  Only skips a
    model entirely when *all* requested opponents are already evaluated
    (unless --force is used, which re-evaluates everything).
    """
    from eval_checkpoints import evaluate_checkpoints

    completed = load_completed_keys()

    # Find entries that have checkpoints, determine which opponents are new
    to_eval = []
    for key, entry in completed.items():
        if entry.get('board_size') != board_size:
            continue
        ckpt = entry.get('checkpoint_path')
        if not ckpt or not os.path.exists(ckpt):
            continue

        existing_eval = entry.get('eval', {})
        if force:
            # Re-evaluate all opponents
            needed_opponents = list(eval_opponents)
        else:
            # Only evaluate opponents we haven't seen yet
            needed_opponents = [
                opp for opp in eval_opponents if opp not in existing_eval
            ]

        if not needed_opponents:
            print(f'  SKIP eval  {key}  (all opponents already evaluated)')
            continue

        to_eval.append((key, entry, ckpt, needed_opponents))

    if not to_eval:
        print('[eval] No checkpoints to evaluate.')
        return

    for key, entry, ckpt, needed_opponents in to_eval:
        label = f'{entry.get("encoder", "?")}/{entry.get("network", "?")}'
        print(f'\n  EVAL  {label}  vs  {needed_opponents}')
        try:
            rows, _ = evaluate_checkpoints(
                [ckpt],
                needed_opponents,
                num_games=eval_games,
                board_size=board_size,
                reference_elos=REFERENCE_ELOS,
                verbose=True,
            )
        except Exception as e:
            print(f'    [ERROR] {e}')
            continue

        # Merge new eval results into existing eval dict
        eval_results = entry.get('eval', {})
        for row in rows:
            opp = row['opponent']
            eval_results[opp] = {
                'win_rate': row['win_pct'] / 100.0,
                'elo_delta': row['elo_diff'],
            }

        # Update the entry in the JSONL file by rewriting
        entry['eval'] = eval_results
        _update_jsonl_entry(key, entry)
        print(f'  [OK] eval for {key}: {list(eval_results.keys())}')


def _backfill_metrics(entry):
    """Add best_val_loss / best_epoch / final_train_* to an old JSONL entry."""
    history = entry.get('history', {})

    def _last(key):
        vals = history.get(key, [])
        return float(vals[-1]) if vals else None

    entry.setdefault('final_train_loss', _last('loss'))
    entry.setdefault('final_train_accuracy', _last('accuracy'))
    entry.setdefault('final_val_loss', _last('val_loss'))
    entry.setdefault('final_val_accuracy', _last('val_accuracy'))

    val_loss_series = history.get('val_loss', [])
    if val_loss_series:
        entry.setdefault('best_val_loss', float(min(val_loss_series)))
        entry.setdefault('best_epoch',
                         int(val_loss_series.index(min(val_loss_series))) + 1)
    return entry


def _update_jsonl_entry(key, updated_entry):
    """Rewrite experiments.jsonl, replacing the entry with matching key."""
    if not os.path.exists(RESULTS_FILE):
        append_result(updated_entry)
        return

    lines = []
    found = False
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get('key') == key:
                    lines.append(json.dumps(updated_entry))
                    found = True
                else:
                    lines.append(line)
            except json.JSONDecodeError:
                lines.append(line)

    if not found:
        lines.append(json.dumps(updated_entry))

    with open(RESULTS_FILE, 'w') as f:
        for line in lines:
            f.write(line + '\n')


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(board_size, combos, optimizer, training_method, tag='supervised'):
    """Show SKIP/RESUME/RUN status for each combo."""
    completed = load_completed_keys()
    print(f'\nDry run — board_size={board_size}, training_method={training_method}\n')

    skip_count = 0
    resume_count = 0
    run_count = 0
    for encoder, network in combos:
        key = make_key(board_size, encoder, network, optimizer, training_method)
        if key in completed:
            ts = completed[key].get('timestamp', '?')
            print(f'  SKIP    {key:<55s}  (completed {ts})')
            skip_count += 1
        else:
            ckpt = latest_checkpoint(tag, encoder, network, optimizer, board_size)
            if ckpt:
                epoch = int(Path(ckpt).stem.rsplit('_', 1)[-1])
                print(f'  RESUME  {key:<55s}  (epoch {epoch}, {os.path.basename(ckpt)})')
                resume_count += 1
            else:
                print(f'  RUN     {key}')
                run_count += 1

    print(f'\nTotal: {skip_count} skip, {resume_count} resume, {run_count} fresh, '
          f'{skip_count + resume_count + run_count} combos')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Unified thesis experiment runner for AlphiGo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--board-size', type=int, default=9, choices=[9, 19],
                        help='Board size (default: 9)')
    parser.add_argument('--phase', default='all',
                        choices=['all', 'data', 'train', 'eval', 'report'],
                        help='Pipeline phase (default: all)')
    parser.add_argument('--encoders', '--encoder', nargs='+', default=None,
                        help='Filter to specific encoders')
    parser.add_argument('--networks', '--network', nargs='+', default=None,
                        help='Filter to specific networks')
    parser.add_argument('--optimizer', default='adam',
                        help='Optimizer (default: adam)')
    parser.add_argument(
        '--training-method',
        choices=['supervised', 'pg', 'q', 'ac', 'alphazero',
                 'policy_gradient', 'actor_critic'],
        default='supervised',
        help='Training method: supervised | pg (REINFORCE) | '
             'q (book Ch. 11 two-input) | ac (book Ch. 12 actor-critic). '
             'Legacy names policy_gradient/actor_critic are translated; '
             'q-dqn was removed in Plan C.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs (default: 20)')

    # RL-specific options (used when --training-method is a RL method)
    parser.add_argument('--rl-rounds', type=int, default=10,
                        help='RL rounds (default: 10)')
    parser.add_argument('--rl-games', type=int, default=500,
                        help='Self-play games per RL round (default: 500)')
    parser.add_argument('--rl-lr', type=float, default=0.0001,
                        help='RL learning rate (default: 0.0001)')
    parser.add_argument('--rl-epsilon', type=float, default=None,
                        help='RL exploration rate — probability of a '
                             'random move. Default: 0.0 for pg/ac, '
                             '0.1 for q')
    parser.add_argument('--rl-bootstrap', default=None, metavar='CHECKPOINT',
                        help='Bootstrap RL from a supervised checkpoint '
                             '(pg or alphazero)')
    parser.add_argument('--rl-mcts-simulations', type=int, default=50,
                        help='AlphaZero PUCT simulations per move '
                             '(default: 50; only used by --training-method alphazero)')
    parser.add_argument('--rl-c-puct', type=float, default=1.5,
                        help='AlphaZero PUCT exploration constant '
                             '(default: 1.5; alphazero only)')
    parser.add_argument('--rl-seed', type=int, default=0,
                        help='RL run seed for variance studies. '
                             '0 = legacy unseeded (default); non-zero '
                             'isolates per-seed checkpoints under '
                             'runs/{slug}_seed{N}/.')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size (default: 512)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--eval-games', type=int, default=50,
                        help='Games per opponent for eval (default: 50)')
    parser.add_argument('--eval-opponents', nargs='+',
                        default=['gnugo', 'pachi:playouts=500'],
                        help='GTP opponents for eval')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if key exists in experiments.jsonl')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print matrix with SKIP/RUN status, do nothing')
    parser.add_argument('--report', action='store_true',
                        help='Generate tables and figures from experiments.jsonl')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable FP16 mixed precision (use pure FP32)')
    parser.add_argument('--no-xla', action='store_true',
                        help='Disable XLA JIT compilation')
    parser.add_argument('--num-games', type=int, default=0,
                        help='Number of 19x19 training games (0 = all). '
                             'The game list is drawn once and reused for all '
                             'encoder/network combos.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Plan C: translate legacy method names; reject q-dqn.
    _LEGACY_METHOD_ALIASES = {'policy_gradient': 'pg', 'actor_critic': 'ac'}
    _REMOVED_METHODS = {'q-dqn', 'q_dqn'}
    if args.training_method in _REMOVED_METHODS:
        parser.error(
            f'--training-method {args.training_method!r} was removed in Plan C. '
            f'The Q-DQN variant is no longer supported.')
    if args.training_method in _LEGACY_METHOD_ALIASES:
        canonical = _LEGACY_METHOD_ALIASES[args.training_method]
        print(f'[deprecation] --training-method '
              f'{args.training_method!r} renamed to {canonical!r}.',
              file=sys.stderr)
        args.training_method = canonical

    # Configure logging
    log_level = logging.DEBUG if os.environ.get('ALPHIGO_DEBUG') else logging.INFO
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    handlers = [logging.StreamHandler()]

    log_file = os.path.join(RESULTS_DIR, 'run.log')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a'))

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info('AlphiGo thesis run started')

    # Report mode
    if args.report or args.phase == 'report':
        from thesis_report import generate_report
        generate_report()
        return

    # Build combo list
    combos = build_combos(args.board_size, args.encoders, args.networks,
                          args.optimizer, args.training_method)

    # Dry run
    if args.dry_run:
        dry_run(args.board_size, combos, args.optimizer, args.training_method,
                tag=args.training_method)
        return

    # Phase execution
    if args.phase in ('all', 'data'):
        print(f'\n{"=" * 60}')
        print(f'PHASE: DATA  (board_size={args.board_size})')
        print('=' * 60)
        phase_data(args.board_size)

    if args.phase in ('all', 'train'):
        # Configure GPU (mixed precision, TF32, cuDNN autotune) before any
        # model is built. Must happen before importing train module.
        from train import _configure_gpu
        _configure_gpu(mixed_precision=not args.no_mixed_precision)

        print(f'\n{"=" * 60}')
        print(f'PHASE: TRAIN  (board_size={args.board_size})')
        print('=' * 60)

        if args.training_method in ('pg', 'q', 'ac', 'alphazero'):
            from train_rl import run_rl_loop
            from alphigo.training.run import Run

            epsilon = args.rl_epsilon
            if epsilon is None:
                epsilon = 0.1 if args.training_method == 'q' else 0.0

            if args.rl_bootstrap and args.training_method not in ('pg',
                                                                  'alphazero'):
                parser.error('--rl-bootstrap is only supported for '
                             '--training-method pg or alphazero')

            method_kwargs = {}
            if args.training_method == 'alphazero':
                method_kwargs['num_simulations'] = args.rl_mcts_simulations
                method_kwargs['c_puct'] = args.rl_c_puct

            for encoder, network in combos:
                # AlphaZero only supports the alphazero encoder + alphazero
                # network. Other combos in the matrix are skipped (logged),
                # not crashed.
                if args.training_method == 'alphazero' and \
                        (encoder != 'alphazero' or network != 'alphazero'):
                    print(f'  [skip] {encoder}+{network} not compatible '
                          f'with alphazero method (requires '
                          f'alphazero+alphazero)')
                    continue

                run = Run(method=args.training_method, encoder=encoder,
                          network=network, lr=args.rl_lr,
                          board_size=args.board_size,
                          seed=args.rl_seed)
                run_rl_loop(
                    run=run,
                    num_rounds=args.rl_rounds,
                    games_per_round=args.rl_games,
                    clipnorm=1.0,
                    batch_size=args.batch_size,
                    epsilon=epsilon,
                    eval_games=args.eval_games,
                    eval_opponents=args.eval_opponents,
                    bootstrap_from=args.rl_bootstrap,
                    min_win_rate=0.6,
                    method_kwargs=method_kwargs,
                )
        else:
            # If --num-games is set, draw the game list ONCE before the loop
            # so all encoder/network combos train on the exact same games.
            train_games_file = None
            if args.board_size == 19 and args.num_games > 0:
                train_games_file = _resolve_train_games(args.num_games,
                                                        args.seed)
            phase_train(args.board_size, combos, args.optimizer,
                        args.training_method, args.epochs, args.patience,
                        args.batch_size, args.force,
                        tag=args.training_method,
                        xla=not args.no_xla, seed=args.seed,
                        train_games_file=train_games_file)

    if args.phase in ('all', 'eval'):
        print(f'\n{"=" * 60}')
        print(f'PHASE: EVAL  (board_size={args.board_size})')
        print('=' * 60)
        phase_eval(args.board_size, args.eval_games, args.eval_opponents,
                   args.force)

    print('\nDone.')


if __name__ == '__main__':
    main()
