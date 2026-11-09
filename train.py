import argparse
import json
import logging
import os
import random
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping

logger = logging.getLogger(__name__)


def _configure_gpu(mixed_precision=True):
    # cuDNN auto-tune: benchmarks convolution algorithms and caches the fastest.
    # Safe because all inputs have fixed spatial dims (board_size × board_size).
    os.environ.setdefault('TF_CUDNN_USE_AUTOTUNE', '1')

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # TF32 for Ampere+ GPUs: ~2× faster FP32 matmuls with negligible
        # precision loss (19-bit mantissa vs 23-bit).
        tf.config.experimental.enable_tensor_float_32_execution(True)

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print(f'[GPU] {len(gpus)} device(s) — mixed_float16, TF32, cuDNN autotune')
        else:
            print(f'[GPU] {len(gpus)} device(s) — float32, TF32, cuDNN autotune')
    else:
        print('[GPU] No GPU detected — training on CPU')


# Defer _configure_gpu() until args are parsed (see __main__ block).
# When imported as a library (experiment.py, thesis_run.py), callers should
# invoke _configure_gpu() themselves before building models.

def set_seed(seed):
    """Set random seeds for Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


from alphigo.agent.deep_learning import DeepLearningAgent, load_prediction_agent
from alphigo.data.networks import NETWORK_MAP
from alphigo.encoders.base import get_encoder_by_name


def _history_path(checkpoint_dir, tag, encoder_name, network, optimizer, board_size):
    """Path for the JSON sidecar that stores training history across resumes."""
    name = f'{tag}_{encoder_name}_{network}_{optimizer}_b{board_size}_history.json'
    return os.path.join(checkpoint_dir, name)


class AgentCheckpoint(Callback):
    """Saves each epoch as a DeepLearningAgent-compatible HDF5 file.

    Also writes a JSON sidecar with accumulated training metrics so that
    resumed runs can seed early-stopping from the full history.
    """

    def __init__(self, filepath, encoder, history_path=None, prior_history=None):
        super().__init__()
        self.filepath = filepath
        self.encoder = encoder
        self.history_path = history_path
        self._accumulated = {k: list(v) for k, v in (prior_history or {}).items()}

    def on_epoch_end(self, epoch, logs=None):
        path = self.filepath.format(epoch=epoch + 1)
        agent = DeepLearningAgent(self.model, self.encoder)
        with h5py.File(path, 'w') as h5:
            agent.serialize(h5)
        logger.info('Saved: %s', os.path.basename(path))

        # Append this epoch's metrics and persist to sidecar
        if self.history_path and logs:
            for k, v in logs.items():
                self._accumulated.setdefault(k, []).append(float(v))
            with open(self.history_path, 'w') as f:
                json.dump(self._accumulated, f)


class ResumableEarlyStopping(EarlyStopping):
    """EarlyStopping seeded with history from a prior (interrupted) run.

    On train_begin, sets ``self.best`` and ``self.wait`` from the prior
    val_loss (or whichever monitor) so that patience accounting is
    continuous across resumes.
    """

    def __init__(self, prior_history=None, **kwargs):
        super().__init__(**kwargs)
        self._prior_history = prior_history or {}

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        values = self._prior_history.get(self.monitor, [])
        if values:
            if self.monitor_op == np.less:
                self.best = min(values)
                best_idx = values.index(min(values))
            else:
                self.best = max(values)
                best_idx = values.index(max(values))
            self.wait = len(values) - best_idx - 1

class TrainingSession:
    """Manages a single encoder/network training run with resume support.

    Handles: checkpoint discovery, history sidecar load/merge, callbacks,
    early stopping, and final model save. Data loading is delegated to
    run_npy() or run_sgf() depending on data source.
    """

    def __init__(self, encoder_name, network, optimizer, board_size,
                 batch_size=512, epochs=5, checkpoint_dir='./checkpoints',
                 tag='run', patience=None, xla=True, seed=42):
        self.encoder_name = encoder_name
        self.network = network
        self.optimizer = optimizer
        self.board_size = board_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.tag = tag
        self.patience = patience
        self.xla = xla
        self.seed = seed

        set_seed(seed)
        self._hist_path = _history_path(
            checkpoint_dir, tag, encoder_name, network, optimizer, board_size)
        self._initial_epoch = 0
        self._prior_history = {}
        self._model = None
        self._encoder = None

    def _resolve_resume(self, resume_from):
        """Load checkpoint and prior history. Returns True if already complete."""
        if resume_from is None:
            return False

        self._initial_epoch = int(Path(resume_from).stem.rsplit('_', 1)[-1])

        if os.path.exists(self._hist_path):
            with open(self._hist_path) as f:
                self._prior_history = json.load(f)

        if self._initial_epoch >= self.epochs:
            logger.info('SKIP: already at epoch %d >= target %d',
                        self._initial_epoch, self.epochs)
            with h5py.File(resume_from, 'r') as h5:
                agent = load_prediction_agent(h5)
            self._model = agent.model
            from tensorflow.keras.callbacks import History
            h = History()
            h.history = self._prior_history
            self._model.history = h
            return True

        logger.info('RESUME: %s — continuing from epoch %d to %d',
                    os.path.basename(resume_from),
                    self._initial_epoch, self.epochs)
        with h5py.File(resume_from, 'r') as h5:
            agent = load_prediction_agent(h5)
        self._model = agent.model
        self._encoder = get_encoder_by_name(self.encoder_name, self.board_size)
        return False

    def _build_new_model(self, loss='categorical_crossentropy'):
        self._model, self._encoder = _build_model(
            self.encoder_name, self.network, self.optimizer,
            self.board_size, xla=self.xla)
        if loss != 'categorical_crossentropy':
            self._model.compile(
                loss=loss, optimizer=self.optimizer,
                metrics=['accuracy'], jit_compile=self.xla)

    def _recompile(self, loss='categorical_crossentropy'):
        self._model.compile(
            loss=loss, optimizer=self.optimizer,
            metrics=['accuracy'], jit_compile=self.xla)

    def _build_callbacks(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'{self.tag}_{self.encoder_name}_{self.network}_'
            f'{self.optimizer}_b{self.board_size}_model_epoch_{{epoch}}.h5',
        )
        callbacks = [AgentCheckpoint(
            checkpoint_path, self._encoder,
            history_path=self._hist_path,
            prior_history=self._prior_history,
        )]
        if self.patience is not None:
            callbacks.append(ResumableEarlyStopping(
                monitor='val_loss', patience=self.patience,
                restore_best_weights=True, verbose=1,
                prior_history=self._prior_history,
            ))
        return callbacks, checkpoint_path

    def _merge_history(self):
        if self._prior_history and hasattr(self._model, 'history') \
                and self._model.history:
            merged = {}
            all_keys = set(list(self._prior_history.keys()) +
                           list(self._model.history.history.keys()))
            for k in all_keys:
                merged[k] = (self._prior_history.get(k, []) +
                             self._model.history.history.get(k, []))
            self._model.history.history = merged

    def _save_final(self, checkpoint_path):
        final_path = os.path.join(
            self.checkpoint_dir,
            f'{self.tag}_{self.encoder_name}_{self.network}_'
            f'{self.optimizer}_b{self.board_size}'
            f'_model_epoch_{self.epochs}.h5',
        )
        with h5py.File(final_path, 'w') as h5:
            DeepLearningAgent(self._model, self._encoder).serialize(h5)
        logger.info('Saved: %s', os.path.basename(final_path))
        return final_path

    def run_npy(self, features_train, labels_train,
                features_test, labels_test, resume_from=None):
        """Train from pre-computed NPY feature arrays."""
        if self._resolve_resume(resume_from):
            return self._model

        if self._model is None:
            self._build_new_model(loss='categorical_crossentropy')
        else:
            self._recompile(loss='categorical_crossentropy')

        features = np.load(features_train)
        labels = np.load(labels_train)
        feature_test = np.load(features_test)
        label_test = np.load(labels_test)

        # C++ generator outputs NCHW (N, planes, H, W); Keras expects NHWC.
        # Detect by checking if channel dim is in position 1.
        if features.ndim == 4 and features.shape[1] == self._encoder.num_planes:
            features = np.transpose(features, (0, 2, 3, 1))
            feature_test = np.transpose(feature_test, (0, 2, 3, 1))

        train_ds = (
            tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(buffer_size=min(len(features), 50_000))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((feature_test, label_test))
            .batch(self.batch_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        callbacks, checkpoint_path = self._build_callbacks()
        self._model.fit(
            train_ds, epochs=self.epochs,
            initial_epoch=self._initial_epoch,
            verbose=1, validation_data=val_ds,
            callbacks=callbacks,
        )
        self._merge_history()
        self._save_final(checkpoint_path)

        score = self._model.evaluate(val_ds, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return self._model

    def run_sgf(self, num_games=100, data_dir='data',
                train_games_file=None, resume_from=None):
        """Train from SGF game archive via generator."""
        from alphigo.data.processor import GoDataProcessor

        if self._resolve_resume(resume_from):
            return self._model

        if self._model is None:
            self._build_new_model(loss='sparse_categorical_crossentropy')
        else:
            self._recompile(loss='sparse_categorical_crossentropy')

        num_classes = self.board_size * self.board_size
        input_shape = (self.board_size, self.board_size,
                       self._encoder.num_planes)
        n = None if num_games == 0 else num_games

        processor = GoDataProcessor(
            encoder=self._encoder.name(), data_directory=data_dir,
            board_size=self.board_size)
        fixed_train = None
        if train_games_file:
            with open(train_games_file) as f:
                fixed_train = [line.strip() for line in f if line.strip()]

        generator = processor.load_go_data('train', n, use_generator=True,
                                            fixed_games=fixed_train)
        test_generator = processor.load_go_data('test', n,
                                                  use_generator=True)
        num_train = generator.get_num_samples(self.batch_size, num_classes)
        num_test = test_generator.get_num_samples(self.batch_size, num_classes)

        output_sig = (
            tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        train_ds = tf.data.Dataset.from_generator(
            lambda: generator.generate(self.batch_size, num_classes),
            output_signature=output_sig,
        ).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_generator(
            lambda: test_generator.generate(self.batch_size, num_classes),
            output_signature=output_sig,
        ).prefetch(tf.data.AUTOTUNE)

        callbacks, checkpoint_path = self._build_callbacks()
        self._model.fit(
            train_ds, epochs=self.epochs,
            initial_epoch=self._initial_epoch,
            steps_per_epoch=num_train // self.batch_size,
            validation_data=val_ds,
            validation_steps=num_test // self.batch_size,
            callbacks=callbacks,
        )
        self._merge_history()
        self._save_final(checkpoint_path)

        score = self._model.evaluate(
            val_ds, steps=num_test // self.batch_size)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return self._model


def _build_model(encoder_name, network, optimizer, board_size, xla=True):
    from alphigo.data.networks import NETWORK_MAP

    encoder = get_encoder_by_name(encoder_name, board_size)
    num_classes = board_size * board_size
    input_shape = (board_size, board_size, encoder.num_planes)

    if network not in NETWORK_MAP:
        raise ValueError(f'Unknown network: {network!r}. '
                         f'Choose from: {list(NETWORK_MAP)}')

    model = NETWORK_MAP[network](input_shape, num_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'],
        jit_compile=xla,
    )
    return model, encoder


def train_from_npy(
    features_train, labels_train, features_test, labels_test,
    encoder_name='oneplane', network='small', optimizer='adadelta',
    board_size=19, batch_size=512, epochs=5, checkpoint_dir='./checkpoints',
    tag='npy', patience=None, xla=True, resume_from=None, seed=42,
):
    session = TrainingSession(
        encoder_name=encoder_name, network=network, optimizer=optimizer,
        board_size=board_size, batch_size=batch_size, epochs=epochs,
        checkpoint_dir=checkpoint_dir, tag=tag, patience=patience,
        xla=xla, seed=seed,
    )
    return session.run_npy(features_train, labels_train,
                           features_test, labels_test,
                           resume_from=resume_from)


def train_from_sgf(
    num_games=100, encoder_name='oneplane', network='small',
    optimizer='adagrad', board_size=19, batch_size=512, epochs=5,
    checkpoint_dir='./checkpoints', tag='sgf', data_dir='data',
    patience=None, xla=True, resume_from=None, seed=42,
    train_games_file=None,
):
    session = TrainingSession(
        encoder_name=encoder_name, network=network, optimizer=optimizer,
        board_size=board_size, batch_size=batch_size, epochs=epochs,
        checkpoint_dir=checkpoint_dir, tag=tag, patience=patience,
        xla=xla, seed=seed,
    )
    return session.run_sgf(
        num_games=num_games, data_dir=data_dir,
        train_games_file=train_games_file, resume_from=resume_from,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a Go move-prediction model',
    )

    npy_group = parser.add_argument_group('NPY mode (pre-computed data)')
    npy_group.add_argument('--features-train',
                           help='Training features .npy path')
    npy_group.add_argument('--labels-train',
                           help='Training labels .npy path')
    npy_group.add_argument('--features-test',
                           help='Test features .npy path')
    npy_group.add_argument('--labels-test',
                           help='Test labels .npy path')

    sgf_group = parser.add_argument_group('SGF mode (generator-based)')
    sgf_group.add_argument('--num-games', type=int,
                           help='Number of SGF games to train on (0 = all available)')
    sgf_group.add_argument('--data-dir', default='data',
                           help='Root data directory for SGF extraction and per-game '
                                '.npy files (default: data)')

    parser.add_argument('--encoder', default='oneplane',
                        choices=['oneplane', 'sevenplane', 'elevenplane',
                                 'nineplane', 'agplane', 'alphazero'])
    parser.add_argument('--network', default='medium',
                        choices=['small', 'medium', 'large',
                                 'small_orig', 'medium_orig', 'large_orig',
                                 'alphazero'])
    parser.add_argument('--optimizer', default='adam',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'])
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--tag', default=None,
                        help='Data-source tag for checkpoint filenames '
                             '(default: "mcts" for NPY mode, "sgf" for SGF mode)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (epochs without val_loss improvement). '
                             'Omit to disable early stopping.')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable FP16 mixed precision (use pure FP32)')
    parser.add_argument('--no-xla', action='store_true',
                        help='Disable XLA JIT compilation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    _configure_gpu(mixed_precision=not args.no_mixed_precision)

    npy_paths = [args.features_train, args.labels_train,
                 args.features_test, args.labels_test]
    has_npy = any(p is not None for p in npy_paths)
    has_sgf = args.num_games is not None

    if has_npy and has_sgf:
        parser.error('Provide either NPY paths or --num-games, not both.')
    if has_npy and not all(p is not None for p in npy_paths):
        parser.error('All four NPY paths are required: '
                     '--features-train, --labels-train, '
                     '--features-test, --labels-test')
    if not has_npy and not has_sgf:
        parser.error('Provide NPY paths (--features-train etc.) '
                     'or --num-games.')

    common = dict(
        encoder_name=args.encoder,
        network=args.network,
        optimizer=args.optimizer,
        board_size=args.board_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        xla=not args.no_xla,
    )

    if has_npy:
        tag = args.tag if args.tag else 'mcts'
        train_from_npy(
            args.features_train, args.labels_train,
            args.features_test, args.labels_test,
            tag=tag, patience=args.patience, seed=args.seed, **common,
        )
    else:
        tag = args.tag if args.tag else 'sgf'
        train_from_sgf(
            num_games=args.num_games, tag=tag,
            data_dir=args.data_dir, patience=args.patience,
            seed=args.seed, **common,
        )
