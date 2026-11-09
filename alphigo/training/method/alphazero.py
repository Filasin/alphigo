"""AlphaZero Method — PUCT-guided self-play.

Self-play uses ZeroAgent's PUCT search; experience is a ZeroReplayBuffer
of (encoded_state, visit-count policy_target, terminal value_target).
Training fits the two-headed network with categorical cross-entropy on
the policy head and MSE on the value head.
"""
from pathlib import Path

import h5py
import numpy as np
from tensorflow.keras.optimizers import SGD


class AlphaZeroMethod:
    name = 'alphazero'
    default_epsilon = 0.0
    allows_bootstrap_from_sl = True

    def __init__(self, num_simulations: int = 50, c_puct: float = 1.5,
                 temperature_moves: int = None, num_workers: int = 1,
                 base_seed: int = 0, **_kwargs):
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.temperature_moves = (None if temperature_moves is None
                                  else int(temperature_moves))
        # Multiprocess self-play (book parallel_self_play pattern). When
        # num_workers > 1 the learner is dumped to a temp .h5 each round,
        # workers reload it, run sequential self-play on a slice of games,
        # and the main process combines the resulting buffers. Per-worker
        # seed = base_seed*1000 + worker_id + 1 to keep RNG streams disjoint.
        self.num_workers = max(1, int(num_workers))
        self.base_seed = int(base_seed)
        # Bumped each call to self_play so seeds across rounds also differ.
        self._round_counter = 0

    # ── construction / IO ─────────────────────────────────────────────

    def build_agent(self, encoder, network_name: str, board_size: int):
        if network_name != 'alphazero':
            raise ValueError(
                f"AlphaZeroMethod requires network='alphazero' "
                f"(two-headed); got {network_name!r}.")
        from alphigo.agent.zero_agent import ZeroAgent
        from alphigo.data.networks import build_alphazero_two_head

        model = build_alphazero_two_head(
            encoder.shape(), encoder.num_points())
        return ZeroAgent(model, encoder,
                         num_simulations=self.num_simulations,
                         c_puct=self.c_puct)

    def load_agent(self, path: Path):
        from alphigo.agent.zero_agent import load_zero_agent
        with h5py.File(path, 'r') as h5:
            agent = load_zero_agent(h5)
        # Override on-disk values with the Method-configured ones so
        # CLI flags like --mcts-simulations take effect every round.
        agent._num_simulations = self.num_simulations
        agent._c_puct = self.c_puct
        return agent

    def load_experience(self, path: Path):
        from alphigo.rl.zero_experience import load_zero_buffer
        with h5py.File(path, 'r') as h5:
            return load_zero_buffer(h5)

    # ── self-play / train / evaluate ──────────────────────────────────

    def self_play(self, learner, reference, board_size, num_games,
                  epsilon, batch_size=64):
        """Self-play a single agent against itself (reference is unused).

        Routes through ``multiprocess_self_play_zero`` when num_workers > 1
        (book parallel_self_play pattern). For num_workers == 1, takes the
        in-process path for parity with the original behavior.
        """
        from alphigo.rl.simulate import (batched_self_play_zero,
                                         multiprocess_self_play_zero)
        # `reference` and `epsilon` are unused for AlphaZero; signature kept
        # for Method protocol compatibility.
        _ = reference, epsilon, batch_size

        self._round_counter += 1
        round_seed = self.base_seed + 10_000 * self._round_counter

        if self.num_workers <= 1:
            return batched_self_play_zero(
                learner, board_size, num_games,
                num_simulations=self.num_simulations,
                temperature_moves=self.temperature_moves,
                verbose=False,
            )

        # Multiprocess path: snapshot learner to temp .h5 for workers to
        # reload. Cheap (~1s) compared to multi-minute self-play.
        import tempfile
        import os
        import h5py
        tmpdir = tempfile.mkdtemp(prefix='az_mp_agent_')
        agent_path = os.path.join(tmpdir, 'learner.h5')
        with h5py.File(agent_path, 'w') as h5:
            learner.serialize(h5)
        try:
            return multiprocess_self_play_zero(
                agent_path=agent_path,
                board_size=board_size,
                num_games=num_games,
                num_workers=self.num_workers,
                num_simulations=self.num_simulations,
                temperature_moves=self.temperature_moves,
                base_seed=round_seed,
            )
        finally:
            try:
                os.remove(agent_path)
            except OSError:
                pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

    def train(self, agent, experience, lr, clipnorm, batch_size):
        agent._model.compile(
            optimizer=SGD(learning_rate=lr, clipnorm=clipnorm),
            loss={'policy_output': 'categorical_crossentropy',
                  'value_output': 'mse'},
            loss_weights={'policy_output': 1.0, 'value_output': 1.0},
            jit_compile=False,
        )
        agent._model.fit(
            experience.encoded_states,
            {'policy_output': experience.policy_targets,
             'value_output': experience.value_targets.reshape(-1, 1)},
            batch_size=batch_size,
            epochs=1,
            verbose=2,
        )

    # ── SL → AZ warm start ─────────────────────────────────────────────

    def bootstrap_from_sl(self, sl_path, encoder, board_size: int):
        """Build a fresh two-head agent and copy SL policy weights in.

        SL checkpoint must use ``encoder='alphazero'`` and an alphazero
        residual trunk. Trunk + policy-head weights are copied; the
        value head stays randomly initialised.
        """
        from alphigo.agent.deep_learning import load_prediction_agent
        from alphigo.agent.zero_agent import ZeroAgent
        from alphigo.data.networks import build_alphazero_two_head

        with h5py.File(sl_path, 'r') as h5:
            sl_agent = load_prediction_agent(h5)

        sl_encoder_name = sl_agent.encoder.name()
        if sl_encoder_name != 'alphazero':
            raise ValueError(
                f"--bootstrap for alphazero requires an SL checkpoint "
                f"trained with the 'alphazero' encoder; got "
                f"{sl_encoder_name!r}.")
        if (sl_agent.encoder.board_width != board_size or
                sl_agent.encoder.board_height != board_size):
            raise ValueError(
                f"--bootstrap board size mismatch: SL checkpoint is "
                f"{sl_agent.encoder.board_width}x"
                f"{sl_agent.encoder.board_height}, run is "
                f"{board_size}x{board_size}.")

        two_head = build_alphazero_two_head(
            encoder.shape(), encoder.num_points())

        # Greedy weight copy: walk both layer lists; for each SL layer
        # with weights, advance through the two-head model until shapes
        # match, then copy. This transfers the trunk and policy head;
        # the value-head layers (different shapes) are skipped.
        sl_weighted = [L for L in sl_agent.model.layers if L.get_weights()]
        th_layers = list(two_head.layers)
        j = 0
        copied = 0
        for L_sl in sl_weighted:
            sig_sl = [w.shape for w in L_sl.get_weights()]
            matched = False
            while j < len(th_layers):
                L_th = th_layers[j]
                w_th = L_th.get_weights()
                j += 1
                if not w_th:
                    continue
                if [w.shape for w in w_th] == sig_sl:
                    L_th.set_weights(L_sl.get_weights())
                    copied += 1
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"SL checkpoint is incompatible with the AlphaZero "
                    f"two-headed network: could not match SL layer "
                    f"{L_sl.name} with weight shapes {sig_sl}. "
                    f"The SL checkpoint must use --network alphazero "
                    f"with the same trunk topology.")

        if copied == 0:
            raise ValueError(
                "bootstrap_from_sl: no weights were copied — SL "
                "checkpoint is incompatible with the AlphaZero "
                "architecture.")

        return ZeroAgent(two_head, encoder,
                         num_simulations=self.num_simulations,
                         c_puct=self.c_puct)

    EVAL_TEMPERATURE_PLIES = 10

    def evaluate(self, new_agent, old_agent, board_size, num_games):
        """Head-to-head with color swap and AGZ-style temperature schedule.

        T=1 for the first ``EVAL_TEMPERATURE_PLIES`` plies of each game
        (visit-count-weighted move sampling, mirrors AGZ self-play),
        T=0 (deterministic argmax over visit counts) thereafter. The
        opening stochasticity gives continuous win rates instead of the
        {0, 0.5, 1.0} bimodal pattern that pure-greedy MCTS produces.

        When ``num_workers > 1``, dispatches across CPU workers via
        ``multiprocess_evaluate_zero``. Eval is otherwise single-process
        and dominated the per-round wall (~22 min/round at sims=25,
        50 games on RTX 3090) — 4× worse than parallelized self-play.
        """
        if self.num_workers <= 1 or num_games <= 0:
            return self._evaluate_serial(new_agent, old_agent,
                                         board_size, num_games)

        # Multiprocess path: snapshot both agents to temp h5 files so
        # workers can reload them cleanly under spawn (no TF/CUDA state
        # inheritance). Cleanup happens in `finally`.
        import tempfile
        import os
        import h5py
        from alphigo.rl.simulate import multiprocess_evaluate_zero

        tmpdir = tempfile.mkdtemp(prefix='az_eval_agents_')
        new_path = os.path.join(tmpdir, 'new.h5')
        old_path = os.path.join(tmpdir, 'old.h5')
        try:
            with h5py.File(new_path, 'w') as h5:
                new_agent.serialize(h5)
            with h5py.File(old_path, 'w') as h5:
                old_agent.serialize(h5)
            # Distinct seed range from self-play (which uses
            # base_seed + 10_000 * round_counter); +5_000_000 offset keeps
            # eval and self-play RNG streams from overlapping.
            eval_seed = (self.base_seed
                         + 10_000 * self._round_counter
                         + 5_000_000)
            return multiprocess_evaluate_zero(
                new_agent_path=new_path,
                old_agent_path=old_path,
                board_size=board_size,
                num_games=num_games,
                num_workers=self.num_workers,
                num_simulations=self.num_simulations,
                eval_temperature_plies=self.EVAL_TEMPERATURE_PLIES,
                base_seed=eval_seed,
            )
        finally:
            for p in (new_path, old_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

    def _evaluate_serial(self, new_agent, old_agent, board_size, num_games):
        from alphigo.alphigo_types import Player
        from alphigo.alphigo_board import GameState

        def play_one(black_player, white_player):
            game = GameState.new_game(board_size)
            agents = {Player.black: black_player, Player.white: white_player}
            ply = 0
            while not game.is_over() and ply < 400:
                t = 1.0 if ply < self.EVAL_TEMPERATURE_PLIES else 0.0
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
        win_rate = a_wins / max(num_games, 1)
        return win_rate, {'a_wins': a_wins, 'total_games': num_games}
