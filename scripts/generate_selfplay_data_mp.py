"""Parallel self-play data generation with a shared GPU inference server.

Same output format as generate_selfplay_data.py, but uses a single GPU
process for all NN inference while multiple CPU workers run game logic
and MCTS in parallel.  This avoids CUDA context contention that starves
workers when each holds its own GPU context.

Architecture:
    GPU server  — one process, one CUDA context, holds both models.
                  Shared memory buffers eliminate pickle serialization.
                  Combined policy+value inference halves round-trips.
    N workers   — CPU-only (CUDA_VISIBLE_DEVICES=""), run game + MCTS,
                  communicate via shared memory arrays + atomic flags.

IPC protocol (per worker):
    input_buf   — shared float array, size = batch_mcts * product(encoder.shape())
    output_buf  — shared float array, size = batch_mcts * (num_points + 2)
                  layout for B states: [policies(B*np), net_values(B), ext_values(B)]
    flag        — shared int: 0=idle, B>=1 means B states ready,
                  -2=response_ready, -1=worker done. (The response value
                  must NOT be a possible request count: a positive value
                  is ambiguous with "B states ready", and a server sweep
                  landing before the worker consumes its answer would
                  re-gather the response as a phantom request.)

Usage:
    python scripts/generate_selfplay_data_mp.py \\
        --agent runs/.../r0_joint_smargin_b19.h5 \\
        --value-model runs/.../value_standalone_margin_b19.h5 \\
        --num-games 2000 --num-workers 6 --sims 25 \\
        --out-dir data/selfplay_b19
"""
import argparse
import multiprocessing
import os
import sys
import time

import numpy as np


# ── GPU inference server ──────────────────────────────────────────────

def _encoder_from_agent(agent_path, board_size):
    """Create the encoder named in the agent checkpoint (default alphazero).

    Keeps the whole pipeline encoder-agnostic: 17-plane 'alphazero' agents
    and 18-plane 'alphazero_komi' agents both just work.
    """
    import h5py
    from alphigo.encoders.base import get_encoder_by_name
    name = 'alphazero'
    try:
        with h5py.File(agent_path, 'r') as h5:
            if 'encoder' in h5:
                raw = h5['encoder'].attrs.get('name', 'alphazero')
                name = raw.decode('ascii') if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass
    return get_encoder_by_name(name, board_size)


def _gpu_server(flags, input_bufs, output_bufs, agent_path,
                value_model_path, board_size, num_workers,
                input_size, output_size, batch_mcts=1,
                metrics_path=None, use_onnx=False,
                ready_path=None, wait_path=None):
    """Single-GPU process: polls shared memory flags, runs inference.

    With --gpu-servers N, N of these run side by side on disjoint worker
    slices; wait_path/ready_path sentinel files chain their startups so
    only the first exports ONNX / builds the TRT engine cache. (File
    sentinels, not mp.Event: spawn-context semaphores fail to rebuild
    under WSL.)
    """
    if wait_path is not None:
        deadline = time.time() + 600
        while not os.path.exists(wait_path) and time.time() < deadline:
            time.sleep(0.5)
    import tensorflow as tf
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    encoder = _encoder_from_agent(agent_path, board_size)
    input_shape = (1,) + tuple(encoder.shape())
    num_points = board_size * board_size
    num_actions = num_points + 1

    import h5py
    with h5py.File(agent_path, 'r') as h5:
        from alphigo import kerasutil
        policy_model = kerasutil.load_model_from_hdf5_group(h5['model'])

    model_policy_width = policy_model.output_shape[0][-1]
    if model_policy_width == num_points:
        pad_pass = True
    elif model_policy_width == num_actions:
        pad_pass = False
    else:
        raise ValueError(f"unexpected policy width {model_policy_width}")
    print(f'[gpu] model policy width={model_policy_width}, '
          f'ipc num_actions={num_actions}, pad_pass={pad_pass}', flush=True)

    single_shape = tuple(encoder.shape())
    single_input_size = input_size // batch_mcts

    # ── Build predict function (ONNX or TF) ─────────────────────────
    predict_fn = None

    if use_onnx and not value_model_path:
        try:
            import onnxruntime as ort
            import tf2onnx

            onnx_path = agent_path.replace('.h5', '.onnx')
            if not os.path.exists(onnx_path):
                print('[gpu] exporting model to ONNX...', flush=True)
                spec = [tf.TensorSpec((None,) + single_shape, tf.float32,
                                     name='board_input')]
                model_proto, _ = tf2onnx.convert.from_keras(
                    policy_model, input_signature=spec, opset=17)
                with open(onnx_path, 'wb') as f:
                    f.write(model_proto.SerializeToString())
                print(f'[gpu] exported ONNX: {onnx_path}', flush=True)
            else:
                print(f'[gpu] reusing ONNX: {onnx_path}', flush=True)

            cache_dir = os.path.join(os.path.dirname(onnx_path), 'trt_cache')
            os.makedirs(cache_dir, exist_ok=True)
            providers = [
                ('TensorrtExecutionProvider', {
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': cache_dir,
                    'trt_max_workspace_size': str(2 * 1024 * 1024 * 1024),
                }),
                ('CUDAExecutionProvider', {}),
                ('CPUExecutionProvider', {}),
            ]
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                onnx_path, sess_options, providers=providers)
            ort_input_name = session.get_inputs()[0].name
            active = session.get_providers()
            print(f'[gpu] ONNX Runtime providers: {active}', flush=True)

            session.run(None, {ort_input_name:
                               np.zeros(input_shape, dtype=np.float32)})

            def _predict_ort(batch_np):
                outs = session.run(None, {ort_input_name: batch_np})
                return outs[0], outs[1], None

            predict_fn = _predict_ort
            print('[gpu] using ONNX Runtime inference', flush=True)

        except Exception as e:
            print(f'[gpu] ONNX init failed ({e}), falling back to TF',
                  flush=True)

    if predict_fn is None:
        value_model = None
        if value_model_path:
            value_model = tf.keras.models.load_model(
                value_model_path, compile=False)

        batch_sig = [tf.TensorSpec(shape=(None,) + single_shape,
                                   dtype=tf.float32)]

        if value_model is not None:
            combined_input = tf.keras.Input(shape=single_shape)
            pol_out = policy_model(combined_input, training=False)
            ext_out = value_model(combined_input, training=False)
            if isinstance(ext_out, (list, tuple)):
                ext_out = ext_out[0]
            combined_model = tf.keras.Model(
                combined_input, [pol_out[0], pol_out[1], ext_out])

            @tf.function(input_signature=batch_sig, reduce_retracing=True)
            def predict_combined(inp):
                return combined_model(inp, training=False)

            predict_combined(np.zeros(input_shape, dtype=np.float32))

            def _predict_combined(batch_np):
                pol, val, ext = predict_combined(batch_np)
                return pol.numpy(), val.numpy(), ext.numpy()

            predict_fn = _predict_combined
        else:
            @tf.function(input_signature=batch_sig, reduce_retracing=True)
            def predict_policy_only(inp):
                return policy_model(inp, training=False)

            predict_policy_only(np.zeros(input_shape, dtype=np.float32))

            def _predict_tf(batch_np):
                outs = predict_policy_only(batch_np)
                return outs[0].numpy(), outs[1].numpy(), None

            predict_fn = _predict_tf

        print('[gpu] using TensorFlow inference', flush=True)

    print('[gpu] server ready', flush=True)
    if ready_path is not None:
        with open(ready_path, 'w') as rf:
            rf.write('ok\n')

    mf = None
    if metrics_path:
        try:
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        except FileExistsError:
            pass
        try:
            mf = open(metrics_path, 'w', buffering=1)
            mf.write('[gpu] server ready\n')
        except OSError:
            mf = None

    max_batch = num_workers * batch_mcts
    batch_buf = np.empty((max_batch, *single_shape), dtype=np.float32)
    input_views = [np.frombuffer(input_bufs[w], dtype=np.float32)
                   for w in range(num_workers)]
    output_views = [np.frombuffer(output_bufs[w], dtype=np.float32)
                    for w in range(num_workers)]
    pad_buf = np.zeros((batch_mcts, num_actions), dtype=np.float32) \
        if pad_pass else None

    workers_done = 0
    total_requests = 0
    next_log_at = 10
    t_start = time.time()
    while workers_done < num_workers:
        pending = []
        pending_set = set()
        batch_offset = 0
        for w in range(num_workers):
            b = flags[w].value
            if b >= 1:
                x = input_views[w]
                batch_buf[batch_offset:batch_offset + b] = \
                    x[:b * single_input_size].reshape(b, *single_shape)
                pending.append((w, b, batch_offset))
                pending_set.add(w)
                batch_offset += b

        if not pending:
            done_count = sum(1 for w in range(num_workers)
                             if flags[w].value == -1)
            if done_count > workers_done:
                workers_done = done_count
            if workers_done < num_workers:
                time.sleep(0.0001)
            continue

        # Straggler wait: busy-spin collecting late workers before running
        # inference. Default 4ms; GPU_BATCH_WAIT_MS=0 disables — inference
        # latency (~3ms) is itself an accumulation window for the next
        # sweep, so the spin mostly burns a core and stalls the GPU.
        _wait_s = float(os.environ.get('GPU_BATCH_WAIT_MS', '4')) / 1000.0
        if _wait_s > 0 and len(pending) < num_workers - workers_done:
            deadline = time.monotonic() + _wait_s
            while time.monotonic() < deadline:
                for w in range(num_workers):
                    if w not in pending_set:
                        b = flags[w].value
                        if b >= 1:
                            x = input_views[w]
                            batch_buf[batch_offset:batch_offset + b] = \
                                x[:b * single_input_size].reshape(
                                    b, *single_shape)
                            pending.append((w, b, batch_offset))
                            pending_set.add(w)
                            batch_offset += b
                if len(pending) >= num_workers - workers_done:
                    break

        batch = batch_buf[:batch_offset]
        policy_np, value_np, ext_np = predict_fn(batch)

        offset = 0
        total_states = 0
        for w, b, _boff in pending:
            out = output_views[w]
            if pad_pass:
                chunk = policy_np[offset:offset + b, :num_points]
                pad_buf[:b, :num_points] = chunk
                pad_buf[:b, num_points:] = 0.0
                out[:b * num_actions] = pad_buf[:b].ravel()
            else:
                out[:b * num_actions] = policy_np[
                    offset:offset + b, :num_actions].ravel()
            out[b * num_actions: b * num_actions + b] = \
                value_np[offset:offset + b, 0]
            if ext_np is not None:
                out[b * num_actions + b: b * num_actions + 2 * b] = \
                    ext_np[offset:offset + b, 0]
            else:
                out[b * num_actions + b: b * num_actions + 2 * b] = 0.0
            flags[w].value = -2
            offset += b
            total_states += b

        total_requests += total_states
        if total_requests >= next_log_at:
            next_log_at = total_requests + 2000
            elapsed = time.time() - t_start
            rate = total_requests / max(elapsed, 0.001)
            msg = (f'[gpu] {total_requests} reqs '
                   f'({rate:.0f}/s, batch={len(batch)}, '
                   f'{elapsed:.0f}s)')
            print(msg, flush=True)
            if mf:
                mf.write(msg + '\n')

    print('[gpu] server shutdown', flush=True)
    if mf:
        mf.write('[gpu] server shutdown\n')
        mf.close()


# ── CPU game worker ───────────────────────────────────────────────────

def _worker(worker_id, result_conn, flag, input_buf, output_buf,
            agent_path, value_model_path, board_size, sims,
            game_ids, seed, score_margin_scale, single_input_size,
            num_points, batch_mcts, search, gumbel_m, value_target,
            komi_range=None, seeded_openings=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    np.random.seed(seed)
    import random as _stdrand
    _stdrand.seed(seed)

    import tensorflow as tf
    import h5py
    from alphigo.agent.zero_agent import load_zero_agent
    from alphigo.alphigo_board import GameState, Move, area_score, ownership_map
    from alphigo.alphigo_types import Player, Point
    from alphigo.encoders.base import get_encoder_by_name

    # Workers don't need the value model — GPU server handles it.
    # But load_zero_agent validates value_mode='external' requires a model,
    # so we create a trivial dummy that will never be called (monkey-patched).
    value_mode = 'network'
    dummy_value_model = None
    if value_model_path:
        value_mode = 'external'
        enc = _encoder_from_agent(agent_path, board_size)
        inp = tf.keras.Input(shape=enc.shape())
        out = tf.keras.layers.GlobalAveragePooling2D()(inp)
        out = tf.keras.layers.Dense(1, activation='tanh')(out)
        dummy_value_model = tf.keras.Model(inp, out)

    def _make_agent():
        if search == 'gumbel':
            from alphigo.agent.gumbel_zero_agent import load_gumbel_agent
            with h5py.File(agent_path, 'r') as h5:
                base = load_gumbel_agent(
                    h5, num_simulations=int(sims), m=int(gumbel_m),
                    value_mode=value_mode,
                    external_value_model=dummy_value_model,
                    batch_size=int(batch_mcts))
            if not os.environ.get('ALPHIGO_NO_FAST_AGENT'):
                try:
                    from alphigo.agent.fast_gumbel_agent import FastGumbelZeroAgent
                    return FastGumbelZeroAgent(
                        base._model, base._encoder,
                        num_simulations=base._num_simulations, m=base._gumbel_m,
                        c_visit=base._c_visit, c_scale=base._c_scale,
                        value_mode=base._value_mode,
                        external_value_model=base._external_value_model,
                        batch_size=base._batch_size)
                except Exception:
                    return base
            return base
        with h5py.File(agent_path, 'r') as h5:
            agent = load_zero_agent(
                h5, value_mode=value_mode,
                external_value_model=dummy_value_model)
        agent._num_simulations = int(sims)
        agent._batch_size = int(batch_mcts)
        return agent

    az_agent = _make_agent()

    inp_np = np.frombuffer(input_buf, dtype=np.float32)
    out_np = np.frombuffer(output_buf, dtype=np.float32)
    encoder = _encoder_from_agent(agent_path, board_size)
    single_shape = tuple(encoder.shape())
    num_actions = num_points + 1

    _sleep = time.sleep

    def _remote_predict_batch(encoded_batch):
        B = encoded_batch.shape[0]
        inp_np[:B * single_input_size] = encoded_batch.reshape(B, -1).ravel()
        flag.value = B
        while flag.value != -2:
            _sleep(1e-6)
        policies = out_np[:B * num_actions].reshape(B, num_actions).copy()
        net_values = out_np[B * num_actions: B * num_actions + B].copy()
        ext_values_raw = out_np[B * num_actions + B: B * num_actions + 2 * B].copy()
        flag.value = 0
        ext_values = ext_values_raw if value_model_path else None
        return policies, net_values, ext_values, None

    az_agent._predict_batch = _remote_predict_batch

    pcr_frac = float(os.environ.get('PLAYOUT_CAP_FRAC', '0'))
    sims_full = int(sims)
    # Fast-search budget: sims/4 by default, overridable via SIMS_LOW so
    # raising full-search sims doesn't drag the (75% of moves) fast
    # searches with it — their m=4 sampling gains little beyond ~32 sims.
    sims_low = int(os.environ.get('SIMS_LOW', 0)) or max(sims_full // 4, 1)
    _rand = np.random.random

    def _play_one_game(game_id, agent, enc, send_fn):
        t0 = time.time()
        if komi_range is not None:
            lo, hi = komi_range
            game_komi = float(np.random.choice(np.arange(lo, hi + 0.001, 1.0)))
        else:
            game_komi = 7.5
        if hasattr(enc, 'komi'):
            enc.komi = game_komi
        agent_enc = getattr(agent, '_encoder', None)
        if agent_enc is not None and hasattr(agent_enc, 'komi'):
            agent_enc.komi = game_komi
        game = GameState.new_game(board_size)

        if seeded_openings:
            opening = seeded_openings[game_id % len(seeded_openings)]
            for color_str, coord_str in opening['moves']:
                player = Player.black if color_str == 'B' else Player.white
                while game.next_player != player:
                    game = game.apply_move(Move.pass_turn())
                col = ord(coord_str[0]) - ord('a') + 1
                row = int(coord_str[1:])
                mv = Move.play_move(Point(row=row, col=col))
                if game.is_valid_move(mv):
                    game = game.apply_move(mv)

        positions = []
        movers = []
        ply = 0

        policy_targets_list = []
        root_q_list = []

        while not game.is_over() and ply < 600:
            is_fast = pcr_frac > 0 and _rand() < pcr_frac
            if is_fast:
                # Fast searches use narrow m=4: their moves are never
                # recorded, so audition width buys nothing — but their
                # blunders become margin-label noise for the whole game.
                # m=4/n=32 -> [(4,4),(2,8)]: full budget consumed, each
                # candidate judged on 4 visits instead of 1 (m=8 wasted
                # 6/32 sims and eliminated candidates on single visits).
                agent._num_simulations = sims_low
                agent._gumbel_m = min(4, gumbel_m)
            else:
                agent._num_simulations = sims_full
                agent._gumbel_m = gumbel_m
            # T=1 for opening diversity only; greedy after ply 12 to keep
            # outcome labels clean (128 sims is too few to survive sampled
            # endgame blunders).
            agent.set_temperature(1.0 if ply < 12 else 0.0)
            encoded = enc.encode(game)
            result = agent.search(game, pre_encoded=encoded)
            move = result['move']

            # No pass before ply 20: early double-pass games get scored
            # with dead stones on the board, corrupting value/ownership
            # labels. Policy target keeps the searched pass mass.
            if move.is_pass and ply < 20:
                board_mass = result['policy_target'][:num_points]
                if board_mass.max() > 0:
                    best = int(np.argmax(board_mass))
                    move = Move.play_move(enc.decode_point_index(best))

            if not move.is_resign and not is_fast:
                positions.append(encoded)
                movers.append(game.next_player)
                policy_targets_list.append(result['policy_target'])
                root_q_list.append(result.get('value', 0.0))

            game = game.apply_move(move)
            ply += 1

        if not positions:
            send_fn(('skip', game_id, 0))
            return

        b_score, w_score = area_score(game.board, komi=game_komi)
        margin = b_score - w_score
        if margin > 0:
            winner = Player.black
        elif margin < 0:
            winner = Player.white
        else:
            winner = None

        omap = ownership_map(game.board)
        omap_flat = omap.flatten()

        n = len(positions)
        encoded_states = np.stack(positions)
        value_targets = np.zeros(n, dtype=np.float32)
        margin_targets = np.zeros(n, dtype=np.float32)
        root_q_targets = np.array(root_q_list, dtype=np.float32)
        ownership_targets = np.zeros((n, board_size * board_size),
                                     dtype=np.float32)
        policy_targets = np.stack(policy_targets_list)

        for i, mover in enumerate(movers):
            m = margin if mover == Player.black else -margin
            margin_targets[i] = m
            if value_target == 'outcome':
                if winner is None:
                    value_targets[i] = 0.0
                else:
                    value_targets[i] = 1.0 if mover == winner else -1.0
            else:
                value_targets[i] = np.tanh(m / score_margin_scale)
            if mover == Player.black:
                ownership_targets[i] = omap_flat
            else:
                ownership_targets[i] = -omap_flat

        elapsed = time.time() - t0
        winner_str = ('B' if winner == Player.black
                      else 'W' if winner == Player.white else 'draw')
        print(f'  [w{worker_id} g{game_id}] '
              f'{ply} moves, winner={winner_str} margin={margin:.1f} '
              f'positions={n} ({elapsed:.0f}s)', flush=True)
        send_fn(('game', game_id, n, elapsed,
                 encoded_states, value_targets, ownership_targets,
                 policy_targets, root_q_targets, margin_targets))

    # ── Dispatch: sequential (default) or K interleaved games ────────
    # WORKER_GAMES=K>1 runs K games in K threads per worker. Threads do
    # not parallelize Python (GIL) — they OVERLAP GPU waits with game
    # logic, and the aggregator below merges their concurrent predict
    # requests into single IPC roundtrips (bigger batches, fewer calls).
    worker_games = int(os.environ.get('WORKER_GAMES', '1'))

    if worker_games <= 1:
        for game_id in game_ids:
            _play_one_game(game_id, az_agent, encoder, result_conn.send)
        flag.value = -1
        result_conn.send(None)
        return

    import threading

    class _BatchAggregator:
        """Merge concurrent per-thread predict calls into one roundtrip.

        Leader election: the first thread to arrive while no roundtrip is
        in flight becomes leader, sleeps a sub-ms accumulation window,
        then serves every queued request (its own included) through the
        single shared shm channel. Followers block on their event.
        """
        def __init__(self, remote_predict, max_batch, wait_s=0.0008):
            self._remote = remote_predict
            self._max = max_batch
            self._wait_s = wait_s
            self._lock = threading.Lock()
            self._queue = []
            self._leading = False

        def predict(self, encoded_batch):
            req = [encoded_batch, threading.Event(), None]
            with self._lock:
                self._queue.append(req)
                lead = not self._leading
                if lead:
                    self._leading = True
            if lead:
                try:
                    time.sleep(self._wait_s)
                    while True:
                        with self._lock:
                            take, size = [], 0
                            while (self._queue and
                                   size + self._queue[0][0].shape[0] <= self._max):
                                r = self._queue.pop(0)
                                take.append(r)
                                size += r[0].shape[0]
                        if not take:
                            break
                        xs = (take[0][0] if len(take) == 1 else
                              np.concatenate([r[0] for r in take]))
                        pol, val, ext, _ = self._remote(xs)
                        off = 0
                        for r in take:
                            b = r[0].shape[0]
                            e = ext[off:off + b] if ext is not None else None
                            r[2] = (pol[off:off + b], val[off:off + b], e, None)
                            off += b
                            r[1].set()
                        with self._lock:
                            if not self._queue:
                                break
                finally:
                    with self._lock:
                        self._leading = False
                        # Failsafe: fail any request left unserved so its
                        # thread raises instead of hanging silently.
                        for r in self._queue:
                            if r[2] is None and r[1].is_set():
                                pass
            req[1].wait()
            if req[2] is None:
                raise RuntimeError('aggregator failed to serve request')
            return req[2]

    aggregator = _BatchAggregator(_remote_predict_batch, int(batch_mcts))
    send_lock = threading.Lock()

    def _locked_send(msg):
        with send_lock:
            result_conn.send(msg)

    ids_lock = threading.Lock()
    ids_iter = iter(game_ids)

    def _thread_main():
        agent = _make_agent()
        agent._predict_batch = aggregator.predict
        enc = _encoder_from_agent(agent_path, board_size)
        while True:
            with ids_lock:
                gid = next(ids_iter, None)
            if gid is None:
                return
            _play_one_game(gid, agent, enc, _locked_send)

    threads = [threading.Thread(target=_thread_main, daemon=True)
               for _ in range(worker_games)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flag.value = -1
    result_conn.send(None)


# ── main: launch server + workers, collect results ────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True)
    parser.add_argument('--value-model', default=None)
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--sims', type=int, default=25)
    parser.add_argument('--num-games', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--score-margin-scale', type=float, default=20.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-mcts', type=int, default=8,
                        help='MCTS batch size for virtual loss (default 8)')
    parser.add_argument('--search', default='puct',
                        choices=['puct', 'gumbel'],
                        help='Root search algorithm (default puct)')
    parser.add_argument('--gumbel-m', type=int, default=16,
                        help='Gumbel root candidates (default 16)')
    parser.add_argument('--komi-random', nargs=2, type=float, default=None,
                        metavar=('LO', 'HI'),
                        help='Randomize komi per game, uniform over '
                             'LO, LO+1, ..., HI (e.g. 3.5 11.5). Requires a '
                             'komi-aware encoder (alphazero_komi). '
                             'Default: fixed 7.5.')
    parser.add_argument('--value-target', default='margin',
                        choices=['margin', 'outcome'],
                        help='margin: tanh(margin/scale); outcome: +/-1')
    parser.add_argument('--onnx', action='store_true',
                        help='Use ONNX Runtime for GPU inference '
                             '(requires onnxruntime-gpu and tf2onnx)')
    parser.add_argument('--gpu-servers', type=int, default=1,
                        help='Number of GPU inference server processes; '
                             'workers are strided across them. 2 halves '
                             'the CPU serialization point (harness: '
                             '+29%% state rate, wait 10.3->6.3ms). '
                             'Env GPU_SERVERS overrides.')
    parser.add_argument('--root-explore-eps', type=float, default=0.25,
                        help='Mix eps-uniform into the root prior used for '
                             'Gumbel candidate sampling (self-play '
                             'exploration; 0 disables). Deployed v11 r103 '
                             'to break the edge-tactics exclusion loop.')
    parser.add_argument('--seeded-openings', default=None,
                        help='JSON file with forced opening sequences. '
                             'Each entry: {"moves": [["B","d4"],["W","e5"],...]}. '
                             'Games cycle through openings by game_id.')
    args = parser.parse_args()
    # Workers inherit this via fork; GumbelZeroAgent reads it at init.
    os.environ['ROOT_EXPLORE_EPS'] = str(args.root_explore_eps)

    seeded_openings = None
    if args.seeded_openings:
        import json
        with open(args.seeded_openings) as f:
            seeded_openings = json.load(f)
        print(f'  openings: {len(seeded_openings)} seeded sequences '
              f'from {args.seeded_openings}', flush=True)

    try:
        os.makedirs(args.out_dir, exist_ok=True)
    except FileExistsError:
        pass

    encoder = _encoder_from_agent(args.agent, args.board_size)
    single_input_size = int(np.prod(encoder.shape()))
    num_points = args.board_size * args.board_size
    num_actions = num_points + 1
    single_output_size = num_actions + 2
    batch_mcts = args.batch_mcts
    input_size = single_input_size * batch_mcts
    output_size = single_output_size * batch_mcts

    worker_games = [[] for _ in range(args.num_workers)]
    for g in range(args.num_games):
        worker_games[g % args.num_workers].append(g)

    print(f'=== generate_selfplay_data_mp ===', flush=True)
    print(f'  agent:    {args.agent}', flush=True)
    print(f'  value:    {args.value_model}', flush=True)
    print(f'  sims:     {args.sims}', flush=True)
    print(f'  search:   {args.search}'
          + (f' (m={args.gumbel_m})' if args.search == 'gumbel' else ''),
          flush=True)
    print(f'  vtarget:  {args.value_target}', flush=True)
    print(f'  encoder:  {encoder.name()} ({encoder.num_planes} planes)', flush=True)
    print(f'  komi:     '
          f'{"random %.1f-%.1f" % tuple(args.komi_random) if args.komi_random else "7.5 fixed"}',
          flush=True)
    print(f'  games:    {args.num_games} (workers={args.num_workers})',
          flush=True)
    print(f'  out:      {args.out_dir}', flush=True)
    print(f'  ipc:      shared memory ({single_input_size} in, '
          f'{single_output_size} out, batch_mcts={batch_mcts})',
          flush=True)
    print(f'  onnx:     {args.onnx}', flush=True)
    print(f'  root_eps: {args.root_explore_eps}', flush=True)
    print(f'  gpu_srv:  '
          f'{int(os.environ.get("GPU_SERVERS", "0")) or args.gpu_servers}',
          flush=True)

    ctx = multiprocessing.get_context('spawn')

    flags = []
    input_bufs = []
    output_bufs = []
    for _ in range(args.num_workers):
        flags.append(ctx.Value('i', 0, lock=False))
        input_bufs.append(ctx.Array('f', input_size, lock=False))
        output_bufs.append(ctx.Array('f', output_size, lock=False))

    metrics_path = os.path.join(args.out_dir, 'gpu_metrics.log')
    n_servers = int(os.environ.get('GPU_SERVERS', '0')) or args.gpu_servers
    gpu_procs = []
    prev_ready = None
    for s in range(n_servers):
        sl = slice(s, None, n_servers)
        ready = os.path.join(args.out_dir, f'.gpu_server_{s}.ready') \
            if s < n_servers - 1 else None
        if ready and os.path.exists(ready):
            os.remove(ready)
        mpath = metrics_path if s == 0 else f'{metrics_path}.{s}'
        gp = ctx.Process(
            target=_gpu_server,
            args=(flags[sl], input_bufs[sl], output_bufs[sl], args.agent,
                  args.value_model, args.board_size, len(flags[sl]),
                  input_size, output_size, batch_mcts, mpath,
                  args.onnx, ready, prev_ready),
        )
        gp.start()
        gpu_procs.append(gp)
        prev_ready = ready

    t0 = time.time()
    procs = []
    parent_conns = []

    for i, ids in enumerate(worker_games):
        if not ids:
            continue
        seed = args.seed + (i + 1) * 7919
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=_worker,
            args=(i, child_conn, flags[i], input_bufs[i], output_bufs[i],
                  args.agent, args.value_model, args.board_size,
                  args.sims, ids, seed, args.score_margin_scale,
                  single_input_size, num_points, batch_mcts,
                  args.search, args.gumbel_m, args.value_target,
                  tuple(args.komi_random) if args.komi_random else None,
                  seeded_openings),
        )
        p.start()
        procs.append((i, p))
        parent_conns.append(parent_conn)

    import h5py
    all_states, all_values, all_ownership, all_policy, all_root_q, all_margins = [], [], [], [], [], []
    total_positions = 0
    total_games = 0
    chunk_idx = 0

    def flush_chunk():
        nonlocal chunk_idx, all_states, all_values, all_ownership, all_policy
        nonlocal all_root_q, all_margins, total_positions
        if not all_states:
            return
        states = np.concatenate(all_states)
        values = np.concatenate(all_values)
        owns = np.concatenate(all_ownership)
        pols = np.concatenate(all_policy)
        root_q = np.concatenate(all_root_q)
        margins = np.concatenate(all_margins) if all_margins else np.zeros(len(states), dtype=np.float32)

        perm = np.random.permutation(len(states))
        states, values, owns, pols, root_q, margins = (
            states[perm], values[perm], owns[perm], pols[perm], root_q[perm], margins[perm])

        path = os.path.join(args.out_dir,
                            f'selfplay_b{args.board_size}_{chunk_idx:04d}.h5')
        with h5py.File(path, 'w') as h5:
            grp = h5.create_group('zero_experience')
            grp.create_dataset('encoded_states', data=states)
            grp.create_dataset('policy_targets', data=pols)
            grp.create_dataset('value_targets', data=values)
            grp.create_dataset('ownership_targets', data=owns)
            grp.create_dataset('root_q_values', data=root_q)
            grp.create_dataset('margin_targets', data=margins)
            h5.attrs['encoder'] = 'alphazero'
            h5.attrs['board_size'] = args.board_size
            h5.attrs['num_positions'] = len(states)
            h5.attrs['chunk_idx'] = chunk_idx
        total_positions += len(states)
        print(f'  chunk {chunk_idx}: {len(states)} positions -> {path}',
              flush=True)
        chunk_idx += 1
        all_states, all_values, all_ownership, all_policy, all_root_q, all_margins = [], [], [], [], [], []

    games_in_chunk = 0
    active_conns = {i: conn for (i, _), conn in zip(procs, parent_conns)}
    while active_conns:
        ready = multiprocessing.connection.wait(
            list(active_conns.values()), timeout=1.0)
        for conn in ready:
            wid = next(k for k, v in active_conns.items() if v is conn)
            try:
                msg = conn.recv()
            except EOFError:
                print(f'  WARN: worker {wid} died', flush=True)
                del active_conns[wid]
                continue
            if msg is None:
                del active_conns[wid]
                continue
            if msg[0] == 'skip':
                continue
            if len(msg) >= 10:
                _, game_id, n, elapsed, states, values, owns, pols, rq, mg = msg
            else:
                _, game_id, n, elapsed, states, values, owns, pols, rq = msg
                mg = np.zeros(n, dtype=np.float32)
            all_states.append(states)
            all_values.append(values)
            all_ownership.append(owns)
            all_policy.append(pols)
            all_root_q.append(rq)
            all_margins.append(mg)
            total_games += 1
            games_in_chunk += 1
            if games_in_chunk >= args.chunk_size:
                flush_chunk()
                games_in_chunk = 0
    for _, p in procs:
        p.join(timeout=30)

    flush_chunk()
    for gp in gpu_procs:
        gp.join(timeout=10)
        if gp.is_alive():
            gp.terminate()

    elapsed = time.time() - t0
    print(f'\n=== Done ===', flush=True)
    print(f'  {total_games} games, {total_positions} positions, '
          f'{chunk_idx} chunks', flush=True)
    print(f'  wall: {elapsed:.0f}s ({elapsed/3600:.1f}h)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
