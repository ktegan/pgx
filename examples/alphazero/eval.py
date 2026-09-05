# Copyright 2026 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tournament evaluation of backgammon AlphaZero checkpoints.

Loads a set of model checkpoints, plays them against each other and reports
which model is best along with a rough win percentage of the best model against
the second / third / fourth best.

Tournament format:
  1. Pick --num-models checkpoints (evenly spaced over training time).
  2. Round robin the pool, eliminate the --eliminate worst models and repeat
     until 4 models remain.
  3. Round robin the remaining 4 to rank them.
  4. The best model plays the 2nd / 3rd / 4th with --final-games games each.

Every model plays with a value-head lookahead player (--mode 1ply or 2ply):
candidate moves are enumerated with the same legality rules the training
strategies use, the resulting boards are scored by each model's value head and
an action is sampled from a temperature softmax over candidate equities.  Dice
rolls are chance nodes: they are sampled with the true dice probabilities
(1/36 for doubles, 2/36 otherwise) and never come from either network.

Example:
    python examples/alphazero/eval.py --checkpoints 'checkpoints/selfplay_*.pkl'

Runtime with the defaults (10 models, 128 games per matchup, 1-ply lookahead,
bfloat16) is roughly 3-4 hours on an RTX 3070 class GPU; halve
--games-per-matchup for a quicker (noisier) pass, or run --mode 2ply for
stronger (much slower) tournament games.
"""

import argparse
import dataclasses
import glob
import json
import math
import os
import pickle
import re
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy

import pgx
import pgx.core
from pgx._src import struct
from pgx.models.aznet import AZNet
from pgx.backgammon import (
    ALL_GAME_POSITIONS,
    NOOP_ACTION_IDX,
    _arr_legal_action_mask_details,
    _arr_make_new_boards,
    _make_observation,
)

NEG = jnp.finfo(jnp.float32).min
UTILITIES = jnp.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], dtype=jnp.float32)


@struct.dataclass
class NNConfig:
    """Evaluator config for make_nn_evaluator_cls.

    must use pgx's flax-style struct.dataclass so jax.tree_util treats it
    correctly when the strategies jit through core.broadcast_config():
    should_broadcast=False short-circuits the (pointless) config repetition
    and micro_batch_size sizes the strategies' chunked evaluation."""
    micro_batch_size: int = struct.field(pytree_node=False, default=512)
    should_broadcast: bool = struct.field(pytree_node=False, default=False)


def make_value_to_scalar(num_value_channels: int):
    """Turn the value head output into a scalar equity from the mover's perspective."""
    if num_value_channels == 6:
        def value_to_scalar(value):
            probs = jax.nn.softmax(value.astype(jnp.float32), axis=-1)
            return jnp.sum(probs * UTILITIES, axis=-1)
        return value_to_scalar

    def value_to_scalar(value):
        return value[..., 0].astype(jnp.float32)
    return value_to_scalar


class Model:
    """A loaded checkpoint: params + the haiku forward built from its own config."""

    def __init__(self, path, iteration, params, bn_state, forward, num_value_channels, arch):
        self.path = path
        self.iteration = iteration
        self.params = params
        self.bn_state = bn_state
        self.forward = forward
        self.num_value_channels = num_value_channels
        self.arch = arch
        self.value_to_scalar = make_value_to_scalar(num_value_channels)

    @property
    def name(self):
        return os.path.basename(self.path)


def make_nn_evaluator_cls(model: Model):
    """Build a pgx.core.Evaluator class driven by a loaded checkpoint's value head.

    The pgx two-ply strategies (BackgammonTwoPlyStrategy,
    BackgammonTwoPlyChunkedStrategy) take an evaluator *class* and construct it
    with broadcast/sliced configs; instances are then called as
    eval(state, idx=...) where state.observation holds the (B, 24, 1, 12)
    boards to score.  This wrapper adapts a checkpoint's haiku forward to that
    interface so the strategies can be benchmarked/played with the real
    networks (the SimpleBackgammonEvaluator handcrafted weights are ~1000x
    cheaper, which hides strategy-level inefficiencies)."""
    value_to_scalar = model.value_to_scalar
    forward = model.forward

    class NeuralBackgammonEvaluator(pgx.core.Evaluator):
        # neural forward >> compaction overhead: strategies should skip illegal rows
        expensive_evaluation = True

        def __init__(self, config=None):
            super().__init__(config)

        def eval(self, state: pgx.core.State, idx=None):
            (_, v), _ = forward.apply(model.params, model.bn_state, state.observation, is_eval=True)
            return value_to_scalar(v)

    return NeuralBackgammonEvaluator


def load_model(path: str, env, dtype=jnp.bfloat16) -> Model:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "model" in data:
        params, bn_state = data["model"]
        cfg = data.get("config", {})
        iteration = int(data.get("iteration", -1))
        env_version = data.get("env_version", None)
    else:
        params, bn_state = data
        cfg, iteration, env_version = {}, -1, None

    current_version = env.version
    if env_version is not None and env_version != current_version:
        raise ValueError(
            f"{path}: checkpoint env_version {env_version} != current {current_version}; "
            "the observation encoding changed between versions so this checkpoint "
            "cannot be compared against current ones"
        )

    # checkpoints are trained in bfloat16; evaluating in the training dtype keeps
    # numerics faithful and is much faster on GPUs with tensor cores
    def cast(tree):
        return jax.tree_util.tree_map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            tree,
        )

    params = cast(params)
    bn_state = cast(bn_state)

    num_channels = int(cfg.get("num_channels", 256))
    num_layers = int(cfg.get("num_layers", 12))
    resnet_v2 = bool(cfg.get("resnet_v2", True))
    num_value_channels = int(cfg.get("num_value_channels", 1))

    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=num_channels,
            num_blocks=num_layers,
            resnet_v2=resnet_v2,
            # only the value head is used for evaluation; a policy head is never
            # built so checkpoints without one also load
            train_policy_network=False,
            num_value_channels=num_value_channels,
            dtype=dtype,
        )
        return net(x, is_training=not is_eval, test_local_stats=False)

    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
    arch = (num_channels, num_layers, resnet_v2, num_value_channels, str(dtype))
    return Model(path, iteration, params, bn_state, forward, num_value_channels, arch)


def _iteration_of(path: str) -> int:
    m = re.search(r"_(\d+)\.pkl$", path)
    return int(m.group(1)) if m else -1


def select_checkpoints(files, num_models: int):
    """Pick num_models checkpoints evenly spaced over the sorted (by iteration) list."""
    files = sorted(files, key=lambda p: (_iteration_of(p), p))
    if len(files) <= num_models:
        return files
    idxs = numpy.linspace(0, len(files) - 1, num_models).astype(int)
    idxs = sorted(set(int(i) for i in idxs))
    return [files[i] for i in idxs]


class EvalConfig:
    def __init__(self, mode="1ply", temperature=0.1, batch_size=128, max_steps=1024,
                 eval_chunk=8192):
        self.mode = mode          # "1ply" or "2ply" value-head lookahead
        self.temperature = temperature  # sampling temperature over candidate equities (0 = greedy)
        self.batch_size = batch_size    # games played in parallel per jitted chunk
        self.max_steps = max_steps      # games reaching this many steps count as draws
        self.eval_chunk = eval_chunk    # boards per forward pass (memory bound for 2ply)


def eval_equity(model, params, bn_state, obs, chunk: int):
    """Value-head equity for a batch of observations, processed in chunks so that
    wide candidate expansions (2-ply) do not blow up device memory."""
    n = obs.shape[0]
    pad = (-n) % chunk
    if pad:
        obs = jnp.pad(obs, ((0, pad), (0, 0), (0, 0), (0, 0)))
    n_chunks = obs.shape[0] // chunk

    def score(o):
        (_, v), _ = model.forward.apply(params, bn_state, o, is_eval=True)
        return model.value_to_scalar(v)

    if n_chunks == 1:
        return score(obs)[:n]
    return jax.lax.map(score, obs.reshape(n_chunks, chunk, *obs.shape[1:])).reshape(-1)[:n]


def make_choose_actions(env, model_a, model_b, cfg: EvalConfig):
    """Action selection for a batch of games where model A and model B each own one side."""
    batch = cfg.batch_size
    game_range = jnp.arange(batch)

    def choose_actions(state, key, params_a, bn_a, params_b, bn_b, pid_a):
        key1, key2, key3 = jax.random.split(key, 3)

        # --- chance node?  then the next action is a dice roll -------------
        # chance logits carry the true roll probabilities (log(2/36), log(1/36))
        chance_logits = state.get_chance_logits()
        is_chance = state.has_chance_logits(chance_logits)

        # --- candidate moves (same legality rules as the training strategies)
        details = jax.vmap(_arr_legal_action_mask_details)(state._board, state._playable_dice)
        one_legal = details.one_move_legal_per_die                # (B, 52)
        two_legal = details.two_move_legal_per_die                # (B, 52, 52)
        has_two = two_legal.any(axis=(-2, -1))
        legal = jnp.where(has_two[:, None], two_legal.any(axis=-1), one_legal)

        # --- evaluate candidate boards with both models, select per game ---
        # candidate boards stay in the mover's (unflipped) perspective, exactly
        # like BackgammonTwoPlyStrategy, so equities are the mover's equities.
        boards1 = _arr_make_new_boards(state._board[:, None, :], details.candidate_diffs)
        obs1 = jax.vmap(_make_observation)(boards1.reshape((-1, ALL_GAME_POSITIONS)))
        eq1_a = eval_equity(model_a, params_a, bn_a, obs1, cfg.eval_chunk)
        eq1_b = eval_equity(model_b, params_b, bn_b, obs1, cfg.eval_chunk)
        eq1 = jnp.where(
            (state.current_player == pid_a)[:, None],
            eq1_a.reshape(boards1.shape[:-1]),
            eq1_b.reshape(boards1.shape[:-1]),
        )

        if cfg.mode == "2ply":
            # second move expansion; per-game candidate diffs keep the (first, second)
            # indexing aligned with two_legal
            boards2 = _arr_make_new_boards(
                boards1[:, :, None, :], details.candidate_diffs[:, None, :, :]
            )
            obs2 = jax.vmap(_make_observation)(boards2.reshape((-1, ALL_GAME_POSITIONS)))
            eq2_a = eval_equity(model_a, params_a, bn_a, obs2, cfg.eval_chunk)
            eq2_b = eval_equity(model_b, params_b, bn_b, obs2, cfg.eval_chunk)
            eq2 = jnp.where(
                (state.current_player == pid_a)[:, None, None],
                eq2_a.reshape(boards2.shape[:-1]),
                eq2_b.reshape(boards2.shape[:-1]),
            )
            eq2 = jnp.where(two_legal, eq2, NEG)
            eq2 = eq2.max(axis=-1)  # best continuation per first candidate
            eq = jnp.where(has_two[:, None], eq2, eq1)
        else:
            eq = eq1

        eq = jnp.where(legal, eq, NEG)
        eq = jnp.where(is_chance[:, None], 0.0, eq)  # neutralize garbage candidates at chance nodes

        # --- sample a move --------------------------------------------------
        if cfg.temperature > 0:
            idx = jax.random.categorical(key1, eq / jnp.maximum(cfg.temperature, 1e-6), axis=-1)
        else:
            idx = jnp.argmax(eq, axis=-1)
        action = details.candidate_action_indices[game_range, idx]
        # no legal move at all -> pass (NOOP); it is never among the candidates
        action = jnp.where(legal.any(axis=-1), action, NOOP_ACTION_IDX)
        # dice rolls are sampled from the chance logits with the true odds
        chance_action = jax.random.categorical(key2, chance_logits, axis=-1)
        action = jnp.where(is_chance, chance_action, action)
        return action, key3

    return choose_actions


_RUNNER_CACHE = {}


def make_chunk_fn(env, model_a, model_b, cfg: EvalConfig):
    """Jitted runner that plays cfg.batch_size games of model_a vs model_b.

    model_a owns player id `pid_a` (its net picks the moves of that side) and
    the fn returns (reward of model_a per game, terminated per game).
    """
    cache_key = (model_a.arch, model_b.arch, cfg.mode, cfg.batch_size, cfg.max_steps)
    if cache_key in _RUNNER_CACHE:
        return _RUNNER_CACHE[cache_key]

    batch = cfg.batch_size
    choose_actions = make_choose_actions(env, model_a, model_b, cfg)

    def chunk_fn(key, params_a, bn_a, params_b, bn_b, pid_a):
        key, init_key = jax.random.split(key)
        state = jax.vmap(env.init)(jax.random.split(init_key, batch))

        def cond(carry):
            _, state, _, i = carry
            return (~state.terminated.all()) & (i < cfg.max_steps)

        def body(carry):
            key, state, rewards, i = carry
            action, key = choose_actions(state, key, params_a, bn_a, params_b, bn_b, pid_a)
            key, step_key = jax.random.split(key)
            state = jax.vmap(env.step)(state, action, jax.random.split(step_key, batch))
            # backgammon has no intermediate rewards and terminated states are
            # stepped in place with zero rewards, so each game outcome is
            # accumulated exactly once
            rewards = rewards + state.rewards[:, pid_a]
            return key, state, rewards, i + 1

        _, state, rewards, _ = jax.lax.while_loop(
            cond, body, (key, state, jnp.zeros(batch), jnp.int32(0))
        )
        return rewards, state.terminated

    fn = jax.jit(chunk_fn)
    _RUNNER_CACHE[cache_key] = fn
    return fn


def play_matchup(model_a, model_b, n_games, key, env, cfg: EvalConfig):
    """Play n_games of model_a vs model_b.  Sides (player ids) are swapped every
    chunk of games so neither model is tied to a player id.

    Games are played in chunks of min(cfg.batch_size, remaining games) so no
    compute is spent on games whose result is discarded."""
    rewards_a, terminated = [], []
    batch = cfg.batch_size
    played = 0
    while played < n_games:
        eff_batch = min(batch, n_games - played)
        if eff_batch != cfg.batch_size:
            chunk_cfg = EvalConfig(cfg.mode, cfg.temperature, eff_batch, cfg.max_steps,
                                   cfg.eval_chunk)
        else:
            chunk_cfg = cfg
        key, sub = jax.random.split(key)
        # alternate which model owns player id 0 for symmetry; the chunk fn
        # always returns the rewards of model_a (which owns player id pid_a)
        pid_a = jnp.int32(0 if played % (2 * batch) == 0 else 1)
        r, t = make_chunk_fn(env, model_a, model_b, chunk_cfg)(
            sub, model_a.params, model_a.bn_state, model_b.params, model_b.bn_state, pid_a
        )
        rewards_a.append(numpy.asarray(r))
        terminated.append(numpy.asarray(t))
        played += eff_batch
    rewards_a = numpy.concatenate(rewards_a)[:n_games]
    terminated = numpy.concatenate(terminated)[:n_games]

    return {
        "wins_a": int(((rewards_a > 0) & terminated).sum()),
        "wins_b": int(((rewards_a < 0) & terminated).sum()),
        "draws": int((~terminated).sum()),  # reached max_steps without terminating
        "points_a": float(rewards_a.sum()),
        "games": n_games,
    }


def round_robin(models, active, games_per_matchup, key, env, cfg, log=print):
    """Play every pair in `active` (indices into `models`).  Returns results
    keyed by (i, j) with i < j; wins_a always belongs to models[i]."""
    results = {}
    total = len(active) * (len(active) - 1) // 2
    done = 0
    st0 = time.time()
    for a_pos in range(len(active)):
        for b_pos in range(a_pos + 1, len(active)):
            i, j = active[a_pos], active[b_pos]
            # deterministic but distinct RNG per stage key and pair
            sub = jax.random.fold_in(key, i * len(models) + j)
            res = play_matchup(models[i], models[j], games_per_matchup, sub, env, cfg)
            results[(i, j)] = res
            done += 1
            log(
                f"    [{done}/{total}] it={models[i].iteration} vs it={models[j].iteration}: "
                f"{res['wins_a']}-{res['wins_b']}"
                + (f" ({res['draws']} draws)" if res["draws"] else "")
                + f"  [{time.time() - st0:.0f}s]"
            )
    return results


def aggregate_scores(active, results):
    """Per-model wins (draws count 0.5), points and games over the given results."""
    score = {i: 0.0 for i in active}
    points = {i: 0.0 for i in active}
    games = {i: 0 for i in active}
    for (i, j), res in results.items():
        score[i] += res["wins_a"] + 0.5 * res["draws"]
        score[j] += res["wins_b"] + 0.5 * res["draws"]
        points[i] += res["points_a"]
        points[j] -= res["points_a"]
        games[i] += res["games"]
        games[j] += res["games"]
    rate = {i: (score[i] / games[i] if games[i] else 0.0) for i in active}
    return score, points, games, rate


def ranking_key(models, i, score, points):
    """Sort key: better models first (win rate, then points, then newer)."""
    return (-score[i], -points[i], -models[i].iteration)


def win_rate_stderr(p, n):
    return math.sqrt(max(p * (1.0 - p), 1e-9) / n) if n > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="checkpoints/selfplay_*.pkl",
                        help="glob of checkpoint files to select from")
    parser.add_argument("--models", nargs="*", default=None,
                        help="explicit checkpoint paths (overrides --checkpoints/--num-models)")
    parser.add_argument("--num-models", type=int, default=10,
                        help="number of checkpoints picked evenly across training time")
    parser.add_argument("--games-per-matchup", type=int, default=128)
    parser.add_argument("--eliminate", type=int, default=2,
                        help="models eliminated after each round robin")
    parser.add_argument("--final-games", type=int, default=512,
                        help="games for best vs 2nd/3rd/4th")
    parser.add_argument("--mode", choices=["1ply", "2ply"], default="1ply",
                        help="value-head lookahead depth used by both players "
                             "(2ply is ~8x slower per game)")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                        help="compute dtype for the value networks")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="sampling temperature over candidate equities (0 = greedy)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="games played in parallel per device chunk")
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    env = pgx.make("backgammon")
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    cfg = EvalConfig(mode=args.mode, temperature=args.temperature,
                     batch_size=args.batch_size, max_steps=args.max_steps)

    if args.models:
        paths = list(args.models)
    else:
        paths = select_checkpoints(glob.glob(args.checkpoints), args.num_models)
    if len(paths) < 2:
        raise SystemExit("need at least 2 checkpoints to run a tournament")

    print(f"loading {len(paths)} checkpoints...")
    models = []
    for p in paths:
        try:
            models.append(load_model(p, env, dtype=dtype))
            print(f"  loaded {os.path.basename(p)} (iteration {models[-1].iteration}, arch {models[-1].arch})")
        except ValueError as e:
            print(f"  SKIPPED {e}")
    if len(models) < 2:
        raise SystemExit("not enough evaluable checkpoints")

    key = jax.random.PRNGKey(args.seed)
    results_json = {
        "args": vars(args),
        "models": [
            {"path": m.path, "iteration": m.iteration, "arch": list(m.arch)} for m in models
        ],
    }

    # ------------------------------------------------------------------
    # stage 1: round robin with elimination until 4 models remain
    # ------------------------------------------------------------------
    active = list(range(len(models)))
    round_no = 1
    while len(active) > 4:
        print(f"\nround robin round {round_no}: {len(active)} models, "
              f"{len(active) * (len(active) - 1) // 2} matchups x {args.games_per_matchup} games")
        results = round_robin(models, active, args.games_per_matchup,
                              jax.random.fold_in(key, round_no), env, cfg)
        score, points, games, rate = aggregate_scores(active, results)
        order = sorted(active, key=lambda i: ranking_key(models, i, score, points))
        print("  standings:")
        for pos, i in enumerate(order):
            print(f"    {pos + 1:2d}. it={models[i].iteration:5d} "
                  f"win_rate={rate[i]:.3f} points={points[i]:+.1f} games={games[i]} "
                  f"  {os.path.basename(models[i].path)}")
        n_elim = min(args.eliminate, len(active) - 4)
        eliminated = order[-n_elim:]
        for i in eliminated:
            print(f"  eliminated it={models[i].iteration} (win_rate={rate[i]:.3f})")
        active = [i for i in active if i not in eliminated]
        results_json.setdefault("stage1_rounds", []).append({
            "round": round_no,
            "results": {f"{i}_vs_{j}": r for (i, j), r in results.items()},
            "eliminated": eliminated,
        })
        round_no += 1

    # ------------------------------------------------------------------
    # stage 2: final round robin among the last 4
    # ------------------------------------------------------------------
    print(f"\nfinal round robin: 4 models, 6 matchups x {args.games_per_matchup} games")
    final_results = round_robin(models, active, args.games_per_matchup,
                                jax.random.fold_in(key, 100), env, cfg)
    score, points, games, rate = aggregate_scores(active, final_results)
    final_order = sorted(active, key=lambda i: ranking_key(models, i, score, points))
    print("  final ranking:")
    for pos, i in enumerate(final_order):
        print(f"    {pos + 1}. it={models[i].iteration:5d} win_rate={rate[i]:.3f} "
              f"points={points[i]:+.1f}  {os.path.basename(models[i].path)}")
    results_json["stage2_final_round_robin"] = {
        f"{i}_vs_{j}": r for (i, j), r in final_results.items()
    }

    # ------------------------------------------------------------------
    # stage 3: best vs 2nd / 3rd / 4th with many games
    # ------------------------------------------------------------------
    best = final_order[0]
    h2h = {}
    print(f"\nhead to head: best (it={models[best].iteration}) vs the rest, "
          f"{args.final_games} games each")
    for pos, other in enumerate(final_order[1:], start=2):
        sub = jax.random.fold_in(key, 1000 + pos)
        res = play_matchup(models[best], models[other], args.final_games, sub, env, cfg)
        p = (res["wins_a"] + 0.5 * res["draws"]) / res["games"]
        h2h[pos] = {
            "opponent_iteration": models[other].iteration,
            "opponent_path": models[other].path,
            "best_wins": res["wins_a"],
            "opponent_wins": res["wins_b"],
            "draws": res["draws"],
            "games": res["games"],
            "best_win_rate": p,
            "stderr": win_rate_stderr(p, res["games"]),
            "best_points_per_game": res["points_a"] / res["games"],
        }
        print(
            f"    best vs #{pos} (it={models[other].iteration}): "
            f"win_rate={p:.3f} +/- {1.96 * h2h[pos]['stderr']:.3f} (95% CI), "
            f"score {res['wins_a']}-{res['wins_b']}"
            + (f" ({res['draws']} draws)" if res["draws"] else "")
            + f", points/game {h2h[pos]['best_points_per_game']:+.3f}"
        )
    results_json["stage3_best_vs_rest"] = h2h

    # ------------------------------------------------------------------
    # summary: does strength increase with training time?
    # ------------------------------------------------------------------
    print("\nsummary (best to worst):")
    iters = [models[i].iteration for i in final_order]
    for pos, i in enumerate(final_order):
        print(f"    {pos + 1}. iteration {models[i].iteration:5d}  {os.path.basename(models[i].path)}")
    print(f"    iterations by rank: {iters}")
    try:
        import scipy.stats
        ranks = [len(final_order) - pos for pos in range(len(final_order))]
        rho, pval = scipy.stats.spearmanr(iters, ranks)
        print(f"    rank/iteration spearman correlation: rho={rho:.2f} (p={pval:.3f}) "
              f"-> {'models improve with training time' if rho > 0 else 'no clear improvement with training time'}")
        results_json["rank_iteration_spearman"] = {"rho": float(rho), "p": float(pval)}
    except ImportError:
        pass

    results_json["final_ranking"] = [
        {"rank": pos + 1, "iteration": models[i].iteration, "path": models[i].path,
         "win_rate": rate[i], "points": points[i], "games": games[i]}
        for pos, i in enumerate(final_order)
    ]

    with open(args.out, "w") as f:
        json.dump(results_json, f, indent=2, default=float)
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
