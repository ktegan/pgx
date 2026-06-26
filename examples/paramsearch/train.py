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

from typing import Tuple, Optional
import time
import scipy.stats
import dataclasses
import jax
import jax.numpy as jnp
import wandb
import pgx
import optuna
from optuna.trial import create_trial, TrialState
from optuna.distributions import FloatDistribution
from pgx.backgammon import (
    SimpleEquityPredictor,
    SimpleEquityPredictorConfig,
    _observe,
    OBSERVATION_SIZE,
)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    num_champions: int = 2
    num_candidates: int = 6
    batch_size: int = 4000             # number of games played in parallel per batch
    num_generations: int = 100
    initial_rng_key: int = 0
    abs_stddev: float = 0.02
    rel_stddev: float = 0.05
    mutate_field_prob: float = 1.0
    p_threshold: float = 0.05
    max_games_per_generation: int = 4096 * 16
    min_games_per_generation: int = 4096
    stddev_shrink: float = 0.8
    mutate_parameters: Optional[Tuple[str]] = None
    freeze_parameters: Optional[Tuple[str]] = None

    @property
    def num_pairs_per_generation(self):
        return (self.num_candidates * (self.num_candidates - 1)) // 2


@dataclasses.dataclass(frozen=True)
class OptunaConfig:
    num_generations: int = 100
    trials_per_generation: int = 20
    champion_pool_size: int = 4         # we make a pool of recent champions (one per generation) to evaluate candidates
    batch_size: int = 4000
    min_games_per_trial: int = 2000     # minimum number of games to play each optuna trial (split across each champion)
    max_games_per_trial: int = 20000
    initial_rng_key: int = 0
    rel_bounds: float = 0.3
    reduction_factor: int = 3
    rolling_history_generations: int = 8   # We feed in trials from recent generations to give the optuna sampler more information
    mutate_parameters: Optional[Tuple[str]] = None
    freeze_parameters: Optional[Tuple[str]] = None


@jax.jit
def select_active_config(parent_player, config_0, config_1):
    new_fields = {}
    is_player_0 = (parent_player == 0)
    for field in dataclasses.fields(config_0):
        c0 = getattr(config_0, field.name)
        c1 = getattr(config_1, field.name)
        if len(c0.shape) > 1:
            mask = is_player_0
            for _ in range(len(c0.shape) - 1):
                mask = mask[:, jnp.newaxis]
            new_fields[field.name] = jnp.where(mask, c0, c1)
        else:
            new_fields[field.name] = jnp.where(is_player_0, c0, c1)
    return SimpleEquityPredictorConfig(**new_fields)


@jax.jit(static_argnames=('repeat_factor',))
def broadcast_config(config, repeat_factor):
    new_fields = {}
    for field in dataclasses.fields(config):
        val = getattr(config, field.name)
        repeated = jnp.repeat(val[:, jnp.newaxis, ...], repeat_factor, axis=1)
        new_fields[field.name] = repeated.reshape((-1,) + val.shape[1:])
    return SimpleEquityPredictorConfig(**new_fields)


@jax.jit
def compute_match_score(rewards, init_player):
    cum_0 = jnp.cumsum(init_player == 0)
    cum_1 = jnp.cumsum(init_player == 1)
    min_count = jnp.minimum(cum_0[-1], cum_1[-1])
    min_count = jnp.maximum(1, min_count)
    keep_mask = jnp.where(init_player == 0, cum_0 <= min_count, cum_1 <= min_count)
    return jnp.sum(rewards * keep_mask) / jnp.maximum(1.0, jnp.sum(keep_mask))


@jax.jit(static_argnames=('batch_size', 'env_id', 'model_cls'))
def play_games_jax(batch_size, env_id, model_cls, rng_key, model_params, p0_indices, p1_indices):
    """ this runs one set of batch_size games in parallel """
    env = pgx.make(env_id)
    num_matchups = p0_indices.shape[0]
    games_per_matchup = batch_size // num_matchups
    batch_size = num_matchups * games_per_matchup   # to ensure divisibility

    param_0_matchups = jax.tree_util.tree_map(lambda x: x[p0_indices], model_params)
    param_1_matchups = jax.tree_util.tree_map(lambda x: x[p1_indices], model_params)

    params_0 = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[:, jnp.newaxis, ...], games_per_matchup, axis=1).reshape((-1,) + x.shape[1:]),
        param_0_matchups
    )
    params_1 = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[:, jnp.newaxis, ...], games_per_matchup, axis=1).reshape((-1,) + x.shape[1:]),
        param_1_matchups
    )

    key, subkey = jax.random.split(rng_key)
    init_keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(init_keys)

    init_player = state.current_player

    def cond_fn(val):
        _, state, _ = val
        return ~(state.terminated.all())

    def body_fn(val):
        key, state, total_rewards = val

        chance_logits = jax.vmap(lambda s: s.get_chance_logits())(state)
        is_chance_node = jax.vmap(lambda s: s.has_chance_logits_recalc())(state)

        parent_player = state.current_player
        active_params = select_active_config(parent_player, params_0, params_1)

        # generate all possible next states for every current state and have the model evaluate them
        actions_all = jnp.arange(env.num_actions)
        key, lookahead_rng, step_rng = jax.random.split(key, 3)
        lookahead_keys = jax.random.split(lookahead_rng, batch_size)
        next_states = jax.vmap(jax.vmap(env.step, in_axes=(None, 0, None)), in_axes=(0, None, 0))(state, actions_all, lookahead_keys)

        obs = jax.vmap(jax.vmap(lambda s, p: _observe(s, p), in_axes=(0, None)), in_axes=(0, 0))(next_states, parent_player)

        broad_config = broadcast_config(active_params, env.num_actions)

        equities_flat = model_cls(broad_config).eval(obs.reshape((-1, OBSERVATION_SIZE)))
        equities = equities_flat.reshape((batch_size, env.num_actions))

        masked_equities = jnp.where(state.legal_action_mask, equities, jnp.finfo(equities.dtype).min)

        key, key1, key2 = jax.random.split(key, 3)
        chance_action = jax.random.categorical(key1, chance_logits, axis=-1)
        move_action = jnp.argmax(masked_equities, axis=-1)
        action = jnp.where(is_chance_node, chance_action, move_action)

        step_keys = jax.random.split(step_rng, batch_size)
        state = jax.vmap(env.step)(state, action, step_keys)

        # record reward for player 0
        total_rewards = total_rewards + state.rewards[:, 0]

        return key, state, total_rewards

    _, _, total_rewards = jax.lax.while_loop(cond_fn, body_fn, (key, state, jnp.zeros(batch_size)))

    rewards_match = total_rewards.reshape((num_matchups, games_per_matchup))
    init_player_match = init_player.reshape((num_matchups, games_per_matchup))

    match_scores = jax.vmap(compute_match_score)(rewards_match, init_player_match)
    var_rewards = jnp.var(total_rewards)
    return match_scores, var_rewards


def mutate_config(config, rng_key, abs_stddev, rel_stddev, mutate_field_prob=1.0,
                  mutate_parameters=None, freeze_parameters=None):
    keys = jax.random.split(rng_key, len(dataclasses.fields(config)) + 1)
    new_fields = {}

    for i, field in enumerate(dataclasses.fields(config)):
        val = jnp.asarray(getattr(config, field.name))
        k, subkey = jax.random.split(keys[i])

        stddev = abs_stddev + jnp.abs(val) * rel_stddev
        if (((mutate_parameters is not None) and (field.name not in mutate_parameters)) or
                ((freeze_parameters is not None) and (field.name in freeze_parameters))):
            stddev = 0.0
        stddev = jnp.where(jax.random.uniform(subkey) < mutate_field_prob, stddev, 0.0)
        noise = jax.random.normal(k, shape=val.shape) * stddev
        mutated_val = jnp.clip(val + noise, 0.0, None)

        new_fields[field.name] = mutated_val

    return SimpleEquityPredictorConfig(**new_fields), keys[-1]


def run_tournament(env_id:str, model_cls:pgx.core.Base, config: TrainConfig, candidate_config_lst=None):

    rng_key = jax.random.PRNGKey(config.initial_rng_key)
    default_config = SimpleEquityPredictor.get_default_config()

    if candidate_config_lst is None:
        pool = [default_config]
        for i in range(1, config.num_candidates):
            rng_key, subkey = jax.random.split(rng_key)
            mutated, _ = mutate_config(default_config, subkey, config.abs_stddev, config.rel_stddev,
                mutate_field_prob=config.mutate_field_prob, mutate_parameters=config.mutate_parameters, freeze_parameters=config.freeze_parameters)
            pool.append(mutated)
    else:
        pool = candidate_config_lst
        config = dataclasses.replace(config, num_candidates = len(pool))

    # Initialize wandb
    wandb.init(
        project="pgx-paramsearch",
        config={
            "num_candidates": config.num_candidates,
            "num_champions": config.num_champions,
            "default_config": dataclasses.asdict(default_config),
        },
        mode='offline'
    )

    p0_indices = []
    p1_indices = []
    for i in range(config.num_candidates):
        for j in range(i + 1, config.num_candidates):
            p0_indices.append(i)
            p1_indices.append(j)
    p0_indices = jnp.array(p0_indices)
    p1_indices = jnp.array(p1_indices)
    num_matchups = p0_indices.shape[0]

    print("=" * 60)
    print("         Backgammon Parameter Search Tournament")
    print("=" * 60)
    print(f"Candidates pool size: {config.num_candidates}")
    print(f"Champions selected per round: {config.num_champions}")
    print(f"Total round-robin matchups: {num_matchups}")
    print(f"Games per matchup per batch: {config.batch_size // num_matchups}")
    print(f"Dynamic stop: Worst champ vs best non-champ diff significant at p < {config.p_threshold}")
    print("=" * 60)

    current_abs_stddev = config.abs_stddev
    current_rel_stddev = config.rel_stddev
    z_crit = float(scipy.stats.norm.ppf(1.0 - config.p_threshold))

    generation = 0
    repeat_champ = 0
    while generation < config.num_generations:
        generation += 1
        st = time.time()
        print(f"\n[Generation {generation}] Playing tournament...")

        pool_configs_batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *pool)

        # Play batches of games dynamically
        match_scores = jnp.zeros(num_matchups)
        var_rewards_accum = 0.0
        b_idx = 0
        games_per_batch = config.batch_size // num_matchups  # games played per matchup, not per candidate

        while True:
            b_idx += 1
            rng_key, subkey = jax.random.split(rng_key)
            batch_scores, batch_var = play_games_jax(config.batch_size, env_id, model_cls, subkey, pool_configs_batched, p0_indices, p1_indices)
            match_scores = match_scores + batch_scores
            var_rewards_accum = var_rewards_accum + batch_var

            # Compute current running stats
            running_match_scores = match_scores / b_idx
            running_var = var_rewards_accum / b_idx

            # Compute candidate scores
            candidate_scores = jnp.zeros(config.num_candidates)
            candidate_scores = candidate_scores.at[p0_indices].add(running_match_scores)
            candidate_scores = candidate_scores.at[p1_indices].add(-running_match_scores)
            candidate_scores = candidate_scores / (config.num_candidates - 1)

            sorted_indices = jnp.argsort(candidate_scores)[::-1]
            best_champ_idx = sorted_indices[0]
            best_non_champ_idx = sorted_indices[config.num_champions]

            diff = candidate_scores[best_champ_idx] - candidate_scores[best_non_champ_idx]
            N = b_idx * games_per_batch
            std_diff = jnp.sqrt(2.0 * config.num_candidates * jnp.maximum(1e-8, running_var) / N) / (config.num_candidates - 1)
            threshold = z_crit * std_diff

            # Explicitly retrieve values to evaluate on host Python
            diff_val = float(diff)
            threshold_val = float(threshold)
            std_diff_val = float(std_diff)

            games_per_candidate = b_idx * games_per_batch * config.num_candidates
            print(f"    Played batch {b_idx} : N={games_per_candidate} games/candidate (min {config.min_games_per_generation}, max {config.max_games_per_generation}). Best champ vs best non-champ diff = {diff_val:.4f} (threshold = {threshold_val:.4f}, std_diff = {std_diff_val:.4f})")

            # Check stopping criteria
            if games_per_candidate >= config.min_games_per_generation and diff_val > threshold_val:
                print(f"    Difference {diff_val:.4f} > threshold {threshold_val:.4f} is statistically significant (p < {config.p_threshold}). Stopping.")
                break
            if games_per_candidate > config.max_games_per_generation:
                break

        match_scores = running_match_scores

        # Block to ensure execution time is measured correctly
        match_scores.block_until_ready()
        et = time.time()

        candidate_scores = jnp.zeros(config.num_candidates)
        candidate_scores = candidate_scores.at[p0_indices].add(match_scores)
        candidate_scores = candidate_scores.at[p1_indices].add(-match_scores)
        candidate_scores = candidate_scores / (config.num_candidates - 1)

        sorted_indices = jnp.argsort(candidate_scores)[::-1]
        top_indices = sorted_indices[:config.num_champions]

        print(f"Finished generation {generation} in {et - st:.2f}s")
        print("\nRanking:")
        for rank, idx in enumerate(sorted_indices):
            print(f"  {rank+1:2d}. Candidate {idx:02d}: Score = {candidate_scores[idx]:.4f}")

        print("\nSelected Champions for next round:")
        champions = []
        for rank, idx in enumerate(top_indices):
            champions.append(pool[idx])
            c = pool[idx]
            print(f"  Champion {rank+1} (Candidate {idx}): Score = {candidate_scores[idx]:.4f}")
            print(f"    Config: {c}")

        # Log to wandb
        log = {
            "generation": generation,
            "time_taken": et - st,
        }
        for rank, idx in enumerate(sorted_indices):
            log[f"scores/candidate_{idx}_score"] = float(candidate_scores[idx])

        best_champ = champions[0]
        log.update({
            "champion_1/score": float(candidate_scores[top_indices[0]]),
            "champion_1/pip_diff_weight": float(best_champ.pip_diff_weight),
            "champion_1/born_off_weight": float(best_champ.born_off_weight),
            "champion_1/made_points_weight": float(best_champ.made_points_weight),
            "champion_1/blots_weight": float(best_champ.blots_weight),
            "champion_1/bar_weight": float(best_champ.bar_weight),
            "champion_1/prime_weight": float(best_champ.prime_weight),
        })
        for i, r in enumerate(best_champ.prime_reward):
            log[f"champion_1/prime_reward_{i}"] = float(r)

        wandb.log(log)

        # Check if the top champion did not change, and shrink standard deviations
        if generation > 1 and sorted_indices[0] == 0:
            if repeat_champ >= 3:
                current_abs_stddev *= config.stddev_shrink
                current_rel_stddev *= config.stddev_shrink
                repeat_champ = 0
                print(f"    Top champion did not change. Shrinking mutation stddev by {config.stddev_shrink:.3f}: abs_stddev={current_abs_stddev:.6f}, rel_stddev={current_rel_stddev:.6f}")
            else:
                repeat_champ += 1
        else:
            repeat_champ = 0

        # Mutate to build new pool
        new_pool = []
        for champ in champions:
            new_pool.append(champ)

        for i in range(config.num_candidates - config.num_champions):
            champ = champions[i % config.num_champions]
            rng_key, subkey = jax.random.split(rng_key)
            mutated, _ = mutate_config(champ, subkey, current_abs_stddev, current_rel_stddev,
                mutate_field_prob=config.mutate_field_prob, mutate_parameters=config.mutate_parameters, freeze_parameters=config.freeze_parameters)
            new_pool.append(mutated)

        pool = new_pool

        time.sleep(2)


def iter_params_schema(config, mutate_parameters, freeze_parameters):
    """Generates schema information for each parameter in the config.

    Yields:
        name (str): The flat parameter name. For array parameters, this will be formatted
            as '{base_name}_{index}' for each element (e.g., 'param_0').
        val (float or jnp.ndarray or int): The default/current value of the parameter or
            parameter element.
        idx (int): None if this is a scalar, otherwise the index in the flattened array
        is_frozen (bool): True if the parameter should remain unchanged (frozen) during mutation.
    """
    for field in dataclasses.fields(config):
        name = field.name
        val = getattr(config, name)

        is_frozen = False
        if freeze_parameters is not None and name in freeze_parameters:
            is_frozen = True
        if mutate_parameters is not None and name not in mutate_parameters:
            is_frozen = True

        if is_frozen:
            yield name, val, None, True
        else:
            if isinstance(val, (int, float, jnp.ndarray)) and (not hasattr(val, "shape") or val.shape == ()):
                yield name, float(val), None, False
            elif hasattr(val, "shape") and len(val.shape) > 0:
                val_flat = jnp.ravel(val)
                for idx, val_elem in enumerate(val_flat):
                    yield name, float(val_elem), idx, False


def get_param_range(val, rel_bounds):
    if abs(val) < 1e-5:
        return -0.5, 0.5
    else:
        low = min(val * (1.0 - rel_bounds), val * (1.0 + rel_bounds))
        high = max(val * (1.0 - rel_bounds), val * (1.0 + rel_bounds))
    if val >= 0:   # in our case all weights are non-negative
        low = max(0.0, low)
    return low, high


def dict_to_config(trial_or_params, default_config, mutate_parameters, freeze_parameters, rel_bounds=0.2):
    is_trial = hasattr(trial_or_params, "suggest_float")
    params = trial_or_params.params if hasattr(trial_or_params, "params") else (trial_or_params if isinstance(trial_or_params, dict) else None)

    config_dict = {}
    for name, default_val, optional_idx, is_frozen in iter_params_schema(default_config, mutate_parameters, freeze_parameters):
        if is_frozen:
            assert not isinstance(default_val, list)
            config_dict[name] = default_val
            continue

        if is_trial:
            low, high = get_param_range(default_val, rel_bounds)
            suggested = trial_or_params.suggest_float(name, low, high)
        else:
            assert not isinstance(default_val, list)
            suggested = params[name]

        if optional_idx is None:
            config_dict[name] = jnp.float32(suggested)
        else:
            if name not in config_dict:
                config_dict[name] = [None] * jnp.ravel(getattr(default_config, name)).shape[0]
            config_dict[name][optional_idx] = suggested

    for name, val in config_dict.items():
        if isinstance(val, list):
            orig_shape = getattr(default_config, name).shape
            config_dict[name] = jnp.array(val, dtype=jnp.float32).reshape(orig_shape)

    return SimpleEquityPredictorConfig(**config_dict)


def config_to_dict(config, mutate_parameters, freeze_parameters):
    params = {}
    for name, val, is_frozen, _ in iter_params_schema(config, mutate_parameters, freeze_parameters):
        if not is_frozen:
            params[name] = val
    return params


def get_current_distributions(default_config, mutate_parameters, freeze_parameters, rel_bounds=0.2):
    distributions = {}
    for name, default_val, optional_idx, is_frozen in iter_params_schema(default_config, mutate_parameters, freeze_parameters):
        if not is_frozen:
            low, high = get_param_range(default_val, rel_bounds)
            distributions[name] = FloatDistribution(low=low, high=high)
    return distributions


def optuna_param_search(env_id: str, model_cls: pgx.core.Base, search_config: OptunaConfig, initial_params=None):
    """
    This runs a sequence of optuna trials which we call generations to search for an optimal
    set of model parameters.  In each generation we use the optuna CMA-ES sampler to come
    up with new candidate model parameters based on the observed performance (and pair-wise
    interaction) of past models.  Then we evalaute the new models relative to a pool of
    previous champion models that showed the best performance in previous generations.
    """
    rng_key = jax.random.PRNGKey(search_config.initial_rng_key)

    if initial_params is None:
        initial_params = SimpleEquityPredictor.get_default_config()

    champion_pool = [initial_params]

    wandb.init(
        project="pgx-paramsearch-optuna",
        config=dataclasses.asdict(search_config),
        mode='offline'
    )

    rolling_history = []
    prev_best_score = 0.0
    prev_k = 0
    prev_dropped_score = 0.0
    will_drop = False

    for generation in range(1, search_config.num_generations + 1):
        rng_key, generation_key = jax.random.split(rng_key)
        gen_seed = int(generation_key[0])
        print("=" * 60)
        print(f"   Optuna Parameter Search - Generation {generation}/{search_config.num_generations}")
        print("=" * 60)
        print(f"Current champion pool size: {len(champion_pool)}")
        for idx, champ in enumerate(champion_pool):
            print(f"  Champ {idx}: {champ}")
        print("-" * 60)

        k = len(champion_pool)

        # Apply score shift based on the fact that the new pool of champions is better
        if generation > 1 and search_config.rolling_history_generations > 0:
            if will_drop:
                delta = -(prev_best_score - prev_dropped_score) / prev_k
            else:
                delta = -prev_best_score / (prev_k + 1)

            print(f"Applying score shift of {delta:.4f} to historical trials.")

            new_history = []
            for gen_trials in rolling_history:
                shifted_gen = []
                for score, params in gen_trials:
                    shifted_gen.append((score + delta, params))
                new_history.append(shifted_gen)
            rolling_history = new_history

        def run_trial(trial):
            """ Run a single optuna trial.  This will play games between the candidate and the champions in the pool. """
            nonlocal rng_key

            # use CMA-ES algo to propose new parameters via trial.suggest_float()
            candidate = dict_to_config(trial, initial_params, search_config.mutate_parameters, search_config.freeze_parameters, search_config.rel_bounds)

            pool_configs = [candidate] + champion_pool
            pool_configs_batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *pool_configs)

            p0_indices = jnp.zeros(k, dtype=jnp.int32)
            p1_indices = jnp.arange(1, k + 1, dtype=jnp.int32)

            match_scores = jnp.zeros(k)
            b_idx = 0

            while True:
                b_idx += 1
                rng_key, subkey = jax.random.split(rng_key)

                # run one batch of games for the current candidate against the pool of champions
                batch_scores, _ = play_games_jax(
                    search_config.batch_size, env_id, model_cls, subkey, pool_configs_batched, p0_indices, p1_indices
                )

                # batch score is average score within the batch
                match_scores = match_scores + batch_scores
                running_match_scores = match_scores / b_idx
                avg_score = float(jnp.mean(running_match_scores))
                N = b_idx * search_config.batch_size

                # Check if this trial should be terminated early by Optuna (pruning)
                trial.report(avg_score, step=b_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                if N >= search_config.max_games_per_trial:
                    break

            return avg_score

        games_per_batch = search_config.batch_size // k
        min_resource_batches = int(jnp.ceil(search_config.min_games_per_trial / games_per_batch))
        study = optuna.create_study(
            sampler=optuna.samplers.CmaEsSampler(seed=gen_seed),
            direction="maximize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(
                min_resource=min_resource_batches,
                reduction_factor=search_config.reduction_factor
            )
        )

        # inject trials (parameters and results) from recent generations (not including past champions)
        current_dists = get_current_distributions(initial_params, search_config.mutate_parameters, search_config.freeze_parameters, search_config.rel_bounds)
        injected_count = 0
        if search_config.rolling_history_generations > 0:
            for gen_trials in rolling_history:
                for score, params in gen_trials:
                    # Clip parameters to the current search space bounds to prevent any out-of-range issues
                    clipped_params = {}
                    for name, val in params.items():
                        if name in current_dists:
                            dist = current_dists[name]
                            clipped_params[name] = float(max(dist.low, min(dist.high, val)))

                    trial = create_trial(
                        state=TrialState.COMPLETE,
                        value=score,
                        params=clipped_params,
                        distributions=current_dists
                    )
                    study.add_trial(trial)
                    injected_count += 1
            if injected_count > 0:
                print(f"Injected {injected_count} historical trials from the past generations.")

        # warm-start (seed) the study with the champions of the most recent generations
        for champ in champion_pool:
            champ_params = config_to_dict(champ, search_config.mutate_parameters, search_config.freeze_parameters)
            study.enqueue_trial(champ_params)

        study.optimize(run_trial, n_trials=search_config.trials_per_generation)

        best_trial = study.best_trial
        print(f"\nGeneration {generation} complete!")
        print(f"Best Trial Score: {best_trial.value:.4f}")
        print(f"Best Parameters: {best_trial.params}")

        best_candidate = dict_to_config(best_trial, initial_params, search_config.mutate_parameters, search_config.freeze_parameters, search_config.rel_bounds)

        log_dict = {
            "generation": generation,
            "best_score": best_trial.value,
        }
        for name, val in best_trial.params.items():
            log_dict[f"best_param/{name}"] = val
        wandb.log(log_dict)

        # Record shifting stats before updating pool
        prev_best_score = best_trial.value
        prev_k = k
        if len(champion_pool) + 1 > search_config.champion_pool_size:
            # The first champion in the pool (index 0) will be dropped
            # Its score in the current generation's study is study.trials[injected_count].value
            prev_dropped_score = study.trials[injected_count].value
            will_drop = True
        else:
            prev_dropped_score = 0.0
            will_drop = False

        # Collect completed trials from this generation to add to rolling history
        if search_config.rolling_history_generations > 0:
            current_gen_trials = []
            for trial in study.trials:
                if trial.state == TrialState.COMPLETE:
                    # Skip injected trials (< injected_count) and enqueued champion trials (< injected_count + k)
                    if trial.number >= injected_count + k:
                        current_gen_trials.append((trial.value, trial.params))

            rolling_history.append(current_gen_trials)
            if len(rolling_history) > search_config.rolling_history_generations:
                rolling_history.pop(0)

        champion_pool.append(best_candidate)
        if len(champion_pool) > search_config.champion_pool_size:
            champion_pool.pop(0)

        initial_params = best_candidate


def main_manual():
    mult_range = [0.94, 0.97, 0.99, 1.0, 1.01, 1.03, 1.06]
    config_dict = dataclasses.asdict(SimpleEquityPredictorConfig())
    for i, field in enumerate(config_dict.keys()):
        config = TrainConfig(
            num_generations=1,
            num_candidates=6,
            num_champions=2,
            batch_size=4000,
            abs_stddev=0.0,
            rel_stddev=0.03,
            mutate_field_prob=0.3,
            p_threshold=0.05,
            max_games_per_generation=50000,
            min_games_per_generation=2000,
            stddev_shrink=0.8,
            mutate_parameters=None,
            freeze_parameters=('prime_reward',),
        )
        config_lst = []
        for mult in mult_range:
            cur_config = config_dict.copy()
            cur_config[field] = config_dict[field] * mult
            config_lst.append(SimpleEquityPredictorConfig(**cur_config))

        print(f'STARTING FOR {field=}')
        run_tournament('backgammon', SimpleEquityPredictor, config, candidate_config_lst=config_lst)
        print(f'ENDING FOR {field=}')


def main_optuna():
    search_config = OptunaConfig(
        freeze_parameters=('prime_reward',),
    )
    initial_params = SimpleEquityPredictorConfig(
        pip_diff_weight=jnp.float32(1.0),
        born_off_weight=jnp.float32(1.0),
        made_points_old_weight=jnp.float32(1.0),
        made_points_weight=jnp.float32(1.0),
        blots_weight=jnp.float32(1.0),
        bar_weight=jnp.float32(1.0),
        end_game_home_board_weight=jnp.float32(1.0),
        end_game_pip_diff_weight=jnp.float32(1.0),
        dancing_weight=jnp.float32(1.0),
        flexibility_weight=jnp.float32(1.0),
        prime_weight=jnp.float32(1.0),
        prime_checker_offset=jnp.float32(1.0),
        prime_reward=jnp.array([0.0, 0.03, 0.06, 0.2, 0.3, 0.5, 1.0, 1.0], dtype=jnp.float32)
    )
    optuna_param_search('backgammon', SimpleEquityPredictor, search_config, initial_params=initial_params)


if __name__ == "__main__":
    # main_manual()
    main_optuna()
