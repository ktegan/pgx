# Copyright 2023 The Pgx Authors. All Rights Reserved.
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

from pgx.backgammon import BOARD_DTYPE
from pgx.backgammon import ALL_GAME_POSITIONS
from pgx.backgammon import PLAYER_CHECKERS
from pgx.backgammon import HOME_BOARD_LENGTH
from pydantic import BaseModel, field_validator, ConfigDict
import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple, ClassVar, Optional, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy
import mctx
import optax
import pgx
import wandb
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel, field_validator, ConfigDict

from pgx.models.aznet import AZNet
from pgx.backgammon import (
    BackgammonTwoPlyStrategy,
    BackgammonFullTurnStrategy,
    SimpleBackgammonEvaluator,
    SimpleBackgammonEvaluatorConfig,
    ACTION_TOTAL_LENGTH,
    NOOP_ACTION_IDX,
    _arr_legal_action_mask,
    _arr_legal_action_mask_details,
    _arr_make_new_boards,
    _make_observation,
    _is_no_contact,
    OFF_IDX,
    START_POSITIONS,
    BOARD_LENGTH,
    HOME_BOARD_LENGTH,
    ALL_GAME_POSITIONS,
    BOARD_DTYPE,
)
from pgx._src.struct import dataclass, field
from pgx._src.types import Array
import pgx.core as core

# Initialize distributed environment safely
try:
    jax.distributed.initialize()
except Exception:
    pass

devices = jax.local_devices()
num_devices = len(devices)


# possible things to try: Regret Matching


SNAPSHOT_ITERS = 4


@dataclass
class NNConfig:
    model_params: Array
    model_state: Array
    should_broadcast: ClassVar[bool] = False
    micro_batch_size: Optional[int] = field(pytree_node=False, default=None)


class AZNetEvaluator(core.Evaluator):
    # neural forward >> compaction overhead: strategies should skip illegal rows
    expensive_evaluation = True

    def __init__(self, forward, value_to_scalar_fn, config=None):
        super().__init__(config)
        self.forward = forward
        self.value_to_scalar_fn = value_to_scalar_fn

    def eval(self, state: pgx.State, idx=None) -> jnp.ndarray:
        (logits, value), _ = self.forward.apply(
            self.config.model_params, self.config.model_state, state.observation, is_eval=True
        )
        return self.value_to_scalar_fn(value)


class Config(BaseModel):
    env_id: pgx.EnvId = "backgammon"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 256   # aka filters
    num_layers: int = 12      # aka residual blocks
    resnet_v2: bool = True
    num_value_channels: int = 1
    model_dtype: str = "bfloat16"
    # selfplay params
    selfplay_batch_size: int = 128
    strategy_name: str = "fullturn"   # "two_ply" or "fullturn": move-selection strategy for selfplay and pool generation

    @field_validator("strategy_name")
    @classmethod
    def _check_strategy_name(cls, v):
        if v not in ("two_ply", "fullturn"):
            raise ValueError(f"strategy_name must be 'two_ply' or 'fullturn', got {v!r}")
        return v

    @field_validator("temperature_schedule")
    @classmethod
    def _check_temperature_schedule(cls, v):
        return _validate_temperature_schedule(v, "temperature_schedule")

    @field_validator("sval_temperature_schedule")
    @classmethod
    def _check_sval_temperature_schedule(cls, v):
        return _validate_temperature_schedule(v, "sval_temperature_schedule")
    temperature: float = 0.1     # when this is zero we take the best move every time, higher values increase randomness
    # per-game temperature mix: [(fraction, temperature), ...], fractions sum to 1.
    # Each game draws its temperature once (fixed for the whole game) - low
    # temperature games give predictable high-quality moves, high temperature
    # games explore lower-equity continuations.  None = use `temperature` for
    # every game.
    temperature_schedule: Optional[List[Tuple[float, float]]] = None
    # eval/vs_champions is a yardstick, not training data: default 0 = argmax
    eval_temperature: float = 0.0
    num_simulations: int = 32    # only active if train_policy_network is True
    max_num_steps: int = 1024
    # training params
    training_batch_size: int = 128
    learning_rate: float = 0.001
    td_lambda: float = 0.9        # reduce this to 0.75 as training advances, 0.0 is pure temporal differencing (learning from value estimate diffs during the game), 1.0 is pure monte carlo sampling (learning only from terminal states at the end of the game)
    # eval params
    eval_interval: int = 1
    load_checkpoint_path: str = ""
    checkpoint_base_path: str = "checkpoints/selfplay"
    train_policy_network: bool = True
    values_nodes_at_turn_end: bool = False
    sval_max_steps: int = 100
    sval_custom_max_steps: int = 3
    sval_random_prob: float = 0.5   # fraction of SVAL warmup games acted with uniform-random legal moves instead of strategy moves (off-corridor position diversity)
    # SVAL warmup strategy-move temperature: 0 = greedy argmax.  Per-game schedule
    # works like temperature_schedule (drawn once per game, fixed for the warmup)
    sval_temperature: float = 0.0
    sval_temperature_schedule: Optional[List[Tuple[float, float]]] = None
    num_champions: int = 4
    new_champion_iters: int = 5
    micro_batch_size: Optional[int] = 8192
    # seed-position pool mix (sums <= 1; the remainder plays the canonical opening)
    seed_backgame_prob: float = 0.10
    seed_blitz_prob: float = 0.10
    seed_race_prob: float = 0.00
    blitz_min_steps: int = 20    # a step is moving one checker and/or re-rolling dice, both players completing a turn usually takes 6 steps
    blitz_max_steps: int = 30
    race_num_moves: int = 150

    model_config = ConfigDict(extra='forbid')


def _validate_temperature_schedule(v, field_name):
    if v is None:
        return v
    if abs(sum(f for f, _ in v) - 1.0) > 1e-6:
        raise ValueError(f"{field_name} fractions must sum to 1, got {v}")
    if any(t < 0 for _, t in v):
        raise ValueError(f"{field_name} temperatures must be >= 0, got {v}")
    return v


def _generate_back_game_board(rng=None):
    """Random reachable back-game position: black has anchors deep in white's
    home board plus checkers scattered through the outer boards, white holds
    the rest of the board (including its home).  Everything is one uniform
    random composition of 15 checkers per side, so no structure is
    over-represented and no side is ever over-strength."""
    rng = rng if rng is not None else numpy.random
    board = numpy.zeros(ALL_GAME_POSITIONS, dtype=BOARD_DTYPE)

    # black: 1-3 anchors in white's home (points 0-4), the rest of the checkers
    # split uniformly over points 16-22
    num_anchors = int(rng.choice([1, 2, 3]))
    anchor_points = rng.choice(5, size=num_anchors, replace=False)
    for pos in anchor_points:
        board[pos] = int(rng.choice([1, 2, 3]))
    black_anchor_total = int(board[:5].sum())
    front_positions = rng.choice(7, size=PLAYER_CHECKERS - black_anchor_total, replace=True) + 16
    for pos in front_positions:
        board[pos] += 1

    # white: all points not occupied by black anchors are available; pick how
    # many points white holds (4..15), place one checker on each, then spread
    # the rest.  Extra white checkers go on points 5-11, keeping white's deep
    # home board sparse so black's anchors can still be hit out.
    free_points = [p for p in range(BOARD_LENGTH) if board[p] == 0]
    num_white_points = int(rng.choice(range(4, min(len(free_points), PLAYER_CHECKERS) + 1)))
    white_points = rng.choice(free_points, size=num_white_points, replace=False)
    for p in white_points:
        board[p] = -1
    spill_points = [p for p in white_points if p >= 5] or list(white_points)
    white_left = PLAYER_CHECKERS - num_white_points
    while white_left > 0:
        board[int(rng.choice(spill_points))] -= 1
        white_left -= 1

    assert board[BOARD_LENGTH:].sum() == 0  # nothing on bar/off
    assert board.sum() == 0                 # exactly 15 checkers per side
    return board


def gen_and_select_boards(rng_key, env, strategy, min_steps, max_steps, count, selection_fn):
    batch_size = 8192
    evaluator_config = SimpleBackgammonEvaluatorConfig()
    evaluator_config = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
        evaluator_config
    )
    collected = []

    while True:
        rng_key, subkey = jax.random.split(rng_key)

        # Sample a single step count for this batch
        steps = int(numpy.random.randint(min_steps, max_steps + 1))

        @jax.jit
        def run_steps(key):
            init_key, loop_key = jax.random.split(key)
            state = jax.vmap(env.init)(jax.random.split(init_key, batch_size))

            def body_fn(i, carry_state):
                state_carry, k = carry_state
                k_step, k_next = jax.random.split(k)
                chance_logits = state_carry.get_chance_logits()
                is_chance = state_carry.has_chance_logits(chance_logits)

                chance_probs = jax.nn.softmax(chance_logits, axis=-1)
                chance_action = jax.random.categorical(k_step, jnp.log(chance_probs + 1e-8), axis=-1)

                normal_action = strategy.get_next_action_batch(
                    state_carry, k_step, evaluator_config, SimpleBackgammonEvaluator
                )

                action = jnp.where(is_chance, chance_action, normal_action)

                step_keys = jax.random.split(k_next, batch_size)
                next_state = jax.vmap(env.step)(state_carry, action, step_keys)
                return next_state, k_next

            final_state, _ = jax.lax.fori_loop(0, steps, body_fn, (state, loop_key))
            mask = jax.vmap(selection_fn)(final_state._board)
            return final_state._board, mask

        boards, mask = run_steps(subkey)

        valid_boards = boards[mask]
        collected_np = jnp.array(valid_boards)
        if jax.process_index() == 0:
            print('    num collected ', collected_np.shape)
        if collected_np.shape[0] > 0:
            collected.append(collected_np)
            total_collected = sum(x.shape[0] for x in collected)
            if total_collected >= count:
                break

    all_collected = jnp.concatenate(collected, axis=0)
    return jnp.array(all_collected[:count], dtype=BOARD_DTYPE)


def collect_blitz_boards(rng_key, env, strategy, min_steps, max_steps, count):
    def selection_fn(board):
        made_points_mask = board[BOARD_LENGTH - HOME_BOARD_LENGTH : BOARD_LENGTH] >= 2
        made_points_count = made_points_mask.sum()
        return made_points_count >= 3

    return gen_and_select_boards(rng_key, env, strategy, min_steps, max_steps, count, selection_fn)


def collect_race_boards(rng_key, env, strategy, num_moves, count):
    def selection_fn(board):
        no_contact = _is_no_contact(board)
        no_born_off = (board[OFF_IDX] == 0) & (board[OFF_IDX + 1] == 0)

        white_checkers = jnp.where(board[HOME_BOARD_LENGTH:BOARD_LENGTH] < 0, -board[HOME_BOARD_LENGTH:BOARD_LENGTH], 0)
        white_ok = white_checkers.sum() >= 2

        black_checkers = jnp.where(board[0:BOARD_LENGTH - HOME_BOARD_LENGTH] > 0, board[0:BOARD_LENGTH - HOME_BOARD_LENGTH], 0)
        black_ok = black_checkers.sum() >= 2

        return no_contact & no_born_off & white_ok & black_ok

    return gen_and_select_boards(rng_key, env, strategy, num_moves, num_moves, count, selection_fn)


def generate_positions_pool(rng_key, env, strategy, config):
    pool_size = config.selfplay_batch_size
    if jax.process_index() == 0:
        print('  generating back games...')
    # 1. Generate back game boards on CPU
    backgame_count = int(pool_size * config.seed_backgame_prob)
    backgame_boards = [ _generate_back_game_board() for _ in range(backgame_count) ]
    backgame_boards = jnp.array(backgame_boards) if backgame_count > 0 else jnp.zeros((0, ALL_GAME_POSITIONS), dtype=BOARD_DTYPE)

    if jax.process_index() == 0:
        print('  generating blitz games...')
    # 2. Generate blitz boards
    blitz_rng_key, race_rng_key = jax.random.split(rng_key)
    blitz_count = int(pool_size * config.seed_blitz_prob)
    blitz_boards = collect_blitz_boards(
        blitz_rng_key, env, strategy, config.blitz_min_steps, config.blitz_max_steps, blitz_count
    ) if blitz_count > 0 else jnp.zeros((0, ALL_GAME_POSITIONS), dtype=BOARD_DTYPE)

    if jax.process_index() == 0:
        print('  generating race games...')
    # 3. Generate race boards
    race_count = int(pool_size * config.seed_race_prob)
    race_boards = collect_race_boards(
        race_rng_key, env, strategy, config.race_num_moves, race_count
    ) if race_count > 0 else jnp.zeros((0, ALL_GAME_POSITIONS), dtype=BOARD_DTYPE)

    if jax.process_index() == 0:
        print('  generating normal games...')
    # 4. Generate normal start boards
    normal_count = pool_size - len(backgame_boards) - len(blitz_boards) - len(race_boards)
    normal_board = jnp.array(START_POSITIONS, dtype=BOARD_DTYPE)
    normal_boards = jnp.tile(normal_board, (normal_count, 1))

    # Combine all
    starting_boards = numpy.concatenate([backgame_boards, blitz_boards, race_boards, normal_boards], axis=0)
    numpy.random.shuffle(starting_boards)

    return jax.device_put(jnp.array(starting_boards))


def make_recurrent_fn(value_to_scalar_fn, forward, env, config):
    def recurrent_fn(model_config, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        # model_config: NNConfig
        # state: embedding
        model_params = model_config.model_params
        model_state = model_config.model_state

        previous_player = state.current_player
        # backgammon requires a PRNGKey per game even though dice rolls are
        # chance actions; the env dynamics themselves are deterministic
        step_keys = jax.random.split(rng_key, state.observation.shape[0])
        state = jax.vmap(env.step)(state, action, step_keys)
        current_player = state.current_player

        (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
        if logits is None:
            logits = jnp.zeros((state.observation.shape[0], env.num_actions), dtype=jnp.float32)
        value = value_to_scalar_fn(value)

        # mask invalid actions
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        logits = state.get_normal_or_chance_logits_recalc(logits)

        discount = jnp.where(previous_player == current_player, 1.0, -1.0) * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        rewards = state.rewards[jnp.arange(state.rewards.shape[0]), previous_player]
        # NOTE: values_nodes_at_turn_end only selects which nodes the value loss
        # is computed on (see loss_fn); it must never touch the search backup,
        # where a +1 discount is required to propagate values through dice rolls
        # and same-player moves.  Zeroing it here collapsed MCTS node values.
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return recurrent_fn_output, state
    return recurrent_fn


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray
    is_chance_node: jnp.ndarray
    value: jnp.ndarray
    is_turn_end: jnp.ndarray


def strategy_action_and_weights(state, key, strategy, config, eval_cls, temperature):
    """
    Sample actions and policy weights for a batch of states using a backgammon
    lookahead strategy.

    Chance nodes (start of a turn, dice not yet rolled) are sampled from the
    chance logits so rolls follow the true dice probabilities.  At no-move
    nodes every candidate equity is masked to finfo.min, which overflows to
    -inf after the temperature division, so the softmax would be NaN: fall
    back to the strategy's best action (the NOOP pass) there and zero their
    policy weights.  At partially-legal nodes the masked candidates stay at
    -inf and get zero sampling probability.
    """
    best_action, candidate_equities, candidate_action_indices = \
        strategy.get_next_action_and_equities_batch(state, key, config, eval_cls)
    key_move, key_dice = jax.random.split(key)

    chance_logits = state.get_chance_logits()
    is_chance = state.has_chance_logits(chance_logits)
    dice_action = jax.random.categorical(key_dice, chance_logits, axis=-1)

    B = state.observation.shape[0]
    temperature = jnp.asarray(temperature) * jnp.ones((B,))  # scalar or per-game (B,)
    # mask BEFORE the temperature division: masked candidates hold finfo.min
    # (finite!), which only overflows to -inf for small temperatures - at
    # temperature ~1 the mask would stay finite and make illegal candidates
    # sampleable at no-move nodes
    logits = jnp.where(
        jnp.isfinite(candidate_equities), candidate_equities, jnp.float32(-jnp.inf)
    ) / jnp.maximum(temperature, 1e-6)[:, jnp.newaxis]
    has_candidates = jnp.isfinite(logits).any(axis=-1)
    game_range = jnp.arange(state.observation.shape[0])

    idx = jax.random.categorical(key_move, logits, axis=-1)
    move_action = candidate_action_indices[game_range, idx]
    move_action = jnp.where(has_candidates, move_action, best_action)
    action = jnp.where(is_chance, dice_action, move_action)

    probs = jax.nn.softmax(logits, axis=-1)
    probs = jnp.where(has_candidates[:, jnp.newaxis], probs, 0.0)
    action_weights = jnp.zeros((state.observation.shape[0], ACTION_TOTAL_LENGTH), dtype=jnp.float32)
    # scatter-max: candidate_action_indices can contain duplicates (doubles and
    # single-die states give die1_idx == die2_idx); .set() would clobber a legal
    # candidate's prob with a later duplicate's 0
    action_weights = action_weights.at[game_range[:, jnp.newaxis], candidate_action_indices].max(probs)
    return action, action_weights, is_chance


def make_selfplay_fn(forward, value_to_scalar_fn, env, config, strategy, eval_cls):
    recurrent_fn = make_recurrent_fn(value_to_scalar_fn, forward, env, config)

    @partial(jax.pmap, in_axes=(0, 0, None))
    def selfplay(model_config: NNConfig, rng_key: jnp.ndarray, pool_of_start_boards: jnp.ndarray) -> SelfplayOutput:
        batch_size = config.selfplay_batch_size // jax.device_count()

        def custom_init_fn(key):
            key1, key2 = jax.random.split(key)
            state = env._init(key1)

            idx = jax.random.randint(key2, shape=(), minval=0, maxval=pool_of_start_boards.shape[0])
            selected_board = pool_of_start_boards[idx]

            legal_action_mask = _arr_legal_action_mask(selected_board, state._playable_dice)

            state = state.replace(
                _board=selected_board,
                legal_action_mask=legal_action_mask
            )

            obs = env.observe(state)
            return state.replace(observation=obs)

        def step_fn(state, key) -> SelfplayOutput:
            key1, key2 = jax.random.split(key)
            observation = state.observation

            (logits, value), _ = forward.apply(
                model_config.model_params, model_config.model_state, state.observation, is_eval=True
            )
            if logits is None:
                logits = jnp.zeros((state.observation.shape[0], env.num_actions), dtype=jnp.float32)
            value_scalar = value_to_scalar_fn(value)

            chance_logits = state.get_chance_logits()
            is_chance_node = state.has_chance_logits(chance_logits)
            if config.train_policy_network:
                # if chance logits are available (meaning upcoming action is stochastic) use those instead
                logits = state.get_normal_or_chance_logits(logits, chance_logits)

                root = mctx.RootFnOutput(prior_logits=logits, value=value_scalar, embedding=state)

                policy_output = mctx.gumbel_muzero_policy(
                    params=model_config,
                    rng_key=key1,
                    root=root,
                    recurrent_fn=recurrent_fn,
                    num_simulations=config.num_simulations,
                    invalid_actions=~state.legal_action_mask,
                    qtransform=mctx.qtransform_completed_by_mix_value,
                    gumbel_scale=1.0,
                )
                action = policy_output.action
                action_weights = policy_output.action_weights
            else:
                action, action_weights, is_chance_node = strategy_action_and_weights(
                    state, key1, strategy, model_config, eval_cls, game_temperatures
                )

            actor = state.current_player
            keys = jax.random.split(key2, batch_size)
            previous_player = state.current_player
            state = jax.vmap(auto_reset(env.step, custom_init_fn))(state, action, keys)
            discount = jnp.where(previous_player == state.current_player, 1.0, -1.0) * jnp.ones_like(value_scalar)
            discount = jnp.where(state.terminated, 0.0, discount)

            is_turn_end = (previous_player != state.current_player) | state.terminated

            return state, SelfplayOutput(
                obs=observation,
                action_weights=action_weights,
                reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
                terminated=state.terminated,
                discount=discount,
                is_chance_node=is_chance_node,
                value=value,
                is_turn_end=is_turn_end,
            )

        rng_key, sub_key = jax.random.split(rng_key)
        keys = jax.random.split(sub_key, batch_size)
        state = jax.vmap(custom_init_fn)(keys)

        # per-game temperature: each game draws its temperature once from the
        # schedule (or uses the scalar `temperature` when no schedule is set)
        # and keeps it for the whole game, so low-temperature games stay
        # consistent and high-temperature games genuinely explore
        if config.temperature_schedule is not None:
            fractions = jnp.array([f for f, _ in config.temperature_schedule])
            temps = jnp.array([t for _, t in config.temperature_schedule])
            key_temp, rng_key = jax.random.split(rng_key)
            schedule_idx = jax.random.categorical(key_temp, jnp.log(fractions), axis=-1, shape=(batch_size,))
            game_temperatures = jnp.array([t for _, t in config.temperature_schedule])[schedule_idx]
        else:
            game_temperatures = config.temperature

        # per-game SVAL warmup temperature: drawn once per game (like the
        # collection temperature) so warmup corridors vary between greedy
        # (t=0), near-greedy and exploratory - on top of the random-move
        # games chosen by sval_random_prob
        if config.sval_temperature_schedule is not None:
            sval_fractions = jnp.array([f for f, _ in config.sval_temperature_schedule])
            key_temp, rng_key = jax.random.split(rng_key)
            sval_idx = jax.random.categorical(key_temp, jnp.log(sval_fractions), axis=-1, shape=(batch_size,))
            sval_game_temperatures = jnp.array([t for _, t in config.sval_temperature_schedule])[sval_idx]
        else:
            sval_game_temperatures = config.sval_temperature

        # this is a simplified version of Apply Self-Supervised Value Alignment (SVAL)
        # where we warmup all games within the batch by a number of randomly chosen steps based on position type
        max_warmup_steps = max(config.sval_max_steps, config.sval_custom_max_steps)
        if max_warmup_steps > 0:
            rng_key, sval_key = jax.random.split(rng_key)
            is_initial = (state._board == jnp.array(START_POSITIONS)).all(axis=-1)

            # Sample step limits for standard initial boards vs custom created boards
            random_steps_initial = jax.random.randint(sval_key, (batch_size,), 0, config.sval_max_steps + 1)
            random_steps_custom = jax.random.randint(sval_key, (batch_size,), 0, config.sval_custom_max_steps + 1)
            limit_steps = jnp.where(is_initial, random_steps_initial, random_steps_custom)

            # a per-game coin flip (fixed for the whole warmup) decides whether
            # the game is warmed up with random legal moves instead of greedy
            # strategy play: purely greedy warmups keep every collected state
            # on a greedy corridor, random play seeds the off-corridor
            # positions the value net must also judge (and punish)
            key_mask, rng_key = jax.random.split(rng_key)
            is_random_game = jax.random.uniform(key_mask, (batch_size,)) < config.sval_random_prob

            def sval_body(i, carry):
                state_carry, key_carry = carry
                key_act, key_rand, key_reset, key_next = jax.random.split(key_carry, 4)

                # strategy moves at the game's warmup temperature (0 = argmax);
                # chance nodes roll from the true dice odds inside
                strategy_action, _, _ = strategy_action_and_weights(
                    state_carry, key_act, strategy, model_config, eval_cls, sval_game_temperatures
                )

                # random: uniform over legal moves; at chance nodes the roll
                # is sampled from the true dice odds
                key_move, key_dice = jax.random.split(key_rand)
                move_logits = jnp.where(
                    state_carry.legal_action_mask, 0.0, jnp.float32(-jnp.inf)
                )
                random_move = jax.random.categorical(key_move, move_logits, axis=-1)
                chance_logits = state_carry.get_chance_logits()
                random_dice = jax.random.categorical(key_dice, chance_logits, axis=-1)
                random_action = jnp.where(
                    state_carry.has_chance_logits(chance_logits), random_dice, random_move
                )

                action = jnp.where(is_random_game, random_action, strategy_action)

                next_state = jax.vmap(auto_reset(env.step, custom_init_fn))(
                    state_carry, action, jax.random.split(key_reset, batch_size)
                )

                # Conditionally step only if the game has not reached its specific warmup steps limit
                should_step = i < limit_steps
                def leaf_fn(x, y):
                    mask = should_step.reshape((should_step.shape[0],) + (1,) * (x.ndim - 1))
                    return jnp.where(mask, x, y)

                state_next = jax.tree_util.tree_map(leaf_fn, next_state, state_carry)
                return state_next, key_next

            state, rng_key = jax.lax.fori_loop(0, max_warmup_steps, sval_body, (state, rng_key))

        key_seq = jax.random.split(rng_key, config.max_num_steps)
        _, data = jax.lax.scan(step_fn, state, key_seq)

        return data
    return selfplay


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray
    is_chance_node: jnp.ndarray
    is_turn_end: jnp.ndarray


def make_compute_loss_input_fn(reward_transform_fn, config):
    @jax.pmap
    def compute_loss_input(data: SelfplayOutput) -> Sample:
        batch_size = config.selfplay_batch_size // jax.device_count()
        # If episode is truncated, there is no value target
        # So when we compute value loss, we need to mask it
        value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

        # Compute value target
        value_next = jnp.concatenate([data.value[1:], jnp.zeros_like(data.value[:1])], axis=0)

        rewards_transformed = jax.vmap(reward_transform_fn)(data.reward)

        def body_fn(carry, i):
            ix = config.max_num_steps - i - 1
            # use value stabilization with temporal differencing targets
            v = rewards_transformed[ix] + data.discount[ix][..., jnp.newaxis] * (
                (1.0 - config.td_lambda) * value_next[ix] + config.td_lambda * carry
            )
            return v, v

        _, value_tgt = jax.lax.scan(
            body_fn,
            jnp.zeros((batch_size, config.num_value_channels)),
            jnp.arange(config.max_num_steps),
        )
        value_tgt = value_tgt[::-1, :, :]

        masked_weights = jnp.where(data.is_chance_node[..., jnp.newaxis], 0.0, data.action_weights)

        return Sample(
            obs=data.obs,
            policy_tgt=masked_weights,
            value_tgt=value_tgt,
            mask=value_mask,
            is_chance_node=data.is_chance_node,
            is_turn_end=data.is_turn_end,
        )
    return compute_loss_input





def make_loss_fn(forward, env, config):
    def loss_fn(model_params, model_state, samples: Sample):
        (logits, value), model_state = forward.apply(
            model_params, model_state, samples.obs, is_eval=False
        )

        if config.train_policy_network:
            policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
            policy_loss = jnp.where(samples.is_chance_node, 0.0, policy_loss)  # do not count chance nodes in policy loss
            if config.values_nodes_at_turn_end:
                policy_loss = jnp.where(samples.is_turn_end, policy_loss, 0.0)
            policy_loss = jnp.mean(policy_loss)
        else:
            policy_loss = jnp.float32(0.0)

        value_loss = optax.l2_loss(value, samples.value_tgt)
        v_mask = samples.mask
        v_mask = jnp.where(samples.is_chance_node, 0.0, v_mask)  # do not count chance nodes in value loss
        if config.values_nodes_at_turn_end:
            v_mask = jnp.where(samples.is_turn_end, v_mask, 0.0)
        value_loss = jnp.mean(value_loss * v_mask[..., jnp.newaxis])

        return policy_loss + value_loss, (model_state, policy_loss, value_loss)
    return loss_fn


def make_train_fn(optimizer, loss_fn):
    @partial(jax.pmap, axis_name="i")
    def train(model, opt_state, data: Sample):
        model_params, model_state = model
        grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
            model_params, model_state, data
        )
        grads = jax.lax.pmean(grads, axis_name="i")
        updates, opt_state = optimizer.update(grads, opt_state)
        model_params = optax.apply_updates(model_params, updates)
        model = (model_params, model_state)
        return model, opt_state, policy_loss, value_loss
    return train


def _sample_action_with_dice(state, key, logits):
    """Sample an action from legal-masked `logits`, but at chance nodes sample
    the dice roll from the chance logits so rolls follow the true 1/36, 2/36
    dice probabilities instead of the network's distribution over the 21 pairs."""
    key_move, key_dice, key_next = jax.random.split(key, 3)
    move_action = jax.random.categorical(key_move, logits, axis=-1)
    chance_logits = state.get_chance_logits()
    is_chance = state.has_chance_logits(chance_logits)
    dice_action = jax.random.categorical(key_dice, chance_logits, axis=-1)
    return jnp.where(is_chance, dice_action, move_action), key_next



def _value_lookahead_logits(forward, value_to_scalar_fn, my_params, my_state, opp_params, opp_state,
                            is_my_turn, state, num_actions, temperature):
    """Per-move-action logits from the value head: every legal first-move
    candidate is enumerated with the same rules the strategies use and scored
    by the side-to-move's value net (1-ply lookahead, the same player type the
    checkpoint tournament uses).  Sampling from these logits with
    _sample_action_with_dice makes eval games value-head contests instead of
    the uniform-random play a value-only net's zero logits would give."""
    B = state._board.shape[0]
    details = jax.vmap(_arr_legal_action_mask_details)(state._board, state._playable_dice)
    has_two = details.two_move_legal_per_die.any(axis=(-2, -1))
    legal = jnp.where(has_two[:, jnp.newaxis],
                      details.two_move_legal_per_die.any(axis=-1),
                      details.one_move_legal_per_die)                    # (B, 52)

    boards = _arr_make_new_boards(state._board[:, jnp.newaxis, :], details.candidate_diffs)  # (B, 52, 28)
    obs = jax.vmap(_make_observation)(boards.reshape((-1, ALL_GAME_POSITIONS)))              # (B*52, ...)

    # my net is one model: evaluate all candidate boards in a single pass
    (_, v_my), _ = forward.apply(my_params, my_state, obs, is_eval=True)
    eq_my = value_to_scalar_fn(v_my).reshape(B, 52)

    # opponents: k distinct nets, games laid out champion-major
    k = jax.tree_util.tree_leaves(opp_params)[0].shape[0]
    games_per_champ = B // k
    obs_by_champ = obs.reshape(k, games_per_champ * 52, *state.observation.shape[1:])
    (_, v_opp), _ = jax.vmap(
        lambda p, s, o: forward.apply(p, s, o, is_eval=True)
    )(opp_params, opp_state, obs_by_champ)
    eq_opp = value_to_scalar_fn(v_opp).reshape(B, 52)

    eq = jnp.where(is_my_turn[:, jnp.newaxis], eq_my, eq_opp)

    # mask candidate equities before scattering into the full action axis
    eq = jnp.where(legal, eq, jnp.finfo(eq.dtype).min)
    logits = jnp.full((B, num_actions), jnp.finfo(jnp.float32).min)
    game_range = jnp.arange(B)
    # scatter-max, not scatter-set: with doubles (or a single playable die) the
    # sorted dice give die1_idx == die2_idx, so first- and second-die candidates
    # map to the same action index; a later illegal duplicate would clobber the
    # legal candidate's equity with .set() (the env's own legal mask unions the
    # duplicates via .add(), so this must match that semantics)
    logits = logits.at[game_range[:, jnp.newaxis], details.candidate_action_indices].max(eq)
    # no legal move at all: the only pass is NOOP
    logits = logits.at[game_range, NOOP_ACTION_IDX].set(
        jnp.where(legal.any(axis=-1), jnp.float32(-jnp.inf), jnp.float32(0.0))
    )
    # argmax by default (eval_temperature=0): the eval is a yardstick, not a
    # sample of the selfplay temperature schedule
    return logits / jnp.maximum(temperature, 1e-6)


def make_evaluate_fn(forward, value_to_scalar_fn, env, config):
    @jax.pmap
    def evaluate(champions, rng_key, model_config: NNConfig):
        my_player = 0
        my_model_params = model_config.model_params
        my_model_state = model_config.model_state
        champs_params, champs_states = champions

        key, subkey = jax.random.split(rng_key)
        batch_size = config.selfplay_batch_size // jax.device_count()

        k = jax.tree_util.tree_leaves(champs_params)[0].shape[0]
        games_per_champ = batch_size // k
        eval_batch_size = k * games_per_champ

        keys = jax.random.split(subkey, eval_batch_size)
        state = jax.vmap(env.init)(keys)

        def body_fn(val):
            key, state, R, terminated = val
            is_my_turn = (state.current_player == my_player)

            logits = _value_lookahead_logits(
                forward, value_to_scalar_fn,
                my_model_params, my_model_state,
                champs_params, champs_states,
                is_my_turn, state, env.num_actions, config.eval_temperature
            )

            action, key = _sample_action_with_dice(state, key, logits)
            key, subkey2 = jax.random.split(key)
            step_keys = jax.random.split(subkey2, eval_batch_size)
            state = jax.vmap(env.step)(state, action, step_keys)
            new_terminated = state.terminated
            reward_mask = new_terminated & ~terminated
            R = R + state.rewards[jnp.arange(eval_batch_size), my_player] * reward_mask
            return (key, state, R, new_terminated)

        _, _, R, _ = jax.lax.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (key, state, jnp.zeros(eval_batch_size), jnp.zeros(eval_batch_size, dtype=jnp.bool_))
        )
        return R
    return evaluate


def main_selfplay():
    #conf_dict = OmegaConf.from_cli()
    #load_checkpoint_path='checkpoints/selfplay_20260705_21:11:04_00033.pkl',
    #max_num_iters=1,
    config = Config(
        train_policy_network=False,
        values_nodes_at_turn_end=True,
        num_value_channels=6,
        # collection temperature mix: keep the signal regime dominant while the
        # value head's spread is still small - 70% of games at <=0.01
        # (0.001 ~ argmax, 0.01 mostly-best), 30% exploratory slices
        temperature_schedule=[(0.40, 0.001), (0.30, 0.01), (0.20, 0.1), (0.10, 0.5)],
        # warmup corridors: half argmax, quarters at 0.01 / 0.1 (the
        # uniform-random half is separate, via sval_random_prob)
        sval_temperature_schedule=[(0.5, 0.0), (0.25, 0.01), (0.25, 0.1)],
        #selfplay_batch_size=8,
        load_checkpoint_path='checkpoints/selfplay_20260902_19:26:54_00028.pkl',
        #load_checkpoint_path='checkpoints/distill_20260702_16:06:35_03000.pkl',
        #load_checkpoint_path='checkpoints/selfplay_20260703_17:33:28_00035.pkl',
        #load_checkpoint_path='checkpoints/selfplay_20260704_11:32:50_00008.pkl',
        #load_checkpoint_path='checkpoints/selfplay_20260705_21:11:04_00033.pkl',
        #load_checkpoint_path='checkpoints/selfplay_20260711_16:51:20_00133.pkl',
        #max_num_iters=4,
        #load_checkpoint_champion_paths=[
        #    'checkpoints/selfplay_20260704_11:32:50_00008.pkl',
        #    'checkpoints/selfplay_20260704_11:32:50_00008.pkl',
        #    'checkpoints/selfplay_20260705_21:11:04_00000.pkl',
        #]
    )
    if jax.process_index() == 0:
        print(config)

    env = pgx.make(config.env_id)
    #baseline = pgx.make_baseline_model(config.env_id + "_v0")


    # this is specific to backgammon
    def value_to_scalar(value):
        probs = jax.nn.softmax(value, axis=-1)
        utilities = jnp.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], dtype=jnp.float32)
        return jnp.sum(probs * utilities, axis=-1)
    #def value_to_scalar(value):
    #    return value[..., 0]

    # this is specific to backgammon
    def reward_transform(rewards_current):
        rewards_current = jnp.round(rewards_current)
        term_ch0 = jnp.where(rewards_current == 3, 1.0, 0.0)
        term_ch1 = jnp.where(rewards_current == 2, 1.0, 0.0)
        term_ch2 = jnp.where(rewards_current == 1, 1.0, 0.0)
        term_ch3 = jnp.where(rewards_current == -1, 1.0, 0.0)
        term_ch4 = jnp.where(rewards_current == -2, 1.0, 0.0)
        term_ch5 = jnp.where(rewards_current == -3, 1.0, 0.0)
        return jnp.stack([term_ch0, term_ch1, term_ch2, term_ch3, term_ch4, term_ch5], axis=-1)
    #def reward_transform(rewards_current):
    #    return rewards_current[..., jnp.newaxis]

    target_dtype = jnp.float32
    if config.model_dtype == "float16":
        target_dtype = jnp.float16
    elif config.model_dtype == "bfloat16":
        target_dtype = jnp.bfloat16

    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=config.num_channels,
            num_blocks=config.num_layers,
            resnet_v2=config.resnet_v2,
            train_policy_network=config.train_policy_network,
            num_value_channels=config.num_value_channels,
            dtype=target_dtype,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out

    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
    optimizer = optax.adam(learning_rate=config.learning_rate)

    # both strategies drive selfplay moves and seed-position generation alike
    if config.strategy_name == "fullturn":
        print("Using full turn strategy")
        strategy = BackgammonFullTurnStrategy(env)
    elif config.strategy_name == "two_ply":
        print("Using 2-ply strategy")
        strategy = BackgammonTwoPlyStrategy(env)
    else:
        raise ValueError(f"unknown strategy_name: {config.strategy_name!r} (expected 'two_ply' or 'fullturn')")

    class AZNetEvaluatorWrapper(AZNetEvaluator):
        def __init__(self, config=None):
            nonlocal forward, value_to_scalar
            super().__init__(forward, value_to_scalar, config)

    selfplay = make_selfplay_fn(forward, value_to_scalar, env, config, strategy, AZNetEvaluatorWrapper)
    compute_loss_input = make_compute_loss_input_fn(reward_transform, config)

    loss_fn = make_loss_fn(forward, env, config)
    train = make_train_fn(optimizer, loss_fn)
    evaluate = make_evaluate_fn(forward, value_to_scalar, env, config)

    if jax.process_index() == 0:
        wandb.init(project="pgx-az", config=config.model_dump(), mode='offline')

    # Initialize model and opt_state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)

    if config.load_checkpoint_path:
        if jax.process_index() == 0:
            print(f"Loading checkpoint from {config.load_checkpoint_path}...")
        with open(config.load_checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
            model = checkpoint_data["model"]
        else:
            model = checkpoint_data

    # Cast weights and batch norm state to target dtype if different from float32
    if target_dtype != jnp.float32:
        if jax.process_index() == 0:
            print(f"Casting model parameters and state to {config.model_dtype}...")
        def cast_floating_leaves(tree, dtype):
            return jax.tree_util.tree_map(
                lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                tree
            )
        model = cast_floating_leaves(model, target_dtype)

    opt_state = optimizer.init(params=model[0])

    mesh = jax.sharding.Mesh(numpy.array(devices), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x',))
    model, opt_state = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding),
        (model, opt_state)
    )

    champion_pool = [jax.tree_util.tree_map(lambda x: x[0], model)] * config.num_champions
    # training iteration at which each pool member became a champion ("init" =
    # the loaded starting checkpoint); parallel to champion_pool, oldest first
    champion_tags = ["init"] * config.num_champions

    start_time = datetime.datetime.now().astimezone() # use local host timezone
    start_time = start_time.strftime("%Y%m%d_%H:%M:%S")

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    host_seed = config.seed + jax.process_index()
    rng_key = jax.random.PRNGKey(host_seed)
    while True:
        if iteration > 0 and iteration % config.new_champion_iters == 0:
            model_0 = jax.tree_util.tree_map(lambda x: x[0], model)
            champion_pool.append(model_0)
            champion_tags.append(iteration)
            if len(champion_pool) > config.num_champions:
                champion_pool.pop(0)
                champion_tags.pop(0)

        if iteration % config.eval_interval == 0:
            # Stack champions in the pool
            stacked_champions = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *champion_pool)
            # Replicate champions to all devices
            champions_replicated = jax.tree_util.tree_map(
                lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding),
                stacked_champions
            )

            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            model_config = NNConfig(model_params=model[0], model_state=model[1], micro_batch_size=config.micro_batch_size)
            R = evaluate(champions_replicated, keys, model_config)

            # per-champion breakdown: games are laid out champion-major within
            # each device, so R (D, k*g) reshapes to (D, k, g); pos0 = oldest
            # champion, pos{k-1} = newest.  A healthy run shows win_rate
            # increasing against OLDER champions while staying ~50% vs the
            # newest one.
            k = len(champion_pool)
            games_per_champ = R.shape[-1] // k
            R_by_champ = numpy.asarray(R).reshape(R.shape[0], k, games_per_champ)
            champ_log = {
                f"now": datetime.datetime.now().astimezone().strftime("%Y%m%d_%H:%M:%S"),
                f"eval/vs_champions/avg_R": R.mean().item(),
                f"eval/vs_champions/win_rate": ((R > 0).sum() / R.size).item(),
                f"eval/vs_champions/draw_rate": ((R == 0).sum() / R.size).item(),
                f"eval/vs_champions/lose_rate": ((R < 0).sum() / R.size).item(),
            }
            for pos in range(k):
                R_c = R_by_champ[:, pos, :]
                champ_log[f"eval/vs_champions/pos{pos}_win_rate"] = ((R_c > 0).sum() / R_c.size).item()
                champ_log[f"eval/champions/pos{pos}_iter"] = str(champion_tags[pos])
            log.update(champ_log)

            # Store checkpoints
            if jax.process_index() == 0:
                snapshot_iteration = SNAPSHOT_ITERS * (iteration // SNAPSHOT_ITERS)
                ckpt_path = f"{config.checkpoint_base_path}_{start_time}_{snapshot_iteration:05d}.pkl"
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
                with open(ckpt_path, "wb") as f:
                    dic = {
                        "config": config.model_dump(),
                        "rng_key": rng_key,
                        "model": jax.device_get(model_0),
                        "opt_state": jax.device_get(opt_state_0),
                        "iteration": iteration,
                        "frames": frames,
                        "hours": hours,
                        "pgx.__version__": pgx.__version__,
                        "env_id": env.id,
                        "env_version": env.version,
                    }
                    pickle.dump(dic, f)
                log['checkpoint_path'] = ckpt_path

        if jax.process_index() == 0:
            print(log)
            wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        if jax.process_index() == 0:
            print('  generating positions...')
        rng_key, subkey, pool_key = jax.random.split(rng_key, 3)
        starting_positions_pool = generate_positions_pool(pool_key, env, strategy, config)

        if jax.process_index() == 0:
            print('  starting selfplay...')
        keys = jax.random.split(subkey, num_devices)
        model_config = NNConfig(model_params=model[0], model_state=model[1], micro_batch_size=config.micro_batch_size)
        data: SelfplayOutput = selfplay(model_config, keys, starting_positions_pool)
        samples: Sample = compute_loss_input(data)

        # selfplay diagnostics: game length and finish rate react to the
        # temperature schedule and seed mix (high-temperature games finish
        # faster and more decisively; unfinished games hit max_num_steps)
        if jax.process_index() == 0:
            term = numpy.asarray(data.terminated)[0]  # (steps, batch) of device 0
            first_term = numpy.where(term.any(axis=0), term.argmax(axis=0), -1)
            finished = first_term >= 0
            log["selfplay/mean_game_length"] = float(first_term[finished].mean()) if finished.any() else 0.0
            log["selfplay/unfinished_rate"] = float((~finished).mean())

        if jax.process_index() == 0:
            print('  shuffle make minibatches...')
        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2] * jax.process_count()
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        local_training_batch_size = config.training_batch_size // jax.process_count()
        num_updates = samples.obs.shape[0] // local_training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        if jax.process_index() == 0:
            print('  starting training...')
        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )


if __name__ == "__main__":
    main_selfplay()
