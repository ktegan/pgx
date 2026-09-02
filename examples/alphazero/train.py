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
from pydantic import ConfigDict
import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple, ClassVar, Optional

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
from pydantic import BaseModel

from pgx.models.aznet import AZNet
from pgx.backgammon import (
    BackgammonTwoPlyStrategy,
    BackgammonTwoPlyChunkedStrategy,
    SimpleBackgammonEvaluator,
    SimpleBackgammonEvaluatorConfig,
    ACTION_TOTAL_LENGTH,
    _arr_legal_action_mask,
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
    def __init__(self, forward, value_to_scalar_fn, config=None):
        super().__init__(config)
        self.forward = forward
        self.value_to_scalar_fn = value_to_scalar_fn

    def eval(self, state: pgx.State) -> jnp.ndarray:
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
    temperature: float = 0.1     # when this is zero we take the best move every time, higher values increase randomness
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
    sval_max_steps: int = 50
    sval_custom_max_steps: int = 3
    num_champions: int = 4
    new_champion_iters: int = 5
    micro_batch_size: Optional[int] = 8192
    seed_backgame_prob: float = 0.05
    seed_blitz_prob: float = 0.05
    seed_race_prob: float = 0.00
    blitz_min_steps: int = 20    # a step is moving one checker and/or re-rolling dice, both players completing a turn usually takes 6 steps
    blitz_max_steps: int = 30
    race_num_moves: int = 150

    model_config = ConfigDict(extra='forbid')


def _generate_back_game_board() -> Array:
    board = numpy.zeros(ALL_GAME_POSITIONS, dtype=BOARD_DTYPE)

    # 1. Randomly select 1 to 3 board positions within the first 5 board positions (0 to 4)
    num_anchors = int(numpy.random.choice([1, 2, 3], p=[1/3, 1/3, 1/3]))
    anchor_positions = numpy.random.choice(5, size=num_anchors, replace=False)
    for pos in anchor_positions:
        board[pos] = int(numpy.random.choice([1, 2, 3], p=[1/3, 1/3, 1/3]))

    # 2. Assign black checkers to positions 16 through 22
    for pos in range(16, 23):
        board[pos] = int(numpy.random.choice([0, 1, 2, 3], p=[0.2, 0.1, 0.5, 0.2]))

    # 3. Adjust black checkers to exactly 15
    total_black = board.sum()
    if total_black > PLAYER_CHECKERS:
        while board.sum() > PLAYER_CHECKERS:
            pos = int(numpy.random.randint(0, BOARD_LENGTH))
            if board[pos] > 0:
                board[pos] -= 1
    elif total_black < PLAYER_CHECKERS:
        while board.sum() < PLAYER_CHECKERS:
            pos = int(numpy.random.randint(0, BOARD_LENGTH))
            board[pos] += 1

    # 4. Add white checkers (negative values)
    # For white checkers from board position starting at 0 if there are no black checkers, add white checkers
    for pos in range(24):
        if board[pos] == 0:
            num_white = int(numpy.random.choice([0, 1, 2, 3], p=[0.2, 0.1, 0.5, 0.2]))
            current_white = -int(board[board < 0].sum())
            if current_white + num_white > PLAYER_CHECKERS:
                num_white = PLAYER_CHECKERS - current_white
            board[pos] = -num_white
            if -board[board < 0].sum() == PLAYER_CHECKERS:
                break

    # If white has less than 15 checkers, randomly add white checkers where there are no black checkers
    current_white = -int(board[board < 0].sum())
    while current_white < PLAYER_CHECKERS:
        pos = int(numpy.random.randint(0, 24))
        if board[pos] <= 0:  # empty or already white
            board[pos] -= 1
            current_white += 1

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
    nodes there are no legal candidate moves (all candidate equities are -inf),
    so fall back to the strategy's best action, which is the NOOP pass there.
    The move sampling sanitizes -inf candidate equities to avoid NaNs in the
    softmax; those nodes' weights are never trained on (is_chance_node /
    is_turn_end masking in the loss).
    """
    best_action, candidate_equities, candidate_action_indices = \
        strategy.get_next_action_and_equities_batch(state, key, config, eval_cls)
    key_move, key_dice = jax.random.split(key)

    chance_logits = state.get_chance_logits()
    is_chance = state.has_chance_logits(chance_logits)
    dice_action = jax.random.categorical(key_dice, chance_logits, axis=-1)

    finite = jnp.isfinite(candidate_equities)
    has_candidates = finite.any(axis=-1)
    logits = jnp.where(finite, candidate_equities, 0.0) / jnp.maximum(temperature, 1e-6)
    probs = jax.nn.softmax(logits, axis=-1)
    game_range = jnp.arange(state.observation.shape[0])

    idx = jax.random.categorical(key_move, logits, axis=-1)
    move_action = candidate_action_indices[game_range, idx]
    move_action = jnp.where(has_candidates, move_action, best_action)
    action = jnp.where(is_chance, dice_action, move_action)

    action_weights = jnp.zeros((state.observation.shape[0], ACTION_TOTAL_LENGTH), dtype=jnp.float32)
    action_weights = action_weights.at[game_range[:, jnp.newaxis], candidate_action_indices].set(probs)
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
                    state, key1, strategy, model_config, eval_cls, config.temperature
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

        # Run selfplay for max_num_steps by batch
        rng_key, sub_key = jax.random.split(rng_key)
        keys = jax.random.split(sub_key, batch_size)
        state = jax.vmap(custom_init_fn)(keys)

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

            def sval_body(i, carry):
                state_carry, key_carry = carry
                key_act, key_reset, key_next = jax.random.split(key_carry, 3)

                # Query the lookahead strategy with the neural network evaluator
                action = strategy.get_next_action_batch(state_carry, key_act, model_config, eval_cls)

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


def make_evaluate_baseline_fn(forward, env, config):
    @jax.pmap
    def evaluate_baseline(baseline, rng_key, my_model):
        """A simplified evaluation by sampling. Only for debugging.
        Please use MCTS and run tournaments for serious evaluation."""
        my_player = 0
        my_model_params, my_model_state = my_model

        key, subkey = jax.random.split(rng_key)
        batch_size = config.selfplay_batch_size // jax.device_count()
        keys = jax.random.split(subkey, batch_size)
        state = jax.vmap(env.init)(keys)

        def body_fn(val):
            key, state, R, terminated = val
            (my_logits, _), _ = forward.apply(
                my_model_params, my_model_state, state.observation, is_eval=True
            )
            opp_logits, _ = baseline(state.observation)
            is_my_turn = (state.current_player == my_player).reshape((-1, 1))
            logits = jnp.where(is_my_turn, my_logits, opp_logits)
            key, subkey = jax.random.split(key)
            action = jax.random.categorical(subkey, logits, axis=-1)
            state = jax.vmap(env.step)(state, action)
            new_terminated = state.terminated
            reward_mask = new_terminated & ~terminated
            R = R + state.rewards[jnp.arange(batch_size), my_player] * reward_mask
            return (key, state, R, new_terminated)

        _, _, R, _ = jax.lax.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (key, state, jnp.zeros(batch_size), jnp.zeros(batch_size, dtype=jnp.bool_))
        )
        return R
    return evaluate_baseline


def make_evaluate_fn(forward, env, config):
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

        opponent_params = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, games_per_champ, axis=0),
            champs_params
        )
        opponent_states = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, games_per_champ, axis=0),
            champs_states
        )

        my_params_batch = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[jnp.newaxis, ...], eval_batch_size, axis=0),
            my_model_params
        )
        my_state_batch = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[jnp.newaxis, ...], eval_batch_size, axis=0),
            my_model_state
        )

        def body_fn(val):
            key, state, R, terminated = val
            is_my_turn = (state.current_player == my_player)

            def select_leaf(mask, val_my, val_opp):
                extra_dims = val_my.ndim - 1
                for _ in range(extra_dims):
                    mask = mask[:, jnp.newaxis]
                return jnp.where(mask, val_my, val_opp)

            active_params = jax.tree_util.tree_map(
                lambda m, o: select_leaf(is_my_turn, m, o),
                my_params_batch, opponent_params
            )
            active_state = jax.tree_util.tree_map(
                lambda m, o: select_leaf(is_my_turn, m, o),
                my_state_batch, opponent_states
            )

            (logits, _), _ = jax.vmap(
                lambda p, s, obs: forward.apply(p, s, obs[jnp.newaxis, ...], is_eval=True)
            )(active_params, active_state, state.observation)

            if logits is None:
                logits = jnp.zeros((state.observation.shape[0], env.num_actions), dtype=jnp.float32)
            else:
                logits = logits.squeeze(1)

            logits = logits - jnp.max(logits, axis=-1, keepdims=True)
            logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

            key, subkey1, subkey2 = jax.random.split(key, 3)
            action = jax.random.categorical(subkey1, logits, axis=-1)
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
        #selfplay_batch_size=8,
        load_checkpoint_path='checkpoints/distill_20260702_16:06:35_03000.pkl',
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

    if os.environ['STRAT'] == 'chunk':
        print("Using chunked strategy")
        strategy = BackgammonTwoPlyChunkedStrategy(env)
    else:
        print("Using normal strategy")
        strategy = BackgammonTwoPlyStrategy(env)

    class AZNetEvaluatorWrapper(AZNetEvaluator):
        def __init__(self, config=None):
            nonlocal forward, value_to_scalar
            super().__init__(forward, value_to_scalar, config)

    selfplay = make_selfplay_fn(forward, value_to_scalar, env, config, strategy, AZNetEvaluatorWrapper)
    compute_loss_input = make_compute_loss_input_fn(reward_transform, config)

    loss_fn = make_loss_fn(forward, env, config)
    train = make_train_fn(optimizer, loss_fn)
    evaluate_baseline = make_evaluate_baseline_fn(forward, env, config)
    evaluate = make_evaluate_fn(forward, env, config)

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
            if len(champion_pool) > config.num_champions:
                champion_pool.pop(0)

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
            log.update(
                {
                    f"now": datetime.datetime.now().astimezone().strftime("%Y%m%d_%H:%M:%S"),
                    f"eval/vs_champions/avg_R": R.mean().item(),
                    f"eval/vs_champions/win_rate": ((R > 0).sum() / R.size).item(),
                    f"eval/vs_champions/draw_rate": ((R == 0).sum() / R.size).item(),
                    f"eval/vs_champions/lose_rate": ((R < 0).sum() / R.size).item(),
                }
            )

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
