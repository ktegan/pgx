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

from functools import partial
from typing import NamedTuple
import pickle
import time
import os

import wandb
from pydantic import BaseModel

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import pgx
from pgx.backgammon import ACTION_CHANCE_LENGTH, ACTION_MOVE_LENGTH, SimpleBackgammonEvaluator, SimpleBackgammonEvaluatorConfig
from pgx.experimental import auto_reset
from pgx.models.aznet import AZNet


class Config(BaseModel):
    env_id: pgx.EnvId = "backgammon"
    seed: int = 0
    max_num_iters: int = 100
    # network params
    num_channels: int = 256   # aka filters
    num_layers: int = 12      # aka residual blocks
    resnet_v2: bool = True
    # selfplay/distill params
    selfplay_batch_size: int = 2048
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 2048
    learning_rate: float = 0.001
    # evaluation config normalization/scaling
    evaluator_scale: float = 0.1
    temperature: float = 0.5
    checkpoint_path: str = "checkpoints/distilled_aznet.ckpt"

    class Config:
        extra = "forbid"


class DistillSample(NamedTuple):
    obs: jnp.ndarray
    target_policy: jnp.ndarray
    target_value: jnp.ndarray
    is_chance_node: jnp.ndarray


def get_forward(env, config):
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=config.num_channels,
            num_blocks=config.num_layers,
            resnet_v2=config.resnet_v2,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out
    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def get_distill_fn(env, config):
    evaluator_config = SimpleBackgammonEvaluatorConfig()
    evaluator = SimpleBackgammonEvaluator(evaluator_config)

    def step_fn(state, key) -> tuple:
        key1, key2 = jax.random.split(key)

        chance_logits = state.get_chance_logits()
        is_chance_node = state.has_chance_logits(chance_logits)

        # 1. Value Target
        raw_eval = evaluator.eval(state)  # shape [B]
        rewards_current = state.rewards[jnp.arange(state.rewards.shape[0]), state.current_player]
        value_target = jnp.where(state.terminated, rewards_current, jnp.tanh(config.evaluator_scale * raw_eval))

        # 2. Policy Target
        actions_all = jnp.arange(ACTION_MOVE_LENGTH)
        B = state.current_player.shape[0]

        # 1-ply lookahead using env.step
        keys = jax.random.split(key1, B)
        next_states = jax.vmap(
            jax.vmap(env.step, in_axes=(None, 0, None)),
            in_axes=(0, None, 0)
        )(state, actions_all, keys)

        # Flatten next_states to batch-evaluate
        flat_next_states = jax.tree_util.tree_map(
            lambda x: x.reshape((B * ACTION_MOVE_LENGTH,) + x.shape[2:]), next_states
        )
        flat_equities = evaluator.eval(flat_next_states)
        equities = flat_equities.reshape((B, ACTION_MOVE_LENGTH))

        # Adjust equities if player alternates or next state is terminated
        is_same_player = next_states.current_player == state.current_player[:, jnp.newaxis]
        rewards_next = next_states.rewards
        batch_idx = jnp.arange(B)[:, jnp.newaxis]
        player_idx = state.current_player[:, jnp.newaxis]
        rewards_for_current = rewards_next[batch_idx, jnp.arange(ACTION_MOVE_LENGTH), player_idx]

        q_values = jnp.where(
            next_states.terminated,
            rewards_for_current,
            jnp.tanh(config.evaluator_scale * equities) * jnp.where(is_same_player, 1.0, -1.0)
        )

        # Pad with chance actions to get full actions shape
        padded_q = jnp.concatenate([
            q_values,
            jnp.full((B, ACTION_CHANCE_LENGTH), -jnp.inf, dtype=q_values.dtype)
        ], axis=-1)

        # Mask illegal actions
        masked_q = jnp.where(state.legal_action_mask, padded_q, -jnp.inf)

        # Softmax to get target policy probabilities
        policy_target = jax.nn.softmax(masked_q / config.temperature, axis=-1)

        # Handle Chance Nodes (replace policy target with real chance probabilities)
        chance_probs = jax.nn.softmax(chance_logits, axis=-1)
        policy_target = jnp.where(
            is_chance_node[:, jnp.newaxis],
            chance_probs,
            policy_target
        )

        # 3. Step Environment
        action = jax.random.categorical(key2, jnp.log(policy_target + 1e-8), axis=-1)
        step_keys = jax.random.split(key2, B)
        next_state = jax.vmap(auto_reset(env.step, env.init))(state, action, step_keys)

        return next_state, DistillSample(
            obs=state.observation,
            target_policy=policy_target,
            target_value=value_target,
            is_chance_node=is_chance_node
        )

    def distill_rollout(rng_key: jnp.ndarray) -> DistillSample:
        batch_size = config.selfplay_batch_size // jax.device_count()
        rng_key, sub_key = jax.random.split(rng_key)
        keys = jax.random.split(sub_key, batch_size)
        state = jax.vmap(env.init)(keys)

        key_seq = jax.random.split(rng_key, config.max_num_steps)
        _, data = jax.lax.scan(step_fn, state, key_seq)
        return data

    return distill_rollout


def get_loss_fn(forward):
    def loss_fn(model_params, model_state, samples: DistillSample):
        (logits, value), model_state = forward.apply(
            model_params, model_state, samples.obs, is_eval=False
        )

        policy_loss = optax.softmax_cross_entropy(logits, samples.target_policy)
        policy_loss = jnp.where(samples.is_chance_node, 0.0, policy_loss)
        policy_loss = jnp.mean(policy_loss)

        value_loss = optax.l2_loss(value, samples.target_value)
        value_loss = jnp.mean(value_loss)

        return policy_loss + value_loss, (model_state, policy_loss, value_loss)
    return loss_fn


def get_train_fn(optimizer, loss_fn):
    @partial(jax.pmap, axis_name="i")
    def train(model, opt_state, data: DistillSample):
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


def train_distill(env, config):
    if jax.process_index() == 0:
        try:
            wandb.init(project="pgx-distill", config=config.model_dump())
        except Exception:
            pass
    distill_rollout = get_distill_fn(env, config)
    distill_rollout_pmapped = jax.pmap(distill_rollout)

    forward = get_forward(env, config)
    optimizer = optax.adam(learning_rate=config.learning_rate)
    loss_fn = get_loss_fn(forward)
    train = get_train_fn(optimizer, loss_fn)

    devices = jax.local_devices()
    num_devices = len(devices)

    # Initialize model and opt_state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
    opt_state = optimizer.init(params=model[0])
    # replicates to all devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Prepare checkpoint dir
    if jax.process_index() == 0:
        os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    host_seed = config.seed + jax.process_index()
    rng_key = jax.random.PRNGKey(host_seed)

    while True:
        if jax.process_index() == 0:
            print(log)
            try:
                wandb.log(log)
            except Exception:
                pass

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Distill rollout to generate targets using SimpleBackgammonEvaluator
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: DistillSample = distill_rollout_pmapped(keys)

        # Shuffle samples and make minibatches
        samples = jax.device_get(data)  # (#devices, batch, max_num_steps, ...)
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

        # Distillation Updates
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: DistillSample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "distill/policy_loss": policy_loss,
                "distill/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )

    # Save serialized results of the distilled neural network
    if jax.process_index() == 0:
        model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
        with open(config.checkpoint_path, "wb") as f:
            dic = {
                "config": config.model_dump(),
                "model": jax.device_get(model_0),
                "opt_state": jax.device_get(opt_state_0),
                "pgx.__version__": pgx.__version__,
            }
            pickle.dump(dic, f)
        print(f"Serialized distilled neural network checkpoint to {config.checkpoint_path}")
