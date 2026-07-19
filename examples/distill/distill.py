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

import datetime
from functools import partial
from typing import NamedTuple
import pickle
import time
import os

import wandb
from pydantic import BaseModel, ConfigDict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy
import optax
import pgx
from pgx.backgammon import ACTION_TOTAL_LENGTH
from pgx.experimental import auto_reset
from pgx.models.aznet import AZNet


SNAPSHOT_ITERS = 500


class Config(BaseModel):
    env_id: pgx.EnvId = "backgammon"
    seed: int = 0
    max_num_iters: int = 8000
    # network params
    num_channels: int = 256   # aka filters
    num_layers: int = 12      # aka residual blocks
    resnet_v2: bool = True
    num_value_channels: int = 1
    # selfplay/distill params
    selfplay_batch_size: int = 256
    max_num_steps: int = 512
    # training params
    training_batch_size: int = 2048
    learning_rate: float = 0.001
    # evaluation config normalization/scaling
    train_policy_network: bool = True      # do we want to train an alpha zero policy network
    values_nodes_at_turn_end: bool = False   # when True we only train the value network on the last action of a turn
    checkpoint_base_path: str = "checkpoints/distill"

    model_config = ConfigDict(extra='forbid')


class DistillSample(NamedTuple):
    obs: jnp.ndarray
    target_policy: jnp.ndarray
    target_value: jnp.ndarray
    is_chance_node: jnp.ndarray
    is_value_node: jnp.ndarray


def get_forward(env, config):
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=config.num_channels,
            num_blocks=config.num_layers,
            resnet_v2=config.resnet_v2,
            train_policy_network=config.train_policy_network,
            num_value_channels=config.num_value_channels,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out
    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def get_tanh_scale(scores):
    return 1.0 / (jnp.std(scores) + 1e-5)


def get_distill_fn(env, config, evaluator, strategy, target_transform_fn=None, rewards_transform_fn=None):

    def step_fn(state, key) -> tuple:
        key1, key2 = jax.random.split(key)

        chance_logits = state.get_chance_logits()
        is_chance_node = state.has_chance_logits(chance_logits)

        # get value targets
        raw_eval = evaluator.eval(state)  # shape [B]
        if target_transform_fn is not None:
            eval_transformed = target_transform_fn(raw_eval)
        else:
            eval_transformed = raw_eval[..., jnp.newaxis]

        rewards_current = state.rewards[jnp.arange(state.rewards.shape[0]), state.current_player]
        if rewards_transform_fn is not None:
            rewards_transformed = rewards_transform_fn(rewards_current)
        else:
            rewards_transformed = rewards_current[..., jnp.newaxis]

        value_target = jnp.where(state.terminated[..., jnp.newaxis], rewards_transformed, eval_transformed)

        # get the next beset action
        B = state.current_player.shape[0]
        batched_eval_config = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, 0), B, axis=0),
            evaluator.get_config()
        )
        best_actions = strategy.get_next_action_batch(state, key1, batched_eval_config, evaluator.__class__)
        chance_probs = jax.nn.softmax(chance_logits, axis=-1)

        if config.train_policy_network:
            # update the policy target to predict the next best action
            policy_target = jax.nn.one_hot(best_actions, ACTION_TOTAL_LENGTH)
            policy_target = jnp.where(
                is_chance_node[:, jnp.newaxis],
                chance_probs,
                policy_target
            )
            action = jax.random.categorical(key2, jnp.log(policy_target + 1e-8), axis=-1)
        else:
            # we are not training a policy graph
            policy_target = jnp.zeros((B, ACTION_TOTAL_LENGTH), dtype=jnp.float32)
            chance_action = jax.random.categorical(key2, jnp.log(chance_probs + 1e-8), axis=-1)
            # Choose best action directly without categorical sampling if it is not a chance node
            action = jnp.where(is_chance_node, chance_action, best_actions)

        step_keys = jax.random.split(key2, B)
        next_state = jax.vmap(auto_reset(env.step, env.init))(state, action, step_keys)

        if config.values_nodes_at_turn_end:
            is_value_node = (state.current_player != next_state.current_player) | next_state.terminated
        else:
            is_value_node = jnp.ones(B, dtype=jnp.bool_)

        return next_state, DistillSample(
            obs=state.observation,
            target_policy=policy_target,
            target_value=value_target,
            is_chance_node=is_chance_node,
            is_value_node=is_value_node
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


def get_loss_fn(forward, config):
    def loss_fn(model_params, model_state, samples: DistillSample):
        (logits, value), model_state = forward.apply(
            model_params, model_state, samples.obs, is_eval=False
        )

        value_loss = optax.l2_loss(value, samples.target_value)
        value_loss = jnp.where(samples.is_value_node[:, jnp.newaxis], value_loss, 0.0)
        value_loss = jnp.mean(value_loss)

        if config.train_policy_network:
            policy_loss = optax.softmax_cross_entropy(logits, samples.target_policy)
            policy_loss = jnp.where(samples.is_chance_node, 0.0, policy_loss)
            policy_loss = jnp.mean(policy_loss)
        else:
            policy_loss = jnp.float32(0.0)

        return value_loss + policy_loss, (model_state, policy_loss, value_loss)
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


def train_distill(env, config, evaluator, strategy, target_transform_fn=None, rewards_transform_fn=None):
    if jax.process_index() == 0:
        try:
            wandb.init(project="pgx-distill", config=config.model_dump(), mode='offline')
        except Exception:
            pass
    distill_rollout = get_distill_fn(env, config, evaluator, strategy, target_transform_fn, rewards_transform_fn)
    distill_rollout_pmapped = jax.pmap(distill_rollout)

    forward = get_forward(env, config)
    optimizer = optax.adam(learning_rate=config.learning_rate)
    loss_fn = get_loss_fn(forward, config)
    train = get_train_fn(optimizer, loss_fn)

    devices = jax.local_devices()
    num_devices = len(devices)

    # Initialize model and opt_state
    dummy_rng_key = jax.random.PRNGKey(config.seed)
    dummy_state = jax.vmap(env.init)(jax.random.split(dummy_rng_key, 2))
    dummy_input = dummy_state.observation
    model = forward.init(dummy_rng_key, dummy_input)  # (params, state)
    opt_state = optimizer.init(params=model[0])

    mesh = jax.sharding.Mesh(numpy.array(devices), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x',))
    model, opt_state = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding),
        (model, opt_state)
    )

    # Prepare checkpoint dir
    if jax.process_index() == 0:
        os.makedirs(os.path.dirname(config.checkpoint_base_path), exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = 'starting distillation'

    host_seed = config.seed + jax.process_index()
    rng_key = jax.random.PRNGKey(host_seed)

    start_time = datetime.datetime.now().astimezone()  # use local host timezone
    start_time = start_time.strftime("%Y%m%d_%H:%M:%S")

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
        if jax.process_index() == 0 and iteration % 10 == 0:
            snapshot_iteration = SNAPSHOT_ITERS * (iteration // SNAPSHOT_ITERS)
            ckpt_path = f'{config.checkpoint_base_path}_{start_time}_{snapshot_iteration:05d}.pkl'
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            with open(ckpt_path, "wb") as f:
                dic = {
                    "config": config.model_dump(),
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "pgx.__version__": pgx.__version__,
                }
                pickle.dump(dic, f)
            log['checkpoint_path'] = ckpt_path
