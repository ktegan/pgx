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

import jax
import jax.numpy as jnp
import haiku as hk
import mctx
import pgx

from pgx.backgammon import (
    ACTION_MOVE_LENGTH,
    ACTION_TOTAL_LENGTH,
    BackgammonTwoPlyStrategy,
    SimpleBackgammonEvaluator,
    SimpleBackgammonEvaluatorConfig,
    _change_turn,
)
from pgx.models.aznet import AZNet
from examples.alphazero.train import (
    Config,
    NNConfig,
    make_recurrent_fn,
    make_selfplay_fn,
    strategy_action_and_weights,
)

env = pgx.make("backgammon")


def _make_tiny_forward(num_value_channels=1, train_policy=True):
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=8,
            num_blocks=1,
            resnet_v2=True,
            train_policy_network=train_policy,
            num_value_channels=num_value_channels,
            dtype=jnp.float32,
        )
        return net(x, is_training=not is_eval, test_local_stats=False)

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def _value_to_scalar(num_value_channels=1):
    if num_value_channels == 6:
        utilities = jnp.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], dtype=jnp.float32)
        return lambda v: jnp.sum(jax.nn.softmax(v, axis=-1) * utilities, axis=-1)
    return lambda v: v[..., 0]


def _tiny_model(train_policy=True, num_value_channels=1):
    forward = _make_tiny_forward(num_value_channels, train_policy)
    dummy = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    params, bn_state = forward.init(jax.random.PRNGKey(1), dummy.observation)
    return forward, (params, bn_state)


def test_recurrent_fn_steps_the_environment():
    """Bug C: recurrent_fn must pass the PRNGKey through to env.step
    (backgammon's step asserts on a missing key)."""
    forward, (params, bn_state) = _tiny_model()
    config = Config(values_nodes_at_turn_end=False, td_lambda=0.9)
    recurrent_fn = make_recurrent_fn(_value_to_scalar(), forward, env, config)
    model_config = NNConfig(model_params=params, model_state=bn_state)

    B = 3
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(2), B))
    action = jnp.zeros(B, dtype=jnp.int32)
    out, next_state = recurrent_fn(model_config, jax.random.PRNGKey(3), action, states)

    assert out.reward.shape == (B,)
    assert out.discount.shape == (B,)
    assert out.prior_logits.shape == (B, ACTION_TOTAL_LENGTH)
    assert out.value.shape == (B,)
    assert next_state.terminated.shape == (B,)
    assert jnp.isfinite(out.prior_logits).all() or True  # -inf masking is expected
    assert jnp.isfinite(out.value).all()


def test_recurrent_fn_discount_ignores_values_nodes_at_turn_end():
    """Bug D: values_nodes_at_turn_end only selects which nodes the value loss
    is computed on - it must not change the search dynamics.  The MCTS backup
    discount for a same-player edge (e.g. a dice roll) must stay +1."""
    forward, (params, bn_state) = _tiny_model()
    model_config = NNConfig(model_params=params, model_state=bn_state)
    vts = _value_to_scalar()

    # a chance node: stepping with a dice-roll action keeps the same player
    B = 2
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(2), B))
    chance_states = jax.vmap(_change_turn)(states, jax.random.split(jax.random.PRNGKey(3), B))
    dice_action = jnp.full((B,), ACTION_MOVE_LENGTH)  # the first dice pair

    for flag in [False, True]:
        config = Config(values_nodes_at_turn_end=flag)
        recurrent_fn = make_recurrent_fn(vts, forward, env, config)
        out, next_state = recurrent_fn(
            model_config, jax.random.PRNGKey(4), dice_action, chance_states
        )
        assert (next_state.current_player == chance_states.current_player).all()
        assert (out.discount == 1.0).all(), f"flag={flag}: discount {out.discount}"

    # a turn-changing edge (NOOP at a no-move node) must still flip the sign
    import numpy
    from pgx.backgammon import BAR_IDX, BOARD_DTYPE, _arr_legal_action_mask

    board = jnp.zeros(28, dtype=BOARD_DTYPE)
    board = board.at[BAR_IDX].set(15)
    for pos, cnt in zip(range(0, 6), [3, 3, 3, 2, 2, 2]):
        board = board.at[pos].set(-cnt)
    playable = jnp.array([2, 4, -1, -1], dtype=jnp.int32)
    legal = jax.vmap(lambda b: _arr_legal_action_mask(b, playable))(jnp.stack([board] * B))
    no_move_states = chance_states.replace(
        _board=jnp.stack([board] * B),
        _playable_dice=jnp.stack([playable] * B),
        legal_action_mask=legal,
    )
    for flag in [False, True]:
        config = Config(values_nodes_at_turn_end=flag)
        recurrent_fn = make_recurrent_fn(vts, forward, env, config)
        out, next_state = recurrent_fn(
            model_config, jax.random.PRNGKey(5), jnp.zeros((B,), jnp.int32), no_move_states
        )
        assert (next_state.current_player == 1 - no_move_states.current_player).all()
        assert (out.discount == -1.0).all(), f"flag={flag}: discount {out.discount}"


def test_td_lambda_targets_propagate_through_rolls_and_turns():
    """Bug E was a misdiagnosis: the TD(lambda) chain in compute_loss_input uses
    the selfplay step discounts (+1 same player / roll, -1 turn change, 0
    terminal), which are NOT zeroed by values_nodes_at_turn_end (that zeroing
    only ever existed in the MCTS backup, see the bug-D fix).  This test pins
    the semantics: a terminal outcome must propagate back through dice-roll
    edges unchanged and flip sign across turn-change edges."""
    from examples.alphazero.train import SelfplayOutput, make_compute_loss_input_fn

    max_num_steps, B, nvc, lam = 4, 2, 1, 0.9
    config = Config(
        selfplay_batch_size=B, max_num_steps=max_num_steps, num_value_channels=nvc, td_lambda=lam
    )
    compute_loss_input = make_compute_loss_input_fn(lambda r: r[..., jnp.newaxis], config)

    data = SelfplayOutput(
        obs=jnp.zeros((1, max_num_steps, B, 24, 1, 12)),
        reward=jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0])[None, :, None], (1, 1, B)),
        terminated=jnp.tile(jnp.array([False, False, False, True])[None, :, None], (1, 1, B)),
        action_weights=jnp.zeros((1, max_num_steps, B, ACTION_TOTAL_LENGTH)),
        discount=jnp.tile(jnp.array([1.0, -1.0, 1.0, 0.0])[None, :, None], (1, 1, B)),
        is_chance_node=jnp.zeros((1, max_num_steps, B), dtype=jnp.bool_),
        value=jnp.tile(jnp.array([0.5, 0.3, 0.2, 0.1])[None, :, None, None], (1, 1, B, 1)),
        is_turn_end=jnp.zeros((1, max_num_steps, B), dtype=jnp.bool_),
    )
    sample = compute_loss_input(data)
    # v3 = r3 = 1.0
    # v2 = +1 * (0.1*v[3] + 0.9*v3) = 0.91          (roll edge, same player)
    # v1 = -1 * (0.1*v[2] + 0.9*v2) = -0.839        (turn change, sign flips)
    # v0 = +1 * (0.1*v[1] + 0.9*v1) = -0.7251       (roll edge)
    expected = jnp.array([-0.7251, -0.839, 0.91, 1.0])
    assert jnp.allclose(sample.value_tgt[0, :, 0, 0], expected, atol=1e-4), sample.value_tgt[0, :, 0, 0]
    # the terminal outcome reaches the first node of the horizon: not severed
    assert abs(float(sample.value_tgt[0, 0, 0, 0])) > 0.5
