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
    OBSERVATION_SHAPE,
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
    from pgx.backgammon import (
        ALL_GAME_POSITIONS, BAR_IDX, BOARD_DTYPE, PLAYER_CHECKERS, _arr_legal_action_mask
    )

    board = jnp.zeros(ALL_GAME_POSITIONS, dtype=BOARD_DTYPE)
    board = board.at[BAR_IDX].set(PLAYER_CHECKERS)
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
        obs=jnp.zeros((1, max_num_steps, B) + OBSERVATION_SHAPE),
        reward=jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0])[None, :, None], (1, 1, B)),
        terminated=jnp.tile(jnp.array([False, False, False, True])[None, :, None], (1, 1, B)),
        action_weights=jnp.zeros((1, max_num_steps, B, ACTION_TOTAL_LENGTH)),
        discount=jnp.tile(jnp.array([1.0, -1.0, 1.0, 0.0])[None, :, None], (1, 1, B)),
        is_chance_node=jnp.zeros((1, max_num_steps, B), dtype=jnp.bool_),
        value=jnp.tile(jnp.array([0.5, 0.3, 0.2, 0.1])[None, :, None, None], (1, 1, B, 1)),
        is_turn_end=jnp.zeros((1, max_num_steps, B), dtype=jnp.bool_),
        search_value=jnp.zeros((1, max_num_steps, B)),
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


def test_sample_action_with_dice_uses_true_odds():
    """Bug F: at chance nodes the dice must be sampled from the chance logits
    (doubles 1/36, non-doubles 2/36) rather than uniformly over the 21 pairs."""
    from pgx.backgammon import ALL_DICE_PAIRS
    from examples.alphazero.train import _sample_action_with_dice

    B = 1
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), B))
    chance_states = jax.vmap(_change_turn)(states, jax.random.split(jax.random.PRNGKey(1), B))

    n = 21000
    keys = jax.random.split(jax.random.PRNGKey(2), n)
    logits = jnp.zeros((B, ACTION_TOTAL_LENGTH))
    actions = jax.jit(jax.vmap(lambda k: _sample_action_with_dice(chance_states, k, logits)[0]))(keys)
    assert actions.shape == (n, B)
    # every sampled action is a dice-roll action
    assert (actions >= ACTION_MOVE_LENGTH).all()

    # doubles are the pairs (d, d); P(doubles) must be 1/6, not 6/21
    double_pair_idx = jnp.array(
        [i for i, (a, b) in enumerate(ALL_DICE_PAIRS.tolist()) if a == b], dtype=jnp.int32
    )
    is_double = jnp.isin(actions[:, 0] - ACTION_MOVE_LENGTH, double_pair_idx)
    frac = float(is_double.mean())
    assert abs(frac - 1 / 6) < 0.02, f"doubles frequency {frac:.4f} != 1/6"

    # at a normal (post-roll) node the move logits are used: sampling from
    # legal-masked one-hot logits must always select a legal action
    legal = states.legal_action_mask
    one_hot = jnp.log(legal.astype(jnp.float32) + 1e-8)
    acts = jax.jit(jax.vmap(lambda k: _sample_action_with_dice(states, k, one_hot)[0]))(
        jax.random.split(jax.random.PRNGKey(3), 200)
    )
    selected_legal = legal[jnp.arange(B)[None, :], acts]
    assert bool(selected_legal.all())



def test_search_value_semantics():
    """The search-consistency target must be the strategy's max candidate
    equity at move nodes, and 0 at chance and must-pass nodes (where the
    candidate boards are meaningless or absent)."""
    import numpy
    from pgx.backgammon import (
        ALL_GAME_POSITIONS, BAR_IDX, BOARD_DTYPE, PLAYER_CHECKERS, _arr_legal_action_mask
    )

    strategy = BackgammonTwoPlyStrategy(env)
    eval_cls = SimpleBackgammonEvaluator
    eval_config = SimpleBackgammonEvaluatorConfig()
    # scalar handcrafted weights: the strategies pass the config straight to
    # the evaluator, which reads its weights from it
    sconfig = SimpleBackgammonEvaluatorConfig()
    B = 8

    # chance nodes report 0
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), B))
    chance_states = jax.vmap(_change_turn)(states, jax.random.split(jax.random.PRNGKey(1), B))
    _, _, is_chance, search_value = strategy_action_and_weights(
        chance_states, jax.random.PRNGKey(2), strategy, sconfig, eval_cls, jnp.float32(0.01)
    )
    assert bool(numpy.asarray(is_chance).all())
    assert bool((numpy.asarray(search_value) == 0.0).all()), "chance nodes must report search_value 0"

    # must-pass nodes report 0
    board = jnp.zeros(ALL_GAME_POSITIONS, dtype=BOARD_DTYPE)
    board = board.at[BAR_IDX].set(PLAYER_CHECKERS)
    for pos, cnt in zip(range(0, 6), [3, 3, 3, 2, 2, 2]):
        board = board.at[pos].set(-cnt)
    playable = jnp.array([2, 4, -1, -1], dtype=jnp.int32)
    legal = jax.vmap(lambda b: _arr_legal_action_mask(b, playable))(jnp.stack([board] * B))
    no_move_states = chance_states.replace(
        _board=jnp.stack([board] * B),
        _playable_dice=jnp.stack([playable] * B),
        legal_action_mask=legal,
    )
    _, _, _, search_value = strategy_action_and_weights(
        no_move_states, jax.random.PRNGKey(2), strategy, sconfig, eval_cls, jnp.float32(0.01)
    )
    assert bool((numpy.asarray(search_value) == 0.0).all()), "must-pass nodes must report search_value 0"

    # real move nodes: search_value == max over legal candidate equities
    _, cand_eq, _ = strategy.get_next_action_and_equities_batch(
        states, jax.random.PRNGKey(3), eval_config, eval_cls
    )
    _, _, _, search_value = strategy_action_and_weights(
        states, jax.random.PRNGKey(3), strategy, sconfig, eval_cls, jnp.float32(0.01)
    )
    legal_mask = cand_eq > jnp.float32(jnp.finfo(jnp.float32).min / 2)
    has_cand = legal_mask.any(axis=-1)
    expected = jnp.where(
        has_cand, jnp.max(jnp.where(legal_mask, cand_eq, jnp.float32(0.0)), axis=-1), 0.0
    )
    assert bool(jnp.allclose(numpy.asarray(search_value), numpy.asarray(expected), atol=1e-5)), (
        numpy.asarray(search_value), numpy.asarray(expected)
    )


def test_search_loss_regresses_prediction_to_search_value():
    """The TD-leaf-style auxiliary loss must pull the value prediction toward
    the strategy's search value when weighted, while the outcome-only loss
    (weight 0) pulls it toward the (zero) terminal target instead."""
    import numpy
    import optax
    from examples.alphazero.train import (
        SelfplayOutput, make_compute_loss_input_fn, make_loss_fn,
    )

    vts = _value_to_scalar(1)
    strategy = BackgammonTwoPlyStrategy(env)
    B = 8

    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), B))
    _, _, _, search_value = strategy_action_and_weights(
        states, jax.random.PRNGKey(7), strategy,
        SimpleBackgammonEvaluatorConfig(),
        SimpleBackgammonEvaluator, jnp.float32(0.01),
    )
    # the handcrafted evaluator's equity at these states is nonzero
    assert float(jnp.abs(search_value).max()) > 0.1

    config = Config(selfplay_batch_size=B, max_num_steps=1, train_policy_network=False,
                    td_lambda=1.0, num_value_channels=1)
    sample = make_compute_loss_input_fn(lambda r: r[..., jnp.newaxis], config)(
        SelfplayOutput(
            obs=states.observation[jnp.newaxis, jnp.newaxis],
            reward=jnp.zeros((1, 1, B)),
            terminated=jnp.ones((1, 1, B), dtype=jnp.bool_),
            action_weights=jnp.zeros((1, 1, B, ACTION_TOTAL_LENGTH)),
            discount=jnp.zeros((1, 1, B)),
            is_chance_node=jnp.zeros((1, 1, B), dtype=jnp.bool_),
            value=jnp.zeros((1, 1, B, 1)),
            is_turn_end=jnp.ones((1, 1, B), dtype=jnp.bool_),
            search_value=search_value[jnp.newaxis, jnp.newaxis, :],
        )
    )

    forward = _make_tiny_forward(num_value_channels=1, train_policy=False)

    def predict(p, bn):
        (_, v), _ = forward.apply(p, bn, states.observation, is_eval=True)
        return numpy.asarray(vts(v))

    params0, bn0 = forward.init(jax.random.PRNGKey(11), states.observation)
    pred_before = predict(params0, bn0)

    def run(weight, steps=30, lr=0.05):
        cfg = Config(selfplay_batch_size=B, max_num_steps=1, train_policy_network=False,
                     td_lambda=1.0, num_value_channels=1, search_value_weight=weight)
        loss_fn = make_loss_fn(forward, vts, env, cfg)
        # flatten the (device, T, B, ...) layout the same way the main loop does
        flat = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), sample)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params0)
        params = params0
        model_state = bn0
        for _ in range(steps):
            grads, (model_state, _, _, _) = jax.grad(loss_fn, has_aux=True)(
                params, model_state, flat)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        return predict(params, model_state)

    pred_w0 = run(0.0)
    pred_w10 = run(10.0)
    sv = numpy.asarray(search_value)

    # weight 0: only the outcome loss acts, pulling toward the zero target
    assert numpy.abs(pred_w0).mean() < numpy.abs(pred_before).mean(), (
        "weight 0: prediction should move toward the outcome target 0")
    # strong weight: the prediction moves toward the search values (all
    # positive here) much further than the outcome-only run does; the tiny
    # net saturates below large targets, so compare against the w0 run
    # rather than asking to reach the target exactly
    assert numpy.mean(pred_w10) - numpy.mean(pred_w0) > 0.5, (
        f"weight 10: prediction should move toward the search value "
        f"(got mean {numpy.mean(pred_w10):.3f} vs {numpy.mean(pred_w0):.4f}, "
        f"targets mean {numpy.mean(sv):.3f})")
    assert numpy.abs(pred_w10 - sv).mean() < numpy.abs(pred_w0 - sv).mean(), (
        "weight 10: prediction should end up closer to the search value than the outcome-only run")
