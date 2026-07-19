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
import pgx

from examples.distill.distill import Config, train_distill
from omegaconf import OmegaConf
from pgx.backgammon import SimpleBackgammonEvaluator, BackgammonTwoPlyStrategy


TANH_SCALE = 1.0
GAMMON_FRAC = 0.15
BACKGAMMON_FRAC = 0.01
WIN_FRAC = 1.0 - GAMMON_FRAC - BACKGAMMON_FRAC

if __name__ == "__main__":
    # Initialize distributed environment safely
    try:
        jax.distributed.initialize()
    except Exception:
        pass

    #conf_dict = OmegaConf.from_cli()

    # for Backgammon we only want to train a value network and we don't want to
    # look at intermediate board states during a turn
    config = Config(
        train_policy_network=False,
        values_nodes_at_turn_end=True,
        num_value_channels=6
    )
    if jax.process_index() == 0:
        print(config)

    env = pgx.make(config.env_id)

    evaluator = SimpleBackgammonEvaluator()
    strategy = BackgammonTwoPlyStrategy(env)

    def reward_transform(rewards_current):
        rewards_current = jnp.round(rewards_current)
        term_ch0 = jnp.where(rewards_current == 3, 1.0, 0.0)
        term_ch1 = jnp.where(rewards_current == 2, 1.0, 0.0)
        term_ch2 = jnp.where(rewards_current == 1, 1.0, 0.0)
        term_ch3 = jnp.where(rewards_current == -1, 1.0, 0.0)
        term_ch4 = jnp.where(rewards_current == -2, 1.0, 0.0)
        term_ch5 = jnp.where(rewards_current == -3, 1.0, 0.0)
        return jnp.stack([term_ch0, term_ch1, term_ch2, term_ch3, term_ch4, term_ch5], axis=-1)

    def target_transform(raw_eval):
        assert len(raw_eval) > 10
        std_val = jnp.std(raw_eval)

        my_prob  = jnp.tanh(raw_eval * TANH_SCALE / std_val) * 0.5 + 0.5
        opp_prob = 1.0 - my_prob

        ch0 = my_prob  * BACKGAMMON_FRAC
        ch1 = my_prob  * GAMMON_FRAC
        ch2 = my_prob  * WIN_FRAC
        ch3 = opp_prob * WIN_FRAC
        ch4 = opp_prob * GAMMON_FRAC
        ch5 = opp_prob * BACKGAMMON_FRAC
        return jnp.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=-1)

    train_distill(env, config, evaluator, strategy, target_transform, reward_transform)
