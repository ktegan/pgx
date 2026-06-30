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

import dataclasses

from pgx.backgammon import BackgammonTwoPlyStrategy
from pgx.backgammon import SimpleBackgammonEvaluator, SimpleBackgammonEvaluatorConfig

from search import TrainConfig, OptunaConfig, run_tournament, optuna_param_search


def main_manual():
    config_lst = [
        SimpleBackgammonEvaluatorConfig(),
        SimpleBackgammonEvaluatorConfig(blots_weight=0.0, bar_weight=0.0, flexibility_weight=0.0),
    ]
    config = TrainConfig(
        num_generations=1,
        num_candidates=6,
        num_champions=2,
        batch_size=2000,
        abs_stddev=0.0,
        rel_stddev=0.03,
        mutate_field_prob=0.3,
        p_threshold=0.1,
        max_games_per_generation=40000,
        min_games_per_generation=80000,
        stddev_shrink=0.8,
        mutate_parameters=None,
        freeze_parameters=('prime_reward',),
    )

    mult_range = [0.7, 0.85, 1.0, 1.0/0.85, 1.0/0.7]
    config_dict = dataclasses.asdict(SimpleBackgammonEvaluatorConfig())
    for field in config_dict.keys():
        config_lst = []
        for mult in mult_range:
            cur_config = config_dict.copy()
            cur_config[field] = config_dict[field] * mult
            config_lst.append(SimpleBackgammonEvaluatorConfig(**cur_config))

        strategy_factory = lambda env: BackgammonTwoPlyStrategy(env)

        print(f'STARTING FOR {field=}')
        run_tournament('backgammon', SimpleBackgammonEvaluator, config, strategy_factory=strategy_factory, candidate_config_lst=config_lst)
        print(f'ENDING FOR {field=}')


def main_optuna():
    search_config = OptunaConfig()
    initial_params = SimpleBackgammonEvaluatorConfig()
    #strategy = OnePlyStrategy(env)
    strategy_factory = lambda env: BackgammonTwoPlyStrategy(env)
    optuna_param_search('backgammon', SimpleBackgammonEvaluator, search_config, strategy_factory=strategy_factory, initial_params=initial_params)

if __name__ == "__main__":
    # main_manual()
    main_optuna()
