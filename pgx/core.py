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

import warnings
import abc
import dataclasses
from typing import Literal, Optional, Tuple, get_args

import jax
import jax.numpy as jnp

from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# Pgx environments are versioned like OpenAI Gym or Brax.
# OpenAI Gym forces user to specify version (e.g., `MountainCar-v0`); while Brax does not (e.g., `ant`)
# We follow the way of Brax. One can check the environment version by `Env.version`.
# We do not explicitly include version in EnvId for three reasons:
# (1) In game domain, performance measure is not the score in environment but
#     the comparison to other agents (i.e., environment version is less important),
# (2) we do not provide older versions (as with OpenAI Gym), and
# (3) it is tedious to remember and write version numbers.
#
# Naming convention:
# Hyphen - is used to represent that there is a different original game source, and
# Underscore - is used for the other cases.
EnvId = Literal[
    "2048",
    "animal_shogi",
    "backgammon",
    "bridge_bidding",
    "chess",
    "connect_four",
    "gardner_chess",
    "go_9x9",
    "go_19x19",
    "hex",
    "kuhn_poker",
    "leduc_holdem",
    # "mahjong",
    "minatar-asterix",
    "minatar-breakout",
    "minatar-freeway",
    "minatar-seaquest",
    "minatar-space_invaders",
    "othello",
    "shogi",
    "sparrow_mahjong",
    "tic_tac_toe",
]


@dataclass
class State(abc.ABC):
    """Base state class of all Pgx game environments. Basically an immutable (frozen) dataclass.
    A basic usage is generating via `Env.init`:

        state = env.init(jax.random.PRNGKey(0))

    and `Env.step` receives and returns this state class:

        state = env.step(state, action)

    Serialization via `flax.struct.serialization` is supported.
    There are 6 common attributes over all games:

    Attributes:
        current_player (Array): id of agent to play.
            Note that this does NOT represent the turn (e.g., black/white in Go).
            This ID is consistent over the parallel vmapped states.
        observation (Array): observation for the current state.
            `Env.observe` is called to compute.
        rewards (Array): the `i`-th element indicates the intermediate reward for
            the agent with player-id `i`. If `Env.step` is called for a terminal state,
            the following `state.rewards` is zero for all players.
        terminated (Array): denotes that the state is terminal state. Note that
            some environments (e.g., Go) have an `max_termination_steps` parameter inside
            and will terminate within a limited number of states (following AlphaGo).
        truncated (Array): indicates that the episode ends with the reason other than termination.
            Note that current Pgx environments do not invoke truncation but users can use `TimeLimit` wrapper
            to truncate the environment. In Pgx environments, some MinAtar games may not terminate within a finite timestep.
            However, the other environments are supposed to terminate within a finite timestep with probability one.
        legal_action_mask (Array): Boolean array of legal actions. If illegal action is taken,
            the game will terminate immediately with the penalty to the palyer.
    """

    current_player: Array
    observation: Array
    rewards: Array
    terminated: Array
    truncated: Array
    legal_action_mask: Array
    _step_count: Array


    def get_chance_logits(self) -> Array:
        """
        Return all -INF values if the following action is a normal player action.
        If the following action is based on chance this will be overridden by the
        subclass to return an array with values other than -INF.

        TODO it would require larger changes but it probably would be better to
        add chance_logits as a member variable instead of using this method.
        """
        return jnp.full(self.current_player.shape, -jnp.inf, dtype=jnp.float32)

    def has_chance_logits(self, chance_logits) -> Array:
        return (~jnp.isneginf(chance_logits)).any(axis=-1)

    def has_chance_logits_recalc(self) -> Array:
        return self.has_chance_logits(self.get_chance_logits())

    def get_normal_or_chance_logits(self, logits:Array, chance_logits:Array) -> Array:
        """
        Assuming that every node is either a chance node or an action node this
        returns chance logits if any are available, otherwise non-chance logits.
        """
        return jnp.where(self.has_chance_logits(chance_logits), chance_logits, logits)

    def get_normal_or_chance_logits_recalc(self, logits:Array) -> Array:
        return self.get_normal_or_chance_logits(logits, self.get_chance_logits())

    def observation_chance_elements(self) -> int:
        return 0

    @property
    @abc.abstractmethod
    def env_id(self) -> EnvId:
        """Environment id (e.g. "go_19x19")"""
        ...

    def _repr_html_(self) -> str:
        return self.to_svg()

    def to_svg(
        self,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> str:
        """Return SVG string. Useful for visualization in notebook.

        Args:
            color_theme (Optional[Literal["light", "dark"]]): xxx see also global config.
            scale (Optional[float]): change image size. Default(None) is 1.0

        Returns:
            str: SVG string
        """
        from pgx._src.visualizer import Visualizer

        v = Visualizer(color_theme=color_theme, scale=scale)
        return v.get_dwg(states=self).tostring()

    def save_svg(
        self,
        filename,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> None:
        """Save the entire state (not observation) to a file.
        The filename must end with `.svg`

        Args:
            color_theme (Optional[Literal["light", "dark"]]): xxx see also global config.
            scale (Optional[float]): change image size. Default(None) is 1.0

        Returns:
            None
        """
        from pgx._src.visualizer import save_svg

        save_svg(self, filename, color_theme=color_theme, scale=scale)


class Env(abc.ABC):
    """Environment class API.

    !!! example "Example usage"

        ```py
        env: Env = pgx.make("tic_tac_toe")
        state = env.init(jax.random.PRNGKey(0))
        action = jax.random.int32(4)
        state = env.step(state, action)
        ```

    """

    def __init__(self): ...

    def init(self, key: PRNGKey) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.

        Args:
            key: pseudo-random generator key in JAX. Consumed in this function.

        Returns:
            State: initial state of environment

        """
        state = self._init(key)
        observation = self.observe(state)
        return state.replace(observation=observation)  # type: ignore

    def step(
        self,
        state: State,
        action: Array,
        key: Optional[Array] = None,
    ) -> State:
        """Step function."""
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),  # type: ignore
            lambda: self._step(state.replace(_step_count=state._step_count + 1), action, key),  # type: ignore
        )

        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        # All legal_action_mask elements are **TRUE** at terminal state
        # This is to avoid zero-division error when normalizing action probability
        # Taking any action at terminal state does not give any effect to the state
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(legal_action_mask=jnp.ones_like(state.legal_action_mask)),  # type: ignore
            lambda: state,
        )

        observation = self.observe(state)
        state = state.replace(observation=observation)  # type: ignore

        return state

    def observe(self, state: State, player_id: Optional[Array] = None) -> Array:
        """Observation function."""
        if player_id is None:
            player_id = state.current_player
        else:
            warnings.warn("[Pgx] `player_id` in `observe` is deprecated. This argument will be removed in the future.", DeprecationWarning)
        obs = self._observe(state, player_id)
        return jax.lax.stop_gradient(obs)

    @abc.abstractmethod
    def _init(self, key: PRNGKey) -> State:
        """Implement game-specific init function here."""
        ...

    @abc.abstractmethod
    def _step(self, state, action, key) -> State:
        """Implement game-specific step function here."""
        ...

    @abc.abstractmethod
    def _observe(self, state: State, player_id: Array) -> Array:
        """Implement game-specific observe function here."""
        ...

    @property
    @abc.abstractmethod
    def id(self) -> EnvId:
        """Environment id."""
        ...

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Environment version. Updated when behavior, parameter, or API is changed.
        Refactoring or speeding up without any expected behavior changes will NOT update the version number.
        """
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        """Number of players (e.g., 2 in Tic-tac-toe)"""
        ...

    @property
    def num_actions(self) -> int:
        """Return the size of action space (e.g., 9 in Tic-tac-toe)"""
        state = self.init(jax.random.PRNGKey(0))
        return int(state.legal_action_mask.shape[0])

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        state = self.init(jax.random.PRNGKey(0))
        obs = self._observe(state, state.current_player)
        return obs.shape

    @property
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        return -1.0

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        penalty = self._illegal_action_penalty
        reward = jnp.ones_like(state.rewards) * (-1 * penalty) * (self.num_players - 1)
        reward = reward.at[loser].set(penalty)
        return state.replace(rewards=reward, terminated=TRUE)  # type: ignore


def available_envs() -> Tuple[EnvId, ...]:
    """List up all environment id available in `pgx.make` function.

    !!! example "Example usage"

        ```py
        pgx.available_envs()
        ('2048', 'animal_shogi', 'backgammon', 'chess', 'connect_four', 'go_9x9', 'go_19x19', 'hex', 'kuhn_poker', 'leduc_holdem', 'minatar-asterix', 'minatar-breakout', 'minatar-freeway', 'minatar-seaquest', 'minatar-space_invaders', 'othello', 'shogi', 'sparrow_mahjong', 'tic_tac_toe')
        ```


    !!! note "`BridgeBidding` environment"

        `BridgeBidding` environment requires the domain knowledge of bridge game.
        So we forbid users to load the bridge environment by `make("bridge_bidding")`.
        Use `BridgeBidding` class directly by `from pgx.bridge_bidding import BridgeBidding`.

    """
    games = get_args(EnvId)
    games = tuple(filter(lambda x: x != "bridge_bidding", games))
    return games


def make(env_id: EnvId):  # noqa: C901
    """Load the specified environment.

    !!! example "Example usage"

        ```py
        env = pgx.make("tic_tac_toe")
        ```

    !!! note "`BridgeBidding` environment"

        `BridgeBidding` environment requires the domain knowledge of bridge game.
        So we forbid users to load the bridge environment by `make("bridge_bidding")`.
        Use `BridgeBidding` class directly by `from pgx.bridge_bidding import BridgeBidding`.

    """
    # NOTE: BridgeBidding environment requires the domain knowledge of bridge
    # So we forbid users to load the bridge environment by `make("bridge_bidding")`.
    if env_id == "2048":
        from pgx.play2048 import Play2048

        return Play2048()
    elif env_id == "animal_shogi":
        from pgx.animal_shogi import AnimalShogi

        return AnimalShogi()
    elif env_id == "backgammon":
        from pgx.backgammon import Backgammon

        return Backgammon()
    elif env_id == "chess":
        from pgx.chess import Chess

        return Chess()
    elif env_id == "connect_four":
        from pgx.connect_four import ConnectFour

        return ConnectFour()
    elif env_id == "gardner_chess":
        from pgx.gardner_chess import GardnerChess

        return GardnerChess()
    elif env_id == "go_9x9":
        from pgx.go import Go

        return Go(size=9, komi=7.5)
    elif env_id == "go_19x19":
        from pgx.go import Go

        return Go(size=19, komi=7.5)
    elif env_id == "hex":
        from pgx.hex import Hex

        return Hex()
    elif env_id == "kuhn_poker":
        from pgx.kuhn_poker import KuhnPoker

        return KuhnPoker()
    elif env_id == "leduc_holdem":
        from pgx.leduc_holdem import LeducHoldem

        return LeducHoldem()
    # elif env_id == "mahjong":
    #     from pgx.mahjong import Mahjong

    #     return Mahjong()
    elif env_id == "minatar-asterix":
        from pgx.minatar.asterix import MinAtarAsterix  # type: ignore

        return MinAtarAsterix()
    elif env_id == "minatar-breakout":
        from pgx.minatar.breakout import MinAtarBreakout  # type: ignore

        return MinAtarBreakout()
    elif env_id == "minatar-freeway":
        from pgx.minatar.freeway import MinAtarFreeway  # type: ignore

        return MinAtarFreeway()
    elif env_id == "minatar-seaquest":
        from pgx.minatar.seaquest import MinAtarSeaquest  # type: ignore

        return MinAtarSeaquest()
    elif env_id == "minatar-space_invaders":
        from pgx.minatar.space_invaders import MinAtarSpaceInvaders  # type: ignore

        return MinAtarSpaceInvaders()
    elif env_id == "othello":
        from pgx.othello import Othello

        return Othello()
    elif env_id == "shogi":
        from pgx.shogi import Shogi

        return Shogi()
    elif env_id == "sparrow_mahjong":
        from pgx.sparrow_mahjong import SparrowMahjong

        return SparrowMahjong()
    elif env_id == "tic_tac_toe":
        from pgx.tic_tac_toe import TicTacToe

        return TicTacToe()
    else:
        envs = "\n".join(available_envs())
        raise ValueError(f"Wrong env_id '{env_id}' is passed. Available ids are: \n{envs}")


class Strategy(abc.ABC):
    def get_next_action_batch(self, state: State, rng_key: Array, config, model_cls) -> Array:
        best_action, _equities, _action_indices = self.get_next_action_and_equities_batch(state, rng_key, config, model_cls)
        return best_action 

    def get_next_action_and_equities_batch(self, state: State, rng_key: Array, config, model_cls) -> tuple:
        raise NotImplementedError("This strategy does not support get_next_action_and_equities_batch.")


@jax.jit(static_argnames=('repeat_factor',))
def broadcast_config(config, repeat_factor: int):
    if not getattr(config, "should_broadcast", True):
        return config

    def repeat_leaf(x):
        val = jnp.atleast_1d(x)
        repeated = jnp.repeat(val[:, jnp.newaxis, ...], repeat_factor, axis=1)
        return repeated.reshape((-1,) + val.shape[1:])

    new_fields = {}
    for field in dataclasses.fields(config):
        val = getattr(config, field.name)
        new_fields[field.name] = jax.tree_util.tree_map(repeat_leaf, val)
    return type(config)(**new_fields)


class OnePlyStrategy(Strategy):
    def __init__(self, env):
        self.env = env

    def get_next_action_batch(self, state: State, rng_key: Array, config, model_cls) -> Array:
        num_chance = state.observation_chance_elements()
        num_move_actions = self.env.num_actions - num_chance
        broad_config = broadcast_config(config, num_move_actions)
        model = model_cls(broad_config)

        actions_all = jnp.arange(num_move_actions)
        batch_size = state.current_player.shape[0]
        lookahead_rng, _ = jax.random.split(rng_key, 2)
        lookahead_keys = jax.random.split(lookahead_rng, batch_size)

        # 1-ply lookahead using env.step
        next_states = jax.vmap(jax.vmap(self.env.step, in_axes=(None, 0, None)), in_axes=(0, None, 0))(state, actions_all, lookahead_keys)
        parent_player = state.current_player
        obs = jax.vmap(jax.vmap(self.env.observe, in_axes=(0, None)), in_axes=(0, 0))(next_states, parent_player)

        obs_size = self.env.observation_shape[-1]
        equities_flat = model.eval(obs.reshape((-1, obs_size)))
        equities = equities_flat.reshape((batch_size, num_move_actions))

        masked_equities = jnp.where(state.legal_action_mask[:, :num_move_actions], equities, jnp.finfo(equities.dtype).min)
        return jnp.argmax(masked_equities, axis=-1)


class Evaluator(abc.ABC):
    def __init__(self, config=None):
        self.config = config

    def get_config(self):
        return self.config

    @abc.abstractmethod
    def eval(self, state: State, idx: Optional[Array] = None) -> Array:
        pass
