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

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from pgx._src.utils import chunked_map


TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


NUM_PLAYERS        = 2
PLAYER_CHECKERS    = 15
DICE_SIDES         = 6
BOARD_LENGTH       = 24
BAR_POSITIONS      = NUM_PLAYERS
OFF_POSITIONS      = NUM_PLAYERS
HOME_BOARD_LENGTH  = 6
NUM_DICE           = 2
NON_DOUBLE_MOVES   = 2
DOUBLE_MOVES       = 4
MAX_MOVES          = DOUBLE_MOVES

NO_MOVE            = -1
NO_MOVE_SUM        = MAX_MOVES * NO_MOVE

IN_GAME_POSITIONS  = BOARD_LENGTH + BAR_POSITIONS
ALL_GAME_POSITIONS = IN_GAME_POSITIONS + OFF_POSITIONS

# Hashing weight vectors for collision-free board deduplication (pseudorandom 32-bit odd integers)
HASH_WEIGHTS_1 = jnp.array([
    1099087573, 2147483647, 391583921, 1438902821, 827391823,
    918273911, 283719283, 192837123, 723192831, 381928371,
    938102931, 481920391, 102938109, 392810293, 839201923,
    192830192, 482910293, 928301923, 283910293, 102938192,
    382910293, 839201928, 482910291, 928301922, 192830191,
    392810291, 839201921, 283910291
], dtype=jnp.int32)

HASH_WEIGHTS_2 = jnp.array([
    1827391823, 93810293, 839201923, 192830192, 482910293,
    928301923, 283910293, 102938192, 382910293, 839201928,
    482910291, 928301922, 192830191, 392810291, 839201921,
    283910291, 1099087573, 2147483647, 391583921, 1438902821,
    827391823, 918273911, 283719283, 192837123, 723192831,
    381928371, 938102931, 481920391
], dtype=jnp.int32)

BOARD_ENCODE_ELEM  = 8                  # number of observation elements per board point
DICE_ENCODE_ELEM   = 2                  # number of observation elements per die side
OBSERVATION_SIZE   = BOARD_LENGTH * BOARD_ENCODE_ELEM + BAR_POSITIONS + OFF_POSITIONS

START_BOARD        = (2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2)
START_POSITIONS    = START_BOARD + (0, 0) + (0, 0)

BAR_IDX            = BOARD_LENGTH       # black is BAR_IDX, white is BAR_IDX + 1, code assumes current player is playing with black checkers
OFF_IDX            = IN_GAME_POSITIONS  # black is OFF_IDX, white is OFF_IDX + 1, code assumes current player is playing with black checkers

BOARD_RANGE        = (0, BOARD_LENGTH)
BAR_RANGE          = (BAR_IDX, BAR_IDX + BAR_POSITIONS)
OFF_RANGE          = (OFF_IDX, OFF_IDX + OFF_POSITIONS)
IDX_RANGES         = (BOARD_RANGE, BAR_RANGE, OFF_RANGE)

SRC_NO_MOVE        = 0
SRC_BAR            = 1
SRC_BOARD_OFFSET   = 2
SRC_LENGTH         = BOARD_LENGTH + SRC_BOARD_OFFSET

ALL_DICE_PAIRS     = jnp.array([(d1, d2) for d1 in range(DICE_SIDES) for d2 in range(DICE_SIDES) if d1 <= d2], dtype=jnp.int32)  # only store pairs where the first number is less than or equal to the second
NUM_DICE_PAIRS     = len(ALL_DICE_PAIRS)

ACTION_MOVE_LENGTH    = SRC_LENGTH * DICE_SIDES
ACTION_CHANCE_LENGTH  = NUM_DICE_PAIRS
ACTION_TOTAL_LENGTH   = ACTION_MOVE_LENGTH + ACTION_CHANCE_LENGTH

INIT_DICE_PATTERN     = jnp.array([(d1, d2) for d1 in range(DICE_SIDES) for d2 in range(DICE_SIDES) if d1 != d2], dtype=jnp.int32)  # cannot have doubles on first roll

DICE_ROLL_LOGITS      = jnp.array([jnp.log(2.0 / (DICE_SIDES * DICE_SIDES)) if (d1 != d2) else jnp.log(1.0 / (DICE_SIDES * DICE_SIDES)) for d1, d2 in ALL_DICE_PAIRS], dtype=jnp.float32)
CHANCE_ACTION_LOGITS  = jnp.concatenate([jnp.full(ACTION_MOVE_LENGTH, -jnp.inf, dtype=jnp.float32), DICE_ROLL_LOGITS])
EMPTY_ACTION_LOGITS   = jnp.full(ACTION_TOTAL_LENGTH, -jnp.inf, dtype=jnp.float32)

BOARD_DTYPE           = jnp.int8  # can store [-128, 127]
MAT_MULT_DTYPE        = jnp.float16  # any fast and small dtype that is supported by hardware matrix mult units

NOOP_ACTION_IDX       = SRC_NO_MOVE * DICE_SIDES
NOOP_ACTION_MASK      = jnp.zeros(ACTION_TOTAL_LENGTH, dtype=jnp.bool_).at[NOOP_ACTION_IDX].set(TRUE)
BOARD_BAR_MASK        = jnp.zeros(ALL_GAME_POSITIONS, dtype=jnp.bool_).at[BAR_IDX].set(TRUE)

MADE_POINT_HEURISTIC  = jnp.array([0.2, 0.2, 0.2, 0.4, 0.5, 0.5] + [0.4, 0.3, 0.2, 0.2, 0.2, 0.2]
                                + [0.2, 0.2, 0.2, 0.3, 0.4, 0.6] + [1.0, 1.0, 0.9, 0.8, 0.7, 0.7], dtype=jnp.float32)

ONE_OF_ONE_DICE_HITS  = 1.0 - ((DICE_SIDES - 1) / DICE_SIDES) ** 1   # chance that a single dice is a specific value
ONE_OF_TWO_DICE_HITS  = 1.0 - ((DICE_SIDES - 1) / DICE_SIDES) ** 2   # chance that either of two dice is a specific value


def _action_is_noop(action):
    return (action // DICE_SIDES) == SRC_NO_MOVE


def _action_to_src(action: Array) -> Array:
    """
    Translate src to board index.  For this function we assume that src is not set to SRC_NO_MOVE.
        no move,         input: action // 6 == 0, output: -2 (-SOURCE_BOARD_OFFSET)
        move from bar,   input: action // 6 == 1, output: BAR_IDX
        move_from board, input: action // 6 >= 2: output: board index (action//6 - 2)
    """
    src_part = action // DICE_SIDES
    return jnp.where(src_part == SRC_BAR, BOARD_DTYPE(BAR_IDX), BOARD_DTYPE(src_part - SRC_BOARD_OFFSET))  # type: ignore


def _action_to_die(action: Array):
    return jnp.where(_action_is_noop(action), BOARD_DTYPE(NO_MOVE), (action % 6 + 1).astype(BOARD_DTYPE))  # transform dice values from [0..5] to [1..6]


def _tgt_from_board(src: Array, die: Array) -> Array:
    """ If the action is a noop (where src equals -SRC_BOARD_OFFSET) we can return anything, we return OFF_IDX. """
    _is_to_board = (src >= 0) & (src + die < BOARD_LENGTH)
    return jnp.where(_is_to_board, BOARD_DTYPE(src + die), BOARD_DTYPE(OFF_IDX))  # type: ignore


def _calc_tgt(src: Array, die: Array) -> Array:
    """
    Translate tgt to board index.  We are either coming in from the
    bar (src >= BOARD_LENGTH) or from the board (src < BOARD_LENGTH).
    When we come in from the bar a roll of 1 translates to the first
    index on the board which is index 0, so we land on index = die - 1.
    """
    is_from_bar = (src == BAR_IDX)
    return jnp.where(is_from_bar, BOARD_DTYPE(die) - 1, _tgt_from_board(src, die))  # type: ignore


def _decompose_action(action: Array):
    """
    Decompose action to src, die, tgt.
    action = src*6 + die
    """
    src = _action_to_src(action)  # -SRC_BOARD_OFFSET means no-op, 0..BOARD_LENGTH move from board, BAR_IDX move from bar
    die = _action_to_die(action)  # 1~6
    tgt = _calc_tgt(src, die)
    return src, die, tgt


def _board_mask_before_src(src: Array):
    """
    Set True values for all board values that are before the src index.
    Src can be BAR_IDX or an index on the board [0 ... BOARD_LENGTH-1].
    """
    indices        = jnp.arange(ALL_GAME_POSITIONS, dtype=jnp.int32)
    src_from_bar   = (src == jnp.int32(BAR_IDX))
    board_at_bar   = (indices == jnp.int32(BAR_IDX))
    board_lt_src   = (indices < src[..., jnp.newaxis]) & (indices < jnp.int32(BOARD_LENGTH))

    return (~src_from_bar[..., jnp.newaxis]) & (board_at_bar | board_lt_src)


def _initialize_arrays():
    ONE_MOVE_BOARD_DIFFS = jnp.zeros((ACTION_MOVE_LENGTH, ALL_GAME_POSITIONS), dtype=BOARD_DTYPE)

    action_indices = jnp.arange(ACTION_MOVE_LENGTH, dtype=jnp.int32)
    ONE_MOVE_SRC, ONE_MOVE_DIE, ONE_MOVE_TGT = jax.vmap(_decompose_action)(action_indices)

    # be careful, src can be negative when action is noop
    def filter_board_indices(indices):
        return jnp.where(indices < 0, ALL_GAME_POSITIONS, indices)

    ONE_MOVE_BOARD_DIFFS  = ONE_MOVE_BOARD_DIFFS.at[action_indices, filter_board_indices(ONE_MOVE_SRC)].set(-1)
    ONE_MOVE_BOARD_DIFFS  = ONE_MOVE_BOARD_DIFFS.at[action_indices, filter_board_indices(ONE_MOVE_TGT)].set(+1)

    ACTION_SRC_BOARD_MASK = jax.vmap(_board_mask_before_src)(ONE_MOVE_SRC).astype(MAT_MULT_DTYPE)

    i_indices = jnp.arange(BOARD_LENGTH)[:, jnp.newaxis]
    j_indices = jnp.arange(BOARD_LENGTH)[jnp.newaxis, :]
    dists = i_indices - j_indices

    one_roll_hit_prob  = jnp.zeros(DICE_SIDES + 1, dtype=jnp.float32).at[1:].set(ONE_OF_TWO_DICE_HITS)
    single_die_prob    = jnp.zeros(DICE_SIDES + 1, dtype=jnp.float32).at[1:].set(ONE_OF_ONE_DICE_HITS)
    two_roll_hit_prob  = jnp.convolve(single_die_prob, single_die_prob)

    def get_board_prob_map(hit_prob):
        safe_dists = jnp.clip(dists, 0, len(hit_prob) - 1)
        return jnp.where((dists > 0) & (dists < len(hit_prob)), hit_prob[safe_dists], 0.0)
    ONE_HIT_DIST_HEURISTIC   = get_board_prob_map(one_roll_hit_prob).astype(MAT_MULT_DTYPE)
    TWO_HIT_DIST_HEURISTIC   = get_board_prob_map(two_roll_hit_prob).astype(MAT_MULT_DTYPE)

    return ONE_MOVE_BOARD_DIFFS, ONE_MOVE_SRC, ONE_MOVE_DIE, ONE_MOVE_TGT, ACTION_SRC_BOARD_MASK, ONE_HIT_DIST_HEURISTIC, TWO_HIT_DIST_HEURISTIC


# ONE_MOVE_BOARD_DIFFS  : ACTION_MOVE_LENGTH x ALL_GAME_POSITIONS
#                         For each action this stores the board state diff that has a single +1 and a single -1
# ONE_MOVE_SRC          : ACTION_MOVE_LENGTH
#                         For each action this stores the src board position for the action
# ONE_MOVE_DIE          : ACTION_MOVE_LENGTH
#                         For each action this stores the die roll for the action
# ONE_MOVE_TGT          : ACTION_MOVE_LENGTH
#                         For each action this stores the target board position for the action
# ACTION_SRC_BOARD_MASK : ACTION_MOVE_LENGTH x ALL_GAME_POSITIONS
#                         For each action this stores a mask that is True for all board positions
#                         before the action src (used in _arr_is_any_earlier())
# ONE_HIT_DIST_HEURISTIC: BOARD_LENGTH x BOARD_LENGTH
#                         For each pair of (src, tgt) board positions this stores the probability of
#                         hitting the tgt from the src in one roll (assuming both dice are available)
# TWO_HIT_DIST_HEURISTIC: BOARD_LENGTH x BOARD_LENGTH
#                         For each pair of (src, tgt) board positions this stores the probability of
#                         hitting the tgt from the src with the sum of two rolls (this ignores the
#                         fact that some board positions might be blocked)

_jit_initialize_arrays = jax.jit(_initialize_arrays)
ONE_MOVE_BOARD_DIFFS, ONE_MOVE_SRC, ONE_MOVE_DIE, ONE_MOVE_TGT, ACTION_SRC_BOARD_MASK, ONE_HIT_DIST_HEURISTIC, TWO_HIT_DIST_HEURISTIC = (
    _jit_initialize_arrays()
)

BOARD_DIFFS_PER_DIE = jnp.transpose(ONE_MOVE_BOARD_DIFFS.reshape(SRC_LENGTH, DICE_SIDES, ALL_GAME_POSITIONS), (1, 0, 2))
SRC_PER_DIE = jnp.transpose(ONE_MOVE_SRC.reshape(SRC_LENGTH, DICE_SIDES), (1, 0))
TGT_PER_DIE = jnp.transpose(ONE_MOVE_TGT.reshape(SRC_LENGTH, DICE_SIDES), (1, 0))
SRC_BOARD_MASK_PER_DIE = jnp.transpose(ACTION_SRC_BOARD_MASK.reshape(SRC_LENGTH, DICE_SIDES, ALL_GAME_POSITIONS), (1, 0, 2))

# The way the actions are arranged SRC_PER_DIE is the same for every die value
SRC_ANY_DIE = SRC_PER_DIE[0]
SRC_BOARD_MASK_ANY_DIE = SRC_BOARD_MASK_PER_DIE[0]




OUTSIDE_HOME_BOARD_MASK = _board_mask_before_src(jnp.int32(BOARD_LENGTH - HOME_BOARD_LENGTH))


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(OBSERVATION_SIZE, dtype=jnp.int32)
    rewards: Array = jnp.float32([0.0] * NUM_PLAYERS)
    terminated: Array = FALSE
    truncated: Array = FALSE
    # micro action = 6 * src + die
    legal_action_mask: Array = jnp.zeros(ACTION_TOTAL_LENGTH, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- Backgammon specific ---
    # _board stores an integer for each board points(24), bar(2) and off(2),
    # positive values are the count of black pieces, negative for white
    _board: Array = jnp.zeros(ALL_GAME_POSITIONS, dtype=BOARD_DTYPE)
    _dice: Array = jnp.zeros(NUM_DICE, dtype=jnp.int32)  # indices 0 to 5 map to dice rolls 1 through 6
    _playable_dice: Array = jnp.full(MAX_MOVES, NO_MOVE, dtype=jnp.int32)  # playable dice, NO_MOVE used for unusable moves
    _played_dice_num: Array = jnp.int32(0)  # the number of dice played
    _turn: Array = jnp.int32(1)  # black: 0 white:1

    @property
    def env_id(self) -> core.EnvId:
        return "backgammon"

    def get_chance_logits(self) -> Array:
        """
        This returns the chance logits for all possible rolls when have flipped
        the board to a new player's turn but have not yet rolled their dice.
        """
        is_start_of_turn = (jnp.sum(self._playable_dice) == NO_MOVE_SUM) & (self._played_dice_num == 0)
        return jnp.where(is_start_of_turn, CHANCE_ACTION_LOGITS, EMPTY_ACTION_LOGITS)

    def observation_chance_elements(self) -> int:
        return ACTION_CHANCE_LENGTH


class Backgammon(core.Env):
    def __init__(self):
        super().__init__()

    def step(self, state: core.State, action: Array, key: Optional[Array] = None) -> core.State:
        assert key is not None, (
            "v2.0.0 changes the signature of step. Please specify PRNGKey at the third argument:\n\n"
            "  * <  v2.0.0: step(state, action)\n"
            "  * >= v2.0.0: step(state, action, key)\n\n"
            "See v2.0.0 release note for more details:\n\n"
            "  https://github.com/sotetsuk/pgx/releases/tag/v2.0.0"
        )
        return super().step(state, action, key)

    def _init(self, key: PRNGKey) -> State:
        return _init(key)

    def _step(self, state: core.State, action: Array, key) -> State:
        assert isinstance(state, State)
        return _step(state, action, key)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "backgammon"

    @property
    def version(self) -> str:
        return "v2.1"

    @property
    def num_players(self) -> int:
        return NUM_PLAYERS

    @property
    def _illegal_action_penalty(self) -> float:
        return -3.0


def _init(rng: PRNGKey) -> State:
    rng1, rng2 = jax.random.split(rng, num=2)
    current_player: Array = jax.random.bernoulli(rng1).astype(jnp.int32)
    board: Array = _make_init_board()
    terminated: Array = FALSE
    dice: Array = _roll_init_dice(rng2)
    playable_dice: Array = _set_playable_dice(dice)
    played_dice_num: Array = jnp.int32(0)
    turn: Array = _init_turn(dice)
    legal_action_mask: Array = _arr_legal_action_mask(board, playable_dice)
    state = State(  # type: ignore
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        _turn=turn,
        legal_action_mask=legal_action_mask,
    )
    return state


def _step(state: State, action: Array, key) -> State:
    """
    Step when not terminated.
    """
    state = _update_by_action(state, action)
    return jax.lax.cond(
        _is_all_off(state._board),
        lambda: _winning_step(state),
        lambda: _no_winning_step(state, action, key),
    )


def _make_observation(board: Array) -> Array:
    """
    Use the gnu backgammon encoding (in eval.c) for the board state which
    is similar to the encoding used by TD Gammon.
            afInput[0] = (nc == 1) ? 1.0f : 0.0f;
            afInput[1] = (nc == 2) ? 1.0f : 0.0f;
            afInput[2] = (nc >= 3) ? 1.0f : 0.0f;
            afInput[3] = nc > 3 ? (float) (nc - 3) / 2.0f : 0.0f;

    Currently we do not include the dice because we assume that any strategy
    will be evaluated on every legal board state.
    """
    board_pts = board[BOARD_RANGE[0]:BOARD_RANGE[1]]
    observations = [
        1.0 * (board_pts == 1),
        1.0 * (board_pts == 2),
        1.0 * (board_pts >= 3),
        1.0 * jnp.maximum(0.0, ((board_pts - 3.0) / 2.0)),
        1.0 * (board_pts == -1),
        1.0 * (board_pts == -2),
        1.0 * (board_pts <= -3),
        1.0 * jnp.maximum(0.0, ((-board_pts - 3.0) / 2.0)),
        jnp.abs(board[BAR_RANGE[0]:BAR_RANGE[1]]) / 2.0,
        jnp.abs(board[OFF_RANGE[0]:OFF_RANGE[1]]) / PLAYER_CHECKERS,
    ]
    ret = jnp.concatenate(observations, axis=None)
    return ret


def _to_playable_dice_count(playable_dice: Array) -> Array:
    """
    Return 6 dim vec which represents the number of playable die
    Examples
    Playable dice: 2, 3, -1, -1   # die values 3 and 4
    Return: [0, 0, 1, 1, 0, 0]

    Playable dice: 4, 4, 4, 4     # double 5's
    Return: [0, 0, 0, 0, 4, 0]
    """
    # -1 values (NO_MOVE) inside of playable_dice will wrap around to index 6 which is filled with zeros
    one_hot = jnp.eye(DICE_SIDES + 1, DICE_SIDES, dtype=jnp.int32)
    return one_hot[playable_dice].sum(axis=0)


def _observe(state: State, _player_id: Array) -> Array:
    return _make_observation(state._board)


def _observation_to_board(observation: Array) -> Array:
    """ this returns the board and an array of size 6 which has the playable dice for each die value """
    BLTH = BOARD_LENGTH
    BRNG = BOARD_RANGE
    obs  = observation

    board_pts = (
         1 * obs[BLTH * 0 + BRNG[0]:BLTH * 0 + BRNG[1]] +
         2 * obs[BLTH * 1 + BRNG[0]:BLTH * 1 + BRNG[1]] +
        jnp.where(obs[BLTH * 2 + BRNG[0]:BLTH * 2 + BRNG[1]] > 0.0,
                 3 + jax.lax.round(2 * obs[BLTH * 3 + BRNG[0]:BLTH * 3 + BRNG[1]]), 0.0) +
        -1 * obs[BLTH * 4 + BRNG[0]:BLTH * 4 + BRNG[1]] +
        -2 * obs[BLTH * 5 + BRNG[0]:BLTH * 5 + BRNG[1]] +
        jnp.where(obs[BLTH * 6 + BRNG[0]:BLTH * 6 + BRNG[1]] > 0.0,
                -3 - jax.lax.round(2 * obs[BLTH * 7 + BRNG[0]:BLTH * 7 + BRNG[1]]), 0.0)
    )
    board = jnp.concatenate([board_pts,
         jax.lax.round(obs[BLTH * 8 + 0] * 2),
        -jax.lax.round(obs[BLTH * 8 + 1] * 2),
         jax.lax.round(obs[BLTH * 8 + 2] * PLAYER_CHECKERS),
        -jax.lax.round(obs[BLTH * 8 + 3] * PLAYER_CHECKERS)
    ], axis=None)
    return board


def _action_is_chance(action):
    return action >= ACTION_MOVE_LENGTH


def _winning_step(
    state: State,
) -> State:
    """
    Step with winner
    """
    win_score = _calc_win_score(state._board)
    winner = state.current_player
    loser = 1 - winner
    reward = jnp.ones_like(state.rewards)
    reward = reward.at[winner].set(win_score)
    reward = reward.at[loser].set(-win_score)
    state = state.replace(terminated=TRUE)  # type: ignore
    return state.replace(rewards=reward)  # type: ignore


def _no_winning_step(state: State, action: Array, key) -> State:
    """
    Step with no winner. Change turn if turn end condition is satisfied.
    If the action was a roll (chance action) then _is_turn_end() should be false.
    """
    return jax.lax.cond(
        (_is_turn_end(state) | _action_is_noop(action)),
        lambda: _change_turn(state, key),
        lambda: state,
    )


def _update_by_action(state: State, action: Array) -> State:
    """
    Update state by action
    """
    is_no_op = _action_is_noop(action)

    def get_state_after_move(state):
        board = _move(state._board, action)
        played_dice_num = jnp.int32(state._played_dice_num + 1)
        playable_dice = _update_playable_dice(state._playable_dice, state._played_dice_num, state._dice, action)
        legal_action_mask = _arr_legal_action_mask(board, playable_dice)
        return state.replace(
            _board=board,
            _played_dice_num=played_dice_num,
            _playable_dice=playable_dice,
            legal_action_mask=legal_action_mask)

    def get_state_after_chance(state):
        roll_idx = action - ACTION_MOVE_LENGTH
        dice = ALL_DICE_PAIRS[roll_idx]
        playable_dice = _set_playable_dice(dice)
        legal_action_mask = _arr_legal_action_mask(state._board, playable_dice)
        return state.replace(
            _dice=dice,
            _playable_dice=playable_dice,
            legal_action_mask=legal_action_mask)

    return jax.lax.cond(is_no_op,
                        lambda: state,
                        lambda: jax.lax.cond(_action_is_chance(action),
                                             lambda: get_state_after_chance(state),
                                             lambda: get_state_after_move(state)))


def _flip_board(board):
    """
    Flip the order of each section of the board, then multiply by -1 to change the color
    of the pieces so that we can always consider the board from perspective of the player
    with black checkers.
    """
    _board = board
    for sidx, eidx in IDX_RANGES:
        board = board.at[sidx:eidx].set(jnp.flip(_board[sidx:eidx]))
    return -1 * board


def _make_init_board() -> Array:
    """
    Initialize the board based on black's perspective.
    """
    board: Array = jnp.array(START_POSITIONS, dtype=BOARD_DTYPE)  # type: ignore
    return board


def _is_turn_end(state: State) -> bool:
    """
    Turn will end if there is no playable dice or no legal action.
    """
    return state._playable_dice.sum() == NO_MOVE_SUM  # type: ignore


def _change_turn(state: State, key) -> State:
    """
    Change turn and return new state.
    NOTE: this is called after a player's turn has ended but the next player
    has not yet rolled.  The next action will set the new dice values (chance action).
    """
    return state.replace(
        _board=_flip_board(state._board),
        _turn=(state._turn + 1) % 2,
        current_player=(state.current_player + 1) % 2,
        legal_action_mask=~jnp.isneginf(CHANCE_ACTION_LOGITS),  # only chance actions are allowed
        _playable_dice=jnp.full(MAX_MOVES, NO_MOVE, dtype=jnp.int32),
        _played_dice_num=jnp.int32(0),
    )


def _roll_init_dice(rng: PRNGKey) -> Array:
    """
    Roll till the dice are different.
    """

    init_dice_pattern: Array = jnp.array(INIT_DICE_PATTERN, dtype=jnp.int32)
    return jax.random.choice(rng, init_dice_pattern)


def _roll_dice(rng: PRNGKey) -> Array:
    roll: Array = jax.random.randint(rng, shape=(1, NUM_DICE), minval=0, maxval=DICE_SIDES, dtype=jnp.int32)
    return roll[0]


def _init_turn(dice: Array) -> Array:
    """
    Decide turn at the beginning of the game.
    Begin with those who have bigger dice
    """
    diff = dice[1] - dice[0]
    return jnp.int32(diff > 0)


def _set_playable_dice(dice: Array) -> Array:
    """
    If the dice match that means we have rolled doubles and we get 4 moves,
    otherwise we get 2 moves.  Array elements that do not correspond to a move
    are filled with NO_MOVE.
    """
    return ((dice[0] == dice[1]) * jnp.array([dice[0]] * DOUBLE_MOVES, dtype=jnp.int32)
          + (dice[0] != dice[1]) * jnp.array([dice[0], dice[1]] + (DOUBLE_MOVES - NON_DOUBLE_MOVES) * [NO_MOVE], dtype=jnp.int32))


def _update_playable_dice(
    playable_dice: Array,
    played_dice_num: Array,
    dice: Array,
    action: Array,
) -> Array:
    _n = played_dice_num
    die_idx = _action_to_die(action) - 1
    return ((dice[0] == dice[1]) * playable_dice.at[(MAX_MOVES - 1) - _n].set(NO_MOVE)
          + (dice[0] != dice[1]) * jnp.where(playable_dice == die_idx, NO_MOVE, playable_dice))


def _arr_black_checker_mask(board_arr: Array):
    """ Return a boolean mask to see where there are Black (+1) or White (-1) checkers. """
    return board_arr > 0


def _arr_is_any_earlier(black_checker_mask: Array, src: Array, src_board_mask: Array = None) -> Array:
    """
    This checks if there are any black checkers before the src position of the relevant action
    (important when bearing off).
    """
    if src.ndim == 0:
        return (black_checker_mask & _board_mask_before_src(src)).any(axis=-1)

    if src_board_mask is None:
        src_board_mask = jax.vmap(_board_mask_before_src)(src).astype(MAT_MULT_DTYPE)
    return (black_checker_mask.astype(MAT_MULT_DTYPE) @ src_board_mask.T) > 0


def _arr_is_illegal_on_board(board_arr, diff):
    assert board_arr.shape[-1] == ALL_GAME_POSITIONS, f"Expected last dimension of board_arr to be {ALL_GAME_POSITIONS}, got {board_arr.shape}"

    diff_pos = (diff > 0).astype(MAT_MULT_DTYPE)
    diff_neg = (diff < 0).astype(MAT_MULT_DTYPE)
    is_illegal_hit_tgt = ((board_arr <= -2).astype(MAT_MULT_DTYPE) @ diff_pos.T) > 0
    is_illegal_src     = ((board_arr <= 0).astype(MAT_MULT_DTYPE) @ diff_neg.T) > 0
    return is_illegal_hit_tgt | is_illegal_src


def _arr_is_illegal_off(black_checker_mask, src, tgt, die, src_board_mask: Array = None):
    is_off              = (tgt == OFF_IDX)
    is_any_earlier      = _arr_is_any_earlier(black_checker_mask, src, src_board_mask)
    is_all_home         = (black_checker_mask.astype(MAT_MULT_DTYPE) @ OUTSIDE_HOME_BOARD_MASK) == 0
    is_exact_off        = (BOARD_LENGTH - src) == die
    return is_off & ((~is_all_home)[..., jnp.newaxis] | ((~is_exact_off) & is_any_earlier))


def _arr_is_move_legal(board_arr: Array, diff: Array, src: Array, die: Array, tgt: Array, src_board_mask: Array = None) -> Array:
    """
    Apply board diffs to board_arr. Wherever diff is positive and board_arr has -1,
    the board_arr value is treated as 0 (simulating hit/blot removal).
    """
    is_illegal_on_board = _arr_is_illegal_on_board(board_arr, diff)
    black_checker_mask  = _arr_black_checker_mask(board_arr)
    is_illegal_off      = _arr_is_illegal_off(black_checker_mask, src, tgt, die, src_board_mask)

    is_illegal_bar      = black_checker_mask[..., BAR_IDX][..., jnp.newaxis] & (src != BAR_IDX)
    is_illegal_noop     = (src < 0)

    is_illegal          = is_illegal_on_board | is_illegal_off | is_illegal_bar | is_illegal_noop
    return ~is_illegal


def _deduplicate_boards(boards: Array, legal: Array, first_move: Array, limit: int) -> Tuple[Array, Array, Array]:
    """
    Deduplicates a set of board states of shape (N, 28) with legality mask (N,),
    returning exactly `limit` unique legal boards of shape (limit, 28), unique_legal mask (limit,),
    and unique_first_move indices (limit,).
    """
    # Sort lexicographically using 2 independent random 32-bit projection hashes.
    # We use ~legal as the primary key (last in the tuple) so that legal boards (0) sort before illegal ones (1).
    hash1 = (boards.astype(jnp.int32) * HASH_WEIGHTS_1).sum(axis=-1)
    hash2 = (boards.astype(jnp.int32) * HASH_WEIGHTS_2).sum(axis=-1)
    sort_keys = (hash1, hash2, (~legal).astype(jnp.int32))
    sort_idx = jnp.lexsort(sort_keys)

    sorted_boards = boards[sort_idx]
    sorted_legal = legal[sort_idx]
    sorted_first_move = first_move[sort_idx]

    # Find duplicates (identical at all board positions)
    is_duplicate = (sorted_boards[1:] == sorted_boards[:-1]).all(axis=-1)
    is_duplicate = jnp.pad(is_duplicate, (1, 0), constant_values=False)

    is_unique_legal = sorted_legal & (~is_duplicate)

    gather_idx = jnp.nonzero(is_unique_legal, size=limit, fill_value=-1)[0]
    safe_gather_idx = jnp.where(gather_idx == -1, 0, gather_idx)

    unique_boards = sorted_boards[safe_gather_idx]
    unique_boards = jnp.where(gather_idx[:, jnp.newaxis] == -1, BOARD_DTYPE(0), unique_boards)

    unique_legal = (gather_idx != -1)

    unique_first_move = sorted_first_move[safe_gather_idx]
    unique_first_move = jnp.where(gather_idx == -1, -1, unique_first_move)

    return unique_boards, unique_legal, unique_first_move


def _arr_make_new_boards(board: Array, diffs: Array) -> Array:
    hit_tgt = (diffs > 0) & (board == -1)
    new_boards = jnp.where(hit_tgt, BOARD_DTYPE(1), board + diffs)

    # Vectorized update of the white bar count (index BAR_IDX + 1) for each candidate board
    hits_per_candidate = hit_tgt.sum(axis=-1).astype(BOARD_DTYPE)
    new_boards         = new_boards - hits_per_candidate[..., jnp.newaxis] * (jnp.arange(ALL_GAME_POSITIONS, dtype=BOARD_DTYPE) == BAR_IDX + 1)

    return new_boards


def _arr_step_one_move_dice_actions(board: Array, die_idx: int) -> Tuple[Array, Array]:
    diffs = BOARD_DIFFS_PER_DIE[die_idx]
    tgt = TGT_PER_DIE[die_idx]
    src = SRC_ANY_DIE
    mask = SRC_BOARD_MASK_ANY_DIE
    die = jnp.full(SRC_LENGTH, die_idx + 1, dtype=BOARD_DTYPE)
    new_boards = _arr_make_new_boards(board, diffs)

    legal = _arr_is_move_legal(board, diffs, src, die, tgt, mask)
    return legal, new_boards


def _arr_one_and_two_moves(board, sorted_dice):
    die1_idx = jnp.clip(sorted_dice[-1], 1, 6) - 1
    die2_idx = jnp.clip(sorted_dice[-2], 1, 6) - 1

    # First move legality and boards (reusing _arr_step_one_move)
    legal_die1, boards_die1 = _arr_step_one_move_dice_actions(board, die1_idx)
    legal_die2, boards_die2 = _arr_step_one_move_dice_actions(board, die2_idx)

    one_move_legal  = jnp.concatenate([legal_die1, legal_die2], axis=0)
    one_move_boards = jnp.concatenate([boards_die1, boards_die2], axis=0)

    # Second move legality (pairwise 52 x 52)
    legal_d2_given_d1, _ = jax.vmap(lambda b: _arr_step_one_move_dice_actions(b, die2_idx))(boards_die1)
    legal_d1_given_d2, _ = jax.vmap(lambda b: _arr_step_one_move_dice_actions(b, die1_idx))(boards_die2)

    top_half       = jnp.concatenate([jnp.zeros((SRC_LENGTH, SRC_LENGTH), dtype=jnp.bool_), legal_d2_given_d1], axis=-1)
    bottom_half    = jnp.concatenate([legal_d1_given_d2, jnp.zeros((SRC_LENGTH, SRC_LENGTH), dtype=jnp.bool_)], axis=-1)
    two_move_legal = jnp.concatenate([top_half, bottom_half], axis=0)

    # return handy values for further computation
    candidate_action_indices = jnp.concatenate([jnp.arange(SRC_LENGTH) * 6 + die1_idx, jnp.arange(SRC_LENGTH) * 6 + die2_idx], axis=0)
    candidate_diffs          = jnp.concatenate([BOARD_DIFFS_PER_DIE[die1_idx], BOARD_DIFFS_PER_DIE[die2_idx]], axis=0) # shape: (52, 28)
    candidate_tgt            = jnp.concatenate([TGT_PER_DIE[die1_idx], TGT_PER_DIE[die2_idx]], axis=0)       # shape: (52,)

    return one_move_legal, one_move_boards, two_move_legal, candidate_action_indices, candidate_tgt, candidate_diffs


def _arr_legal_action_mask_details(board: Array, playable_dice: Array):
    """
    I have not proven mathematically but I believe that for double rolls it is
    sufficient to look two moves ahead.  The "must use the larger die" and
    other subtleties do not come up when every move has the same dice value.
    """
    sorted_dice = jnp.sort(jnp.where(playable_dice == NO_MOVE, NO_MOVE - 1, playable_dice + 1))

    one_move_legal_per_die, one_move_boards_per_die, two_move_legal_raw_per_die, candidate_action_indices, candidate_tgt, candidate_diffs = \
        _arr_one_and_two_moves(board, sorted_dice)

    first_dice  = sorted_dice[-1]  # larger dice
    second_dice = sorted_dice[-2]  # can be NO_MOVE - 1

    first_dice_valid  = (first_dice >= 1)
    second_dice_valid = (second_dice >= 1)

    candidate_die_is_first = jnp.concatenate([
        jnp.ones(SRC_LENGTH, dtype=jnp.bool_),
        jnp.zeros(SRC_LENGTH, dtype=jnp.bool_)
    ])
    candidate_die_is_second = ~candidate_die_is_first

    move1_die1_legal   = one_move_legal_per_die & candidate_die_is_first & first_dice_valid
    move1_die2_legal   = one_move_legal_per_die & candidate_die_is_second & second_dice_valid

    move2_die1_legal   = two_move_legal_raw_per_die & candidate_die_is_first[jnp.newaxis, :] & first_dice_valid
    move2_die2_legal   = two_move_legal_raw_per_die & candidate_die_is_second[jnp.newaxis, :] & second_dice_valid

    two_move_legal_per_die = (move1_die1_legal[:, jnp.newaxis] & move2_die2_legal) | (move1_die2_legal[:, jnp.newaxis] & move2_die1_legal)
    one_move_legal_per_die  = jnp.where(move1_die1_legal.any(), move1_die1_legal, move1_die2_legal)

    return one_move_legal_per_die, one_move_boards_per_die, two_move_legal_per_die, candidate_action_indices, sorted_dice, candidate_tgt, candidate_diffs


def _arr_legal_action_mask(board: Array, playable_dice: Array) -> Array:
    one_move_legal_per_die, one_move_boards_per_die, two_move_legal_per_die, candidate_action_indices, sorted_dice, candidate_tgt, candidate_diffs = \
        _arr_legal_action_mask_details(board, playable_dice)
    any_two_moves      = two_move_legal_per_die.any()
    legal_moves_per_die = jnp.where(any_two_moves, two_move_legal_per_die.any(axis=-1), one_move_legal_per_die)

    legal_moves_int = jnp.zeros(ACTION_MOVE_LENGTH, dtype=jnp.int32).at[candidate_action_indices].add(legal_moves_per_die.astype(jnp.int32))
    legal_moves = (legal_moves_int > 0)
    legal_moves_padded = jnp.pad(legal_moves, (0, ACTION_CHANCE_LENGTH), constant_values=False)

    return jnp.where(legal_moves.any(), legal_moves_padded, NOOP_ACTION_MASK)


def _home_board() -> Array:
    """
    black: [18~23], white: [0~5]: Always black's perspective
    """
    return jnp.arange(BOARD_LENGTH - HOME_BOARD_LENGTH, BOARD_LENGTH, dtype=BOARD_DTYPE)  # type: ignore


def _rear_distance(board: Array) -> Array:
    """
    The distance from the farthest checker to the goal: Always black's perspective
    """
    b = board[:BOARD_LENGTH]
    exists = jnp.where((b > 0), size=BOARD_LENGTH, fill_value=jnp.nan)[0]  # type: ignore
    return BOARD_LENGTH - jnp.min(jnp.nan_to_num(exists, nan=jnp.int32(BOARD_LENGTH + 1)))


def _is_all_on_home_board(board: Array):
    """
    One can bear off if all checkers are on home board.
    """
    home_board: Array = _home_board()
    on_home_board = jnp.minimum(jnp.maximum(board[home_board], 0), PLAYER_CHECKERS).sum()
    off = board[OFF_IDX]  # type: ignore
    return (PLAYER_CHECKERS - off) == on_home_board


def _is_open(board: Array, point: int) -> bool:
    """
    Check if the point is open for the current player: Always black's perspective
    One can move to the point if there is no more than one opponent's checker.
    """
    checkers = board[point]
    return checkers >= -1  # type: ignore


def _exists(board: Array, point: int) -> bool:
    """
    Check if the point has the current player's checker: Always black's perspective
    """
    checkers = board[point]
    return checkers >= 1  # type: ignore





def _is_action_legal(board: Array, action: Array) -> bool:
    """
    Check if the action is legal.  Noop (negative src) is not considered legal in this function.
    action = src * 6 + die
    src = [no op., from bar, 0, .., 23]
    """
    src, die, tgt   = _decompose_action(action)
    is_regular_move = (src >= 0) & (src <= BAR_IDX)   # not noop or chance action
    _is_to_point    = (tgt < BOARD_LENGTH)
    return is_regular_move & jnp.where(_is_to_point, _is_to_point_legal(board, src, tgt),
                                                     _is_to_off_legal(board, src, tgt, die))  # type: ignore


def _distance_to_goal(src: int) -> int:
    """
    The distance from the src to the goal: Always black's perspective
    """
    return BOARD_LENGTH - src  # type: ignore


def _is_to_off_legal(board: Array, src: int, tgt: int, die: int):
    """
    Check if the action is legal when the target is off.
    The conditions are:
    1. src has checkers.
    2. All checkers are on home board.
    3. The distance from the src to the goal is the same as the die or the src is the farthest checker and the die is bigger than the distance.
    """
    r = _rear_distance(board)
    d = _distance_to_goal(src)
    is_regular_move = (src >= 0) & (src <= BAR_IDX)   # not noop or chance action
    return (
        is_regular_move & _exists(board, src) & _is_all_on_home_board(board) & ((d == die) | ((r <= die) & (r == d)))
    )  # type: ignore


def _is_to_point_legal(board: Array, src: int, tgt: int) -> bool:
    """
    Check if the action is legal when the target is point.
    """
    e = _exists(board, src)
    o = _is_open(board, tgt)
    nothing_on_bar = (board[BAR_IDX] == 0)
    is_regular_move = (src >= 0) & (src <= BAR_IDX)   # not noop or chance action
    return e & o & is_regular_move & ((src == BAR_IDX) | nothing_on_bar)


def _move(board: Array, action: Array) -> Array:
    """
    Move checkers based on the action.
    """
    src, _, tgt = _decompose_action(action)
    board = board.at[BAR_IDX + 1].add(
        -1 * (board[tgt] == -1)
    )  # If the opponent has a single checker on this space (blot) it is hit and goes to the bar
    board = board.at[src].add(-1)
    board = board.at[tgt].add(1 + (board[tgt] == -1))  # If hit, the sign changes, so add 1
    return board


def _is_all_off(board: Array) -> bool:
    """
    If all checkers are off, the player wins. Always black's perspective.
    """
    return board[OFF_IDX] == PLAYER_CHECKERS  # type: ignore


def _calc_win_score(board: Array) -> int:
    """
    Normal win: 1 point
    Gammon win: 2 points
    Backgammon win: 3 points
    """
    g = _is_gammon(board)
    return 1 + g + (g & _remains_at_inner(board))


def _is_gammon(board: Array) -> bool:
    """
    If there is no opponent's checker on off, the player wins gammon.
    """
    return board[OFF_IDX + 1] == 0  # type: ignore


def _remains_at_inner(board: Array) -> bool:
    """
    (1) If there is no opponent's checker on off and (2) there is at least one opponent's checker on inner, the player wins backgammon.
    """
    return jnp.take(board, _home_board()).sum() != 0  # type: ignore


def _can_use_other_die(action_die_pair, is_selected, board, playable_dice):
    first_action, last_die = action_die_pair

    def handle_action():
        new_board = _move(board, first_action)
        other_playable_dice = jnp.where(playable_dice == last_die, NO_MOVE, playable_dice)
        next_legal_actions = jax.vmap(partial(_legal_action_mask_for_single_die, board=new_board))(die=other_playable_dice) # return 2D (SRC_LENGTH, DICE_SIDES) array
        return next_legal_actions.any()

    return jax.lax.cond(is_selected & (last_die != NO_MOVE), handle_action, lambda: FALSE)


def _legal_action_mask(board: Array, playable_dice: Array) -> Array:
    start_idx = SRC_NO_MOVE * DICE_SIDES
    no_op_mask = jnp.zeros(ACTION_TOTAL_LENGTH, dtype=jnp.bool_).at[start_idx:start_idx + DICE_SIDES].set(TRUE)
    legal_actions = jax.vmap(partial(_legal_action_mask_for_single_die, board=board))(die=playable_dice) # return 2D (SRC_LENGTH, DICE_SIDES) array
    dice_has_valid_first_moves = legal_actions.any(axis=1)

    # Backgammon movement rules from bkgm.com: A player must use both numbers of a roll if
    #   this is legally possible (or all four numbers of a double). When only one number can
    #   be played, the player must play that number. Or if either number can be played but not
    #   both, the player must play the larger one. When neither number can be used, the player
    #   loses his turn.  In the case of doubles, when all four numbers cannot be played, the
    #   player must play as many numbers as he can.

    # Compute which moves can lead to subsequent second moves
    flat_array_shape = ACTION_TOTAL_LENGTH * MAX_MOVES
    flat_actions = jnp.tile(jnp.arange(ACTION_TOTAL_LENGTH), (MAX_MOVES, 1)).reshape(flat_array_shape)
    flat_last_die = jnp.tile(playable_dice[..., jnp.newaxis], (1, ACTION_TOTAL_LENGTH)).reshape(flat_array_shape)
    flat_legal_moves = legal_actions.reshape(flat_array_shape)
    selection_indices, results = chunked_map(_can_use_other_die, (flat_actions, flat_last_die), flat_legal_moves,
                                             func_kwargs={'board': board, 'playable_dice': playable_dice}, func_uses_is_selected=True)
    action_has_valid_second_moves = jnp.zeros(flat_array_shape, dtype=jnp.bool_)

    action_has_valid_second_moves = action_has_valid_second_moves.at[selection_indices].set(results)
    action_has_valid_second_moves = action_has_valid_second_moves.reshape((MAX_MOVES, ACTION_TOTAL_LENGTH))

    any_valid_second_moves = action_has_valid_second_moves.any()
    two_move_actions = legal_actions & action_has_valid_second_moves
    two_move_actions = two_move_actions.any(axis=0)   # if an action works for any playable dice it is legal

    # We must use the largest roll if no second moves are available
    playable_dice_with_one_move = jnp.where(dice_has_valid_first_moves, playable_dice, NO_MOVE)
    largest_die_with_one_move = jnp.max(playable_dice_with_one_move)  # a playable die will always be greater than NO_MOVE which is negative
    one_move_actions = legal_actions & (playable_dice == largest_die_with_one_move)[..., jnp.newaxis]
    one_move_actions = one_move_actions.any(axis=0)

    out = jnp.where(legal_actions.any(),
                    jnp.where(any_valid_second_moves, two_move_actions, one_move_actions),
                    no_op_mask)
    return out


def _legal_action_mask_for_single_die(board: Array, die: int) -> Array:
    """
    Legal action mask for a single die.
    """
    return jnp.where(die == NO_MOVE, jnp.zeros(ACTION_TOTAL_LENGTH, dtype=jnp.bool_),
                                     _legal_action_mask_for_valid_single_dice(board, die))


def _legal_action_mask_for_valid_single_dice(board: Array, die: int) -> Array:
    """
    Legal action mask for a single die when the die is valid.
    """
    action_src_indices = jnp.arange(SRC_LENGTH, dtype=jnp.int32)  # calc legal action for all src indices

    def _is_legal(idx: Array):
        action = idx * DICE_SIDES + die
        legal_action_mask = jnp.zeros(ACTION_TOTAL_LENGTH, dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[action].set(_is_action_legal(board, action))
        return legal_action_mask

    legal_action_mask = jax.vmap(_is_legal)(action_src_indices).any(axis=0)  # map over ACTION_LENGTH elements
    return legal_action_mask


def _get_abs_board(state: State) -> Array:
    """
    For visualization.
    """
    board: Array = state._board
    turn: Array = state._turn
    return jax.lax.cond(turn == 0, lambda: board, lambda: _flip_board(board))



def max_non_zero_idx_or_fallback(arr: Array, fallback: Array):
    max_idx = (arr.shape[-1] - 1) - jnp.argmax(arr[..., ::-1], axis=-1)
    return jnp.where(arr.any(axis=-1), max_idx, fallback)


def min_non_zero_idx_or_fallback(arr: Array, fallback: Array):
    min_idx = jnp.argmax(arr, axis=-1)
    return jnp.where(arr.any(axis=-1), min_idx, fallback)


def _get_backmost_black_checker_pos(board: Array) -> Array:
    """
    Return the board position for the back most checker for black (smallest index),
    [-1 ... 24].  If there is a checker on the bar we return -1, if all of the
    checkers are off the board we return BOARD_LENGTH.
    """
    has_checkers = (board[..., :BOARD_LENGTH] > 0)
    bar_idx      = BAR_IDX
    bar_position = -1
    has_bar_checkers = (board[..., bar_idx] > 0)
    return jnp.where(has_bar_checkers, bar_position, min_non_zero_idx_or_fallback(has_checkers, BOARD_LENGTH))


def _get_backmost_white_checker_pos(board: Array) -> Array:
    """
    Return the board position for the back most checker for white (largest index),
    [-1 ... 24].  If there is a checker on the bar we return BOARD_LENGTH, if all of the
    checkers are off the board we return -1.
    """
    has_checkers = (board[..., :BOARD_LENGTH] < 0)
    bar_idx      = BAR_IDX + 1
    bar_position = BOARD_LENGTH
    has_bar_checkers = (board[..., bar_idx] < 0)
    return jnp.where(has_bar_checkers, bar_position, max_non_zero_idx_or_fallback(has_checkers, -1))


def _is_no_contact(board: Array) -> Array:
    """ See if we are in the phase of the game where contact is no longer possible """
    return _get_backmost_black_checker_pos(board) > _get_backmost_white_checker_pos(board)


def _calc_pip_diff(board: Array) -> Array:
    """ Calculate the pip difference between the two players """
    my_checkers = jnp.clip(board[..., :BOARD_LENGTH], 0, None)
    opp_checkers = jnp.clip(-board[..., :BOARD_LENGTH], 0, None)
    my_bar = board[..., BAR_IDX]
    opp_bar = -board[..., BAR_IDX + 1]

    point_distances_me = jnp.arange(BOARD_LENGTH, 0, -1)      # 24 down to 1
    point_distances_opp = jnp.arange(1, BOARD_LENGTH + 1, 1)  # 1 up to 24

    my_pip = jnp.sum(my_checkers * point_distances_me, axis=-1) + (my_bar * (BOARD_LENGTH + 1))
    opp_pip = jnp.sum(opp_checkers * point_distances_opp, axis=-1) + (opp_bar * (BOARD_LENGTH + 1))
    return opp_pip - my_pip


def _calc_made_points(board: Array) -> tuple[Array, Array]:
    my_checkers = jnp.clip(board[..., :BOARD_LENGTH], 0, None)
    opp_checkers = jnp.clip(-board[..., :BOARD_LENGTH], 0, None)
    my_made_points = jnp.sum(my_checkers[..., BOARD_LENGTH - HOME_BOARD_LENGTH:BOARD_LENGTH] >= 2, axis=-1)
    opp_made_points = jnp.sum(opp_checkers[..., 0:HOME_BOARD_LENGTH] >= 2, axis=-1)
    return my_made_points, opp_made_points


def _calc_no_contact_home_board_count(board: Array) -> tuple[Array, Array]:
    my_checkers = jnp.clip(board[..., :BOARD_LENGTH], 0, None)
    opp_checkers = jnp.clip(-board[..., :BOARD_LENGTH], 0, None)
    my_home_checkers = jnp.sum(my_checkers[..., BOARD_LENGTH - HOME_BOARD_LENGTH:BOARD_LENGTH], axis=-1)
    opp_home_checkers = jnp.sum(opp_checkers[..., 0:HOME_BOARD_LENGTH], axis=-1)
    my_off_checkers = board[..., OFF_IDX]
    opp_off_checkers = -board[..., OFF_IDX + 1]
    return my_home_checkers + my_off_checkers, opp_home_checkers + opp_off_checkers


def _calc_made_points_heuristic(board: Array) -> tuple[Array, Array]:
    in_no_contact = _is_no_contact(board)

    my_points   = (board[..., :BOARD_LENGTH] >= 2).astype(jnp.float32)
    opp_points  = (board[..., :BOARD_LENGTH] <= -2).astype(jnp.float32)
    my_points_heuristic  = jnp.sum(my_points * MADE_POINT_HEURISTIC, axis=-1)
    opp_points_heuristic = jnp.sum(opp_points * MADE_POINT_HEURISTIC[::-1], axis=-1)
    return jnp.where(in_no_contact, 0, my_points_heuristic), jnp.where(in_no_contact, 0, opp_points_heuristic)

def _calc_blots_heuristic(board: Array) -> tuple[Array, Array]:
    return (
        -jnp.sum(board[..., :BOARD_LENGTH] == 1, axis=-1),   # negate because a blot is bad
        -jnp.sum(board[..., :BOARD_LENGTH] == -1, axis=-1)
    )

def _calc_blots_hit_heuristic(board: Array) -> tuple[Array, Array]:
    my_checkers = jnp.clip(board[..., :BOARD_LENGTH], 0, None)
    opp_checkers = jnp.clip(-board[..., :BOARD_LENGTH], 0, None)
    my_bar = board[..., BAR_IDX]
    opp_bar = -board[..., BAR_IDX + 1]

    indices = jnp.arange(BOARD_LENGTH)
    is_my_home = (indices >= BOARD_LENGTH - HOME_BOARD_LENGTH)
    is_opp_home = (indices < HOME_BOARD_LENGTH)

    is_my_blot = (my_checkers == 1)
    has_one_roll = opp_bar <= 1
    has_two_roll = opp_bar == 0
    has_opp = (opp_checkers > 0).astype(MAT_MULT_DTYPE)
    bar_hit_my = jnp.where((opp_bar[..., jnp.newaxis] > 0) & is_my_home[jnp.newaxis, :], ONE_OF_TWO_DICE_HITS, 0.0)
    prob_my = has_one_roll[..., jnp.newaxis] * (has_opp @ ONE_HIT_DIST_HEURISTIC) + has_two_roll[..., jnp.newaxis] * (has_opp @ TWO_HIT_DIST_HEURISTIC) + bar_hit_my
    my_blot_danger = jnp.sum(is_my_blot * prob_my, axis=-1)

    is_opp_blot = (opp_checkers == 1)
    has_one_roll = my_bar <= 1
    has_two_roll = my_bar == 0
    has_my = (my_checkers > 0).astype(MAT_MULT_DTYPE)
    bar_hit_opp = jnp.where((my_bar[..., jnp.newaxis] > 0) & is_opp_home[jnp.newaxis, :], ONE_OF_TWO_DICE_HITS, 0.0)
    prob_opp = has_one_roll[..., jnp.newaxis] * (has_my @ ONE_HIT_DIST_HEURISTIC.T) + has_two_roll[..., jnp.newaxis] * (has_my @ TWO_HIT_DIST_HEURISTIC.T) + bar_hit_opp
    opp_blot_danger = jnp.sum(is_opp_blot * prob_opp, axis=-1)

    # having vulnerable blots is bad, so negate the sign (negative values are bad)
    return -my_blot_danger, -opp_blot_danger


def _largest_blocking_prime(board: Array) -> tuple[Array, Array, Array, Array]:
    my_checkers = jnp.clip(board[:, :BOARD_LENGTH], 0, None)
    opp_checkers = jnp.clip(-board[:, :BOARD_LENGTH], 0, None)
    my_bar = board[:, BAR_IDX]
    opp_bar = -board[:, BAR_IDX + 1]

    my_made  = (my_checkers >= 2)
    opp_made = (opp_checkers >= 2)

    def longest_consecutive_run(mask_batch):
        inputs = mask_batch.T.astype(jnp.int32)
        def step(carry, x):
            next_carry = (carry + 1) * x
            return next_carry, next_carry
        init_carry = jnp.zeros(mask_batch.shape[0], dtype=jnp.int32)
        _, runs = jax.lax.scan(step, init_carry, inputs)
        runs = runs.T
        max_len = jnp.max(runs, axis=-1)
        end_idx = jnp.argmax(runs, axis=-1)
        start_idx = end_idx - max_len + 1
        return max_len, start_idx, end_idx

    my_largest, my_start_idx, my_end_idx = longest_consecutive_run(my_made)
    opp_largest, opp_start_idx, opp_end_idx = longest_consecutive_run(opp_made)

    indices = jnp.arange(BOARD_LENGTH)

    # Opponent (White) checkers behind Black's prime (indices >= my_start_idx)
    my_is_behind = indices[jnp.newaxis, :] >= my_start_idx[:, jnp.newaxis]
    my_board_count = jnp.sum(jnp.where(my_is_behind, opp_checkers, 0.0), axis=-1)
    my_checkers_behind = jnp.where(my_largest > 0, my_board_count + opp_bar, 0.0)

    # Black checkers behind Opponent's prime (indices <= opp_end_idx)
    opp_is_behind = indices[jnp.newaxis, :] <= opp_end_idx[:, jnp.newaxis]
    opp_board_count = jnp.sum(jnp.where(opp_is_behind, my_checkers, 0.0), axis=-1)
    opp_checkers_behind = jnp.where(opp_largest > 0, opp_board_count + my_bar, 0.0)

    return my_largest, opp_largest, my_checkers_behind, opp_checkers_behind


def _calc_prime_heuristic(board, checker_offset, prime_reward):
    my_prime, opp_prime, my_checkers_behind, opp_checkers_behind = _largest_blocking_prime(board)

    prime_reward = jnp.broadcast_to(prime_reward, my_prime.shape + (prime_reward.shape[-1],))
    my_prime_reward  = prime_reward[jnp.arange(len(my_prime)), my_prime]
    opp_prime_reward = prime_reward[jnp.arange(len(opp_prime)), opp_prime]
    return my_prime_reward * (my_checkers_behind + checker_offset), opp_prime_reward * (opp_checkers_behind + checker_offset)


def _calc_dancing_heuristic(board):
    """
    Chance we are not able to bring a checker off the bar with a pair of dice
    (called dancing), multiplied by the number of checkers on the bar.
    """
    my_bar = board[:, BAR_IDX]
    opp_bar = -board[:, BAR_IDX + 1]
    my_made_points, opp_made_points = _calc_made_points(board)
    my_fraction_blocked  = my_made_points / HOME_BOARD_LENGTH
    opp_fraction_blocked = opp_made_points / HOME_BOARD_LENGTH
    my_force_dance_prob  = 1.0 - (1.0 - my_fraction_blocked) ** NUM_DICE
    opp_force_dance_prob = 1.0 - (1.0 - opp_fraction_blocked) ** NUM_DICE
    return my_force_dance_prob * opp_bar, opp_force_dance_prob * my_bar


def _flexibility_heuristic(board):
    """ This returns a negative number for points with more than 3 checkers (0.5, 1.5, 2.5, etc) """
    my_penalty         = -jnp.sum(jnp.clip(0.5 + (board[..., :BOARD_LENGTH] - 4), 0, None), axis=-1)
    opp_penalty        = -jnp.sum(jnp.clip(0.5 + (-board[..., :BOARD_LENGTH] - 4), 0, None), axis=-1)
    return my_penalty, opp_penalty


@dataclass
class SimpleBackgammonEvaluatorConfig:
    pip_diff_weight: Array = jnp.float32(0.035)
    born_off_weight: Array = jnp.float32(4.0)
    made_points_home_weight: Array = jnp.float32(1.1)
    made_points_weight: Array = jnp.float32(6.4)
    blots_weight: Array = jnp.float32(0.21)
    bar_weight: Array = jnp.float32(0.28)
    flexibility_weight: Array = jnp.float32(0.61)


class SimpleBackgammonEvaluator(core.Evaluator):

    @classmethod
    def get_default_config(cls) -> SimpleBackgammonEvaluatorConfig:
        return SimpleBackgammonEvaluatorConfig()

    def __init__(self, config):
        self.config = config

    def eval(self, state: State) -> Array:
        # in this case we know the internals of the state so grab the board directly
        #board = jax.vmap(_observation_to_board)(state.observations)
        board = state._board

        my_born_off = board[:, OFF_IDX]
        opp_born_off = -board[:, OFF_IDX + 1]
        my_bar_heuristic = -board[:, BAR_IDX]     # negate because having a checker on the bar is bad
        opp_bar_heuristic = board[:, BAR_IDX + 1]

        pip_diff = _calc_pip_diff(board)
        my_blots_heuristic, opp_blots_heuristic = _calc_blots_heuristic(board)
        my_made_points, opp_made_points = _calc_made_points(board)
        my_points_heuristic, opp_points_heuristic = _calc_made_points_heuristic(board)
        my_flexibility_heuristic, opp_flexibility_heuristic = _flexibility_heuristic(board)

        estimated_equity = (
            + (my_born_off - opp_born_off) * self.config.born_off_weight
            + (pip_diff * self.config.pip_diff_weight)
            + (my_blots_heuristic - opp_blots_heuristic) * self.config.blots_weight
            + (my_bar_heuristic - opp_bar_heuristic) * self.config.bar_weight
            + (my_made_points - opp_made_points) * self.config.made_points_home_weight
            + (my_points_heuristic - opp_points_heuristic) * self.config.made_points_weight
            + (my_flexibility_heuristic - opp_flexibility_heuristic) * self.config.flexibility_weight
        )
        return estimated_equity


def _evaluate_boards(boards: Array, mask: Array, model) -> Array:
    orig_shape = boards.shape[:-1]
    flat_boards = boards.reshape((-1, ALL_GAME_POSITIONS))
    dummy_state = State(_board=flat_boards)
    flat_equities = model.eval(dummy_state)
    equities = flat_equities.reshape(orig_shape)
    return jnp.where(mask, equities, jnp.finfo(equities.dtype).min)


class BackgammonTwoPlyStrategy(core.Strategy):
    CHUNK_SIZE = 13
    NUM_CHUNKS = 4

    def __init__(self, env):
        self.env = env
        # if this is not true we need to add code to use padding
        assert self.CHUNK_SIZE * self.NUM_CHUNKS == 2 * SRC_LENGTH

    def _evaluate_2ply_details(self, state: core.State, config, model_cls):
        assert state.current_player.ndim == 1, 'state must be a batched state (jax pytree)'
        B = state.current_player.shape[0]

        one_move_legal_per_die, one_move_boards_per_die, two_move_legal_per_die, candidate_action_indices, sorted_dice, candidate_tgt, candidate_diffs = \
            jax.vmap(_arr_legal_action_mask_details)(state._board, state._playable_dice)

        # Broadcast config to B * 2 * SRC_LENGTH for 1-ply evaluation
        broad_config = core.broadcast_config(config, 2 * SRC_LENGTH)
        model_1ply = model_cls(broad_config)

        # Broadcast config to B * CHUNK_SIZE * 2 * SRC_LENGTH for 2-ply evaluation
        config_2ply = core.broadcast_config(config, self.CHUNK_SIZE * 2 * SRC_LENGTH)
        model_2ply = model_cls(config_2ply)

        # 1. 1-move evaluation
        one_move_equities = _evaluate_boards(one_move_boards_per_die, one_move_legal_per_die, model_1ply) # shape: (B, 52)
        best_one_move_action_idx = jnp.argmax(one_move_equities, axis=-1) # shape: (B,)
        best_one_move_action = candidate_action_indices[jnp.arange(B), best_one_move_action_idx]

        # 2. 2-move evaluation in chunks of 13 along the 52 first actions
        chunk_boards = one_move_boards_per_die.transpose((1, 0, 2)).reshape((self.NUM_CHUNKS, self.CHUNK_SIZE, B, ALL_GAME_POSITIONS))
        chunk_legal = two_move_legal_per_die.transpose((1, 0, 2)).reshape((self.NUM_CHUNKS, self.CHUNK_SIZE, B, 2 * SRC_LENGTH))

        def map_fn(inputs):
            boards, legal = inputs

            two_move_boards = _arr_make_new_boards(
                boards[:, jnp.newaxis, :, :],
                candidate_diffs.transpose((1, 0, 2))[jnp.newaxis, :, :, :]
            )

            flat_two_move_boards = two_move_boards.transpose((2, 0, 1, 3)).reshape((B, self.CHUNK_SIZE * 2 * SRC_LENGTH, ALL_GAME_POSITIONS))
            flat_legal = legal.transpose((1, 0, 2)).reshape((B, self.CHUNK_SIZE * 2 * SRC_LENGTH))

            flat_equities = _evaluate_boards(flat_two_move_boards, flat_legal, model_2ply) # shape: (B, self.CHUNK_SIZE * 52)

            chunk_equities = flat_equities.reshape((B, self.CHUNK_SIZE, 2 * SRC_LENGTH)).transpose((1, 0, 2))
            return chunk_equities

        chunked_equities = jax.lax.map(map_fn, (chunk_boards, chunk_legal))

        two_move_equities = chunked_equities.reshape((2 * SRC_LENGTH, B, 2 * SRC_LENGTH)).transpose((1, 0, 2))

        best_second_move_equity = jnp.max(two_move_equities, axis=-1)
        best_two_move_first_action_idx = jnp.argmax(best_second_move_equity, axis=-1) # shape: (B,)
        best_two_move_first_action = candidate_action_indices[jnp.arange(B), best_two_move_first_action_idx]

        has_two_moves_legal = two_move_legal_per_die.any(axis=(-2, -1))
        has_one_move_legal = one_move_legal_per_die.any(axis=-1)

        best_action_nd = jnp.select([has_two_moves_legal, has_one_move_legal],
                                    [best_two_move_first_action, best_one_move_action],
                                    default=NOOP_ACTION_IDX)

        return (
            one_move_legal_per_die,
            one_move_boards_per_die,
            two_move_legal_per_die,
            candidate_action_indices,
            sorted_dice,
            candidate_tgt,
            candidate_diffs,
            best_action_nd
        )

    def get_next_action_batch(self, state: core.State, _rng_key: Array, config, model_cls) -> Array:
        """ this assumes that state is a jax pytree which has B states """
        _, _, _, _, _, _, _, best_action_nd = self._evaluate_2ply_details(state, config, model_cls)
        return best_action_nd


class BackgammonFullTurnStrategy(core.Strategy):
    LIMIT_2_MOVES = 120
    LIMIT_3_MOVES = 680
    LIMIT_4_MOVES = 3060

    COMBINATIONS_2_MOVES_D = 676
    COMBINATIONS_3_MOVES = 3120
    COMBINATIONS_4_MOVES = 17680

    CHUNK_SIZE_3_MOVES = 20
    NUM_CHUNKS_3_MOVES = 34
    CHUNK_SIZE_4_MOVES = 30
    NUM_CHUNKS_4_MOVES = 102

    def __init__(self, env):
        self.env = env
        self.two_ply_strategy = BackgammonTwoPlyStrategy(env)

    def _propagate_lookahead_step(self, unique_boards_prev, unique_legal_prev, unique_first_move_prev, candidate_diffs, candidate_die, candidate_tgt, limit_next, combinations_next):
        B = unique_boards_prev.shape[0]

        # 1. Generate candidate boards
        boards_next = _arr_make_new_boards(unique_boards_prev[:, :, jnp.newaxis, :], candidate_diffs[:, jnp.newaxis, :SRC_LENGTH, :])

        # 2. Legality check for the step
        def check_legality(boards_prev_single, diffs_single, die_single, tgt_single):
            legals = jax.vmap(
                lambda b: _arr_is_move_legal(b, diffs_single, SRC_ANY_DIE, die_single, tgt_single, SRC_BOARD_MASK_ANY_DIE)
            )(boards_prev_single)
            return legals

        legals_next = jax.vmap(check_legality)(unique_boards_prev, candidate_diffs[:, :SRC_LENGTH], candidate_die[:, :SRC_LENGTH], candidate_tgt[:, :SRC_LENGTH])

        # 3. Combine legality masks
        flat_boards_next = boards_next.reshape((B, combinations_next, ALL_GAME_POSITIONS))
        flat_legal_next = (unique_legal_prev[:, :, jnp.newaxis] & legals_next).reshape((B, combinations_next))

        # 4. Propagate first move action indices
        first_move_next = jnp.repeat(unique_first_move_prev[:, :, jnp.newaxis], SRC_LENGTH, axis=-1).reshape((B, combinations_next))

        # 5. Deduplicate
        unique_boards_next, unique_legal_next, unique_first_move_next = jax.vmap(
            lambda b, l, fm: _deduplicate_boards(b, l, fm, limit_next)
        )(flat_boards_next, flat_legal_next, first_move_next)

        return unique_boards_next, unique_legal_next, unique_first_move_next

    def _evaluate_unique_boards(self, unique_boards, unique_legal, limit, chunk_size, num_chunks, config, model_cls):
        """ evaluate chunks of boards within the unique_boards, stop early when a chunk has no valid boards """
        B = unique_boards.shape[0]
        chunk_boards = unique_boards.transpose((1, 0, 2)).reshape((num_chunks, chunk_size, B, ALL_GAME_POSITIONS))
        chunk_legal = unique_legal.transpose((1, 0)).reshape((num_chunks, chunk_size, B))

        model = model_cls(core.broadcast_config(config, chunk_size))

        init_equities = jnp.full((num_chunks, B, chunk_size), jnp.finfo(jnp.float32).min, dtype=jnp.float32)

        def cond_fn(val):
            c, _ = val
            cond_chunks = c < num_chunks
            safe_c = jnp.minimum(c, num_chunks - 1)
            cond_legal = chunk_legal[safe_c].any()
            return cond_chunks & cond_legal

        def body_fn(val):
            c, equities = val
            boards = chunk_boards[c]
            legal = chunk_legal[c]

            flat_boards = boards.transpose((1, 0, 2)).reshape((-1, ALL_GAME_POSITIONS))
            dummy_state = State(_board=flat_boards)
            flat_equities = model.eval(dummy_state)
            equities_chunk = flat_equities.reshape((B, chunk_size))
            equities_chunk = jnp.where(legal.T, equities_chunk, jnp.finfo(equities_chunk.dtype).min)

            equities = equities.at[c].set(equities_chunk)
            return c + 1, equities

        _, final_equities = jax.lax.while_loop(cond_fn, body_fn, (0, init_equities))

        equities = final_equities.transpose((1, 0, 2)).reshape((B, limit)) # shape: (B, limit)

        return equities

    def get_next_action_batch(self, state: core.State, _rng_key: Array, config, model_cls) -> Array:
        """ this assumes that state is a jax pytree which has B states """
        assert state.current_player.ndim == 1, 'state must be a batched state (jax pytree)'
        B = state.current_player.shape[0]

        # 1. 2-Ply lookahead search details and non-double action
        one_move_legal_per_die, one_move_boards_per_die, two_move_legal_per_die, candidate_action_indices, sorted_dice, candidate_tgt, candidate_diffs, best_action_nd = \
            self.two_ply_strategy._evaluate_2ply_details(state, config, model_cls)

        # For doubles, the die value is the same for all 4 moves.
        double_die_value = sorted_dice[:, -1, jnp.newaxis] # shape: (B, 1)
        candidate_die = jnp.full((B, SRC_LENGTH), double_die_value, dtype=BOARD_DTYPE)

        # We only use the active double roll die (first SRC_LENGTH candidates)
        # Move 2 (COMBINATIONS_2_MOVES_D combinations)
        boards_d = _arr_make_new_boards(one_move_boards_per_die[:, :SRC_LENGTH][:, :, jnp.newaxis, :], candidate_diffs[:, jnp.newaxis, :SRC_LENGTH, :])

        flat_boards_d = boards_d.reshape((B, self.COMBINATIONS_2_MOVES_D, ALL_GAME_POSITIONS))

        # We only consider actual legal Move 2s (no propagation)
        two_move_legal_doubles = two_move_legal_per_die[:, :SRC_LENGTH, SRC_LENGTH:2*SRC_LENGTH]
        flat_legal_d = two_move_legal_doubles.reshape((B, self.COMBINATIONS_2_MOVES_D))

        first_move_idx_d = jnp.repeat(jnp.arange(SRC_LENGTH)[:, jnp.newaxis], SRC_LENGTH, axis=-1).flatten()
        first_move_d = jnp.tile(first_move_idx_d[jnp.newaxis, :], (B, 1))

        # Deduplicate to LIMIT_2_MOVES boards
        unique_boards_2, unique_legal_2, unique_first_move_2 = jax.vmap(
            lambda b, l, fm: _deduplicate_boards(b, l, fm, self.LIMIT_2_MOVES)
        )(flat_boards_d, flat_legal_d, first_move_d)

        # Move 3 propagation
        unique_boards_3, unique_legal_3, unique_first_move_3 = self._propagate_lookahead_step(
            unique_boards_2, unique_legal_2, unique_first_move_2,
            candidate_diffs, candidate_die, candidate_tgt,
            self.LIMIT_3_MOVES, self.COMBINATIONS_3_MOVES
        )

        # Chunked evaluation for Move 3
        move3_equities = self._evaluate_unique_boards(
            unique_boards_3, unique_legal_3,
            self.LIMIT_3_MOVES, self.CHUNK_SIZE_3_MOVES, self.NUM_CHUNKS_3_MOVES,
            config, model_cls
        )

        # Find best 3-move action
        best_3_idx = jnp.argmax(move3_equities, axis=-1)
        best_3_first_move_idx = jax.vmap(lambda fm, idx: fm[idx])(unique_first_move_3, best_3_idx)
        best_action_3 = candidate_action_indices[jnp.arange(B), best_3_first_move_idx]

        # Move 4 propagation
        unique_boards_4, unique_legal_4, unique_first_move_4 = self._propagate_lookahead_step(
            unique_boards_3, unique_legal_3, unique_first_move_3,
            candidate_diffs, candidate_die, candidate_tgt,
            self.LIMIT_4_MOVES, self.COMBINATIONS_4_MOVES
        )

        # Chunked evaluation for Move 4
        move4_equities = self._evaluate_unique_boards(
            unique_boards_4, unique_legal_4,
            self.LIMIT_4_MOVES, self.CHUNK_SIZE_4_MOVES, self.NUM_CHUNKS_4_MOVES,
            config, model_cls
        )

        # Find best double action
        best_4_idx = jnp.argmax(move4_equities, axis=-1)
        best_4_first_move_idx = jax.vmap(lambda fm, idx: fm[idx])(unique_first_move_4, best_4_idx)
        best_action_4 = candidate_action_indices[jnp.arange(B), best_4_first_move_idx]

        # -------------------------------------------------------------
        # 3. Action Selection
        # -------------------------------------------------------------
        has_3_moves = (sorted_dice[:, -3] >= 1)
        has_4_moves = (sorted_dice[:, -4] >= 1)
        has_3_moves_legal = unique_legal_3.any(axis=-1)
        has_4_moves_legal = unique_legal_4.any(axis=-1)

        final_action = jnp.select(
            [has_4_moves & has_4_moves_legal, has_3_moves & has_3_moves_legal],
            [best_action_4, best_action_3],
            default=best_action_nd
        )
        return final_action
