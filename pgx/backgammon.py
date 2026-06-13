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
from typing import Optional

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
OBSERVATION_SIZE   = ALL_GAME_POSITIONS + DICE_SIDES

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
    _board: Array = jnp.zeros(ALL_GAME_POSITIONS, dtype=jnp.int32)
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
    legal_action_mask: Array = _legal_action_mask(board, playable_dice)
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


def _observe(state: State, player_id: Array) -> Array:
    """
    Return observation for player_id which is the board and the dice still
    remaining to be played (represented in the playable dice count format).
    """
    board: Array = state._board
    playable_dice_count_vec: Array = _to_playable_dice_count(
        state._playable_dice
    )  # 6 dim vec which represents the count of playable die.
    return jax.lax.cond(
        player_id == state.current_player,
        lambda: jnp.concatenate((board, playable_dice_count_vec), axis=None),  # type: ignore
        lambda: jnp.concatenate((board, jnp.zeros(DICE_SIDES, dtype=jnp.int32)), axis=None),  # type: ignore
    )


def _action_is_noop(action):
    return (action // DICE_SIDES) == SRC_NO_MOVE

def _action_is_chance(action):
    return action >= ACTION_MOVE_LENGTH

def _to_playable_dice_count(playable_dice: Array) -> Array:
    """
    Return 6 dim vec which represents the number of playable die
    Examples
    Playable dice: 2, 3, -1, -1
    Return: [0, 1, 1, 0, 0, 0]

    Playable dice: 4, 4, 4, 4
    Return: [0, 0, 0, 0, 4, 0]
    """
    dice_indices: Array = jnp.array(list(range(MAX_MOVES)), dtype=jnp.int32)

    def _insert_dice_num(idx: Array, playable_dice: Array) -> Array:
        vec: Array = jnp.zeros(DICE_SIDES, dtype=jnp.int32)
        return (playable_dice[idx] != NO_MOVE) * vec.at[playable_dice[idx]].set(1) + (playable_dice[idx] == NO_MOVE) * vec

    return jax.vmap(_insert_dice_num)(dice_indices, jnp.tile(playable_dice, (MAX_MOVES, 1))).sum(axis=0, dtype=jnp.int32)


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
        legal_action_mask = _legal_action_mask(board, playable_dice)
        return state.replace(
            _board=board,
            _played_dice_num=played_dice_num,
            _playable_dice=playable_dice,
            legal_action_mask=legal_action_mask)

    def get_state_after_chance(state):
        roll_idx = action - ACTION_MOVE_LENGTH
        dice = ALL_DICE_PAIRS[roll_idx]
        playable_dice = _set_playable_dice(dice)
        legal_action_mask = _legal_action_mask(state._board, playable_dice)
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
    board: Array = jnp.array(START_POSITIONS, dtype=jnp.int32)  # type: ignore
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


def _home_board() -> Array:
    """
    black: [18~23], white: [0~5]: Always black's perspective
    """
    return jnp.arange(BOARD_LENGTH - HOME_BOARD_LENGTH, BOARD_LENGTH, dtype=jnp.int32)  # type: ignore


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


def _action_to_src(action: Array) -> int:
    """
    Translate src to board index.  For this function we assume that src is not set to SRC_NO_MOVE.
        no move,         input: action // 6 == 0, output: -2 (-SOURCE_BOARD_OFFSET)
        move from bar,   input: action // 6 == 1, output: BAR_IDX
        move_from board, input: action // 6 >= 2: output: board index (action//6 - 2)
    """
    src_part = action // DICE_SIDES
    return jnp.where(src_part == SRC_BAR, jnp.int32(BAR_IDX), jnp.int32(src_part - SRC_BOARD_OFFSET))  # type: ignore


def _action_to_die(action: Array):
    return action % 6 + 1  # 0~5 -> 1~6


def _calc_tgt(src: int, die) -> int:
    """
    Translate tgt to board index.  We are either coming in from the
    bar (src >= BOARD_LENGTH) or from the board (src < BOARD_LENGTH).
    When we come in from the bar a roll of 1 translates to the first
    index on the board which is index 0, so we land on index = die - 1.
    """
    is_from_bar = (src == BAR_IDX)
    return jnp.where(is_from_bar, jnp.int32(die) - 1, jnp.int32(_tgt_from_board(src, die)))  # type: ignore


def _tgt_from_board(src: int, die: int) -> int:
    """ If the action is a noop (where src equals -SRC_BOARD_OFFSET) we can return anything, we return OFF_IDX. """
    _is_to_board = (src >= 0) & (src + die < BOARD_LENGTH)
    return jnp.where(_is_to_board, jnp.int32(src + die), jnp.int32(OFF_IDX))  # type: ignore


def _decompose_action(action: Array):
    """
    Decompose action to src, die, tgt.
    action = src*6 + die
    """
    src = _action_to_src(action)  # -SRC_BOARD_OFFSET means no-op, 0..BOARD_LENGTH move from board, BAR_IDX move from bar
    die = _action_to_die(action)  # 1~6
    tgt = _calc_tgt(src, die)
    return src, die, tgt


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
