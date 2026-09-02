from collections import namedtuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import pgx.core
from pgx.experimental.utils import act_randomly
from pgx._src.types import Array

from pgx.backgammon import (
    ONE_MOVE_BOARD_DIFFS,
    ONE_MOVE_SRC,
    ONE_MOVE_TGT,
    ONE_MOVE_DIE,
    START_POSITIONS,
    BOARD_DTYPE,
    State,
    Backgammon,
    SimpleBackgammonEvaluator,
    SimpleBackgammonEvaluatorConfig,
    BackgammonTwoPlyStrategy,
    BackgammonTwoPlyChunkedStrategy,
    BackgammonFullTurnStrategy,
    ACTION_CHANCE_LENGTH,
    ACTION_MOVE_LENGTH,
    ACTION_TOTAL_LENGTH,
    BAR_IDX,
    NO_MOVE,
    NOOP_ACTION_IDX,
    _decompose_action,
    _to_playable_dice_count,
    _flip_board,
    _action_to_src,
    _calc_tgt,
    _calc_win_score,
    _change_turn,
    _arr_is_all_on_home_board,
    _arr_is_any_on_opponent_home_board,
    _arr_is_any_earlier,
    _arr_is_move_legal,
    _arr_is_action_legal,
    _arr_one_and_two_moves,
    _move,
    _roll_init_dice,
    _is_turn_end,
    _no_winning_step,
    _set_playable_dice,
    _board_mask_before_src,
    _arr_legal_action_mask,
    _arr_is_illegal_on_board,
    _arr_is_illegal_off,
    _make_observation,
    _observation_to_board,
    _calc_pip_diff,
    _calc_made_points,
    _calc_blots_heuristic,
    _calc_blots_hit_heuristic,
    _largest_blocking_prime,
    _get_backmost_black_checker_pos,
    _get_backmost_white_checker_pos,
    _is_no_contact,
)

seed = 1701
rng = jax.random.PRNGKey(seed)
env = Backgammon()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)
_decompose_action = jax.jit(_decompose_action)
_to_playable_dice_count = jax.jit(_to_playable_dice_count)
_no_winning_step = jax.jit(_no_winning_step)
_action_to_src = jax.jit(_action_to_src)
_calc_tgt = jax.jit(_calc_tgt)
_calc_win_score = jax.jit(_calc_win_score)
_change_turn = jax.jit(_change_turn)
_arr_is_all_on_home_board = jax.jit(_arr_is_all_on_home_board)
_arr_is_any_on_opponent_home_board = jax.jit(_arr_is_any_on_opponent_home_board)
_arr_is_any_earlier = jax.jit(_arr_is_any_earlier)
_arr_is_move_legal = jax.jit(_arr_is_move_legal)
_arr_is_action_legal = jax.jit(_arr_is_action_legal)
_arr_one_and_two_moves = jax.jit(_arr_one_and_two_moves)
_move = jax.jit(_move)
_set_playable_dice = jax.jit(_set_playable_dice)
_board_mask_before_src = jax.jit(_board_mask_before_src)
_arr_legal_action_mask = jax.jit(_arr_legal_action_mask)
_arr_is_illegal_on_board = jax.jit(_arr_is_illegal_on_board)
_arr_is_illegal_off = jax.jit(_arr_is_illegal_off)
_calc_pip_diff = jax.jit(_calc_pip_diff)
_calc_made_points = jax.jit(_calc_made_points)
_calc_blots_heuristic = jax.jit(_calc_blots_heuristic)
_calc_blots_hit_heuristic = jax.jit(_calc_blots_hit_heuristic)
_largest_blocking_prime = jax.jit(_largest_blocking_prime)


def make_test_board():
    board: jnp.ndarray = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0, -2, -1,  0,    0,  0,  0,  0, -5,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0,  0,  0,  0,  0,  0,    0,  5,  1,  2, -3,  0,    0, -4,    7,  0
    ], dtype=BOARD_DTYPE)
    return board


"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
                       +  +  +  -
                       +     +  -
                       +        -
                       +
                       +

    -
    -
    -
    -                     -
    -                  -  -
11 10  9  8  7  6   5  4  3  2  1  0
Bar ----
Off +++++++
"""

def make_test_board_use_2_moves_simple():
    """
    Test that we do both legal moves.  If we roll a 2 and a 4
    then the black checker at index 15 must move to index 19
    then to index 21.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  0,    1,  0, -2,  0, -2,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  1,  0, -2,    0,  0,  2,  0, -2,  0,    0,  0,   11,  -7
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
          +     -         +     -
                -         +     -


    -     -
    -     -     +
11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off +++++++++++   -----------
"""

def make_answer_use_2_moves_simple():
    """
    Give the expected board state where the target number of moves is the,
    final element.  We assume that if an answer goes up to move X it means
    that only X moves are possible this turn.
    """
    return ([2, 4], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    1,  0, -2,  0, -2,  0,  # BAR       OFF     MOVE NUM
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0, -2,    0,  1,  2,  0, -2,  0,    0,  0,   11,  -7,     1],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    1,  0, -2,  0, -2,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0, -2,    0,  0,  2,  1, -2,  0,    0,  0,   11,  -7,     2],
    ], dtype=BOARD_DTYPE))


def make_test_board_no_legal_moves():
    """
    Test that we do both legal moves.  If we roll a 2 and a 4
    then the black checker at index 15 must move to index 19
    then to index 21.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  0,    1,  0, -2,  0, -2,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  1,  0, -2,    0, -2,  2,  0, -2,  0,    0,  0,   11,  -5
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
          +     -      -  +     -
                -      -  +     -


    -     -
    -     -     +
11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off +++++++++++   ---------
"""

def make_answer_no_legal_moves():
    """
    Give the expected board state where the target number of moves is the,
    final element.  We assume that if an answer goes up to move X it means
    that only X moves are possible this turn.
    """
    return ([2, 4], jnp.zeros((0, 28 + 1), dtype=BOARD_DTYPE))


def make_test_board_use_4_moves_simple():
    """
    Test that we do all 4 legal moves.  If we roll double 2's
    then the black checker at index 15 must move to index 23
    hitting on index 21 along the way.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  1,  0,  0,    0,  0,  2, -1, -2,  0,    0,  0,   12, -12
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
          +               +  -  -
                          +     -


11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off ++++++++++++  -------------
"""

def make_answer_use_4_moves_simple():
    return ([2, 2], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  1,    0,  0,  2, -1, -2,  0,    0,  0,   12, -12,     1],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  1,  2, -1, -2,  0,    0,  0,   12, -12,     2],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  0,  2,  1, -2,  0,    0, -1,   12, -12,     3],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  0,  2,  0, -2,  1,    0, -1,   12, -12,     4],
    ], dtype=BOARD_DTYPE))


def make_test_board_use_2_moves_order_matters():
    """
    Check the case where we have to use dice in a specific order to make sure
    both dice are played.  This is an Example from the "Watch It Played" backgammon
    YouTube video channel.  In this case if the player with black pieces rolls a 4 and a 6
    they must play the 6 from index 0 first.  This is the only legal way to use both dice.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           2, -2, -2,  0, -2, -3,    0,  0,  0,  0,  0,  5,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0,  0,  0, -2,  0,  3,    5,  0,  0, -2, -2,  0,    0,  0,    0,  0
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
          -     +   +        -  -
          -     +   +        -  -
                +   +
                    +
                    +

 +
 +
 +                  -
 +                  -  -     -  -  +
 +                  -  -     -  -  +
11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off
"""


def make_answer_use_2_moves_order_matters():
    return ([4, 6], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             1, -2, -2,  0, -2, -3,    1,  0,  0,  0,  0,  5,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27,     28
             0,  0,  0, -2,  0,  3,    5,  0,  0, -2, -2,  0,    0,  0,    0,  0,     1],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             1, -2, -2,  0, -2, -3,    0,  0,  0,  0,  1,  5,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27,     28
             0,  0,  0, -2,  0,  3,    5,  0,  0, -2, -2,  0,    0,  0,    0,  0,     2],
    ], dtype=BOARD_DTYPE))


def make_test_board_use_1_move_higher_roll():
    """
    Check a case where we must use the larger of two rolls.  In this case
    if a 4 and 6 are rolled we must move the black checker from
    index 0 to index 6.  Another example from "Watch It Played".
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           1,  0,  0, -2, -1, -2,    0,  0,  0,  4, -2,  5,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0, -2,  0, -2,  0, -2,    5,  0,  0,  0, -2,  0,    0,  0,    0,  0
    ], dtype=BOARD_DTYPE)


"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
    -     -     -   +           -
    -     -     -   +           -
                    +
                    +
                    +

 +
 +     +
 +     +
 +  -  +            -     -
 +  -  +            -  -  -        +
11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off
"""


def make_answer_use_1_move_higher_roll():
    return ([4, 6], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0, -2, -1, -2,    1,  0,  0,  4, -2,  5,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27,    28
             0, -2,  0, -2,  0, -2,    5,  0,  0,  0, -2,  0,    0,  0,    0,  0,     1],
    ], dtype=BOARD_DTYPE))



def make_test_board_use_1_move_only_available():
    """
    Check a case where only the smaller roll is legal.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           1,  0,  0, -2, -1,  0,   -2,  0,  0,  4, -2,  5,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0, -2,  0, -2,  0, -2,    5,  0,  0,  0, -2,  0,    0,  0,    0,  0
    ], dtype=BOARD_DTYPE)


"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
    -     -     -   +           -
    -     -     -   +           -
                    +
                    +
                    +

 +
 +     +
 +     +
 +  -  +        -         -
 +  -  +        -      -  -        +
11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off
"""


def make_answer_use_1_move_only_available():
    return ([4, 6], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0, -2,  1,  0,   -2,  0,  0,  4, -2,  5,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27,    28
             0, -2,  0, -2,  0, -2,    5,  0,  0,  0, -2,  0,    0, -1,    0,  0,     1],
    ], dtype=BOARD_DTYPE))



def make_test_board_use_3_moves_simple():
    """
    Check that all three valid moves are used.  In this case if we roll double sixes the
    black checker at index 9 must move to index 21 and the checker at
    index 5 must move to index 11 (order within those moves is not restricted).
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  1,    0,  0,  0,  1,  0,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  0,  0, -2,    0,  0,  0, -1,  2, -2,    0,  0,   11, -10
    ], dtype=BOARD_DTYPE)


"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
                -            -  +  -
                -               +  -


       +            +
11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off +++++++++++   ----------
"""


def make_answer_use_3_moves_simple():
    return ([6, 6], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  1,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0, -2,    0,  0,  0,  1,  2, -2,    0, -1,   11, -10,     3],
    ], dtype=BOARD_DTYPE))


def make_test_board_use_4_moves_bear_off_v1():
    """
    Test that we do all 4 legal moves.  If we roll double 4's
    then the black checker at index 9 must move to index 21 (hitting
    on index 17 along the way), at which point one checker bears off
    at index 20.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  0,    0,  0,  0,  1,  0,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  0,  0, -1,    0,  0,  2,  0,  0, -2,    0,  0,   12, -12
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
                -         +        -
                          +        -


       +
11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off ++++++++++++  -------------
"""

def make_answer_use_4_moves_bear_off_v1():
    return ([4, 4], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  1,  0,  0,  0, -1,    0,  0,  2,  0,  0, -2,    0,  0,   12, -12,     1],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  1,    0,  0,  2,  0,  0, -2,    0, -1,   12, -12,     2],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  0,  2,  1,  0, -2,    0, -1,   12, -12,     3],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  0,  1,  1,  0, -2,    0, -1,   13, -12,     4],
    ], dtype=BOARD_DTYPE))



def make_test_board_use_4_moves_bear_off_v2():
    """
    Same as the previous test but the black checker bears off at index 21.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  1,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  0,  0, -1,    0,  0,  0,  2,  0, -2,    0,  0,   12, -12
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
                -            +     -
                             +     -


    +
11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off ++++++++++++  -------------
"""

def make_answer_use_4_moves_bear_off_v2():
    return ([4, 4], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  1,  0,  0, -1,    0,  0,  0,  2,  0, -2,    0,  0,   12, -12,     1],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0, -1,    1,  0,  0,  2,  0, -2,    0,  0,   12, -12,     2],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0, -1,    0,  0,  0,  2,  1, -2,    0,  0,   12, -12,     3],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0, -1,    0,  0,  0,  1,  1, -2,    0,  0,   13, -12,     4],
    ], dtype=BOARD_DTYPE))



def make_test_board_use_2_of_4_moves():
    """
    Test that we do all 2 legal moves.  If we roll double 4's
    then the black checker at index 11 must move to index 19, at which
    point there are no more legal moves.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  1,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  0,  0,  0,    0,  0,  0,  2,  0, -2,    0,  0,   12, -13
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
                             +     -
                             +     -


 +
11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off ++++++++++++  -------------
"""

def make_answer_use_2_of_4_moves():
    return ([4, 4], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  1,  0,  0,    0,  0,  0,  2,  0, -2,    0,  0,   12, -13,     1],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  1,  0,  2,  0, -2,    0,  0,   12, -13,     2],
    ], dtype=BOARD_DTYPE))


def make_test_board_use_3_moves_bear_off_v1():
    """
    Test that we do all 3 legal moves.  If we roll double 4's
    then the black checker at index 15 must move to index 19, at which
    point the two checkers at index 20 bears off.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  1,  0,  0,    0,  0,  2,  0,  0, -2,    0,  0,   12, -13
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
          +               +        -
                          +        -


11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off ++++++++++++  -------------
"""


def make_answer_use_3_moves_bear_off_v1():
    return ([4, 4], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  1,  2,  0,  0, -2,    0,  0,   12, -13,     1],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  1,  1,  0,  0, -2,    0,  0,   13, -13,     2],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  1,  0,  0,  0, -2,    0,  0,   14, -13,     3],
    ], dtype=BOARD_DTYPE))


def make_test_board_use_3_moves_bear_off_v2():
    """
    Test that we do all 3 legal moves.  If we roll double 3's
    then the black checker at index 17 must move to index 20 and
    the black checker at index 18 must move to index 21 and then
    bear off.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  0,  0,  1,    1,  0,  0,  0,  0, -2,    0,  0,   13, -13
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
                +   +              -
                                   -


11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off +++++++++++++ -------------
"""


def make_answer_use_3_moves_bear_off_v2():
    return ([3, 3], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  0,  1,  0,  0, -2,    0,  0,   14, -13,     3],
    ], dtype=BOARD_DTYPE))


def make_test_board_use_2_moves_bear_off_order_matters():
    """
    Test that we use both values when bearing off.  In this case
    if we roll a 6 and 4 we must use the 4 first at index 18, because
    that's the only way to use the 4 and 6 rolls.  In this case
    the black checker at index 18 moves to index 22 and one of the
    checkers at index 19 bears off.  This is another example from Watch It Played.
    """
    return jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           0,  0,  0,  0,  0,  0,    1,  2,  0,  1,  0, -2,    0,  0,   11, -13
    ], dtype=BOARD_DTYPE)

"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
                    +  +     +     -
                       +           -


11 10  9  8  7  6   5  4  3  2  1  0
Bar
Off ++++++++++++  -------------
"""


def make_answer_use_2_moves_bear_off_order_matters():
    return ([6, 4], jnp.array([
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  2,  0,  1,  1, -2,    0,  0,   11, -13,     1],
        [ #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
             0,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,
          # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27,    28
             0,  0,  0,  0,  0,  0,    0,  1,  0,  1,  1, -2,    0,  0,   12, -13,     2],
    ], dtype=BOARD_DTYPE))



def make_test_state(
    current_player: jnp.ndarray,
    board: jnp.ndarray,
    turn: jnp.ndarray,
    dice: jnp.ndarray,
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.ndarray,
    legal_action_mask=jnp.zeros(6 * 26 + 21, dtype=jnp.bool_),
):
    return State(
        current_player=current_player,
        _board=board,
        _turn=turn,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
    )


def test_flip_board():
    test_board = make_test_board()
    board: jnp.ndarray = jnp.zeros(28, dtype=BOARD_DTYPE)
    board = board.at[4].set(-5)
    board = board.at[3].set(-1)
    board = board.at[2].set(-2)
    board = board.at[27].set(-7)
    board = board.at[20].set(2)
    board = board.at[19].set(1)
    board = board.at[13].set(5)
    board = board.at[1].set(3)
    board = board.at[24].set(4)
    flipped_board = _flip_board(test_board)
    print(_flip_board, test_board)
    assert  (flipped_board == board).all()



def test_init():
    state = init(rng)
    assert state._turn == 0 or state._turn == 1


def test_init_roll():
    a = _roll_init_dice(rng)
    assert len(a) == 2
    assert a[0] != a[1]


def test_is_turn_end():
    state = init(rng)
    assert not _is_turn_end(state)

    # white dance
    board: jnp.ndarray = make_test_board()
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    assert _is_turn_end(state)

    # No playable dice
    board: jnp.ndarray = make_test_board()
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(2),
    )
    assert _is_turn_end(state)


def test_change_turn():
    state = init(rng)
    _turn = state._turn
    state = _change_turn(state, jax.random.PRNGKey(0))
    assert state._turn == (_turn + 1) % 2

    test_board: jnp.ndarray = make_test_board()
    board: jnp.ndarray = jnp.zeros(28, dtype=BOARD_DTYPE)
    board = board.at[4].set(-5)
    board = board.at[3].set(-1)
    board = board.at[2].set(-2)
    board = board.at[27].set(-7)
    board = board.at[20].set(2)
    board = board.at[19].set(1)
    board = board.at[13].set(5)
    board = board.at[1].set(3)
    board = board.at[24].set(4)
    state = make_test_state(
        current_player=jnp.int32(0),
        board=test_board,
        turn=jnp.int32(0),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(2),
    )
    state = _change_turn(state, jax.random.PRNGKey(0))
    print(state._board, board)
    assert state._turn == jnp.int32(1)  # Turn changed
    assert (state._board == board).all()  # Flipped.


def test_no_op():
    board: jnp.ndarray = make_test_board()
    legal_action_mask = _arr_legal_action_mask(
        board, jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    )
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
        legal_action_mask=legal_action_mask,
    )
    state = step(state, 0, jax.random.PRNGKey(0))  # execute no-op action
    assert state._turn == jnp.int32(0)  # Turn changes after no-op.


def test_step():
    # 白
    board: jnp.ndarray = make_test_board()
    board = _flip_board(board)  # Flipped
    legal_action_mask = _arr_legal_action_mask(
        board, jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    )
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
        legal_action_mask=legal_action_mask,
    )
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 21, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 0
    ].set(
        True
    )  # 24(bar)->0
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 1
    ].set(
        True
    )  # 24(bar)->1
    assert (expected_legal_action_mask == state.legal_action_mask).all()  # Test legal action

    # White plays die=2 24(bar)->1
    state = step(state=state, action=(1) * 6 + 1, key=jax.random.PRNGKey(0))
    assert (
            state._playable_dice == jnp.array([0, -1, -1, -1], dtype=jnp.int32)
    ).all()  # Is playable dice updated correctly?
    assert state._played_dice_num == 1  # played dice increased?
    assert state._turn == 1  # turn is not changed?
    assert state._board.at[1].get() == 4 and state._board.at[24].get() == 3
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 21, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 0
    ].set(
        True
    )  # 24(bar)->0
    assert (expected_legal_action_mask == state.legal_action_mask).all()  # test legal action
    # White plays die=1 24(off)->0
    state = step(state=state, action=(1) * 6 + 0, key=jax.random.PRNGKey(0))
    assert state._played_dice_num == 0
    assert state._turn == 0  # turn changed to black?
    assert state._board.at[23].get() == -1 and state._board.at[25].get() == -2

    # black
    board: jnp.ndarray = make_test_board()
    legal_action_mask = _arr_legal_action_mask(
        board, jnp.array([4, 5, -1, -1], dtype=jnp.int32)
    )
    state = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([4, 5], dtype=jnp.int32),
        playable_dice=jnp.array([4, 5, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
        legal_action_mask=legal_action_mask,
    )
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 21, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 5
    ].set(
        True
    )  # 19 -> off
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 4
    ].set(
        True
    )  # 19 -> off
    print(jnp.where(state.legal_action_mask==1)[0], jnp.where(expected_legal_action_mask==1)[0])
    assert (expected_legal_action_mask == state.legal_action_mask).all()


def test_observe():
    # NOTE this does not add up to 15 checkers
    board: jnp.ndarray = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0, -6,  1,  0,   -3,  0,  0,  0, -5, -2,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0,  0,  0,  7, -1,  0,    3,  0,  0,  0,  5,  2,    3, -4,    6, -3
    ], dtype=BOARD_DTYPE)

    board_pts = board[0:24]

    # Programmatically build expected features to test _make_observation()
    exp_pts_1 = jnp.array([1.0 if x == 1 else 0.0 for x in board_pts], dtype=jnp.float32)
    exp_pts_2 = jnp.array([1.0 if x == 2 else 0.0 for x in board_pts], dtype=jnp.float32)
    exp_pts_ge3 = jnp.array([1.0 if x >= 3 else 0.0 for x in board_pts], dtype=jnp.float32)
    exp_pts_excess = jnp.maximum(0.0, (board_pts - 3.0) / 2.0)
    exp_pts_neg1 = jnp.array([1.0 if x == -1 else 0.0 for x in board_pts], dtype=jnp.float32)
    exp_pts_neg2 = jnp.array([1.0 if x == -2 else 0.0 for x in board_pts], dtype=jnp.float32)
    exp_pts_le_neg3 = jnp.array([1.0 if x <= -3 else 0.0 for x in board_pts], dtype=jnp.float32)
    exp_pts_neg_excess = jnp.maximum(0.0, (-board_pts - 3.0) / 2.0)

    local_features = jnp.stack([
        exp_pts_1, exp_pts_2, exp_pts_ge3, exp_pts_excess,
        exp_pts_neg1, exp_pts_neg2, exp_pts_le_neg3, exp_pts_neg_excess,
    ], axis=-1)

    my_bar = jnp.abs(board[24]) / 2.0
    opp_bar = jnp.abs(board[25]) / 2.0
    my_off = jnp.abs(board[26]) / 15.0  # PLAYER_CHECKERS = 15
    opp_off = jnp.abs(board[27]) / 15.0

    global_vec = jnp.array([my_bar, opp_bar, my_off, opp_off], dtype=jnp.float32)
    global_features = jnp.tile(global_vec, (24, 1))

    expected_obs = jnp.concatenate([local_features, global_features], axis=-1)
    expected_obs = jnp.expand_dims(expected_obs, axis=1)

    state = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([2, 2, 2, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(1),
    )

    obs = _make_observation(board)
    assert jnp.allclose(obs, expected_obs)

    reconstructed_board = _observation_to_board(obs)
    assert (reconstructed_board == board).all()
    assert jnp.allclose(observe(state), obs)


def test_to_playable_dice_count():
    # Example 1: Playable dice: 2, 3, -1, -1 (representing 0-indexed dice 2 and 3, i.e., face values 3 and 4)
    res1 = _to_playable_dice_count(jnp.array([2, 3, -1, -1], dtype=jnp.int32))
    assert (res1 == jnp.array([0, 0, 1, 1, 0, 0], dtype=jnp.int32)).all()

    # If the user meant 0-indexed dice 1 and 2 (representing face values 2 and 3)
    # Example 2: Playable dice: 4, 4, 4, 4 (representing four 4s)
    res1_alt = _to_playable_dice_count(jnp.array([0, 1, -1, -1], dtype=jnp.int32))
    assert (res1_alt == jnp.array([1, 1, 0, 0, 0, 0], dtype=jnp.int32)).all()

    # Example 2: Playable dice: 4, 4, 4, 4 (representing four 0-indexed 4s, i.e., face values 5)
    res2 = _to_playable_dice_count(jnp.array([4, 4, 4, 4], dtype=jnp.int32))
    assert (res2 == jnp.array([0, 0, 0, 0, 4, 0], dtype=jnp.int32)).all()


def test_estimate_batch_equity():
    from pgx.backgammon import SimpleBackgammonEvaluator
    board = make_test_board()
    state = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs = observe(state)

    predictor = SimpleBackgammonEvaluator()
    batched_state = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], state)
    equity_new = predictor.eval(batched_state)
    assert equity_new.shape == (1,)
    assert jnp.isfinite(equity_new[0])


def test_calc_pip_diff():
    board: jnp.ndarray = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0, -3,  1,  0,   -2,  0,  0,  0,  3, -2,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0,  0,  0,  2, -1,  0,    2,  0,  0,  0, -3,  2,    1, -2,    4, -2
    ], dtype=BOARD_DTYPE)
    assert jnp.sum(jnp.clip(board, 0, None)) ==  15
    assert jnp.sum(jnp.clip(board, None, 0)) == -15

    # Calculate expected pip count for black (me)
    # point_distances_me = [24, 23, ..., 1]
    expected_my_pip = (1 * 20) + (3 * 14) + (2 * 9) + (2 * 6) + (2 * 1) + (1 * 25) # 119

    # Calculate expected pip count for white (opponent)
    # point_distances_opp = [1, 2, ..., 24]
    expected_opp_pip = (3 * 4) + (2 * 7) + (2 * 12) + (1 * 17) + (3 * 23) + (2 * 25) # 186

    expected_pip_diff = expected_opp_pip - expected_my_pip # 186 - 119 = 67

    pip_diff = _calc_pip_diff(board[jnp.newaxis, :])
    assert pip_diff[0] == expected_pip_diff


def test_calc_made_points():
    board: jnp.ndarray = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0, -3,  1,  0,   -2,  0,  0,  0,  3, -2,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0,  0,  0,  2, -1,  0,    2,  0,  0,  0, -3,  2,    1, -2,    4, -2
    ], dtype=BOARD_DTYPE)
    # my home points >= 2 are at: 18 (2), 23 (2). Total = 2 points.
    # opp home points >= 2 are at: 3 (3). Total = 1 point.
    my_made, opp_made = _calc_made_points(board[jnp.newaxis, :])
    assert my_made[0] == 2
    assert opp_made[0] == 1


def test_calc_blots_heuristic():
    board1 = jnp.zeros(28, dtype=BOARD_DTYPE).at[1].set(1).at[2].set(1).at[3].set(-1).at[15].set(-3).at[18].set(2)
    board2 = jnp.zeros(28, dtype=BOARD_DTYPE).at[4].set(-1).at[5].set(-1).at[6].set(-1).at[15].set(-3).at[18].set(2)

    # 1. Call with a single board (shape 1, 28)
    my_blots, opp_blots = _calc_blots_heuristic(board1[jnp.newaxis, :])
    assert my_blots[0] == -2
    assert opp_blots[0] == -1

    # 2. Call with multiple boards (shape 2, 28)
    stacked = jnp.stack([board1, board2], axis=0)
    my_blots, opp_blots = _calc_blots_heuristic(stacked)
    assert (my_blots == jnp.array([-2, 0])).all()
    assert (opp_blots == jnp.array([-1, -3])).all()


def test_calc_blots_hit_heuristic():
    board: jnp.ndarray = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           0,  0,  0, -3,  1,  0,   -2,  0,  1,  0,  3, -2,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0,  0,  0,  2, -1,  0,    2,  0,  0,  0, -3,  2,    0,  0,    5, -4
    ], dtype=BOARD_DTYPE)

    # Base danger probabilities
    my_one_roll = (2.0 * 11.0) / 36.0                       # 4 can be hit in one roll by 6, 8 can be hit in one roll by 11
    my_two_roll = ((1.0 + 6.0 + 1.0) + (2.0 + 5.0)) / 36.0  # 4 can be hit in two rolls at dist 2, 7 and 12, 8 can be hit at dist 3 and 8
    opp_one_roll = (2.0 * 11.0) / 36.0                      # 16 can be hit in one roll by 15 and 10
    opp_two_roll = (5.0 + 5.0 + 1.0) / 36.0                 # 16 can be hit in two rolls at dist 6, 8 and 12

    expected_my_values = [-my_two_roll - my_one_roll, -my_one_roll, 0.0, 0.0]
    expected_opp_values = [-opp_two_roll - opp_one_roll, -opp_one_roll, 0.0, 0.0]

    # Stack 6 test boards: first 3 for opp_bar modifications (danger for me), next 3 for my_bar modifications (danger for opp)
    test_boards = []
    for bar_count in range(len(expected_my_values)):
        test_boards.append(board.at[25].set(-bar_count))
    for bar_count in range(len(expected_opp_values)):
        test_boards.append(board.at[24].set(bar_count))

    stacked_boards = jnp.stack(test_boards, axis=0)

    my_blots, opp_blots = _calc_blots_hit_heuristic(stacked_boards)

    # Verify opp_blots (vulnerability to our attack based on my_bar counts)
    for i, expected_val in enumerate(expected_my_values):
        assert jnp.allclose(my_blots[i], expected_val, atol=1e-3, rtol=1e-3), f"Failed for my_bar={i}: got {my_blots[i]}, expected {expected_val}"

    # Verify my_blots (our vulnerability to opponent's attack based on opp_bar counts)
    for i, expected_val in enumerate(expected_opp_values):
        offset = len(expected_my_values)
        assert jnp.allclose(opp_blots[offset + i], expected_val, atol=1e-3, rtol=1e-3), f"Failed for opp_bar={-i}: got {opp_blots[offset + i]}, expected {expected_val}"

    # test 1D case
    my_blots_0, opp_blots_0 = _calc_blots_hit_heuristic(stacked_boards[0])
    assert jnp.allclose(my_blots_0, my_blots[0], atol=1e-3, rtol=1e-3)
    assert jnp.allclose(opp_blots_0, opp_blots[0], atol=1e-3, rtol=1e-3)


def test_calc_blots_bar_hitting():
    # 1. Opponent has checker on the bar, we have a blot in our home area (point 18)
    board_my = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_my = board_my.at[18].set(1)  # my blot at 18

    for bar_checkers in [0, 1, 2, 3]:
        test_board = board_my.at[25].set(-bar_checkers)  # opponent checker on the bar (opp_bar = 1)
        my_blots, opp_blots = _calc_blots_hit_heuristic(test_board[jnp.newaxis, :])
        assert jnp.allclose(opp_blots[0], 0.0)
        if bar_checkers == 0:
            assert jnp.allclose(my_blots[0], 0.0)
        else:
            assert jnp.allclose(my_blots[0], -11.0 / 36.0)

    # 2. We have a checker on the bar, opponent has a blot in their home area (point 3)
    board_opp = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_opp = board_opp.at[2].set(-1) # opponent blot at 2
    board_opp = board_opp.at[3].set(-2) # opponent made point at 3
    board_opp = board_opp.at[4].set(-1) # opponent blot at 4

    for bar_checkers in [0, 1, 2, 3]:
        test_board = board_opp.at[24].set(bar_checkers)  # my checker on the bar (my_bar = 1)
        my_blots, opp_blots = _calc_blots_hit_heuristic(test_board[jnp.newaxis, :])
        assert jnp.allclose(my_blots[0], 0.0)
        if bar_checkers == 0:
            assert jnp.allclose(opp_blots[0], 0.0)
        else:
            assert jnp.allclose(opp_blots[0], -2.0 * 11.0 / 36.0)


def test_largest_blocking_prime():
    # 1. Test case: Black has a prime of size 3 (points 18, 19, 20)
    # Opponent's farthest back is at 15
    board = jnp.zeros(28, dtype=BOARD_DTYPE)
    board = board.at[10].set(2)
    board = board.at[11].set(3)
    board = board.at[12].set(2)
    board = board.at[13].set(2)

    board = board.at[18].set(2)  # 3 prime (18 - 20) is longest prime after 15
    board = board.at[19].set(3)
    board = board.at[20].set(2)
    board = board.at[21].set(1)
    board = board.at[22].set(2)

    board = board.at[15].set(-1) # White farthest back checker at 15
    board = board.at[16].set(-2)
    board = board.at[17].set(-2)
    board = board.at[23].set(-1)

    # we give the largest blocking prime regardless of whether any checkers
    # are behind the prime
    my_prime, opp_prime, _, _ = _largest_blocking_prime(board[jnp.newaxis, :])
    assert my_prime[0] == 4
    assert opp_prime[0] == 2


def test_is_all_on_home_board():
    board: jnp.ndarray = make_test_board()
    # Black
    assert _arr_is_all_on_home_board(board)
    # White
    board = _flip_board(board)
    assert not _arr_is_all_on_home_board(board)

    board = jnp.zeros(28, dtype=BOARD_DTYPE).at[26].set(13)
    assert not _arr_is_all_on_home_board(board.at[17].set(1))
    assert     _arr_is_all_on_home_board(board.at[18].set(1))
    assert not _arr_is_all_on_home_board(board.at[18].set(1).at[24].set(2))


def test_is_any_on_opponent_home_board():
    board = jnp.zeros(28, dtype=BOARD_DTYPE).at[26].set(13)
    assert     _arr_is_any_on_opponent_home_board(board.at[5].set(1))
    assert not _arr_is_any_on_opponent_home_board(board.at[6].set(1))
    assert     _arr_is_any_on_opponent_home_board(board.at[5].set(1).at[24].set(2))



def test_action_to_src():
    assert _action_to_src(0 * 6) < 0
    assert _action_to_src(1 * 6) == 24
    assert _action_to_src(2 * 6) == 0


def test_calc_tgt():
    assert _calc_tgt(24, 1) == 0  # bar to board (die is transformed from 0~5 -> 1~ 6)
    assert _calc_tgt(6, 2) == 8  # board to board
    assert _calc_tgt(23, 6) == 26  # to off


def test_is_action_legal():
    board: jnp.ndarray = make_test_board()
    # 黒
    assert _arr_is_action_legal(board, (19 + 2) * 6 + 1)  # 19->21
    assert not _arr_is_action_legal(board, (19 + 2) * 6 + 2)  # 19 -> 22
    assert not _arr_is_action_legal(
        board, (19 + 2) * 6 + 2
    )  # 19 -> 22: Some whites on 22
    assert not _arr_is_action_legal(
        board, (22 + 2) * 6 + 2
    )  # 22 -> 25: No black on 22
    assert _arr_is_action_legal(board, (19 + 2) * 6 + 5)  # bear off
    assert not _arr_is_action_legal(
        board, (20 + 2) * 6 + 5
    )  # cannot bear off as some blacks behind
    # white
    board = _flip_board(board)
    assert not _arr_is_action_legal(
        board, (20 + 2) * 6 + 0
    )  # 20->21(after flipped): cannot move checkers as some left on bar
    assert _arr_is_action_legal(board, (1) * 6 + 0)  # bar -> 0(after flipped)
    assert not _arr_is_action_legal(board, (1) * 6 + 2)  # bar -> 2(after flipped)


def test_move():
    # point to point black
    board = make_test_board()
    board = _move(board, (19 + 2) * 6 + 1)  # 19->21
    assert (
        board.at[19].get() == 4
        and board.at[21].get() == 3
        and board.at[25].get() == -4
    )
    # point to off black
    board = make_test_board()
    board = _move(board, (19 + 2) * 6 + 5)  # 19->26
    assert (
        board.at[19].get() == 4
        and board.at[26].get() == 8
        and board.at[25].get() == -4
    )
    # enter white
    board = make_test_board()
    board = _flip_board(board)
    board = _move(board, (1) * 6 + 0)  # 25 -> 0
    assert (
        board.at[24].get() == 3
        and board.at[0].get() == 1
    )
    # hit white
    board = make_test_board()
    board = _flip_board(board)
    board = _move(board, (1 + 2) * 6 + 1)  # 1 -> 3
    print(board)
    assert (
        board.at[1].get() == 2
        and board.at[3].get() == 1
        and board.at[25].get() == -1
    )


def test_board_mask_before_src():
    # Test scalar input
    src_scalar = jnp.int32(5)
    mask_scalar = _board_mask_before_src(src_scalar)
    assert mask_scalar.shape == (28,)
    expected_scalar = jnp.zeros(28, dtype=jnp.bool_).at[jnp.array([0, 1, 2, 3, 4, 24])].set(True)
    assert (mask_scalar == expected_scalar).all()

    # Test 1D array input
    src_array = jnp.array([5, 24], dtype=jnp.int32)
    mask_array = _board_mask_before_src(src_array)
    assert mask_array.shape == (2, 28)

    expected_bar = jnp.zeros(28, dtype=jnp.bool_)
    assert (mask_array[0] == expected_scalar).all()
    assert (mask_array[1] == expected_bar).all()


def test_arr_is_illegal_on_board():
    # Base board: checkers at 3 (2), opponent checkers at 4 (-2), empty at 5 (0), single opponent at 6 (-1)
    board = jnp.zeros(28, dtype=BOARD_DTYPE)
    board = board.at[3].set(2)
    board = board.at[4].set(-2)
    board = board.at[6].set(-1)

    # We evaluate for all actions since the optimized function uses global constants
    is_illegal = _arr_is_illegal_on_board(board, ONE_MOVE_BOARD_DIFFS)

    # Case 1: move from 3 to 5 (empty) -> Action: (3+2)*6 + (2-1) = 31
    assert not is_illegal[31]

    # Case 2: move from 3 to 6 (single opponent - blot hit) -> Action: (3+2)*6 + (3-1) = 32
    assert not is_illegal[32]

    # Case 3: move from 3 to 4 (occupied by opponent >= 2) -> Action: (3+2)*6 + (1-1) = 30
    assert is_illegal[30]

    # Case 4: move from 5 (empty) to 7 (empty) -> Action: (5+2)*6 + (2-1) = 43
    assert is_illegal[43]


def test_arr_is_illegal_off():
    # OFF_IDX is 26, BOARD_LENGTH is 24

    # Case 1: Move is not to off (tgt != 26) -> always legal (not illegal off)
    mask1 = jnp.zeros(28, dtype=jnp.bool_).at[10].set(True) # checker outside home board
    assert (~_arr_is_illegal_off(mask1, jnp.int32(10), jnp.int32(15), jnp.int32(5))).all()

    # Case 2: Move to off, but one checker is outside home board (at index 10)
    mask2 = jnp.zeros(28, dtype=jnp.bool_).at[10].set(True).at[20].set(True)
    assert _arr_is_illegal_off(mask2, jnp.int32(20), jnp.int32(26), jnp.int32(4)).all()

    # Case 3: Move to off, all checkers are in home board, not farthest back but uses an exact die
    mask3 = jnp.zeros(28, dtype=jnp.bool_).at[20].set(True).at[19].set(True)
    assert (~_arr_is_illegal_off(mask3, jnp.int32(20), jnp.int32(26), jnp.int32(4))).all()

    # Case 4: Move to off, all checkers in home board, die is larger, but not farthest back
    mask4 = jnp.zeros(28, dtype=jnp.bool_).at[18].set(True).at[20].set(True)
    assert (_arr_is_illegal_off(mask4, jnp.int32(20), jnp.int32(26), jnp.int32(5))).all()

    # Case 5: Move to off, all checkers in home board, die is larger, and it IS the farthest back
    mask5 = jnp.zeros(28, dtype=jnp.bool_).at[20].set(True)
    assert (~_arr_is_illegal_off(mask5, jnp.int32(20), jnp.int32(26), jnp.int32(5))).all()


def test_arr_is_move_legal():

    @dataclass(frozen=True)
    class CaseApplyDiff:
        board: Array
        src: Array
        tgt: Array
        die: Array
        is_hit: Array
        is_legal: Array
        description: str


    # 28 elements board
    # orig_board has some -1 and other values
    orig_board = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
          -1, -1,  0,  2, -2,  0,   -1,  0,  0,  0,  0,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0,  0,  0,  0,  0,  0,    0,  0,  2,  0,  2, -2,    0,  0,   9, -8
    ], dtype=BOARD_DTYPE)

    home_board    = orig_board.at[3].set(0)
    bar_board     = home_board.at[24].set(1).at[26].set(8)

    tests = [
        CaseApplyDiff(board=orig_board,    src=3,  tgt=5,  die=2, is_hit=False, is_legal=True,  description='move with no hit'),
        CaseApplyDiff(board=orig_board,    src=3,  tgt=6,  die=3, is_hit=True,  is_legal=True,  description='move with legal hit'),
        CaseApplyDiff(board=orig_board,    src=20, tgt=21, die=1, is_hit=False, is_legal=True,  description='move from one point to another'),
        CaseApplyDiff(board=home_board,    src=22, tgt=26, die=2, is_hit=False, is_legal=True,  description='move to off with exact die roll'),
        CaseApplyDiff(board=home_board,    src=20, tgt=26, die=6, is_hit=False, is_legal=True,  description='move to off as farthest back'),
        CaseApplyDiff(board=home_board,    src=20, tgt=26, die=4, is_hit=False, is_legal=True,  description='move to off as farthest back with exact die roll'),
        CaseApplyDiff(board=bar_board,     src=24, tgt=2,  die=3, is_hit=False, is_legal=True,  description='come in from the bar onto empty slot'),
        CaseApplyDiff(board=bar_board,     src=24, tgt=1,  die=2, is_hit=True,  is_legal=True,  description='come in from the bar and hit'),
        CaseApplyDiff(board=bar_board,     src=24, tgt=3,  die=4, is_hit=False, is_legal=True,  description='come in from the bar and land on black checkers'),

        CaseApplyDiff(board=orig_board,    src=3,  tgt=4,  die=1, is_hit=False, is_legal=False, description='cannot land on index 4'),
        CaseApplyDiff(board=orig_board,    src=2,  tgt=5,  die=3, is_hit=False, is_legal=False, description='no black checkers on src'),
        CaseApplyDiff(board=orig_board,    src=22, tgt=26, die=2, is_hit=False, is_legal=False, description='move to off but one checker is not in the home board'),
        CaseApplyDiff(board=bar_board,     src=22, tgt=26, die=2, is_hit=False, is_legal=False, description='move to off but one checker is on the bar'),
        CaseApplyDiff(board=home_board,    src=22, tgt=26, die=3, is_hit=False, is_legal=False, description='move to off but roll is not exact and not farthest back'),
        CaseApplyDiff(board=bar_board,     src=24, tgt=4,  die=5, is_hit=False, is_legal=False, description='come in from bar but cannot land on index 4'),
    ]

    for cur_test in tests:
        cur_diff      = jnp.zeros(28, dtype=BOARD_DTYPE).at[cur_test.src].set(-1).at[cur_test.tgt].set(+1)
        action        = jnp.where(cur_test.src == 24, 6 + cur_test.die - 1, (cur_test.src + 2) * 6 + cur_test.die - 1)
        src, die, tgt = _decompose_action(action)
        assert (src == cur_test.src).all(), cur_test.description
        assert (die == cur_test.die).all(), cur_test.description
        assert (tgt == cur_test.tgt).all(), cur_test.description
        assert ((src >= 0) & (src <= 24)).all(), cur_test.description
        assert (((tgt >= 0) & (tgt < 24)) | (tgt == 26)).all(), cur_test.description
        assert ((die >= 1) & (die <= 6)).all(), cur_test.description
        expected_board = cur_test.board + cur_diff
        if cur_test.is_hit:
            expected_board = expected_board.at[cur_test.tgt].set(1)

        is_legal = _arr_is_move_legal(cur_test.board, jnp.tile(cur_diff, (156, 1)), cur_test.src, cur_test.die, cur_test.tgt)
        is_legal_hit_tgt = (jnp.tile(cur_diff, (156, 1)) > 0) & (cur_test.board == -1)
        new_board = jnp.where(is_legal_hit_tgt[action], BOARD_DTYPE(1), cur_test.board + cur_diff)
        assert is_legal[action] == cur_test.is_legal, cur_test.description
        assert (new_board == expected_board).all(), cur_test.description


def test_arr_one_and_two_moves():
    orig_board = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
          -1, -1,  0,  2, -2,  0,   -1,  0,  0,  0,  0,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0,  0,  0,  0,  0,  0,    0,  0,  0,  2,  0, -2,    0,  0,   11, -8
    ], dtype=BOARD_DTYPE)

    for dice_pair in [[1, 2], [3, 5], [4, 6], [1, 1], [6, 6]]:
        dice = jnp.array(dice_pair, dtype=jnp.int32)
        one_move_legal, one_move_boards, two_move_legal, candidate_action_indices, candidate_tgt, candidate_diffs = \
            _arr_one_and_two_moves(orig_board, dice)

        assert one_move_legal.shape == (52,)
        assert one_move_boards.shape == (52, 28)
        assert two_move_legal.shape == (52, 52)
        assert candidate_action_indices.shape == (52,)
        assert candidate_tgt.shape == (52,)
        assert candidate_diffs.shape == (52, 28)

        # Verify Move 1 legality and boards
        for i in range(52):
            action = candidate_action_indices[i]
            expected_legal = _arr_is_move_legal(orig_board, candidate_diffs[i], ONE_MOVE_SRC[action], ONE_MOVE_DIE[action], ONE_MOVE_TGT[action])
            assert one_move_legal[i] == expected_legal[action]

            expected_hit_tgt = (candidate_diffs[i] > 0) & (orig_board == -1)
            expected_board = jnp.where(expected_hit_tgt, BOARD_DTYPE(1), orig_board + candidate_diffs[i])
            expected_hit_count = expected_hit_tgt.sum().astype(BOARD_DTYPE)
            expected_board = expected_board.at[25].add(-expected_hit_count)

            assert (one_move_boards[i] == expected_board).all()

        # Verify Move 2 legality
        for i in range(52):
            for j in range(52):
                if (i < 26 and j >= 26) or (i >= 26 and j < 26):
                    action_j = candidate_action_indices[j]
                    expected_legal_j = _arr_is_move_legal(
                        one_move_boards[i],
                        candidate_diffs[j],
                        ONE_MOVE_SRC[action_j],
                        ONE_MOVE_DIE[action_j],
                        ONE_MOVE_TGT[action_j]
                    )
                    expected_two_move_legal = expected_legal_j[action_j]
                    assert two_move_legal[i, j] == expected_two_move_legal
                else:
                    assert two_move_legal[i, j] == False


def test_arr_legal_action():
    board = make_test_board()
    # black rolling a 4 and a 3
    playable_dice = jnp.array([3, 2, -1, -1], dtype=jnp.int32)
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 21, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 3
    ].set(
        True
    )  # 19->23
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (20 + 2) + 2
    ].set(
        True
    )  # 20->23
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (20 + 2) + 3
    ].set(
        True
    )  # 20->off
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (21 + 2) + 2
    ].set(
        True
    )  # 21->off
    legal_action_mask = _arr_legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    playable_dice = jnp.array([5, 5, 5, 5], dtype=jnp.int32)
    expected_legal_action_mask = jnp.zeros(6 * 26 + 21, dtype=jnp.bool_)
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 5
    ].set(True)
    legal_action_mask = _arr_legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    # white
    board = _flip_board(board)
    playable_dice = jnp.array([4, 1, -1, -1], dtype=jnp.int32)
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 21, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * 1 + 1].set(
        True
    )
    legal_action_mask = _arr_legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    playable_dice = jnp.array([4, 4, 4, 4], dtype=jnp.int32)
    expected_legal_action_mask = jnp.zeros(
        6 * 26 + 21, dtype=jnp.bool_
    )  # dance
    expected_legal_action_mask = expected_legal_action_mask.at[0].set(
        True
    )  # only no-op
    legal_action_mask = _arr_legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    board_1 = make_test_board_use_2_moves_simple()
    playable_dice = jnp.array([1, 3, -1, -1], dtype=jnp.int32)
    expected_legal_action_mask = jnp.zeros(6 * 26 + 21, dtype=jnp.bool_)
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (15 + 2) + 3
    ].set(True)  # only using the 4 at index 15
    legal_action_mask = _arr_legal_action_mask(board_1, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()


    board_1 = make_test_board_use_2_moves_simple()
    board_1 = board_1.at[15].set(0)
    board_1 = board_1.at[19].set(1)
    playable_dice = jnp.array([1, -1, -1, -1], dtype=jnp.int32)
    expected_legal_action_mask = jnp.zeros(6 * 26 + 21, dtype=jnp.bool_)
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 1
    ].set(True)  # only using the 2 at index 19
    legal_action_mask = _arr_legal_action_mask(board_1, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()


def get_board_answer_pairs():
    return [
        (make_test_board_use_2_moves_simple(),                 make_answer_use_2_moves_simple()),
        (make_test_board_no_legal_moves(),                     make_answer_no_legal_moves()),
        (make_test_board_use_4_moves_simple(),                 make_answer_use_4_moves_simple()),
        (make_test_board_use_2_moves_order_matters(),          make_answer_use_2_moves_order_matters()),
        (make_test_board_use_1_move_higher_roll(),             make_answer_use_1_move_higher_roll()),
        (make_test_board_use_1_move_only_available(),          make_answer_use_1_move_only_available()),
        (make_test_board_use_3_moves_simple(),                 make_answer_use_3_moves_simple()),
        (make_test_board_use_4_moves_bear_off_v1(),            make_answer_use_4_moves_bear_off_v1()),
        (make_test_board_use_4_moves_bear_off_v2(),            make_answer_use_4_moves_bear_off_v2()),
        (make_test_board_use_2_of_4_moves(),                   make_answer_use_2_of_4_moves()),
        (make_test_board_use_3_moves_bear_off_v1(),            make_answer_use_3_moves_bear_off_v1()),
        (make_test_board_use_3_moves_bear_off_v2(),            make_answer_use_3_moves_bear_off_v2()),
        (make_test_board_use_2_moves_bear_off_order_matters(), make_answer_use_2_moves_bear_off_order_matters()),
    ]


def test_forced_moves():
    board_answer_pairs = get_board_answer_pairs()

    for _test_num, (test_board, (dice, answer)) in enumerate(board_answer_pairs):
        expected_boards = answer[:,:28]
        move_nums = answer[:,28]
        max_move = 0 if len(move_nums) == 0 else jnp.max(move_nums)

        rolled_doubles = dice[0] == dice[1]
        theoretical_max_move = 4 if rolled_doubles else 2

        # if we have less than the theoretical maximum number of moves than there is a no-op move
        steps_after_max_move = (2 if (max_move < theoretical_max_move) else 1)
        last_move_to_check = max_move + steps_after_max_move

        black_checker_count = jnp.sum(jnp.where(expected_boards > 0, expected_boards, 0), axis=1)
        white_checker_count = jnp.sum(jnp.where(expected_boards < 0, expected_boards, 0), axis=1)
        assert (black_checker_count == 15).all()
        assert (white_checker_count == -15).all()

        parallel_games = 20

        dice = jnp.array(dice, dtype=jnp.int32) - 1   # dice are encoded as 0 through 5
        playable_dice = _set_playable_dice(dice)

        rng = jax.random.PRNGKey(0)
        rng, subkey = jax.random.split(rng)

        start_state = make_test_state(
            current_player=jnp.int32(0),
            board=test_board,
            turn=jnp.int32(0),
            dice=jnp.array(dice, dtype=jnp.int32),
            playable_dice=jnp.array(playable_dice, dtype=jnp.int32),
            played_dice_num=jnp.int32(0),
            legal_action_mask=_arr_legal_action_mask(test_board, playable_dice)
        )

        # make a batched version of start_state
        s = jax.jit(jax.vmap(lambda _ : start_state))(jnp.arange(parallel_games))
        vmap_step = jax.jit(jax.vmap(step))

        # we need to use vmap for State methods so that the underlying methods can
        # operate on State member data of the expected size instead of a batched
        # version of State (PyTree) that has first dimension size of parallel_games
        vmap_has_chance_logits_recalc = jax.vmap(State.has_chance_logits_recalc)
        vmap_get_chance_logits        = jax.vmap(State.get_chance_logits)

        answer_idx = 0

        # make random moves and verify that the expected boards always match
        for move_num in range(1, last_move_to_check + 1):
            rng, subkey = jax.random.split(rng)
            a = act_randomly(subkey, s.legal_action_mask)

            rng, step_rng = jax.random.split(rng)
            step_keys = jax.random.split(step_rng, parallel_games)

            if move_num == max_move + steps_after_max_move:
                # about to step according to a chance action
                assert (a >= 6 * 26).all()
            elif move_num == max_move + 1:
                # player had to prematurely end their turn, this is a player noop action
                assert (a < 6).all()
            else:
                # about to step according to a player move action
                assert (a < 6 * 26).all()

            s = vmap_step(s, a, step_keys)

            if move_num >= max_move and move_num < max_move + steps_after_max_move:
                if move_num == max_move + steps_after_max_move - 1:
                    # next player is about to roll the dice
                    assert (s.current_player == jnp.array([1], dtype=jnp.int32)).all()
                    assert (s._played_dice_num == jnp.array([0], dtype=jnp.int32)).all()
                    assert (~s.legal_action_mask[..., 0:6*26]).all()     # chance action is next, all player moves are illegal
                    assert jnp.isneginf(vmap_get_chance_logits(s)[..., 0:6*26]).all()   # player move actions have zero probability
                    assert vmap_has_chance_logits_recalc(s).all()        # chance action is next, we are using chance logits
                    assert (s._playable_dice == -1).all()                # new player has not yet rolled
                else:
                    # current player has run out of moves and must make a noop move
                    assert (s.current_player == jnp.array([0], dtype=jnp.int32)).all()
                    assert (s._played_dice_num == jnp.array([max_move], dtype=jnp.int32)).all()
                    assert (~s.legal_action_mask[..., 6:]).all()              # only noop action is allowed
            else:
                if move_num == max_move + steps_after_max_move:
                    # next player is about to make their first move
                    assert (s.current_player == jnp.array([1], dtype=jnp.int32)).all()
                    assert (s._played_dice_num == jnp.array([0], dtype=jnp.int32)).all()
                else:
                    # current player is going to make a move
                    assert (s.current_player == jnp.array([0], dtype=jnp.int32)).all()
                    assert (s._played_dice_num == jnp.array([move_num], dtype=jnp.int32)).all()
                assert (~s.legal_action_mask[..., 6*26:]).all()      # player move action is next, all dice moves are illegal
                assert jnp.isneginf(vmap_get_chance_logits(s)[..., 6*26:]).all()    # chance actions have zero probability
                assert (~vmap_has_chance_logits_recalc(s)).all()     # player move action is next, we are not using chance logits
                assert (s._playable_dice != -1).any(axis=-1).all()   # existing or new now has playable dice

            if len(move_nums) > 0 and move_nums[answer_idx] == move_num:
                # validate that the current board matches all game boards
                cur_board = expected_boards[answer_idx]
                if move_num >= max_move + steps_after_max_move - 1:
                    cur_board = _flip_board(cur_board)
                assert (s._board == cur_board[None,:]).all()
                answer_idx += 1


StrategyTest = namedtuple('StrategyTest', ['board', 'dice', 'src_pos', 'tgt_pos', 'description'])


def _run_strategy_test(strategy, cur_test, eval_cls=SimpleBackgammonEvaluator):
    assert_msg = ', '.join([cur_test.description, str(strategy.__class__)])
    assert (jnp.sum(jnp.clip(cur_test.board, 0, None)) == 15).all()
    assert (jnp.sum(jnp.clip(cur_test.board, None, 0)) == -15).all()

    all_actions = [(src + 2) * 6 + (tgt - src) - 1 for src, tgt in zip(cur_test.src_pos, cur_test.tgt_pos)]

    dice = jnp.array(cur_test.dice, dtype=jnp.int32) - 1   # dice are encoded as 0 through 5
    playable_dice = _set_playable_dice(dice)

    rng = jax.random.PRNGKey(0)

    start_state = make_test_state(
        current_player=jnp.int32(0),
        board=cur_test.board,
        turn=jnp.int32(0),
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=jnp.int32(0),
        legal_action_mask=_arr_legal_action_mask(cur_test.board, playable_dice)
    )
    expected_state = start_state
    for action in all_actions:
        assert expected_state.legal_action_mask[action], assert_msg
        expected_state = step(expected_state, action, rng)

    batch_size = 10
    rng = jax.random.PRNGKey(0)
    config = eval_cls.get_default_config()
    config_batched = jax.tree_util.tree_map(lambda x: jnp.stack([x] * batch_size), config)

    # have the strategy play it's best moves, the moves may be in a different order
    # but should end up with the same final board
    strategy_state = jax.tree_util.tree_map(lambda x: jnp.stack([x] * batch_size), start_state)
    for idx in range(len(all_actions)):
        rng, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, batch_size)
        actions = strategy.get_next_action_batch(strategy_state, keys, config_batched, eval_cls)
        assert strategy_state.legal_action_mask[jnp.arange(batch_size), actions].all(), assert_msg
        strategy_state = jax.vmap(step)(strategy_state, actions, keys)

    vmap_flip_board = jax.vmap(_flip_board)

    assert (strategy_state._playable_dice == -1).all(), assert_msg
    assert (strategy_state.current_player == 1).all(), assert_msg
    # flip the board back to being the first player's turn
    assert (vmap_flip_board(strategy_state._board) == _flip_board(expected_state._board)).all(), assert_msg


def test_strategy_obvious_moves():

    # test whether SimpleBackgammonEvaluator is picking the obvious best move
    start_board = jnp.array(START_POSITIONS, dtype=BOARD_DTYPE)
    starting_move_tests = [
        StrategyTest(start_board, (1, 3), (16, 18), (19, 19), '1 and 3 opening roll'),
        StrategyTest(start_board, (2, 4), (16, 18), (20, 20), '2 and 4 opening roll'),
        StrategyTest(start_board, (6, 1), (11, 16), (17, 17), '1 and 6 opening roll'),
    ]

    hit_make_point_board = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           1,  0,  1,  0,  0, -3,    0, -4,  0,  0,  0,  2,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
          -5,  1,  0,  0,  0,  3,    3,  2,  2,  0,  0, -2,    0,  0,    0,   0   # we will add blot on 21 or 22
    ], dtype=BOARD_DTYPE)
    hit_and_make_home_point_tests = [
        StrategyTest(hit_make_point_board.at[21].set(-1), (4, 3), (17, 18), (21, 21), 'able to hit and make point on idx 21'),
        StrategyTest(hit_make_point_board.at[22].set(-1), (5, 4), (17, 18), (22, 22), 'able to hit and make point on idx 22'),
    ]

    # a NOOP move is encoded as src=-2, tgt=-1
    one_move_board = make_test_board_use_1_move_higher_roll()
    one_move_tests = [
        StrategyTest(one_move_board, (4, 6), (0, -2), (6, -1), 'only one legal move then NOOP'),
    ]

    no_move_board = make_test_board_no_legal_moves()
    no_move_tests = [
        StrategyTest(no_move_board, (2, 4), (-2,), (-1,), 'no legal moves, only NOOP'),
    ]

    all_tests = starting_move_tests + hit_and_make_home_point_tests + one_move_tests + no_move_tests
    all_boards = jnp.concatenate([test.board for test in all_tests])

    strategy_lst = [BackgammonTwoPlyStrategy(env), BackgammonTwoPlyChunkedStrategy(env), BackgammonFullTurnStrategy(env)]
    for strategy in strategy_lst:
        for cur_test in all_tests:
            _run_strategy_test(strategy, cur_test)


def test_strategy_full_move():
    class UnitTestEvaluator(pgx.core.Evaluator):
        @classmethod
        def get_default_config(cls):
            return SimpleBackgammonEvaluator.get_default_config()   # return a dummy config

        def __init__(self, config):
            super().__init__(config)

        def eval(self, state: State) -> Array:
            # create a reward that would require looking 4 moves ahead to discover,
            # if you only look two moves ahead you would move one checker from 7 to 9,
            # then you would move 13 to 15
            board = state._board
            specific_reward  = jnp.where(board[..., 8] == 4, 1e4, 0.0).astype(jnp.float32)
            height_penalty   = -jnp.sum(jnp.clip(board[..., :24] - 1.0, 0, None) ** 2, axis=-1).astype(jnp.float32)
            position_penalty = -jnp.sum(jnp.clip(board[..., :24] * jnp.arange(24, dtype=jnp.float32)[jnp.newaxis, :] * 1e-4, 0, None), axis=-1)
            return specific_reward + position_penalty + height_penalty

    all_blots = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
           1,  1,  1,  1,  1,  1,    1,  2,  1,  0,  1,  1,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26,  27
           1,  1,  1,  0,  0,  0,    0,  0,  0,  0, -8, -7,    0,  0,    0,   0
    ], dtype=BOARD_DTYPE)

    two_ply_tests   = [StrategyTest(all_blots, (1, 1), (7, 8, 13, 14), (8, 9, 14, 15), 'short sighted eval will move from 7 to 9 then 13 to 15')]
    full_move_tests = [StrategyTest(all_blots, (1, 1), (6, 7, 7, 7),   (7, 8, 8, 8),   'correctly move 3 checkers to put 4 checkers on position 8')]
    strategy_lst    = [(BackgammonTwoPlyStrategy(env), two_ply_tests), (BackgammonFullTurnStrategy(env), full_move_tests)]
    for strategy, all_tests in strategy_lst:
        for cur_test in all_tests:
            _run_strategy_test(strategy, cur_test, UnitTestEvaluator)


def test_calc_win_score():
    # backgammon win by black
    back_gammon_board = jnp.zeros(28, dtype=BOARD_DTYPE)
    back_gammon_board = back_gammon_board.at[26].set(15)
    back_gammon_board = back_gammon_board.at[23].set(-15)  # black on home board
    print(_calc_win_score(back_gammon_board))
    assert _calc_win_score(back_gammon_board) == 3

    # gammon win by black
    gammon_board = jnp.zeros(28, dtype=BOARD_DTYPE)
    gammon_board = gammon_board.at[26].set(15)
    gammon_board = gammon_board.at[7].set(-15)
    assert _calc_win_score(gammon_board) == 2

    # single win by black
    single_board = jnp.zeros(28, dtype=BOARD_DTYPE)
    single_board = single_board.at[26].set(15)
    single_board = single_board.at[27].set(-3)
    single_board = single_board.at[3].set(-12)
    assert _calc_win_score(single_board) == 1


def test_black_off():
    board: jnp.ndarray = jnp.zeros(28, dtype=BOARD_DTYPE)
    board = board.at[0].set(15)
    playable_dice = jnp.array([3, 2, -1, -1])
    legal_action_mask = _arr_legal_action_mask(board, playable_dice)
    print("3, 2", jnp.where(legal_action_mask != 0)[0])
    playable_dice = jnp.array([1, 1, -1, -1])
    legal_action_mask = _arr_legal_action_mask(board, playable_dice)
    print("1, 1", jnp.where(legal_action_mask != 0)[0])


def test_board_mask_before_src():
    # Test scalar input
    src_scalar = jnp.int32(5)
    mask_scalar = _board_mask_before_src(src_scalar)
    assert mask_scalar.shape == (28,)
    expected_scalar = jnp.zeros(28, dtype=jnp.bool_).at[jnp.array([0, 1, 2, 3, 4, 24])].set(True)
    assert (mask_scalar == expected_scalar).all()

    # Test 1D array input
    src_array = jnp.array([5, 24], dtype=jnp.int32)
    mask_array = _board_mask_before_src(src_array)
    assert mask_array.shape == (2, 28)

    expected_bar = jnp.zeros(28, dtype=jnp.bool_)
    assert (mask_array[0] == expected_scalar).all()
    assert (mask_array[1] == expected_bar).all()


def test_api():
    import pgx
    env = pgx.make("backgammon")
    pgx.api_test(env, 3, use_key=True)


def test_backmost_checker_pos():
    # 1. Black backmost test
    # Case A: Checker on the bar (BAR_IDX = 24)
    board_a = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_a = board_a.at[24].set(1)
    board_a = board_a.at[5].set(2)
    assert _get_backmost_black_checker_pos(board_a) == -1

    # Case B: No checker on the bar, first checker at index 3
    board_b = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_b = board_b.at[3].set(2)
    board_b = board_b.at[10].set(5)
    assert _get_backmost_black_checker_pos(board_b) == 3

    # Case C: All checkers off the board (or no checkers on the board)
    board_c = jnp.zeros(28, dtype=BOARD_DTYPE)
    assert _get_backmost_black_checker_pos(board_c) == 24

    # 2. White backmost test
    # Case A: Checker on the bar (White bar is index 25)
    board_d = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_d = board_d.at[25].set(-1)
    board_d = board_d.at[15].set(-2)
    assert _get_backmost_white_checker_pos(board_d) == 24

    # Case B: No checker on the bar, backmost checker at index 19
    board_e = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_e = board_e.at[19].set(-2)
    board_e = board_e.at[10].set(-5)
    assert _get_backmost_white_checker_pos(board_e) == 19

    # Case C: All checkers off the board (or no checkers on the board)
    board_f = jnp.zeros(28, dtype=BOARD_DTYPE)
    assert _get_backmost_white_checker_pos(board_f) == -1

    # 3. Batched test (shape (3, 28))
    batched_board = jnp.stack([board_a, board_b, board_c], axis=0)
    assert (_get_backmost_black_checker_pos(batched_board) == jnp.array([-1, 3, 24])).all()

    batched_board_white = jnp.stack([board_d, board_e, board_f], axis=0)
    assert (_get_backmost_white_checker_pos(batched_board_white) == jnp.array([24, 19, -1])).all()


def test_checkers_behind_prime():
    # Setup board
    board = jnp.zeros(28, dtype=BOARD_DTYPE)

    # Black prime: 18, 19, 20 (length 3, starting at 18)
    board = board.at[18].set(2)
    board = board.at[19].set(2)
    board = board.at[20].set(2)

    # White checkers behind Black's prime (indices >= 18 or on bar (25))
    board = board.at[22].set(-2)
    board = board.at[21].set(-1)
    board = board.at[25].set(-1)
    board = board.at[4].set(-1)  # Not behind (4 < 18)

    # White prime: 8, 9 (length 2, ending at 9)
    board = board.at[8].set(-2)
    board = board.at[9].set(-2)

    # Black checkers behind White's prime (indices <= 9 or on bar (24))
    board = board.at[5].set(3)
    board = board.at[7].set(2)
    board = board.at[24].set(2)
    board = board.at[12].set(1)  # Not behind (12 > 9)

    my_prime, opp_prime, my_checkers_behind, opp_checkers_behind = _largest_blocking_prime(board[jnp.newaxis, :])

    assert my_prime[0] == 3
    assert opp_prime[0] == 2
    assert my_checkers_behind[0] == 4  # 2 at 22 + 1 at 21 + 1 on bar
    assert opp_checkers_behind[0] == 7  # 3 at 5 + 2 at 7 + 2 on bar


def test_is_no_contact():
    # Case 1: Standard starting board (definitely contact is possible)
    board_start = jnp.array(START_POSITIONS, dtype=BOARD_DTYPE)
    assert not _is_no_contact(board_start)

    # Case 2: Contact is still possible (e.g. black has checker at 5, white at 8)
    board_contact = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_contact = board_contact.at[5].set(2)
    board_contact = board_contact.at[8].set(-2)
    assert not _is_no_contact(board_contact)

    # Case 3: No contact (black has checker at 12, white at 8. They have crossed paths)
    board_no_contact = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_no_contact = board_no_contact.at[12].set(2)
    board_no_contact = board_no_contact.at[8].set(-2)
    assert _is_no_contact(board_no_contact)

    # Case 4: No contact, with bar checkers
    # If black has a checker on the bar (index 24, pos -1), and white has checkers on the board (e.g., at 0)
    # White's backmost checker is at 0. Black's backmost is -1.
    # -1 > 0 is False. So contact is still possible.
    board_bar_black = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_bar_black = board_bar_black.at[24].set(1)  # black checker on bar
    board_bar_black = board_bar_black.at[0].set(-1)  # white checker at 0
    assert not _is_no_contact(board_bar_black)

    # Case 5: Batched input (shape (3, 28))
    batched_boards = jnp.stack([board_start, board_contact, board_no_contact], axis=0)
    assert (jax.vmap(_is_no_contact)(batched_boards) == jnp.array([False, False, True])).all()


def test_strategy_chunked_vs_standard():
    # Setup some test states
    board_1 = jnp.array(START_POSITIONS, dtype=BOARD_DTYPE)

    board_2 = jnp.zeros(28, dtype=BOARD_DTYPE)
    board_2 = board_2.at[5].set(2)
    board_2 = board_2.at[8].set(-2)
    board_2 = board_2.at[17].set(3)
    board_2 = board_2.at[20].set(-5)

    boards = jnp.stack([board_1, board_2], axis=0)
    dice = jnp.array([[2, 4], [3, 5]], dtype=BOARD_DTYPE)

    # We initialize states
    # Let's map env.init to create a batch of two states, then replace board and playable_dice
    env_instance = Backgammon()
    dummy_states = jax.vmap(env_instance.init)(jax.random.split(jax.random.PRNGKey(42), 2))

    test_states = dummy_states.replace(
        _board=boards,
        _playable_dice=dice
    )

    # evaluators with per-candidate weights need one config row per game
    evaluator_config = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), boards.shape[0], axis=0),
        SimpleBackgammonEvaluatorConfig(),
    )

    # Evaluate using standard 2-ply strategy
    strat_std = BackgammonTwoPlyStrategy(env_instance)
    best_action_std, equities_std, candidate_action_indices_std = strat_std.get_next_action_and_equities_batch(
        test_states, jax.random.PRNGKey(0), evaluator_config, SimpleBackgammonEvaluator
    )

    # Evaluate using chunked 2-ply strategy
    strat_chk = BackgammonTwoPlyChunkedStrategy(env_instance)
    best_action_chk, equities_chk, candidate_action_indices_chk = strat_chk.get_next_action_and_equities_batch(
        test_states, jax.random.PRNGKey(0), evaluator_config, SimpleBackgammonEvaluator
    )

    # Assertions
    assert (best_action_std == best_action_chk).all(), f"Actions mismatch: {best_action_std} vs {best_action_chk}"
    assert jnp.allclose(equities_std, equities_chk, atol=1e-5, equal_nan=True), f"Equities mismatch: {equities_std} vs {equities_chk}"
    assert (candidate_action_indices_std == candidate_action_indices_chk).all()



def test_strategy_action_at_chance_node():
    """At a chance node (turn changed, dice not rolled) the strategy must return
    a legal dice-roll action, sampled with the true dice odds, not a move."""
    B = 4
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), B))
    # force pre-roll chance nodes by ending the current turn
    chance_states = jax.vmap(_change_turn)(states, jax.random.split(jax.random.PRNGKey(1), B))
    assert chance_states.has_chance_logits(chance_states.get_chance_logits()).all()

    evaluator_config = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), B, axis=0), SimpleBackgammonEvaluatorConfig()
    )
    for strategy in [BackgammonTwoPlyStrategy(env), BackgammonTwoPlyChunkedStrategy(env)]:
        action = strategy.get_next_action_batch(
            chance_states, jax.random.PRNGKey(2), evaluator_config, SimpleBackgammonEvaluator
        )
        assert (action >= ACTION_MOVE_LENGTH).all(), f"not a dice-roll action: {action}"
        assert chance_states.legal_action_mask[jnp.arange(B), action].all()

        # stepping with the roll keeps the same player and sets the dice
        next_states = jax.vmap(env.step)(
            chance_states, action, jax.random.split(jax.random.PRNGKey(3), B)
        )
        assert (next_states.current_player == chance_states.current_player).all()
        assert (next_states._playable_dice != NO_MOVE).any(axis=-1).all()


def test_strategy_action_at_no_move_node():
    """At a no-move node the strategy must pass (NOOP) instead of returning an
    illegal candidate move."""
    B = 2
    board = jnp.zeros(28, dtype=BOARD_DTYPE)
    board = board.at[BAR_IDX].set(15)  # all black checkers on the bar
    for pos, cnt in zip(range(0, 6), [3, 3, 3, 2, 2, 2]):
        board = board.at[pos].set(-cnt)  # every entry point blocked by white

    playable_dice = jnp.stack([jnp.array([2, 4, NO_MOVE, NO_MOVE])] * 2)
    legal = jax.vmap(_arr_legal_action_mask)(jnp.stack([board, board]), playable_dice)
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), B)).replace(
        _board=jnp.stack([board, board]),
        _playable_dice=playable_dice,
        legal_action_mask=legal,
    )
    # every dice entry is blocked, so the only legal action is the NOOP pass
    assert (legal.sum(axis=-1) == 1).all() and legal[:, NOOP_ACTION_IDX].all()

    evaluator_config = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), B, axis=0), SimpleBackgammonEvaluatorConfig()
    )
    for strategy in [BackgammonTwoPlyStrategy(env), BackgammonTwoPlyChunkedStrategy(env)]:
        action = strategy.get_next_action_batch(
            states, jax.random.PRNGKey(2), evaluator_config, SimpleBackgammonEvaluator
        )
        assert (action == NOOP_ACTION_IDX).all(), f"expected NOOP, got {action}"
        # stepping with the NOOP passes the turn without terminating
        next_states = jax.vmap(env.step)(
            states, action, jax.random.split(jax.random.PRNGKey(3), B)
        )
        assert (~next_states.terminated).all()
        assert (next_states.current_player == 1 - states.current_player).all()


def test_get_normal_or_chance_logits_batched():
    """get_normal_or_chance_logits must work on batched states directly (no
    vmap): chance rows get the chance logits, other rows keep the given logits."""
    B = 4
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), B))
    chance_states = jax.vmap(_change_turn)(states, jax.random.split(jax.random.PRNGKey(1), B))
    # batch with two chance nodes and two normal nodes
    mixed_states = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate([a[:2], b[:2]], axis=0), chance_states, states
    )

    logits = jnp.arange(B * ACTION_TOTAL_LENGTH, dtype=jnp.float32).reshape(B, ACTION_TOTAL_LENGTH)
    out = mixed_states.get_normal_or_chance_logits(logits, mixed_states.get_chance_logits())

    chance_logits = mixed_states.get_chance_logits()
    expected = jnp.where(mixed_states.has_chance_logits(chance_logits)[:, jnp.newaxis],
                         chance_logits, logits)
    assert jnp.array_equal(out, expected)
    # chance rows carry finite dice logits only on the chance actions
    assert (~jnp.isneginf(out[0, ACTION_MOVE_LENGTH:])).any()
    assert jnp.isneginf(out[0, :ACTION_MOVE_LENGTH]).all()
    # normal rows pass the given logits through untouched
    assert jnp.array_equal(out[2:], logits[2:])
    # recalc variant on a batched state works too
    out2 = mixed_states.get_normal_or_chance_logits_recalc(logits)
    assert jnp.array_equal(out2, expected)
    # and the unbatched case still works
    single = jax.tree_util.tree_map(lambda x: x[2], mixed_states)
    out3 = single.get_normal_or_chance_logits(logits[2], single.get_chance_logits())
    assert jnp.array_equal(out3, logits[2])
