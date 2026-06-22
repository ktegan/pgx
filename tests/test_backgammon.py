from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pgx.experimental.utils import act_randomly
from pgx._src.types import Array

from pgx.backgammon import (
    ONE_MOVE_BOARD_DIFFS,
    ONE_MOVE_SRC,
    ONE_MOVE_TGT,
    ONE_MOVE_DIE,
    BOARD_DTYPE,
    State,
    Backgammon,
    _decompose_action,
    _flip_board,
    _action_to_src,
    _calc_tgt,
    _calc_win_score,
    _change_turn,
    _is_action_legal,
    _is_all_on_home_board,
    _arr_apply_diff,
    _arr_one_and_two_moves,
    _is_open,
    _move,
    _rear_distance,
    _roll_init_dice,
    _distance_to_goal,
    _is_turn_end,
    _no_winning_step,
    _exists,
    _set_playable_dice,
    _board_mask_before_src,
    _arr_legal_action_mask,
)

seed = 1701
rng = jax.random.PRNGKey(seed)
env = Backgammon()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)
_decompose_action = jax.jit(_decompose_action)
_no_winning_step = jax.jit(_no_winning_step)
_action_to_src = jax.jit(_action_to_src)
_calc_tgt = jax.jit(_calc_tgt)
_calc_win_score = jax.jit(_calc_win_score)
_change_turn = jax.jit(_change_turn)
_is_action_legal = jax.jit(_is_action_legal)
_is_all_on_home_board = jax.jit(_is_all_on_home_board)
_arr_apply_diff = jax.jit(_arr_apply_diff)
_arr_one_and_two_moves = jax.jit(_arr_one_and_two_moves)
_is_open = jax.jit(_is_open)
_move = jax.jit(_move)
_rear_distance = jax.jit(_rear_distance)
_exists = jax.jit(_exists)
_set_playable_dice = jax.jit(_set_playable_dice)
_board_mask_before_src = jax.jit(_board_mask_before_src)
_arr_legal_action_mask = jax.jit(_arr_legal_action_mask)


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
               -1         +        -
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
    board: jnp.ndarray = make_test_board()

    # current_player = white, playable_dice = (1, 2)
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([1, 1, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state) == expected_obs).all()

    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([1, 1, 1, 1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([0, 4, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state) == expected_obs).all()

    # current_player = black, playabl_dice = (2)
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(-1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([0, 1, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state) == expected_obs).all()


def test_is_open():
    board = make_test_board()
    # Black
    assert _is_open(board, 9)
    assert _is_open(board, 19)
    assert _is_open(board, 4)
    assert not _is_open(board, 10)
    # White
    board = _flip_board(board)
    assert _is_open(board, 9)
    assert _is_open(board, 8)
    assert not _is_open(board, 2)
    assert not _is_open(board, 4)


def test_exists():
    board = make_test_board()
    # Black
    assert _exists(board, 19)
    assert _exists(board, 20)
    assert not _exists(board, 4)
    # White
    board = _flip_board(board)
    assert _exists(board, 19)
    assert _exists(board, 20)
    assert not _exists(board, 2)


def test_is_all_on_home_board():
    board: jnp.ndarray = make_test_board()
    # Black
    assert _is_all_on_home_board(board)
    # White
    board = _flip_board(board)
    assert not _is_all_on_home_board(board)


def test_rear_distance():
    board = make_test_board()
    turn = jnp.int32(-1)
    # Black
    assert _rear_distance(board) == 5
    # White
    board = _flip_board(board)
    assert _rear_distance(board) == 23


def test_distance_to_goal():
    board = make_test_board()
    # Black
    src = 23
    assert _distance_to_goal(src) == 1
    src = 10
    assert _distance_to_goal(src) == 14
    # Teat at the src where rear_distance is same
    assert _rear_distance(board) == _distance_to_goal(19)


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
    assert _is_action_legal(board, (19 + 2) * 6 + 1)  # 19->21
    assert not _is_action_legal(board, (19 + 2) * 6 + 2)  # 19 -> 22
    assert not _is_action_legal(
        board, (19 + 2) * 6 + 2
    )  # 19 -> 22: Some whites on 22
    assert not _is_action_legal(
        board, (22 + 2) * 6 + 2
    )  # 22 -> 25: No black on 22 
    assert _is_action_legal(board, (19 + 2) * 6 + 5)  # bear off
    assert not _is_action_legal(
        board, (20 + 2) * 6 + 5
    )  # cannot bear off as some blacks behind
    # white
    board = _flip_board(board)
    assert not _is_action_legal(
        board, (20 + 2) * 6 + 0
    )  # 20->21(after flipped): cannot move checkers as some left on bar
    assert _is_action_legal(board, (1) * 6 + 0)  # bar -> 1(after flipped)
    assert not _is_action_legal(board, (1) * 6 + 2)  # bar -> 2(after flipped)


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


def test_arr_apply_diff():

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

        is_legal, new_board = _arr_apply_diff(cur_test.board, cur_diff, cur_test.src, cur_test.die, cur_test.tgt)
        assert (is_legal == cur_test.is_legal).all(), cur_test.description
        assert (new_board == expected_board).all(), cur_test.description


def test_arr_one_and_two_moves():
    orig_board = jnp.array([
        #  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11,
          -1, -1,  0,  2, -2,  0,   -1,  0,  0,  0,  0,  0,
        # 12, 13, 14, 15, 16, 17,   18, 19, 20, 21, 22, 23,   24, 25,   26, 27
           0,  0,  0,  0,  0,  0,    0,  0,  0,  2,  0, -2,    0,  0,   11, -8
    ], dtype=BOARD_DTYPE)

    one_move_legal, one_move_boards, two_move_legal, two_move_boards = \
        _arr_one_and_two_moves(orig_board)

    assert one_move_legal.shape == (156,)
    assert one_move_boards.shape == (156, 28)
    assert two_move_legal.shape == (156, 156)
    assert two_move_boards.shape == (156, 156, 28)

    # Let's manually verify a few combinations
    # For a couple of index pairs (i, j), verify that:
    # two_move_boards[i, j] matches applying the i-th and j-th diff sequentially
    expected_one_move_legal, expected_one_move_boards = _arr_apply_diff(orig_board, ONE_MOVE_BOARD_DIFFS, ONE_MOVE_SRC, ONE_MOVE_DIE, ONE_MOVE_TGT)
    test_actions = list(range((3 + 2) * 6 - 2, (3 + 2) * 6 + 2)) + [(5 + 2) * 6 + 4, (21 + 2) * 6 + 4]
    for i in test_actions:
        for j in test_actions:
            expected_two_move_legal, expected_two_move_board = _arr_apply_diff(expected_one_move_boards[i], ONE_MOVE_BOARD_DIFFS[j], ONE_MOVE_SRC[j], ONE_MOVE_DIE[j], ONE_MOVE_TGT[j])
            assert (two_move_legal[i, j] == expected_two_move_legal).all()
            assert (two_move_boards[i, j] == expected_two_move_board).all()

    # if we roll a 3 and move one checker from index 3 we hit a blot,
    # NOTE this simple function does not update the bar counter when we hit
    hit_blot = orig_board.at[3].set(1)
    hit_blot = hit_blot.at[6].set(1)
    # hit_blot = hit_blot.at[BAR_IDX + 1].set(-1)   # this simple function doesn't do this
    assert (one_move_boards[(3 + 2) * 6 + 2, :] == hit_blot).all()


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


def test_forced_moves():
    board_answer_pairs = [
        (make_test_board_use_2_moves_simple(),                 make_answer_use_2_moves_simple()),
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

    for _test_num, (test_board, (dice, answer)) in enumerate(board_answer_pairs):
        expected_boards = answer[:,:28]
        move_nums = answer[:,28]
        max_move = jnp.max(move_nums)

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

            if move_nums[answer_idx] == move_num:
                # validate that the current board matches all game boards
                cur_board = expected_boards[answer_idx]
                if move_num >= max_move + steps_after_max_move - 1:
                    cur_board = _flip_board(cur_board)
                assert (s._board == cur_board[None,:]).all()
                answer_idx += 1


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


def test_api():
    import pgx
    env = pgx.make("backgammon")
    pgx.api_test(env, 3, use_key=True)
