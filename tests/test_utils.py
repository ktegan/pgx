import jax
import jax.numpy as jnp

from pgx._src.types import Array
from pgx._src.utils import chunked_map



def test_chunked_map_shapes():
    """
    At a basic level this is creating a multi-dimensional input array that has 10
    consecutive integers at each row and the data runs from -70 to 929.
    The function that we apply checks to see that the input shapes and data
    match what it expects and then calls sum() on the inputs (the test_func_is_selected()
    function handles a row with missing data slightly differently).  To try to
    really check all of the corner cases we use different multi-dimensional shapes,
    pass in tuples instead of simple arrays and add additional args and kwargs.
    All of those extra complications are verified but do not affect the core
    processing of the test.

    The chunked_map() function handles CHUNK_SIZE rows at a time and we check that the output
    matches what we expect.
    """
    IN_SHAPE    = (5, 2)
    OUT_SHAPE   = (3, 4, 3)
    CHUNK_SIZE  = 32
    N           = 100
    data        = jnp.arange(1000, dtype=jnp.int32).reshape((N,) + IN_SHAPE) - 70
    selection   = ~(((data < 0) | (data > 305)).any(axis=tuple(1 + x for x in range(len(IN_SHAPE)))))
    BAD_INPUT   = jnp.int32(-99)
    BAD_SHAPE   = jnp.int32(-98)
    BAD_ARGS    = jnp.int32(-96)

    EMPTY_INPUT = jnp.int32(-51)
    MISC_OUT    = jnp.int32(-52)

    ARGS1_ARR   = jnp.arange(4).reshape((2, 2))
    ARGS2_ARR   = jnp.arange(9).reshape((3, 3))
    KWARGS1_ARR = jnp.arange(25).reshape((5, 5))
    KWARGS3_ARR = jnp.arange(49).reshape((7, 7))

    assert (data.shape[0] % CHUNK_SIZE) != 0   # see how we handle case where we have to pad

    def args_good(args1:Array, args2:Array, kwargs1:Array, kwargs3:Array):
        return jnp.logical_and.reduce(jnp.array([
            (args1 == ARGS1_ARR).all(),
            (args2 == ARGS2_ARR).all(),
            (kwargs1 == KWARGS1_ARR).all(),
            (kwargs3 == KWARGS3_ARR).all()]))

    def test_func_simple(input_pair,
                         args1,
                         args2=jnp.zeros_like(ARGS1_ARR),
                         kwargs1=jnp.zeros_like(KWARGS1_ARR),
                         kwargs3=jnp.zeros_like(KWARGS3_ARR)):
        input_row, misc_row = input_pair
        if input_row.shape != IN_SHAPE or misc_row.shape != tuple():
            return BAD_SHAPE
        args_match     = args_good(args1, args2, kwargs1, kwargs3)
        out = jax.lax.cond(args_match,
                           lambda: jnp.sum(input_row),
                           lambda: BAD_ARGS)
        return (jnp.full(OUT_SHAPE, out), MISC_OUT)

    def test_func_is_selected(input_pair,
                              is_selected,
                              args1,
                              args2=jnp.zeros_like(ARGS1_ARR),
                              kwargs1=jnp.zeros_like(KWARGS1_ARR),
                              kwargs3=jnp.zeros_like(KWARGS3_ARR)):
        input_row, misc_row = input_pair
        if input_row.shape != IN_SHAPE or misc_row.shape != tuple():
            return BAD_SHAPE
        args_match     = args_good(args1, args2, kwargs1, kwargs3)
        selected_match = ((input_row < 0) | (input_row > 305)).any() != is_selected

        out = jax.lax.cond(selected_match & args_match,
                           lambda: jax.lax.cond(is_selected,
                                                lambda: jnp.sum(input_row),
                                                lambda: EMPTY_INPUT),
                           lambda: jax.lax.cond(~args_match,
                                                lambda: BAD_ARGS,
                                                lambda: BAD_INPUT))
        return (jnp.full(OUT_SHAPE, out), MISC_OUT)


    for pass_is_selected in [False, True]:
        cur_func = test_func_is_selected if pass_is_selected else test_func_simple
        data_pair = (data, selection) # just to make things more complicated pass selection as well (it is captured as misc_row in test functions)
        indices, orig_result = chunked_map(cur_func, data_pair, selection, chunk_size=CHUNK_SIZE, func_uses_is_selected=pass_is_selected,
                                           func_args=[ARGS1_ARR, ARGS2_ARR], func_kwargs={ 'kwargs1': KWARGS1_ARR, 'kwargs3': KWARGS3_ARR})

        result, misc_out = orig_result
        assert not (result == BAD_INPUT).any()
        assert not (result == BAD_SHAPE).any()
        assert not (result == BAD_ARGS).any()
        assert indices.shape == (data.shape[0],)
        assert result.shape == (data.shape[0],) + OUT_SHAPE
        assert misc_out.shape == (data.shape[0],)
        assert ((misc_out == MISC_OUT) | (misc_out == 0)).all()

        assert (indices != N).sum() == jnp.int32(30)  # 30 selected input rows of 10 elements that go from 0 to 299
        assert (indices == jnp.concat([jnp.arange(7, 37, dtype=jnp.int32), jnp.full(100 - 30, N, dtype=jnp.int32)])).all()

        found_zero  = False
        found_empty = False
        found_first = False
        for out_idx, data_idx in enumerate(indices):
            result_scalar   = result[(out_idx,) + (0,) * len(OUT_SHAPE)]
            misc_out_scalar = misc_out[out_idx]
            assert (result_scalar == result[out_idx]).all()  # we fill result with the same value
            if data_idx == N:
                cur_zero   = (result_scalar == 0)  # if an entire chunk has unselected input chunked_map() returns zero
                found_zero = found_zero  or cur_zero
                if pass_is_selected:
                    cur_empty   = (result_scalar == EMPTY_INPUT)
                    found_empty = found_empty or cur_empty
                    assert cur_zero or cur_empty
                else:
                    cur_first   = (result_scalar == data[0].sum())  # unselected input rows within a chunk are mapped to use the input at index 0 in chunked_map()
                    found_first = found_first or cur_first
                    assert cur_zero or cur_first
                
                assert misc_out_scalar == (0 if cur_zero else MISC_OUT)
            else:
                # for selected input data the output should be the sum of input
                assert result_scalar == data[data_idx].sum()
                assert misc_out_scalar == MISC_OUT

        # we should see examples of different behavior for unselected input depending on pass_is_selected
        assert found_zero
        if pass_is_selected:
            assert found_empty
        else:
            assert found_first



def test_chunked_map_selection():
    """ Call chunked_map() with an alternating set of selection values. """
    EMPTY_INPUT = jnp.int32(-51)
    CHUNK_SIZE  = 32
    ARGS1_VAL   = jnp.int32(17)

    def test_func_is_selected(input_row, is_selected, args1):
        return jax.lax.cond(is_selected, lambda: input_row.sum() + args1, lambda: EMPTY_INPUT)

    N = 100
    data = jnp.arange(1000, dtype=jnp.int32).reshape((N, 10))
    selection = ((data[:, 0] // 20) % 2) == 0    # True for 0 and 10, False for 20 and 30, True for 40 and 50, etc
    indices, result = chunked_map(test_func_is_selected, data, selection, chunk_size=CHUNK_SIZE, func_uses_is_selected=True, func_args=[ARGS1_VAL])

    sum_10_plus_args1 = jnp.arange(10).sum() + ARGS1_VAL

    assert ((result[:50] % 100) == sum_10_plus_args1).all()
    assert ((result[50:] == 0) | (result[50:] == EMPTY_INPUT)).all()
    assert (((indices[:50] // 2) % 2) == 0).all()
    assert (indices[50:] == N).all()

    assert (indices[0] == 0)
    assert (indices[1] == 1)
    assert (indices[2] == 4)
    assert (indices[3] == 5)
    assert (indices[48] == 96)
    assert (indices[49] == 97)
    
    assert (result[0]  == sum_10_plus_args1 + 0)
    assert (result[1]  == sum_10_plus_args1 + 100)
    assert (result[2]  == sum_10_plus_args1 + 400)
    assert (result[3]  == sum_10_plus_args1 + 500)
    assert (result[48] == sum_10_plus_args1 + 9600)
    assert (result[49] == sum_10_plus_args1 + 9700)
