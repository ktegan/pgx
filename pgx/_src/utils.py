import sys
from urllib.request import urlopen
from pgx._src.types import Array
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp


@jax.jit(static_argnames=('func', 'chunk_size', 'func_uses_is_selected'))
def chunked_map(func:Callable,
                data:Array,
                selection:Array,
                func_args:Optional[List[Array]]=None,
                func_kwargs:Optional[Dict[str,Array]]=None,
                chunk_size=64,
                func_uses_is_selected=False):
    """
    This computes idx and func(data[idx]) for every idx where selection[idx] != 0.
    This assumes that selection and data have the same in their first dimension.
    This is useful when func() is a heavy computation.

    Args:
        func (Callable): a function that either takes a row of data (eg data[idx])
            as the only input or (data_row, is_selected) as input when the
            func_uses_is_selected parameter is set to True.  When is_selected is False
            that means data_row is not a selected input and should not be trusted.
            The output should be a jax numpy Array or convertible to a PyTree.
        data (Array): an array (or tuple of arrays, or any jax PyTree) with first
            dimension of size N that will be used as input to func.
        selection (Array): a 1D array of size N that has non-zero elements for
            any data[idx] values we want to pass into func.
        func_args (Optional[List[Array]]): a list of array *args that can be
            of any size and will be passed as is to each call to func().
        func_kwargs (Optional[Dict[str,Array]]): a dict of array **kwargs that can be
            of any size and will be passed as is to each call to func().
        chunk_size (int): this function will call func using vmap across chunks
            of data of size chunk_size.  Larger chunks can be handled better by
            TPUs but will also lead to more func() calls on unselected elements.
        func_uses_is_selected (bool): if True, `func` will receive a second `is_selected`
            argument (a boolean mask for the chunk).

        Returns:
            (Array, Array): returns a pair of arrays, the first array
            contains idx values and the second array contains func(data[idx])
            values.  Both arrays have a first dimension of size N.  All idx values
            less than N correspond to calls to func(data[idx]).  These values will
            be followed by idx values equal to N that signify unselected elements.
            Returning N instead of -1 is convenient since you can write
            prior_results.at[indices].set(results) and only overwrite indices that
            were in the selection array (all indices that are N or greater  will
            be ignored by jax).
    """
    # compute padding necessary for us to processes chunks of chunk_size
    N                         = selection.shape[0]
    remainder                 = N % chunk_size
    padding_needed            = (chunk_size - remainder) % chunk_size
    padded_data_len           = N + padding_needed

    # pad the selection input (simple 1D array)
    padded_selection          = jnp.pad(selection, pad_width=(0, padding_needed), mode='constant', constant_values=0)
    num_chunks                = padded_data_len // chunk_size
    selection_indices,        = jnp.nonzero(padded_selection, size=padded_data_len, fill_value=N)
    chunked_selection_indices = selection_indices.reshape(num_chunks, chunk_size)

    # because data could be an Array, tuple or PyTree we use tree_map to chunk underlying Arrays in input data
    def chunk_data(data_src):
        padded_data       = jnp.pad(data_src, pad_width=((0, padding_needed),) + ((0, 0),) * (data_src.ndim - 1), mode='constant', constant_values=0)
        gathered_data     = padded_data[jnp.where(selection_indices == N, 0, selection_indices)]
        chunked_data      = gathered_data.reshape(num_chunks, chunk_size, *data_src.shape[1:])
        return chunked_data

    chunked_data = jax.tree_util.tree_map(chunk_data, data)

    if func_args is None:
        func_args = []
    if func_kwargs is None:
        func_kwargs = {}

    # jax.vmap() does not want kwargs included in split_axes and then assumes they
    # shoudl be split along axis zero.  Include kwargs using wrapped_func instead.
    def wrapped_func(cur_data, *wrapped_args):
        return func(cur_data, *wrapped_args, **func_kwargs)

    split_axes = (0,) * (2 if func_uses_is_selected else 1) + (None,) * len(func_args)
    vmap_func = jax.vmap(jax.jit(wrapped_func), in_axes=split_axes)

    # use jax.eval_shape() to figure out what the output Array or PyTree will look like
    def create_zero_in_data_chunk(data_src):
        return jnp.zeros((chunk_size, *data_src.shape[1:]), dtype=data_src.dtype)
    zero_in_data_chunk = jax.tree_util.tree_map(create_zero_in_data_chunk, data)

    if func_uses_is_selected:
        sample_out_chunk = jax.eval_shape(vmap_func, zero_in_data_chunk, jnp.zeros((chunk_size,), dtype=jnp.bool_), *func_args)
    else:
        sample_out_chunk = jax.eval_shape(vmap_func, zero_in_data_chunk, *func_args)

    # create zero_out_chunk that has the correct output shape but contains all zero data
    def create_zero_out_chunk(out_src):
        return jnp.zeros(out_src.shape, dtype=out_src.dtype)
    zero_out_chunk = jax.tree_util.tree_map(create_zero_out_chunk, sample_out_chunk)

    def process_chunk(xs):
        cur_indices, cur_data = xs
        is_selected           = cur_indices != N
        any_selected          = is_selected.any()
        if func_uses_is_selected:
            process_lambda    = lambda: vmap_func(cur_data, is_selected, *func_args)
        else:
            process_lambda    = lambda: vmap_func(cur_data, *func_args)
        return jax.lax.cond(any_selected, process_lambda, lambda: zero_out_chunk)

    # Call jax.lax.map() to iterate through chunks and call a vmap'ed version of
    # func to process all rows within a chunk in parallel.
    #
    # If instead of map() we used vmap() below that would make this chunking
    # useless.  If we used vmap() the jax.lax.cond() condition in process_chunk()
    # would have no value because both branches would have to be evaluated.
    # All threads have to stay in lock step when using vmap() (hardware branch
    # divergence), so if we want to short-circuit an entire chunk of unselected
    # data we should use jax.lax.map().
    results = jax.lax.map(process_chunk, (chunked_selection_indices, chunked_data))

    # reshape the output so that it is no longer in chunks and trim to the original length
    def reshape_chunked_results(out_src):
        return out_src.reshape(padded_data_len, *out_src.shape[2:])
    results = jax.tree_util.tree_map(reshape_chunked_results, results)

    def trim_to_size_N(data_src):
        return data_src[:N]

    return trim_to_size_N(selection_indices), jax.tree_util.tree_map(trim_to_size_N, results)




def _download(url, filename):
    try:
        print(f"Downloading from {url} ...", file=sys.stderr)
        data = urlopen(url).read()
        with open(filename, mode="wb") as f:
            f.write(data)
    except Exception as e:
        print(f"Failed to downalod the data from {url}", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
