import numbers
from functools import partial
from typing import Any

import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten


def convert_to_tensor(data: Any, tensor_type: str) -> Any:
    """Convert a tensor in a given tensor_type.

    Parameters
    ----------
    tensor_type
        The targeted tensor type. Can be in ['numpy', 'tensorflow', 'jax', 'torch"].

    Returns
    -------
    data
        data structure with converted tensors.
    """
    if tensor_type == "tensorflow":
        import tensorflow as tf

        if isinstance(data, np.datetime64):
            # datetime not managed by tensorflow
            return data

        return tf.convert_to_tensor(data)

    elif tensor_type == "torch":
        import torch

        if isinstance(data, np.datetime64):
            # datetime not managed by pytorch
            return data
        elif isinstance(data, np.ndarray) and data.dtype.type is np.str_:
            return data
        elif isinstance(data, numbers.Number):
            return data
        new_data = torch.tensor(data)
        return new_data

    elif tensor_type == "jax":
        import jax.numpy as jnp

        if isinstance(data, np.datetime64):
            # datetime not managed by jax
            return data
        elif isinstance(data, np.ndarray) and data.dtype.type is np.str_:
            return data
        new_data = jnp.asarray(data, dtype=data.dtype)
        return new_data

    elif tensor_type == "numpy":
        return np.asarray(data)

    raise ValueError(
        f"tensor_type {tensor_type} must be in ['numpy', 'tensorflow', 'jax', 'torch']"
    )


def convert_to_tensors(data: Any, tensor_type: str) -> Any:
    """Convert tensors in a nested data structure .

    Parameters
    ----------
    tensor_type
        The targeted tensor type. Can be in ['numpy', 'tensorflow', 'jax', 'torch"].

    Returns
    -------
    data
        data structure with converted tensors.
    """
    leaf_values, tree_def = tree_flatten(data)
    leaf_values = list(
        map(partial(convert_to_tensor, tensor_type=tensor_type), leaf_values)
    )
    return tree_unflatten(tree_def, leaf_values)
