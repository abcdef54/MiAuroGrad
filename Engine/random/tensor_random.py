from ..tensor import Tensor
import numpy as np
from numpy.typing import DTypeLike
from typing import List, Tuple


def uniform(shape: Tuple[int, ...] | int, 
            low: float = -1.0, 
            high: float = 1.0, 
            dtype: DTypeLike = np.float32,
            requires_grad: bool = False) -> Tensor:
    """
    Generate a tensor with values uniformly distributed between low and high.
    
    Args:
        shape: Shape of the output tensor
        low: Lower bound (inclusive)
        high: Upper bound (exclusive)
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients
    
    Returns:
        Tensor with uniformly distributed random values
    """

    if isinstance(shape, int):
        shape = (shape, )

    data = np.random.uniform(low, high, shape).astype(dtype) #type:ignore
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)


def randn(*shape: int,
          dtype: DTypeLike = np.float32,
          requires_grad: bool = False) -> Tensor:
    """Generate a tensor with standard normal distribution (PyTorch-style API)."""
    data = np.random.randn(*shape).astype(dtype) #type:ignore
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)


def rand(*shape: int,
         dtype: DTypeLike = np.float32, #type:ignore
         requires_grad: bool = False) -> Tensor:
    """Generate a tensor with uniform distribution [0, 1) (PyTorch-style API)."""
    data = np.random.rand(*shape).astype(dtype) #type:ignore
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)
