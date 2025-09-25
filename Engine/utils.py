import numpy as np
from typing import Tuple

def _unbroadcast(grad: np.ndarray, original_shape: Tuple[int, ...]):
        """
        Un-broadcast gradient to match original tensor shape
        """

        ndim_added = grad.ndim - len(original_shape)

        # collapse the added dimensions
        for i in range(ndim_added):
            grad = grad.sum(axis=0)

        for i, (dim, ori_dim) in enumerate(zip(grad.shape, original_shape)):
            if ori_dim == 1 and dim > 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad