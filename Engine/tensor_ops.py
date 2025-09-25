from typing import List, Tuple, Iterable
from .tensor import Tensor
from .utils import _unbroadcast
import numpy as np
from numpy.typing import DTypeLike


def where(condition, then, otherwise) -> Tensor:
    then = then if isinstance(then, Tensor) else Tensor(then)
    otherwise = otherwise if isinstance(otherwise, Tensor) else Tensor(otherwise)
    
    filtered = np.where(condition, then.data, otherwise.data)
    
    requires_grad = then.requires_grad or otherwise.requires_grad

    out = Tensor(filtered,
                 dtype=then.dtype,
                 requires_grad=requires_grad,
                 _children={then, otherwise},
                 _op='where')
    
   # "Then" get the gradient from the indices where the condition is True
   # "Otherwise" get the gradient from the indices where the condition is False

    def _backward():
        if out.grad is None:
            return
        
        if then.requires_grad:
            then.grad += np.where(condition, out.grad, 0)
        
        if otherwise.requires_grad:
            otherwise.grad += np.where(condition, 0, out.grad)
    
    out._backward = _backward
    return out
    

def concat(tensors: List[Tensor] | Tuple[Tensor], axis=0) -> Tensor:
    if isinstance(tensors, tuple):
        tensors = list(tensors)
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    
    out_data = np.concatenate([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(out_data,
                dtype=tensors[0].dtype,
                requires_grad=requires_grad,
                _children=set(tensors),
                _op='concat')
    
    def _backward():
        if out.grad is None:
            return
        
        # compute the size of each tensor along the concatenation axis
        sizes = [t.data.shape[axis] for t in tensors]
        # split the upstream gradient into pieces
        grads = np.split(out.grad, np.cumsum(sizes[:-1]), axis=axis)

        for tensor, grad in zip(tensors, grads):
            if tensor.requires_grad:
                tensor.grad = (tensor.grad + grad) if tensor.grad is not None else grad
    
    out._backward = _backward
    return out


def stack(tensors: List[Tensor] | Tuple[Tensor], axis=0) -> Tensor:
    if isinstance(tensors, tuple):
        tensors = list(tensors)
    
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]

    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(np.stack([t.data for t in tensors], axis=axis),
                 dtype=tensors[0].dtype,
                 requires_grad=requires_grad,
                 _children=set(tensors),
                 _op='stack')
    
    def _backward():
        if out.grad is None:
            return
        
        # the stack op does change the values inside a tensor so the grad
        # so just unstack (index) the tensor to take the gradients

        for i, tensor in enumerate(tensors):
            grad = np.take(out.grad, i, axis=axis)
            tensor.grad = grad if tensor.grad is None else (tensor.grad + grad)
    
    out._backward = _backward
    return out


def sum(tensors: List[Tensor] | Tuple[Tensor], axis=None, keepdims=False) -> Tensor:
    if isinstance(tensors, tuple):
        tensors = list(tensors)
    
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]

    out_data = np.sum([t.data for t in tensors], axis=0) if axis is None else \
               np.sum(np.stack([t.data for t in tensors], axis=axis), axis=axis, keepdims=keepdims)

    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(out_data,
                 dtype=tensors[0].dtype,
                 requires_grad=requires_grad,
                 _children=set(tensors),
                 _op='sum')

    def _backward():
        if out.grad is None:
            return
        for t in tensors:
            if t.requires_grad:
                if axis is None:
                    grad = np.ones_like(t.data) * out.grad
                else:
                    grad = out.grad
                    if not keepdims:
                        grad = np.expand_dims(grad, axis)
                    grad = np.broadcast_to(grad, t.data.shape)
                t.grad = grad if t.grad is None else t.grad + grad

    out._backward = _backward    
    return out


def product(tensors: List[Tensor] | Tuple[Tensor], axis=None, keepdims=False) -> Tensor:
    if isinstance(tensors, tuple):
        tensors = list(tensors)
    
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]

    out_data = np.prod([t.data for t in tensors], axis=0) if axis is None else \
               np.prod(np.stack([t.data for t in tensors], axis=axis), axis=axis, keepdims=keepdims)

    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(out_data,
                 dtype=tensors[0].dtype,
                 requires_grad=requires_grad,
                 _children=set(tensors),
                 _op='product')

    def _backward():
        if out.grad is None:
            return
        
        total = np.prod([t.data for t in tensors], axis=0)
        for t in tensors:
            if t.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)
                grad = np.broadcast_to(grad, t.data.shape)

                contrib = np.where(t.data != 0, total / t.data, 0.0) * grad
                t.grad = contrib if t.grad is None else t.grad + contrib

    out._backward = _backward
    return out


def broadcast_to(tensor: Tensor,  shape: int | Iterable[int]) -> Tensor:
    assert isinstance(tensor, Tensor)

    out = np.broadcast_to(tensor.data, shape)
    out = Tensor(out,
                 dtype=tensor.dtype,
                 requires_grad=tensor.requires_grad,
                 _children={tensor},
                 _op='broadcast_to')
    
    def _backward():
        if out.grad is None or not tensor.requires_grad:
            return
        
        grad = out.grad
        tensor.grad += _unbroadcast(grad, tensor.data.shape)
        
    out._backward = _backward
    return out

def copy(tensor_to_copy: Tensor) -> Tensor:
    assert isinstance(tensor_to_copy, Tensor)
    return tensor_to_copy.copy()

def zeros_like(tensor: Tensor, dtype: DTypeLike = None, requires_grad: bool = False) -> Tensor:
    assert isinstance(tensor, Tensor)

    if dtype is None:
        dtype = tensor.dtype
    
    out = np.zeros_like(tensor.data, dtype=dtype)
    return Tensor(out, dtype=dtype, requires_grad=requires_grad)

def ones_like(tensor: Tensor, dtype: DTypeLike = None, requires_grad: bool = False) -> Tensor:
    assert isinstance(tensor, Tensor)

    if dtype is None:
        dtype = tensor.dtype
    
    out = np.ones_like(tensor.data, dtype=dtype)
    return Tensor(out, dtype=dtype, requires_grad=requires_grad)

def full_like(tensor: Tensor, fill_value: float, dtype: DTypeLike = None, requires_grad: bool = False) -> Tensor:
    assert isinstance(tensor, Tensor)

    if dtype is None:
        dtype = tensor.dtype
    
    out = np.full_like(tensor.data, fill_value, dtype=dtype)
    return Tensor(out, dtype=dtype, requires_grad=requires_grad)

def eye(n: int, m: int | None = None,  dtype: DTypeLike = None, requires_grad: bool = False) -> Tensor: 
    if m is None:
        m = n
    
    if dtype is None:
        dtype = np.float32
    
    out = np.eye(n, m, dtype=dtype) #type:ignore
    return Tensor(out, dtype=dtype, requires_grad=requires_grad)

def empty_like(tensor: Tensor, dtype: DTypeLike = None, requires_grad: bool = False) -> Tensor:
    assert isinstance(tensor, Tensor)

    if dtype is None:
        dtype = tensor.dtype
    
    out = np.empty_like(tensor.data, dtype=dtype)
    return Tensor(out, dtype=dtype, requires_grad=requires_grad)