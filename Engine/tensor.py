from numpy.typing import DTypeLike
import numpy as np
from typing import Set, Tuple

from .utils import _unbroadcast


class Tensor:
    def __init__(self, data , dtype: DTypeLike = np.float32, requires_grad: bool = False,
                 _children: Set['Tensor'] | None = None, _op: str = '') -> None:
        self.data = data.data if isinstance(data, Tensor) else data if isinstance(data, np.ndarray) else np.array(data, dtype=dtype)

        if self.data.dtype != dtype:
            self.data = self.data.astype(dtype)
        self.dtype = dtype
        self.requires_grad = requires_grad or bool(_children)

        self._children = _children if _children is not None else set()
        self._op = _op
        self.grad = np.zeros_like(self.data, dtype=dtype) if requires_grad else None
        self._backward = lambda: None

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape
    
    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    def backward(self) -> None:
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build(child)
                topo.append(v)
        
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build(child)
                topo.append(v)
        
        build(self)
        for v in topo:
            if v.requires_grad:
                v.grad = np.zeros_like(v.data)

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        sum_data = np.sum(self.data, axis=axis, keepdims=True)

        if not keepdims:
            sum_data = np.squeeze(sum_data, axis=axis)

        out = Tensor(sum_data,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                    _children={self},
                    _op='sum')
        
        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            grad = np.broadcast_to(grad, self.data.shape)

            self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out

    
    def product(self, axis=None, keepdims=False) -> 'Tensor':
        prod_data_keepdims = np.prod(self.data, axis=axis, keepdims=True)
        prod_data = prod_data_keepdims

        if not keepdims:
            prod_data = np.squeeze(prod_data_keepdims, axis=axis)

        out = Tensor(prod_data,
                     dtype=self.dtype,
                     requires_grad=self.requires_grad,
                     _children={self},
                     _op='prod')
        
        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            
            grad = out.grad

            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            
            grad = np.broadcast_to(grad, self.data.shape)
            
            this_grad = np.where(self.data != 0, prod_data_keepdims / self.data, 0.0) * grad
            self.grad = this_grad if self.grad is None else self.grad + this_grad

        out._backward = _backward
        return out
    
    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        out = self.data.reshape(shape)
        out = Tensor(out,
                     dtype=self.dtype,  
                     requires_grad=self.requires_grad,
                     _children={self},
                     _op='reshape')
        
        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out
    
    def transpose(self, axes=None) -> 'Tensor':
        data = np.transpose(self.data, axes=axes)
        out = Tensor(data,
                     dtype=self.dtype,
                     requires_grad=self.requires_grad,
                     _children={self},
                     _op='transpose')
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if axes is None:
                    self.grad += out.grad.T
                else:
                    if isinstance(axes, int):
                        inv_axes = list(range(len(self.data.shape)))
                        inv_axes[0], inv_axes[axes] = inv_axes[axes], inv_axes[0]
                    else:
                        inv_axes = [0] * len(axes)
                        for i, ax in enumerate(axes):
                            inv_axes[ax] = i
                self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out

    def argmax(self, axis=None) -> 'Tensor':
        indices = np.argmax(self.data, axis=axis)
        out = Tensor(indices,
                     dtype=np.int64,
                     requires_grad=False)
        return out

    def argmin(self, axis=None) -> 'Tensor':
        indices = np.argmin(self.data, axis=axis)
        out = Tensor(indices,
                     dtype=np.int64,
                     requires_grad=False)
        return out
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        return self @ other
    
    def abs(self) -> 'Tensor':
        return self.__abs__()
    
    def max(self, axis=None, keepdims=False) -> 'Tensor':
        max_data_keepdims = np.max(self.data, axis=axis, keepdims=True)
        max_data = max_data_keepdims

        if not keepdims:
            max_data = np.squeeze(max_data_keepdims, axis=axis)

        out = Tensor(max_data,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                    _children={self},
                    _op='max')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return

            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)

            # mask of max elements
            max_mask = (self.data == max_data_keepdims)

            if axis is None:
                count = np.sum(max_mask)
            else:
                # count how many max elements per reduced slice
                count = np.sum(max_mask, axis=axis, keepdims=True)

            # distribute grad equally among all maxima
            this_grad = (max_mask / count) * grad
            self.grad = this_grad if self.grad is None else self.grad + this_grad

        out._backward = _backward
        return out


    def min(self, axis=None, keepdims=False) -> 'Tensor':
        min_data_keepdims = np.min(self.data, axis=axis, keepdims=True)
        min_data = min_data_keepdims

        if not keepdims:
            min_data = np.squeeze(min_data_keepdims, axis=axis)
        
        out = Tensor(min_data,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                    _children={self},
                    _op='min')
        
        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            
            # Gradient flows only to the minimum elements
            min_mask = (self.data == min_data_keepdims)

            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)

            if axis is None:
                count = np.sum(min_mask)
            else:
                count = np.sum(min_mask, axis=axis, keepdims=True)

            this_grad = (min_mask / count) * grad
            self.grad = this_grad if self.grad is None else self.grad + this_grad
        
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        mean_data_keepdims = np.mean(self.data, axis=axis, keepdims=True)
        mean_data = mean_data_keepdims
        if not keepdims:
            mean_data = np.squeeze(mean_data_keepdims, axis=axis)
        
        out = Tensor(mean_data,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                    _children={self},
                    _op='mean')
        
        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            
            grad = out.grad
            # Calculate coefficient (1/number_of_elements_averaged)
            if axis is None:
                coeff = 1.0 / self.data.size
            else:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                coeff = 1.0 / self.data.shape[axis]
            
            # Broadcast gradient and apply coefficient
            grad_broadcasted = np.broadcast_to(grad, self.data.shape)
            this_grad = grad_broadcasted * coeff
            self.grad = this_grad if self.grad is None else self.grad + this_grad
        
        out._backward = _backward
        return out

    def cumsum(self, axis=None) -> 'Tensor':
        out = self.data.cumsum(axis=axis)
        out = Tensor(out,
                     dtype=self.dtype,
                     requires_grad=self.requires_grad,
                     _children={self},
                     _op='cumsum')
        
        def  _backward():
            if out.grad is None or not self.requires_grad:
                return
            
            grad = out.grad
            if axis is None:
                # For flattened cumsum, reverse cumsum of flattened gradient
                rev_grad = np.cumsum(grad[::-1])[::-1].reshape(self.data.shape)
            else:
                # The gradient of cumsum is reversed of cumsum of the reverse order, all along the same axis
                rev_grad = np.flip(np.flip(grad, axis=axis).cumsum(axis=axis), axis=axis)
    
            self.grad += rev_grad
        
        out._backward = _backward
        return out

    def cumprod(self, axis: int = -1) -> 'Tensor':
        out_data = np.cumprod(self.data, axis=axis)

        out = Tensor(out_data,
                        dtype=self.dtype,
                        requires_grad=self.requires_grad,
                        _children={self},
                        _op='cumprod')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            
            x = self.data
            y = out.data
            grad_output = out.grad
            
            x_safe = np.where(x != 0, x, 1.0)

            # z = the derivative of each output w.r.t each input (i.e gradient of yi w.r.t xi)
            z = grad_output * (y / x_safe)
            
            # But xi not only affect yi but also affect yj (j >= i)
            #      y0  y1  y2  y3  y4
            # x0    ✓   ✓   ✓   ✓   ✓
            # x1        ✓   ✓   ✓   ✓
            # x2            ✓   ✓   ✓
            # x3                ✓   ✓
            # x4                    ✓
            # perform reverse cumsum to get the true (accumulated) gradient of each xi

            if axis is None:
                z_flat = z.flatten()
                rev_cumsum = np.flip(np.cumsum(np.flip(z_flat))).reshape(x.shape)
            else:
                rev_cumsum = np.flip(np.cumsum(np.flip(z, axis=axis), axis=axis), axis=axis)
            
            # x=0 cases (set gradient to 0)
            grad_input = np.where(x != 0, rev_cumsum, 0.0)
            
            self.grad = grad_input if self.grad is None else self.grad + grad_input

        out._backward = _backward
        return out
    
    def squeeze(self, axis=None) -> 'Tensor':
        out = self.data.squeeze(axis)
        out = Tensor(out,
                     dtype=self.dtype,
                     requires_grad=self.requires_grad,
                     _children={self},
                     _op='squeeze')
        
        def _backward():
            if out.grad is None or  not self.requires_grad:
                return
            
            grad = out.grad
            grad = np.reshape(grad, self.data.shape)
            
            self.grad += grad

        out._backward = _backward
        return out

    def unsqueeze(self, axis) -> 'Tensor':
        out = np.expand_dims(self.data, axis=axis)
        out = Tensor(out,
                     dtype=self.dtype,
                     requires_grad=self.requires_grad,
                     _children={self},
                     _op='unsqueeze')
        
        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            
            # Sum over the inserted axis because multiple gradients could flow into that dimension.
            grad = out.grad
            grad = grad.sum(axis=axis)
            self.grad += grad
        
        out._backward = _backward
        return out

    def to_numpy(self) -> np.ndarray:
        return np.copy(self.data)
    
    def to_list(self) -> list:
        return self.data.tolist()
    
    def flatten(self) -> 'Tensor':
        return Tensor(self.data.flatten(), dtype=self.dtype, requires_grad=self.requires_grad)
    
    def copy(self) -> 'Tensor':
        return Tensor(self.data.copy(), dtype=self.dtype, requires_grad=self.requires_grad, _children=self._children, _op=self._op)

    def exp(self):
        out = Tensor(np.exp(self.data),
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                    _children={self, },
                      _op='exp')

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data),
                     dtype=self.dtype,
                     requires_grad=self.requires_grad,
                     _children={self, }, 
                     _op='log')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self.grad = (1 / self.data) * out.grad if self.grad is None \
                        else self.grad + (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = np.where(self.data > 0, self.data, 0.0)
        out = Tensor(out,
                     dtype=self.dtype,
                     requires_grad=self.requires_grad, 
                     _children={self, }, 
                     _op='relu')

        def _backward():
            if self.requires_grad:
                self.grad += np.where(self.data > 0, 1.0, 0.0) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        sigmoid_data = 1 / (1 + np.exp(-self.data))
        out = Tensor(sigmoid_data,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad, 
                    _children={self}, 
                    _op='sigmoid')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self.grad = sigmoid_data * (1 - sigmoid_data) * out.grad if self.grad is None \
                         else self.grad + sigmoid_data * (1 - sigmoid_data) * out.grad

        out._backward = _backward
        return out
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t,
                     dtype=self.dtype,
                     requires_grad=self.requires_grad, 
                     _children={self, }, 
                     _op='tanh')

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data + other.data, 
                     dtype=self.dtype, 
                     requires_grad=self.requires_grad or other.requires_grad, 
                     _children={self, other}, 
                     _op='+')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(1.0 * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(1.0 * out.grad, other.data.shape)

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(self.data - other.data, 
                     dtype=self.dtype, 
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children={self, other},
                     _op='-')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(1.0 * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-1.0 * out.grad, other.data.shape)

        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return Tensor(other) - self
    
    def __isub__(self, other):
        return self - other
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(self.data * other.data, 
                     dtype=self.dtype, 
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children={self, other},
                     _op='*')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(self.data / other.data, 
                     dtype=self.dtype, 
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children={self, other},
                     _op='/')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast((1 / other.data) * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast((-self.data / other.data ** 2) * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(other.data / self.data, 
                     dtype=self.dtype, 
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children={self, other},
                     _op='/')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast((-other.data / self.data ** 2) * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast((1 / self.data) * out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    def __itruediv__(self, other):
        return self / other
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(self.data @ other.data, 
                     dtype=self.dtype, 
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children={self, other},
                     _op='matmul')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, exponent):
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent)
        
        out = Tensor(self.data ** exponent.data,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad or exponent.requires_grad,
                    _children={self, exponent},  # Include both!
                    _op='pow')
    
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast((exponent.data * self.data ** (exponent.data - 1)) * out.grad, self.data.shape)
            if exponent.requires_grad:
                # Use numpy operations, not tensor operations
                mask = self.data > 0
                exponent.grad += _unbroadcast((out.data * np.log(np.abs(self.data)) * mask) * out.grad, exponent.data.shape)
        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Tensor(-self.data,
                     dtype=self.dtype,
                     requires_grad=self.requires_grad,
                    _children={self, }, 
                    _op='neg')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self.grad = -1.0 * out.grad if self.grad is None else self.grad + (-1.0 * out.grad)  
        out._backward = _backward
        return out
    
    def __abs__(self):
        out = Tensor(np.abs(self.data),
                     dtype=self.dtype,
                     requires_grad=self.requires_grad,
                     _children={self, }, 
                     _op='abs')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self.grad += np.where(self.data > 0,
                                    1.0 * out.grad,
                                    np.where(self.data < 0, -1.0 * out.grad, 0.0)
                                      )
        out._backward = _backward
        return out
    
    __hash__ = object.__hash__
    
    def __lt__(self, other):
        return self.data < other
    
    def __gt__(self, other):
        return self.data > other
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return np.array_equal(self.data, other.data)
        return False
    
    def __le__(self, other):
        return self.data <= other
    
    def __ge__(self, other):
        return self.data >= other
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __iter__(self):
        if self.data.ndim == 0:
            raise TypeError("iteration over a 0-d Tensor")
        for val in self.data:
            yield Tensor(val, dtype=self.dtype, requires_grad=self.requires_grad)
    
    def __repr__(self) -> str:
        # Get the data representation
        data_str = self._format_data()
        
        # Build the main tensor string
        tensor_str = f"Tensor({data_str}"
        
        # Add dtype if it's not the default float32
        if self.dtype != np.float32:
            tensor_str += f", dtype={self._dtype_name()}"
        
        # Add gradient info if applicable
        if self.requires_grad:
            tensor_str += f", requires_grad=True"
        
        # Add shape for multi-dimensional tensors
        if len(self.shape) > 1 or (len(self.shape) == 1 and self.shape[0] > 5):
            tensor_str += f", shape={self.shape}"
        
        if self._op:
            tensor_str += f", op='{self._op}'"
        tensor_str += ")"
        return tensor_str

    def _format_data(self) -> str:
        """Format the data portion of the representation."""
        # Handle scalars
        if self.data.ndim == 0:
            return f"{self.data.item():.4g}" if self.dtype in [np.float16, np.float32, np.float64] else str(self.data.item())
        
        # Handle small arrays - show full data
        if self.data.size <= 6:
            if self.dtype in [np.float16, np.float32, np.float64]:
                # Format floats nicely
                formatted = np.array2string(self.data, precision=4, suppress_small=True, separator=', ')
            else:
                formatted = np.array2string(self.data, separator=', ')
            return formatted
        
        # Handle larger arrays - show abbreviated form
        with np.printoptions(threshold=6, precision=4):
            return np.array2string(self.data, separator=', ')

    def _format_grad(self) -> str:
        """Format the gradient for display."""
        if self.grad is None:
            return "None"
        
        # For small gradients, show full data
        if self.grad.size <= 3:
            return np.array2string(self.grad, precision=4, suppress_small=True, separator=', ')
        
        # For larger gradients, show abbreviated
        with np.printoptions(threshold=3, precision=4):
            return np.array2string(self.grad, separator=', ')

    def _dtype_name(self) -> str:
        """Get a clean dtype name."""
        dtype_map = {
            np.float16: 'float16',
            np.float32: 'float32', 
            np.float64: 'float64',
            np.int8: 'int8',
            np.int16: 'int16',
            np.int32: 'int32',
            np.int64: 'int64',
            np.uint8: 'uint8',
            np.uint16: 'uint16',
            np.uint32: 'uint32',
            np.bool: 'bool',
            np.complex64: 'complex64',
            np.complex128: 'complex128'
        }
        return dtype_map.get(self.dtype, str(self.dtype))