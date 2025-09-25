# MiAuroGrad

A PyTorch-like automatic differentiation engine implemented from scratch in Python using NumPy.
Inspired by Anrej Karpathy's micrograd [github.com/](https://github.com/karpathy/micrograd).

## Features

- **Automatic Differentiation**: Full backward propagation support with computational graph tracking
- **Tensor Operations**: Comprehensive set of tensor operations with gradient computation
- **Broadcasting**: NumPy-style broadcasting for element-wise operations  
- **Neural Network Module**: Base module class for building neural networks
- **Memory Efficient**: Gradient accumulation and zeroing capabilities

## Installation

Clone this repository:
```bash
git clone <repository-url>
cd MiAuroGrad
```

Install dependencies:
```bash
pip install numpy
```

## Quick Start

```python
from Engine import Tensor
import Engine.tensor_ops as ops

# Create tensors with gradient tracking
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0, 6.0], requires_grad=True)

# Perform operations
z = x @ y  # dot product
loss = z.sum()

# Compute gradients
loss.backward()

print(f"x.grad: {x.grad}")  # [4. 5. 6.]
print(f"y.grad: {y.grad}")  # [1. 2. 3.]
```

## Core Components

### Tensor Class
The main `Tensor` class supports:
- Element-wise operations (`+`, `-`, `*`, `/`, `**`)
- Matrix operations (`@` for matrix multiplication)
- Activation functions (`relu`, `sigmoid`, `tanh`)
- Reduction operations (`sum`, `mean`, `max`, `min`)
- Shape manipulation (`reshape`, `transpose`, `squeeze`, `unsqueeze`)

### Tensor Operations Module
Additional operations in `tensor_ops.py`:
- `concat()` - Concatenate tensors along an axis
- `stack()` - Stack tensors along a new axis  
- `broadcast_to()` - Broadcast tensor to a new shape
- Utility functions for tensor creation

### Neural Network Module
Base `Module` class for building neural networks with parameter management.

## Architecture

```
MiAuroGrad/
├── Engine/
│   ├── tensor.py          # Core Tensor class with autograd
│   ├── tensor_ops.py      # Additional tensor operations
│   └── utils.py           # Utility functions
├── nn/
│   └── module.py          # Neural network module base class
└── README.md
```

## Supported Operations

**Arithmetic**: `+`, `-`, `*`, `/`, `**`, `@`  
**Activations**: `relu`, `sigmoid`, `tanh`, `exp`, `log`  
**Reductions**: `sum`, `mean`, `max`, `min`, `product`  
**Shape**: `reshape`, `transpose`, `squeeze`, `unsqueeze`  
**Advanced**: `cumsum`, `cumprod`, `argmax`, `argmin`

## Examples

### Basic Autograd
```python
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
y = x.sum()
y.backward()
print(x.grad)  # [[1. 1.] [1. 1.]]
```

### Chain Rule
```python
x = Tensor(2.0, requires_grad=True)
y = x ** 2
z = y * 3
z.backward()
print(x.grad)  # 12.0 (derivative of 3x²)
```

## License

This project is licensed under the MIT License
