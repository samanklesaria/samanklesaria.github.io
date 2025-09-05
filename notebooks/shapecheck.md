---
title: ShapeCheck: Making Tensor Code Self-Documenting with Runtime Shape Validation
date: 8/30/2025
category: tools
---


Writing neural networks often feels like juggling tensors in the dark. You know that `attention_weights` should be 4-dimensional, but PyTorch won't tell you until your matrix multiplication explodes at runtime. What if your variable names could automatically validate tensor shapes?

Meet [**ShapeCheck**](https://github.com/samanklesaria/shapecheck) – a Python decorator that brings Character AI's shape-suffix convention to life with automatic runtime validation. Pip install `shapecheck` to get started!

## The Character AI Convention

At Character AI, engineers follow a simple but powerful naming convention: append dimension letters to tensor variable names. As Noam Shazeer explains in his [Medium post](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):

> "When known, the name of a tensor should end in a dimension-suffix composed of those letters, e.g. `input_token_id_BL` for a two-dimensional tensor with batch and length dimensions."

This makes code self-documenting. Looking at `query_BLHK`, you immediately know it's a 4D tensor with batch, length, heads, and key dimensions.

## From Convention to Validation

ShapeCheck takes this convention and makes it bulletproof. By analyzing your function's syntax tree, it automatically injects shape checks wherever you use suffixed variable names. Just prefix your function with `@shapecheck`:

```python
import torch
from shapecheck import shapecheck

@shapecheck
def attention(query_BLH, key_BLH, value_BLH):
    # Automatic validation: all tensors must be 3D with matching B,L dimensions
    scores_BLL = torch.matmul(query_BLH, key_BLH.transpose(-2, -1))
    weights_BLL = torch.softmax(scores_BLL, dim=-1)
    output_BLH = torch.matmul(weights_BLL, value_BLH)
    return output_BLH
```

When shapes don't match, ShapeCheck produces a clear error message describing the discrepency. For example:

```python
q = torch.randn(2, 10, 64)
k = torch.randn(2, 12, 64)
v = torch.randn(2, 10, 64)
result = attention(q, k, v)
```

```
AssertionError: Shape mismatch for key_BLH dimension L: expected 10 (from query_BLH), got 12
```

The magic happens through AST transformation. ShapeCheck parses your function, identifies shape-annotated variables, and injects validation code automatically. You write clean, readable code with meaningful names, and get bulletproof shape checking for free.

## Available in Julia too!

The Julia version is called `SizeCheck`. It's available on [GitHub](https://github.com/samanklesaria/SizeCheck) and can be installed via `Pkg.add("SizeCheck")`.
