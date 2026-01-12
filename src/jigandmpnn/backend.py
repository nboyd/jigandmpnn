"""Singledispatch-based conversion from PyTorch to JAX/Equinox modules."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from typing import Any


@singledispatch
def from_torch(x: Any) -> Any:
    """Convert a PyTorch object to its JAX/Equinox equivalent.

    This is the main entry point for conversion. Register new types
    using the @register_from_torch decorator on Equinox module classes.
    """
    raise NotImplementedError(f"from_torch not implemented for {type(x).__name__}: {x}")


# Register primitive types (pass through unchanged)
@from_torch.register(type(None))
@from_torch.register(str)
@from_torch.register(int)
@from_torch.register(float)
@from_torch.register(bool)
def _from_torch_primitive(x: Any) -> Any:
    return x


@from_torch.register(tuple)
def _from_torch_tuple(x: tuple) -> tuple:
    return tuple(from_torch(item) for item in x)


@from_torch.register(list)
def _from_torch_list(x: list) -> list:
    return [from_torch(item) for item in x]


@from_torch.register(dict)
def _from_torch_dict(x: dict) -> dict:
    return {k: from_torch(v) for k, v in x.items()}


# Register torch.Tensor -> jax.Array
@from_torch.register(torch.Tensor)
def _from_torch_tensor(x: torch.Tensor) -> jnp.ndarray:
    return jnp.array(x.detach().cpu().numpy())


# Register nn.ModuleList -> list
@from_torch.register(nn.ModuleList)
def _from_torch_modulelist(x: nn.ModuleList) -> list:
    return [from_torch(m) for m in x]


def register_from_torch(torch_module_type: type):
    """Decorator to register a JAX module's from_torch staticmethod.

    Usage:
        @register_from_torch(torch.nn.Linear)
        class Linear(eqx.Module):
            weight: jax.Array
            bias: jax.Array | None

            @staticmethod
            def from_torch(m: torch.nn.Linear) -> "Linear":
                return Linear(
                    weight=jnp.array(m.weight.detach().numpy()),
                    bias=jnp.array(m.bias.detach().numpy()) if m.bias is not None else None,
                )
    """
    def decorator(cls):
        from_torch.register(torch_module_type, cls.from_torch)
        return cls
    return decorator


# Built-in nn.Linear conversion
@register_from_torch(nn.Linear)
class Linear(eqx.Module):
    """Equinox Linear layer converted from torch.nn.Linear."""

    weight: jnp.ndarray
    bias: jnp.ndarray | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y

    @staticmethod
    def from_torch(m: nn.Linear) -> "Linear":
        return Linear(
            weight=jnp.array(m.weight.detach().cpu().numpy()),
            bias=jnp.array(m.bias.detach().cpu().numpy()) if m.bias is not None else None,
            in_features=m.in_features,
            out_features=m.out_features,
        )


# Built-in nn.LayerNorm conversion
@register_from_torch(nn.LayerNorm)
class LayerNorm(eqx.Module):
    """Equinox LayerNorm layer converted from torch.nn.LayerNorm."""

    weight: jnp.ndarray | None
    bias: jnp.ndarray | None
    normalized_shape: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Compute mean and variance over the last len(normalized_shape) dimensions
        axis = tuple(range(-len(self.normalized_shape), 0))
        mean = jnp.mean(x, axis=axis, keepdims=True)
        var = jnp.var(x, axis=axis, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)

        if self.weight is not None:
            x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias
        return x_norm

    @staticmethod
    def from_torch(m: nn.LayerNorm) -> "LayerNorm":
        normalized_shape = m.normalized_shape
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        return LayerNorm(
            weight=jnp.array(m.weight.detach().cpu().numpy()) if m.weight is not None else None,
            bias=jnp.array(m.bias.detach().cpu().numpy()) if m.bias is not None else None,
            normalized_shape=tuple(normalized_shape),
            eps=m.eps,
        )


# Built-in nn.Embedding conversion
@register_from_torch(nn.Embedding)
class Embedding(eqx.Module):
    """Equinox Embedding layer converted from torch.nn.Embedding."""

    weight: jnp.ndarray
    num_embeddings: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.weight[x]

    @staticmethod
    def from_torch(m: nn.Embedding) -> "Embedding":
        return Embedding(
            weight=jnp.array(m.weight.detach().cpu().numpy()),
            num_embeddings=m.num_embeddings,
            embedding_dim=m.embedding_dim,
        )


# Built-in nn.Dropout conversion (inference mode - identity)
@register_from_torch(nn.Dropout)
class Dropout(eqx.Module):
    """Equinox Dropout layer (inference mode - identity function)."""

    p: float = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Inference mode: no dropout
        return x

    @staticmethod
    def from_torch(m: nn.Dropout) -> "Dropout":
        return Dropout(p=m.p)
