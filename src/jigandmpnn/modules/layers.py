"""Neural network layers for LigandMPNN in JAX/Equinox."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jigandmpnn.backend import Linear, LayerNorm, from_torch, register_from_torch
from jigandmpnn.modules.utils import cat_neighbors_nodes
from jigandmpnn.vendor.ligandmpnn.model_utils import (
    PositionalEncodings as TorchPositionalEncodings,
    PositionWiseFeedForward as TorchPositionWiseFeedForward,
    EncLayer as TorchEncLayer,
    DecLayer as TorchDecLayer,
    DecLayerJ as TorchDecLayerJ,
)

if TYPE_CHECKING:
    pass


def gelu(x):
    """GELU activation matching PyTorch's default (exact, not approximate)."""
    return jnn.gelu(x, approximate=False)


@register_from_torch(TorchPositionalEncodings)
class PositionalEncodings(eqx.Module):
    """Positional encoding layer for relative sequence positions."""

    num_embeddings: int = eqx.field(static=True)
    max_relative_feature: int = eqx.field(static=True)
    linear: Linear

    def __call__(
        self,
        offset: Int[Array, "B N K"],
        mask: Int[Array, "B N K"],
    ) -> Float[Array, "B N K num_embeddings"]:
        """Compute positional encodings.

        Args:
            offset: Relative position offsets
            mask: Mask for same-chain (1) vs cross-chain (0) interactions

        Returns:
            Positional embeddings
        """
        # Clip offset to valid range and apply mask
        d = jnp.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)

        # One-hot encode
        d_onehot = jnn.one_hot(d, 2 * self.max_relative_feature + 1 + 1)

        # Project through linear layer
        E = self.linear(d_onehot.astype(jnp.float32))
        return E

    @staticmethod
    def from_torch(m: TorchPositionalEncodings) -> "PositionalEncodings":
        return PositionalEncodings(
            num_embeddings=m.num_embeddings,
            max_relative_feature=m.max_relative_feature,
            linear=from_torch(m.linear),
        )


@register_from_torch(TorchPositionWiseFeedForward)
class PositionWiseFeedForward(eqx.Module):
    """Position-wise feed-forward network with GELU activation."""

    W_in: Linear
    W_out: Linear

    def __call__(self, h_V: Float[Array, "... D"]) -> Float[Array, "... D"]:
        """Forward pass through FFN.

        Args:
            h_V: Input features

        Returns:
            Transformed features
        """
        h = gelu(self.W_in(h_V))
        h = self.W_out(h)
        return h

    @staticmethod
    def from_torch(m: TorchPositionWiseFeedForward) -> "PositionWiseFeedForward":
        return PositionWiseFeedForward(
            W_in=from_torch(m.W_in),
            W_out=from_torch(m.W_out),
        )


@register_from_torch(TorchEncLayer)
class EncLayer(eqx.Module):
    """Encoder layer with message passing on nodes and edges."""

    num_hidden: int = eqx.field(static=True)
    num_in: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    # Node message passing
    W1: Linear
    W2: Linear
    W3: Linear
    norm1: LayerNorm
    norm2: LayerNorm

    # Edge message passing
    W11: Linear
    W12: Linear
    W13: Linear
    norm3: LayerNorm

    # Feed-forward
    dense: PositionWiseFeedForward

    def __call__(
        self,
        h_V: Float[Array, "B N H"],
        h_E: Float[Array, "B N K H"],
        E_idx: Int[Array, "B N K"],
        mask_V: Float[Array, "B N"] | None = None,
        mask_attend: Float[Array, "B N K"] | None = None,
    ) -> tuple[Float[Array, "B N H"], Float[Array, "B N K H"]]:
        """Forward pass through encoder layer.

        Args:
            h_V: Node features [B, N, H]
            h_E: Edge features [B, N, K, H]
            E_idx: Edge indices [B, N, K]
            mask_V: Node mask [B, N]
            mask_attend: Attention mask [B, N, K]

        Returns:
            Updated (h_V, h_E) tuple
        """
        # Node update
        # Gather neighbor node features and concatenate with edge features
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)  # [B, N, K, 2*H]

        # Expand h_V for concatenation: [B, N, H] -> [B, N, K, H]
        K = h_EV.shape[-2]
        h_V_expand = jnp.broadcast_to(h_V[:, :, None, :], (*h_V.shape[:2], K, h_V.shape[-1]))

        # Concatenate: [B, N, K, 3*H] (h_V + h_E + h_neighbors)
        h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)

        # MLP for message
        h_message = self.W3(gelu(self.W2(gelu(self.W1(h_EV)))))

        # Apply attention mask
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message

        # Aggregate messages (sum over neighbors, scaled)
        dh = jnp.sum(h_message, axis=-2) / self.scale

        # Residual + LayerNorm (inference: dropout is identity)
        h_V = self.norm1(h_V + dh)

        # Feed-forward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + dh)

        # Apply node mask
        if mask_V is not None:
            h_V = mask_V[..., None] * h_V

        # Edge update
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.broadcast_to(h_V[:, :, None, :], (*h_V.shape[:2], K, h_V.shape[-1]))
        h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)

        h_message = self.W13(gelu(self.W12(gelu(self.W11(h_EV)))))

        # Edge residual + LayerNorm
        h_E = self.norm3(h_E + h_message)

        return h_V, h_E

    @staticmethod
    def from_torch(m: TorchEncLayer) -> "EncLayer":
        return EncLayer(
            num_hidden=m.num_hidden,
            num_in=m.num_in,
            scale=m.scale,
            W1=from_torch(m.W1),
            W2=from_torch(m.W2),
            W3=from_torch(m.W3),
            W11=from_torch(m.W11),
            W12=from_torch(m.W12),
            W13=from_torch(m.W13),
            norm1=from_torch(m.norm1),
            norm2=from_torch(m.norm2),
            norm3=from_torch(m.norm3),
            dense=from_torch(m.dense),
        )


@register_from_torch(TorchDecLayer)
class DecLayer(eqx.Module):
    """Decoder layer with message passing on nodes only."""

    num_hidden: int = eqx.field(static=True)
    num_in: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    W1: Linear
    W2: Linear
    W3: Linear
    norm1: LayerNorm
    norm2: LayerNorm
    dense: PositionWiseFeedForward

    def __call__(
        self,
        h_V: Float[Array, "B N H"],
        h_E: Float[Array, "B N K H2"],
        mask_V: Float[Array, "B N"] | None = None,
        mask_attend: Float[Array, "B N K"] | None = None,
    ) -> Float[Array, "B N H"]:
        """Forward pass through decoder layer.

        Args:
            h_V: Node features [B, N, H]
            h_E: Edge/context features [B, N, K, H2]
            mask_V: Node mask [B, N]
            mask_attend: Attention mask [B, N, K]

        Returns:
            Updated h_V
        """
        # Expand h_V for concatenation: [B, N, H] -> [B, N, K, H]
        K = h_E.shape[-2]
        h_V_expand = jnp.broadcast_to(h_V[:, :, None, :], (*h_V.shape[:2], K, h_V.shape[-1]))

        # Concatenate: [B, N, K, H + H2]
        h_EV = jnp.concatenate([h_V_expand, h_E], axis=-1)

        # MLP for message
        h_message = self.W3(gelu(self.W2(gelu(self.W1(h_EV)))))

        # Apply attention mask
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message

        # Aggregate messages (sum over neighbors, scaled)
        dh = jnp.sum(h_message, axis=-2) / self.scale

        # Residual + LayerNorm
        h_V = self.norm1(h_V + dh)

        # Feed-forward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + dh)

        # Apply node mask
        if mask_V is not None:
            h_V = mask_V[..., None] * h_V

        return h_V

    @staticmethod
    def from_torch(m: TorchDecLayer) -> "DecLayer":
        return DecLayer(
            num_hidden=m.num_hidden,
            num_in=m.num_in,
            scale=m.scale,
            W1=from_torch(m.W1),
            W2=from_torch(m.W2),
            W3=from_torch(m.W3),
            norm1=from_torch(m.norm1),
            norm2=from_torch(m.norm2),
            dense=from_torch(m.dense),
        )


@register_from_torch(TorchDecLayerJ)
class DecLayerJ(eqx.Module):
    """Decoder layer for ligand context (4D edge tensor).

    Similar to DecLayer but handles h_E with shape [B, L, M, M, H]
    instead of [B, L, K, H].
    """

    num_hidden: int = eqx.field(static=True)
    num_in: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    W1: Linear
    W2: Linear
    W3: Linear
    norm1: LayerNorm
    norm2: LayerNorm
    dense: PositionWiseFeedForward

    def __call__(
        self,
        h_V: Float[Array, "B L M H"],
        h_E: Float[Array, "B L M M H2"],
        mask_V: Float[Array, "B L M"] | None = None,
        mask_attend: Float[Array, "B L M M"] | None = None,
    ) -> Float[Array, "B L M H"]:
        """Forward pass through DecLayerJ.

        Args:
            h_V: Node features [B, L, M, H]
            h_E: Edge features [B, L, M, M, H2]
            mask_V: Node mask [B, L, M]
            mask_attend: Attention mask [B, L, M, M]

        Returns:
            Updated h_V [B, L, M, H]
        """
        # Expand h_V: [B, L, M, H] -> [B, L, M, M, H]
        M = h_E.shape[-2]
        h_V_expand = jnp.broadcast_to(
            h_V[..., None, :], (*h_V.shape[:-1], M, h_V.shape[-1])
        )

        # Concatenate: [B, L, M, M, H + H2]
        h_EV = jnp.concatenate([h_V_expand, h_E], axis=-1)

        # MLP for message
        h_message = self.W3(gelu(self.W2(gelu(self.W1(h_EV)))))

        # Apply attention mask
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message

        # Aggregate messages (sum over last neighbor dimension, scaled)
        dh = jnp.sum(h_message, axis=-2) / self.scale

        # Residual + LayerNorm
        h_V = self.norm1(h_V + dh)

        # Feed-forward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + dh)

        # Apply node mask
        if mask_V is not None:
            h_V = mask_V[..., None] * h_V

        return h_V

    @staticmethod
    def from_torch(m: TorchDecLayerJ) -> "DecLayerJ":
        return DecLayerJ(
            num_hidden=m.num_hidden,
            num_in=m.num_in,
            scale=m.scale,
            W1=from_torch(m.W1),
            W2=from_torch(m.W2),
            W3=from_torch(m.W3),
            norm1=from_torch(m.norm1),
            norm2=from_torch(m.norm2),
            dense=from_torch(m.dense),
        )
