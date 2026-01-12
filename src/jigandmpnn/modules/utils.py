"""Utility functions for graph operations in JAX."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def gather_edges(
    edges: Float[Array, "B N N C"],
    neighbor_idx: Int[Array, "B N K"],
) -> Float[Array, "B N K C"]:
    """Gather edge features at neighbor indices.

    Args:
        edges: Edge features of shape [B, N, N, C]
        neighbor_idx: Neighbor indices of shape [B, N, K]

    Returns:
        Gathered edge features of shape [B, N, K, C]
    """
    B, N, K = neighbor_idx.shape
    C = edges.shape[-1]

    # Expand neighbor_idx to [B, N, K, C] for gathering
    neighbors = jnp.broadcast_to(neighbor_idx[..., None], (B, N, K, C))

    # Use take_along_axis on dimension 2 (the N dimension we're indexing into)
    edge_features = jnp.take_along_axis(edges, neighbors, axis=2)
    return edge_features


def gather_nodes(
    nodes: Float[Array, "B N C"],
    neighbor_idx: Int[Array, "B N K"],
) -> Float[Array, "B N K C"]:
    """Gather node features at neighbor indices.

    Args:
        nodes: Node features of shape [B, N, C]
        neighbor_idx: Neighbor indices of shape [B, N, K]

    Returns:
        Gathered node features of shape [B, N, K, C]
    """
    B, N, K = neighbor_idx.shape
    C = nodes.shape[-1]

    # Flatten neighbor_idx: [B, N, K] -> [B, N*K]
    neighbors_flat = neighbor_idx.reshape(B, -1)

    # Expand for gathering: [B, N*K] -> [B, N*K, C]
    neighbors_flat = jnp.broadcast_to(neighbors_flat[..., None], (B, N * K, C))

    # Gather along dimension 1
    neighbor_features = jnp.take_along_axis(nodes, neighbors_flat, axis=1)

    # Reshape back: [B, N*K, C] -> [B, N, K, C]
    neighbor_features = neighbor_features.reshape(B, N, K, C)
    return neighbor_features


def gather_nodes_t(
    nodes: Float[Array, "B N C"],
    neighbor_idx: Int[Array, "B K"],
) -> Float[Array, "B K C"]:
    """Gather node features at neighbor indices (transposed version).

    Args:
        nodes: Node features of shape [B, N, C]
        neighbor_idx: Neighbor indices of shape [B, K]

    Returns:
        Gathered node features of shape [B, K, C]
    """
    B, K = neighbor_idx.shape
    C = nodes.shape[-1]

    # Expand idx for gathering: [B, K] -> [B, K, C]
    idx_flat = jnp.broadcast_to(neighbor_idx[..., None], (B, K, C))

    # Gather along dimension 1
    neighbor_features = jnp.take_along_axis(nodes, idx_flat, axis=1)
    return neighbor_features


def cat_neighbors_nodes(
    h_nodes: Float[Array, "B N C1"],
    h_neighbors: Float[Array, "B N K C2"],
    E_idx: Int[Array, "B N K"],
) -> Float[Array, "B N K C1+C2"]:
    """Concatenate gathered node features with neighbor features.

    Args:
        h_nodes: Node features of shape [B, N, C1]
        h_neighbors: Neighbor/edge features of shape [B, N, K, C2]
        E_idx: Edge indices of shape [B, N, K]

    Returns:
        Concatenated features of shape [B, N, K, C1+C2]
    """
    h_nodes_gathered = gather_nodes(h_nodes, E_idx)  # [B, N, K, C1]
    h_nn = jnp.concatenate([h_neighbors, h_nodes_gathered], axis=-1)
    return h_nn
