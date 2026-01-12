"""Tests for utility functions comparing JAX and PyTorch implementations."""

import jax.numpy as jnp
import numpy as np
import torch

from jigandmpnn.modules.utils import (
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
    gather_nodes_t,
)

# Import PyTorch reference implementations
from jigandmpnn.vendor.ligandmpnn import (
    cat_neighbors_nodes as torch_cat_neighbors_nodes,
    gather_edges as torch_gather_edges,
    gather_nodes as torch_gather_nodes,
    gather_nodes_t as torch_gather_nodes_t,
)


def test_gather_edges():
    """Test gather_edges matches PyTorch implementation."""
    B, N, K, C = 2, 10, 5, 8

    # Create random inputs
    torch.manual_seed(42)
    edges_torch = torch.randn(B, N, N, C)
    neighbor_idx_torch = torch.randint(0, N, (B, N, K))

    # Run PyTorch
    result_torch = torch_gather_edges(edges_torch, neighbor_idx_torch)

    # Convert to JAX
    edges_jax = jnp.array(edges_torch.numpy())
    neighbor_idx_jax = jnp.array(neighbor_idx_torch.numpy())

    # Run JAX
    result_jax = gather_edges(edges_jax, neighbor_idx_jax)

    # Compare
    np.testing.assert_allclose(
        np.array(result_jax),
        result_torch.numpy(),
        atol=1e-6,
        err_msg="gather_edges mismatch",
    )


def test_gather_nodes():
    """Test gather_nodes matches PyTorch implementation."""
    B, N, K, C = 2, 10, 5, 8

    # Create random inputs
    torch.manual_seed(42)
    nodes_torch = torch.randn(B, N, C)
    neighbor_idx_torch = torch.randint(0, N, (B, N, K))

    # Run PyTorch
    result_torch = torch_gather_nodes(nodes_torch, neighbor_idx_torch)

    # Convert to JAX
    nodes_jax = jnp.array(nodes_torch.numpy())
    neighbor_idx_jax = jnp.array(neighbor_idx_torch.numpy())

    # Run JAX
    result_jax = gather_nodes(nodes_jax, neighbor_idx_jax)

    # Compare
    np.testing.assert_allclose(
        np.array(result_jax),
        result_torch.numpy(),
        atol=1e-6,
        err_msg="gather_nodes mismatch",
    )


def test_gather_nodes_t():
    """Test gather_nodes_t matches PyTorch implementation."""
    B, N, K, C = 2, 10, 5, 8

    # Create random inputs
    torch.manual_seed(42)
    nodes_torch = torch.randn(B, N, C)
    neighbor_idx_torch = torch.randint(0, N, (B, K))

    # Run PyTorch
    result_torch = torch_gather_nodes_t(nodes_torch, neighbor_idx_torch)

    # Convert to JAX
    nodes_jax = jnp.array(nodes_torch.numpy())
    neighbor_idx_jax = jnp.array(neighbor_idx_torch.numpy())

    # Run JAX
    result_jax = gather_nodes_t(nodes_jax, neighbor_idx_jax)

    # Compare
    np.testing.assert_allclose(
        np.array(result_jax),
        result_torch.numpy(),
        atol=1e-6,
        err_msg="gather_nodes_t mismatch",
    )


def test_cat_neighbors_nodes():
    """Test cat_neighbors_nodes matches PyTorch implementation."""
    B, N, K, C = 2, 10, 5, 8

    # Create random inputs
    torch.manual_seed(42)
    h_nodes_torch = torch.randn(B, N, C)
    h_neighbors_torch = torch.randn(B, N, K, C)
    E_idx_torch = torch.randint(0, N, (B, N, K))

    # Run PyTorch
    result_torch = torch_cat_neighbors_nodes(h_nodes_torch, h_neighbors_torch, E_idx_torch)

    # Convert to JAX
    h_nodes_jax = jnp.array(h_nodes_torch.numpy())
    h_neighbors_jax = jnp.array(h_neighbors_torch.numpy())
    E_idx_jax = jnp.array(E_idx_torch.numpy())

    # Run JAX
    result_jax = cat_neighbors_nodes(h_nodes_jax, h_neighbors_jax, E_idx_jax)

    # Compare
    np.testing.assert_allclose(
        np.array(result_jax),
        result_torch.numpy(),
        atol=1e-6,
        err_msg="cat_neighbors_nodes mismatch",
    )


def test_gather_edges_shape():
    """Test gather_edges output shape."""
    B, N, K, C = 2, 10, 5, 8
    edges = jnp.zeros((B, N, N, C))
    neighbor_idx = jnp.zeros((B, N, K), dtype=jnp.int32)

    result = gather_edges(edges, neighbor_idx)
    assert result.shape == (B, N, K, C)


def test_gather_nodes_shape():
    """Test gather_nodes output shape."""
    B, N, K, C = 2, 10, 5, 8
    nodes = jnp.zeros((B, N, C))
    neighbor_idx = jnp.zeros((B, N, K), dtype=jnp.int32)

    result = gather_nodes(nodes, neighbor_idx)
    assert result.shape == (B, N, K, C)
