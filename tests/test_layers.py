"""Tests for layer modules comparing JAX and PyTorch implementations."""

import jax.numpy as jnp
import numpy as np
import torch

from jigandmpnn.backend import from_torch
from jigandmpnn.modules.layers import PositionalEncodings, PositionWiseFeedForward, EncLayer, DecLayer, DecLayerJ

# Import PyTorch reference implementations
from jigandmpnn.vendor.ligandmpnn import (
    PositionalEncodings as TorchPositionalEncodings,
    PositionWiseFeedForward as TorchPositionWiseFeedForward,
    EncLayer as TorchEncLayer,
    DecLayer as TorchDecLayer,
    DecLayerJ as TorchDecLayerJ,
)


def test_positional_encodings():
    """Test PositionalEncodings matches PyTorch implementation."""
    num_embeddings = 16
    max_relative_feature = 32

    # Create PyTorch module
    torch.manual_seed(42)
    torch_module = TorchPositionalEncodings(num_embeddings, max_relative_feature)
    torch_module.eval()

    # Convert to JAX
    jax_module = from_torch(torch_module)

    # Create test inputs
    B, N, K = 2, 50, 48
    offset = torch.randint(-10, 10, (B, N, K))
    mask = torch.ones(B, N, K, dtype=torch.long)

    # Run PyTorch
    with torch.no_grad():
        result_torch = torch_module(offset, mask)

    # Run JAX
    result_jax = jax_module(jnp.array(offset.numpy()), jnp.array(mask.numpy()))

    # Compare
    np.testing.assert_allclose(
        np.array(result_jax),
        result_torch.numpy(),
        atol=1e-5,
        err_msg="PositionalEncodings mismatch",
    )


def test_positional_encodings_with_mask():
    """Test PositionalEncodings with cross-chain mask."""
    num_embeddings = 16
    max_relative_feature = 32

    torch.manual_seed(42)
    torch_module = TorchPositionalEncodings(num_embeddings, max_relative_feature)
    torch_module.eval()

    jax_module = from_torch(torch_module)

    B, N, K = 2, 50, 48
    offset = torch.randint(-10, 10, (B, N, K))
    # Mix of same-chain (1) and cross-chain (0)
    mask = torch.randint(0, 2, (B, N, K))

    with torch.no_grad():
        result_torch = torch_module(offset, mask)

    result_jax = jax_module(jnp.array(offset.numpy()), jnp.array(mask.numpy()))

    np.testing.assert_allclose(
        np.array(result_jax),
        result_torch.numpy(),
        atol=1e-5,
        err_msg="PositionalEncodings with mask mismatch",
    )


def test_position_wise_feedforward():
    """Test PositionWiseFeedForward matches PyTorch implementation."""
    num_hidden = 128
    num_ff = 512

    torch.manual_seed(42)
    torch_module = TorchPositionWiseFeedForward(num_hidden, num_ff)
    torch_module.eval()

    jax_module = from_torch(torch_module)

    B, N = 2, 50
    h_V = torch.randn(B, N, num_hidden)

    with torch.no_grad():
        result_torch = torch_module(h_V)

    result_jax = jax_module(jnp.array(h_V.numpy()))

    # Allow slightly higher tolerance due to GELU implementation differences
    np.testing.assert_allclose(
        np.array(result_jax),
        result_torch.numpy(),
        atol=1e-5,
        err_msg="PositionWiseFeedForward mismatch",
    )


def test_enc_layer():
    """Test EncLayer matches PyTorch implementation."""
    num_hidden = 128
    num_in = 256  # This is hidden_dim * 2 in ProteinMPNN

    torch.manual_seed(42)
    torch_module = TorchEncLayer(num_hidden, num_in, dropout=0.0)
    torch_module.eval()

    jax_module = from_torch(torch_module)

    B, N, K = 2, 50, 48
    h_V = torch.randn(B, N, num_hidden)
    h_E = torch.randn(B, N, K, num_hidden)
    E_idx = torch.randint(0, N, (B, N, K))
    mask_V = torch.ones(B, N)
    mask_attend = torch.ones(B, N, K)

    with torch.no_grad():
        torch_h_V, torch_h_E = torch_module(h_V, h_E, E_idx, mask_V, mask_attend)

    jax_h_V, jax_h_E = jax_module(
        jnp.array(h_V.numpy()),
        jnp.array(h_E.numpy()),
        jnp.array(E_idx.numpy()),
        jnp.array(mask_V.numpy()),
        jnp.array(mask_attend.numpy()),
    )

    # Allow slightly higher tolerance due to GELU implementation differences
    np.testing.assert_allclose(
        np.array(jax_h_V),
        torch_h_V.numpy(),
        atol=1e-5,
        err_msg="EncLayer h_V mismatch",
    )
    np.testing.assert_allclose(
        np.array(jax_h_E),
        torch_h_E.numpy(),
        atol=1e-5,
        err_msg="EncLayer h_E mismatch",
    )


def test_enc_layer_with_masking():
    """Test EncLayer with partial masking."""
    num_hidden = 128
    num_in = 256

    torch.manual_seed(42)
    torch_module = TorchEncLayer(num_hidden, num_in, dropout=0.0)
    torch_module.eval()

    jax_module = from_torch(torch_module)

    B, N, K = 2, 50, 48
    h_V = torch.randn(B, N, num_hidden)
    h_E = torch.randn(B, N, K, num_hidden)
    E_idx = torch.randint(0, N, (B, N, K))

    # Partial masks
    mask_V = torch.rand(B, N) > 0.2
    mask_V = mask_V.float()
    mask_attend = torch.rand(B, N, K) > 0.2
    mask_attend = mask_attend.float()

    with torch.no_grad():
        torch_h_V, torch_h_E = torch_module(h_V, h_E, E_idx, mask_V, mask_attend)

    jax_h_V, jax_h_E = jax_module(
        jnp.array(h_V.numpy()),
        jnp.array(h_E.numpy()),
        jnp.array(E_idx.numpy()),
        jnp.array(mask_V.numpy()),
        jnp.array(mask_attend.numpy()),
    )

    # Allow slightly higher tolerance due to GELU implementation differences
    np.testing.assert_allclose(
        np.array(jax_h_V),
        torch_h_V.numpy(),
        atol=1e-5,
        err_msg="EncLayer with masking h_V mismatch",
    )
    np.testing.assert_allclose(
        np.array(jax_h_E),
        torch_h_E.numpy(),
        atol=1e-5,
        err_msg="EncLayer with masking h_E mismatch",
    )


def test_enc_layer_no_mask():
    """Test EncLayer without masks (None)."""
    num_hidden = 128
    num_in = 256

    torch.manual_seed(42)
    torch_module = TorchEncLayer(num_hidden, num_in, dropout=0.0)
    torch_module.eval()

    jax_module = from_torch(torch_module)

    B, N, K = 2, 50, 48
    h_V = torch.randn(B, N, num_hidden)
    h_E = torch.randn(B, N, K, num_hidden)
    E_idx = torch.randint(0, N, (B, N, K))

    with torch.no_grad():
        torch_h_V, torch_h_E = torch_module(h_V, h_E, E_idx, None, None)

    jax_h_V, jax_h_E = jax_module(
        jnp.array(h_V.numpy()),
        jnp.array(h_E.numpy()),
        jnp.array(E_idx.numpy()),
        None,
        None,
    )

    # Allow slightly higher tolerance due to GELU implementation differences
    np.testing.assert_allclose(
        np.array(jax_h_V),
        torch_h_V.numpy(),
        atol=1e-5,
        err_msg="EncLayer (no mask) h_V mismatch",
    )
    np.testing.assert_allclose(
        np.array(jax_h_E),
        torch_h_E.numpy(),
        atol=1e-5,
        err_msg="EncLayer (no mask) h_E mismatch",
    )


def test_dec_layer():
    """Test DecLayer matches PyTorch implementation."""
    num_hidden = 128
    # W1 has shape (num_hidden + num_in, num_hidden)
    # h_EV = concat([h_V, h_E]) so h_V.dim + h_E.dim = num_hidden + num_in
    # Therefore h_E.dim = num_in
    num_in = 256

    torch.manual_seed(42)
    torch_module = TorchDecLayer(num_hidden, num_in, dropout=0.0)
    torch_module.eval()

    jax_module = from_torch(torch_module)

    B, N, K = 2, 50, 48
    h_V = torch.randn(B, N, num_hidden)  # [B, N, 128]
    h_E = torch.randn(B, N, K, num_in)   # [B, N, K, 256]
    mask_V = torch.ones(B, N)
    mask_attend = torch.ones(B, N, K)

    with torch.no_grad():
        torch_h_V = torch_module(h_V, h_E, mask_V, mask_attend)

    jax_h_V = jax_module(
        jnp.array(h_V.numpy()),
        jnp.array(h_E.numpy()),
        jnp.array(mask_V.numpy()),
        jnp.array(mask_attend.numpy()),
    )

    np.testing.assert_allclose(
        np.array(jax_h_V),
        torch_h_V.numpy(),
        atol=1e-5,
        err_msg="DecLayer h_V mismatch",
    )


def test_dec_layer_j():
    """Test DecLayerJ matches PyTorch implementation."""
    num_hidden = 128
    num_in = 128  # DecLayerJ uses hidden_dim for num_in

    torch.manual_seed(42)
    torch_module = TorchDecLayerJ(num_hidden, num_in, dropout=0.0)
    torch_module.eval()

    jax_module = from_torch(torch_module)

    B, L, M = 2, 50, 25  # B=batch, L=sequence length, M=ligand atoms
    h_V = torch.randn(B, L, M, num_hidden)
    h_E = torch.randn(B, L, M, M, num_hidden)
    mask_V = torch.ones(B, L, M)
    mask_attend = torch.ones(B, L, M, M)

    with torch.no_grad():
        torch_h_V = torch_module(h_V, h_E, mask_V, mask_attend)

    jax_h_V = jax_module(
        jnp.array(h_V.numpy()),
        jnp.array(h_E.numpy()),
        jnp.array(mask_V.numpy()),
        jnp.array(mask_attend.numpy()),
    )

    np.testing.assert_allclose(
        np.array(jax_h_V),
        torch_h_V.numpy(),
        atol=1e-5,
        err_msg="DecLayerJ h_V mismatch",
    )


def test_dec_layer_j_with_masking():
    """Test DecLayerJ with partial masking."""
    num_hidden = 128
    num_in = 128

    torch.manual_seed(42)
    torch_module = TorchDecLayerJ(num_hidden, num_in, dropout=0.0)
    torch_module.eval()

    jax_module = from_torch(torch_module)

    B, L, M = 2, 50, 25
    h_V = torch.randn(B, L, M, num_hidden)
    h_E = torch.randn(B, L, M, M, num_hidden)

    # Partial masks - simulate some ligand atoms being masked
    mask_V = torch.rand(B, L, M) > 0.3
    mask_V = mask_V.float()
    mask_attend = mask_V[:, :, :, None] * mask_V[:, :, None, :]

    with torch.no_grad():
        torch_h_V = torch_module(h_V, h_E, mask_V, mask_attend)

    jax_h_V = jax_module(
        jnp.array(h_V.numpy()),
        jnp.array(h_E.numpy()),
        jnp.array(mask_V.numpy()),
        jnp.array(mask_attend.numpy()),
    )

    np.testing.assert_allclose(
        np.array(jax_h_V),
        torch_h_V.numpy(),
        atol=1e-5,
        err_msg="DecLayerJ with masking h_V mismatch",
    )
