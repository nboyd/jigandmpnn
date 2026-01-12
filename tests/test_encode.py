"""Tests for ProteinMPNN.encode() comparing JAX and PyTorch implementations."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jigandmpnn import get_weight_path
from jigandmpnn.backend import from_torch
from jigandmpnn.modules.model import ProteinMPNN

# Import PyTorch reference implementation
from jigandmpnn.vendor.ligandmpnn import ProteinMPNN as TorchProteinMPNN


# Path to pretrained checkpoint
CHECKPOINT_PATH = get_weight_path("protein_mpnn")


def create_synthetic_protein(B: int, L: int, seed: int = 42):
    """Create synthetic protein data for testing.

    Args:
        B: Batch size
        L: Sequence length
        seed: Random seed

    Returns:
        Dictionary with protein features
    """
    torch.manual_seed(seed)

    # Generate random backbone coordinates
    X = torch.randn(B, L, 4, 3) * 3.0  # [B, L, 4, 3] for N, CA, C, O

    # Random sequence (0-20)
    S = torch.randint(0, 21, (B, L))

    # Create mask (all valid)
    mask = torch.ones(B, L)

    # Residue indices (sequential)
    R_idx = torch.arange(L).unsqueeze(0).expand(B, -1)

    # Chain labels (single chain)
    chain_labels = torch.zeros(B, L, dtype=torch.long)

    return {
        "X": X,
        "S": S,
        "mask": mask,
        "R_idx": R_idx,
        "chain_labels": chain_labels,
    }


def load_pretrained_model():
    """Load pretrained ProteinMPNN model from checkpoint.

    Returns:
        PyTorch model with pretrained weights
    """
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}. Run get_model_params.sh first.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    k_neighbors = checkpoint["num_edges"]

    model = TorchProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        atom_context_num=1,
        model_type="protein_mpnn",
        ligand_mpnn_use_side_chain_context=0,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def test_encode_pretrained():
    """Test ProteinMPNN.encode() with pretrained weights."""
    # Load pretrained PyTorch model
    torch_model = load_pretrained_model()

    # Convert to JAX
    jax_model = from_torch(torch_model)

    # Create test data
    B, L = 2, 50
    feature_dict = create_synthetic_protein(B, L)

    # Run PyTorch
    with torch.no_grad():
        h_V_torch, h_E_torch, E_idx_torch = torch_model.encode(feature_dict)

    # Convert inputs to JAX
    feature_dict_jax = {k: jnp.array(v.numpy()) for k, v in feature_dict.items()}

    # Run JAX
    h_V_jax, h_E_jax, E_idx_jax = jax_model.encode(feature_dict_jax)

    # Compare E_idx (should be identical)
    np.testing.assert_array_equal(
        np.array(E_idx_jax),
        E_idx_torch.numpy(),
        err_msg="encode() E_idx mismatch",
    )

    # Compare h_V and h_E
    np.testing.assert_allclose(
        np.array(h_V_jax),
        h_V_torch.numpy(),
        atol=1e-4,  # Multiple layers accumulate error
        err_msg="encode() h_V mismatch",
    )

    np.testing.assert_allclose(
        np.array(h_E_jax),
        h_E_torch.numpy(),
        atol=1e-4,
        err_msg="encode() h_E mismatch",
    )


def test_encode_pretrained_different_lengths():
    """Test encode() with pretrained weights on different sequence lengths."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    # Test with different lengths
    for L in [30, 100, 200]:
        B = 2
        feature_dict = create_synthetic_protein(B, L, seed=L)  # Different seed per length

        with torch.no_grad():
            h_V_torch, h_E_torch, E_idx_torch = torch_model.encode(feature_dict)

        feature_dict_jax = {k: jnp.array(v.numpy()) for k, v in feature_dict.items()}
        h_V_jax, h_E_jax, E_idx_jax = jax_model.encode(feature_dict_jax)

        np.testing.assert_array_equal(
            np.array(E_idx_jax),
            E_idx_torch.numpy(),
            err_msg=f"encode() E_idx mismatch for L={L}",
        )

        np.testing.assert_allclose(
            np.array(h_V_jax),
            h_V_torch.numpy(),
            atol=1e-4,
            err_msg=f"encode() h_V mismatch for L={L}",
        )


def test_encode_pretrained_multi_chain():
    """Test encode() with pretrained weights on multi-chain protein."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    B, L = 2, 60
    feature_dict = create_synthetic_protein(B, L, seed=123)

    # Make it multi-chain
    chain_labels = torch.zeros(B, L, dtype=torch.long)
    chain_labels[:, 30:] = 1
    feature_dict["chain_labels"] = chain_labels

    # Reset R_idx per chain
    R_idx = torch.cat([torch.arange(30), torch.arange(30)]).unsqueeze(0).expand(B, -1)
    feature_dict["R_idx"] = R_idx

    with torch.no_grad():
        h_V_torch, h_E_torch, E_idx_torch = torch_model.encode(feature_dict)

    feature_dict_jax = {k: jnp.array(v.numpy()) for k, v in feature_dict.items()}
    h_V_jax, h_E_jax, E_idx_jax = jax_model.encode(feature_dict_jax)

    np.testing.assert_array_equal(
        np.array(E_idx_jax),
        E_idx_torch.numpy(),
        err_msg="encode() multi-chain E_idx mismatch",
    )

    np.testing.assert_allclose(
        np.array(h_V_jax),
        h_V_torch.numpy(),
        atol=1e-4,
        err_msg="encode() multi-chain h_V mismatch",
    )


def test_encode_pretrained_output_shapes():
    """Test that encode() produces correct output shapes with pretrained model."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    hidden_dim = 128
    k_neighbors = 48  # From checkpoint

    B, L = 2, 50
    feature_dict = create_synthetic_protein(B, L)
    feature_dict_jax = {k: jnp.array(v.numpy()) for k, v in feature_dict.items()}

    h_V, h_E, E_idx = jax_model.encode(feature_dict_jax)

    # Check shapes
    assert h_V.shape == (B, L, hidden_dim), f"h_V shape: {h_V.shape}"
    assert h_E.shape == (B, L, k_neighbors, hidden_dim), f"h_E shape: {h_E.shape}"
    assert E_idx.shape == (B, L, k_neighbors), f"E_idx shape: {E_idx.shape}"

    # Check types
    assert h_V.dtype == jnp.float32
    assert h_E.dtype == jnp.float32
    assert E_idx.dtype == jnp.int32


def test_encode_pretrained_batch_consistency():
    """Test that encode() produces consistent results across batches."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    # Create single protein
    feature_dict_single = create_synthetic_protein(B=1, L=50, seed=42)

    # Create batch of 2 with same protein duplicated
    feature_dict_batch = {
        k: torch.cat([v, v], dim=0) for k, v in feature_dict_single.items()
    }

    # Run JAX on both
    feature_dict_single_jax = {k: jnp.array(v.numpy()) for k, v in feature_dict_single.items()}
    feature_dict_batch_jax = {k: jnp.array(v.numpy()) for k, v in feature_dict_batch.items()}

    h_V_single, h_E_single, E_idx_single = jax_model.encode(feature_dict_single_jax)
    h_V_batch, h_E_batch, E_idx_batch = jax_model.encode(feature_dict_batch_jax)

    # First element of batch should match single (small numerical differences from batching)
    np.testing.assert_allclose(
        np.array(h_V_single[0]),
        np.array(h_V_batch[0]),
        atol=1e-5,
        err_msg="Batch inconsistency in h_V[0]",
    )

    # Second element should also match (same input)
    np.testing.assert_allclose(
        np.array(h_V_batch[0]),
        np.array(h_V_batch[1]),
        atol=1e-6,
        err_msg="Batch inconsistency between h_V[0] and h_V[1]",
    )


def test_encode_pretrained_jit():
    """Test ProteinMPNN.encode() works correctly with eqx.filter_jit."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    # JIT compile the encode method
    encode_jit = eqx.filter_jit(jax_model.encode)

    B, L = 2, 50
    feature_dict = create_synthetic_protein(B, L)

    # Run PyTorch for reference
    with torch.no_grad():
        h_V_torch, h_E_torch, E_idx_torch = torch_model.encode(feature_dict)

    feature_dict_jax = {k: jnp.array(v.numpy()) for k, v in feature_dict.items()}

    # First call triggers compilation
    h_V_jax, h_E_jax, E_idx_jax = encode_jit(feature_dict_jax)

    # Verify JIT results match PyTorch
    np.testing.assert_array_equal(
        np.array(E_idx_jax),
        E_idx_torch.numpy(),
        err_msg="JIT encode() E_idx mismatch",
    )

    np.testing.assert_allclose(
        np.array(h_V_jax),
        h_V_torch.numpy(),
        atol=1e-4,
        err_msg="JIT encode() h_V mismatch",
    )

    np.testing.assert_allclose(
        np.array(h_E_jax),
        h_E_torch.numpy(),
        atol=1e-4,
        err_msg="JIT encode() h_E mismatch",
    )

    # Second call uses cached compilation - should give same results
    h_V_jax2, h_E_jax2, E_idx_jax2 = encode_jit(feature_dict_jax)

    np.testing.assert_array_equal(
        np.array(E_idx_jax),
        np.array(E_idx_jax2),
        err_msg="JIT encode() E_idx not deterministic",
    )

    np.testing.assert_allclose(
        np.array(h_V_jax),
        np.array(h_V_jax2),
        atol=1e-6,
        err_msg="JIT encode() h_V not deterministic",
    )
