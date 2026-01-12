"""Tests for LigandMPNN.encode() comparing JAX and PyTorch implementations."""

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
CHECKPOINT_PATH = get_weight_path("ligand_mpnn")


def create_synthetic_ligand_data(B: int, L: int, M: int = 25, seed: int = 42):
    """Create synthetic protein + ligand data for testing.

    Args:
        B: Batch size
        L: Sequence length
        M: Number of ligand atoms per residue
        seed: Random seed

    Returns:
        Dictionary with protein and ligand features
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

    # Ligand coordinates [B, L, M, 3]
    # Place ligand atoms near the backbone
    Y = torch.randn(B, L, M, 3) * 2.0 + X[:, :, 1:2, :]  # Near Ca

    # Ligand atom types [B, L, M] - use common elements (C=6, N=7, O=8)
    Y_t = torch.randint(6, 9, (B, L, M))

    # Ligand mask - make only first few atoms valid
    Y_m = torch.zeros(B, L, M)
    Y_m[:, :, :10] = 1.0  # First 10 atoms per residue are valid

    return {
        "X": X,
        "S": S,
        "mask": mask,
        "R_idx": R_idx,
        "chain_labels": chain_labels,
        "Y": Y,
        "Y_t": Y_t,
        "Y_m": Y_m,
    }


def load_pretrained_ligand_model():
    """Load pretrained LigandMPNN model from checkpoint.

    Returns:
        PyTorch model with pretrained weights
    """
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}. Run get_model_params.sh first.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    k_neighbors = checkpoint["num_edges"]
    atom_context_num = checkpoint.get("atom_context_num", 25)

    model = TorchProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        atom_context_num=atom_context_num,
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=0,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def test_ligand_encode_pretrained():
    """Test LigandMPNN.encode() with pretrained weights."""
    # Load pretrained PyTorch model
    torch_model = load_pretrained_ligand_model()

    # Convert to JAX
    jax_model = from_torch(torch_model)

    # Create test data
    B, L, M = 2, 50, 25
    feature_dict = create_synthetic_ligand_data(B, L, M)

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
        err_msg="ligand_mpnn encode() E_idx mismatch",
    )

    # Compare h_V and h_E
    np.testing.assert_allclose(
        np.array(h_V_jax),
        h_V_torch.numpy(),
        atol=1e-4,  # Multiple layers accumulate error
        err_msg="ligand_mpnn encode() h_V mismatch",
    )

    np.testing.assert_allclose(
        np.array(h_E_jax),
        h_E_torch.numpy(),
        atol=1e-4,
        err_msg="ligand_mpnn encode() h_E mismatch",
    )


def test_ligand_encode_pretrained_output_shapes():
    """Test that ligand_mpnn encode() produces correct output shapes."""
    torch_model = load_pretrained_ligand_model()
    jax_model = from_torch(torch_model)

    hidden_dim = 128
    k_neighbors = 32  # From checkpoint

    B, L, M = 2, 50, 25
    feature_dict = create_synthetic_ligand_data(B, L, M)
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


def test_ligand_encode_different_lengths():
    """Test ligand_mpnn encode() with different sequence lengths."""
    torch_model = load_pretrained_ligand_model()
    jax_model = from_torch(torch_model)

    # Test with different lengths
    for L in [30, 100]:
        B, M = 2, 25
        feature_dict = create_synthetic_ligand_data(B, L, M, seed=L)

        with torch.no_grad():
            h_V_torch, h_E_torch, E_idx_torch = torch_model.encode(feature_dict)

        feature_dict_jax = {k: jnp.array(v.numpy()) for k, v in feature_dict.items()}
        h_V_jax, h_E_jax, E_idx_jax = jax_model.encode(feature_dict_jax)

        np.testing.assert_array_equal(
            np.array(E_idx_jax),
            E_idx_torch.numpy(),
            err_msg=f"ligand_mpnn encode() E_idx mismatch for L={L}",
        )

        np.testing.assert_allclose(
            np.array(h_V_jax),
            h_V_torch.numpy(),
            atol=1e-4,
            err_msg=f"ligand_mpnn encode() h_V mismatch for L={L}",
        )


def test_ligand_encode_varying_ligand_atoms():
    """Test ligand_mpnn encode() with varying numbers of valid ligand atoms."""
    torch_model = load_pretrained_ligand_model()
    jax_model = from_torch(torch_model)

    B, L, M = 2, 50, 25
    feature_dict = create_synthetic_ligand_data(B, L, M, seed=789)

    # Vary the number of valid atoms per residue
    Y_m = torch.zeros(B, L, M)
    Y_m[0, :, :5] = 1.0   # First batch: 5 atoms
    Y_m[1, :, :15] = 1.0  # Second batch: 15 atoms
    feature_dict["Y_m"] = Y_m

    with torch.no_grad():
        h_V_torch, h_E_torch, E_idx_torch = torch_model.encode(feature_dict)

    feature_dict_jax = {k: jnp.array(v.numpy()) for k, v in feature_dict.items()}
    h_V_jax, h_E_jax, E_idx_jax = jax_model.encode(feature_dict_jax)

    np.testing.assert_allclose(
        np.array(h_V_jax),
        h_V_torch.numpy(),
        atol=1e-4,
        err_msg="ligand_mpnn encode() h_V mismatch with varying ligand atoms",
    )


def test_ligand_encode_multi_chain():
    """Test ligand_mpnn encode() with multi-chain protein."""
    torch_model = load_pretrained_ligand_model()
    jax_model = from_torch(torch_model)

    B, L, M = 2, 60, 25
    feature_dict = create_synthetic_ligand_data(B, L, M, seed=123)

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
        err_msg="ligand_mpnn encode() multi-chain E_idx mismatch",
    )

    np.testing.assert_allclose(
        np.array(h_V_jax),
        h_V_torch.numpy(),
        atol=1e-4,
        err_msg="ligand_mpnn encode() multi-chain h_V mismatch",
    )


def test_ligand_encode_pretrained_jit():
    """Test LigandMPNN.encode() works correctly with eqx.filter_jit."""
    torch_model = load_pretrained_ligand_model()
    jax_model = from_torch(torch_model)

    # JIT compile the encode method
    encode_jit = eqx.filter_jit(jax_model.encode)

    B, L, M = 2, 50, 25
    feature_dict = create_synthetic_ligand_data(B, L, M)

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
        err_msg="JIT ligand_mpnn encode() E_idx mismatch",
    )

    np.testing.assert_allclose(
        np.array(h_V_jax),
        h_V_torch.numpy(),
        atol=1e-4,
        err_msg="JIT ligand_mpnn encode() h_V mismatch",
    )

    np.testing.assert_allclose(
        np.array(h_E_jax),
        h_E_torch.numpy(),
        atol=1e-4,
        err_msg="JIT ligand_mpnn encode() h_E mismatch",
    )

    # Second call uses cached compilation - should give same results
    h_V_jax2, h_E_jax2, E_idx_jax2 = encode_jit(feature_dict_jax)

    np.testing.assert_array_equal(
        np.array(E_idx_jax),
        np.array(E_idx_jax2),
        err_msg="JIT ligand_mpnn encode() E_idx not deterministic",
    )

    np.testing.assert_allclose(
        np.array(h_V_jax),
        np.array(h_V_jax2),
        atol=1e-6,
        err_msg="JIT ligand_mpnn encode() h_V not deterministic",
    )
