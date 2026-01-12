"""Tests for ProteinMPNN.sample() autoregressive decoding."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jigandmpnn import get_weight_path
from jigandmpnn.backend import from_torch
from jigandmpnn.modules.model import ProteinMPNN

# Import PyTorch reference implementation
from jigandmpnn.vendor.ligandmpnn import ProteinMPNN as TorchProteinMPNN


# Path to pretrained checkpoints
PROTEINMPNN_CHECKPOINT_PATH = get_weight_path("protein_mpnn")
LIGANDMPNN_CHECKPOINT_PATH = get_weight_path("ligand_mpnn")


def create_sample_feature_dict(B: int, L: int, seed: int = 42):
    """Create synthetic protein data for sampling.

    Args:
        B: Batch size
        L: Sequence length
        seed: Random seed

    Returns:
        Dictionary with protein features for sampling
    """
    torch.manual_seed(seed)

    # Generate random backbone coordinates
    X = torch.randn(B, L, 4, 3) * 3.0

    # Random sequence (0-20)
    S = torch.randint(0, 21, (B, L))

    # Create mask (all valid)
    mask = torch.ones(B, L)

    # Residue indices (sequential)
    R_idx = torch.arange(L).unsqueeze(0).expand(B, -1)

    # Chain labels (single chain)
    chain_labels = torch.zeros(B, L, dtype=torch.long)

    # Chain mask (all positions to be designed)
    chain_mask = torch.ones(B, L)

    # Random numbers for decoding order
    randn = torch.randn(B, L)

    # Amino acid bias (no bias)
    bias = torch.zeros(B, L, 21)

    return {
        "X": X,
        "S": S,
        "mask": mask,
        "R_idx": R_idx,
        "chain_labels": chain_labels,
        "chain_mask": chain_mask,
        "randn": randn,
        "bias": bias,
        "temperature": 1.0,
        "symmetry_residues": [[]],
        "symmetry_weights": [[]],
        "batch_size": 1,
    }


def create_sample_feature_dict_ligand(B: int, L: int, M: int = 25, seed: int = 42):
    """Create synthetic protein + ligand data for sampling."""
    feature_dict = create_sample_feature_dict(B, L, seed)

    torch.manual_seed(seed + 1000)

    X = feature_dict["X"]
    Y = torch.randn(B, L, M, 3) * 2.0 + X[:, :, 1:2, :]
    Y_t = torch.randint(6, 9, (B, L, M))
    Y_m = torch.zeros(B, L, M)
    Y_m[:, :, :10] = 1.0

    feature_dict.update({
        "Y": Y,
        "Y_t": Y_t,
        "Y_m": Y_m,
    })

    return feature_dict


def load_pretrained_proteinmpnn():
    """Load pretrained ProteinMPNN model."""
    if not PROTEINMPNN_CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {PROTEINMPNN_CHECKPOINT_PATH}")

    checkpoint = torch.load(PROTEINMPNN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
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


def load_pretrained_ligandmpnn():
    """Load pretrained LigandMPNN model."""
    if not LIGANDMPNN_CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {LIGANDMPNN_CHECKPOINT_PATH}")

    checkpoint = torch.load(LIGANDMPNN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
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


def test_sample_proteinmpnn_output_shapes():
    """Test that sample() produces correct output shapes."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_sample_feature_dict(B, L)

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    key = jax.random.PRNGKey(42)
    result = jax_model.sample(feature_dict_jax, key=key)

    # Check shapes
    assert result["S"].shape == (B, L), f"S shape: {result['S'].shape}"
    assert result["sampling_probs"].shape == (B, L, 20), f"sampling_probs shape: {result['sampling_probs'].shape}"
    assert result["log_probs"].shape == (B, L, 21), f"log_probs shape: {result['log_probs'].shape}"
    assert result["decoding_order"].shape == (B, L), f"decoding_order shape: {result['decoding_order'].shape}"

    # Check types
    assert result["S"].dtype == jnp.int32
    assert result["sampling_probs"].dtype == jnp.float32
    assert result["log_probs"].dtype == jnp.float32


def test_sample_proteinmpnn_deterministic():
    """Test that same key produces same results."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_sample_feature_dict(B, L)

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    key = jax.random.PRNGKey(42)

    # Same key should produce same results
    result1 = jax_model.sample(feature_dict_jax, key=key)
    result2 = jax_model.sample(feature_dict_jax, key=key)

    np.testing.assert_array_equal(
        np.array(result1["S"]),
        np.array(result2["S"]),
        err_msg="Same key should produce same S",
    )

    np.testing.assert_allclose(
        np.array(result1["sampling_probs"]),
        np.array(result2["sampling_probs"]),
        atol=1e-6,
        err_msg="Same key should produce same sampling_probs",
    )


def test_sample_proteinmpnn_different_keys():
    """Test that different keys produce different results."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_sample_feature_dict(B, L)

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(123)

    result1 = jax_model.sample(feature_dict_jax, key=key1)
    result2 = jax_model.sample(feature_dict_jax, key=key2)

    # Different keys should (very likely) produce different sequences
    # Note: decoding_order should be the same since it's determined by randn
    np.testing.assert_array_equal(
        np.array(result1["decoding_order"]),
        np.array(result2["decoding_order"]),
        err_msg="Decoding order should be deterministic",
    )

    # Sequences should differ (with high probability)
    assert not np.array_equal(
        np.array(result1["S"]),
        np.array(result2["S"]),
    ), "Different keys should produce different sequences"


def test_sample_proteinmpnn_fixed_positions():
    """Test that fixed positions use true sequence."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_sample_feature_dict(B, L, seed=123)

    # Fix some positions
    chain_mask = torch.ones(B, L)
    chain_mask[:, :10] = 0.0  # First 10 positions fixed
    chain_mask[:, 40:] = 0.0  # Last 10 positions fixed
    feature_dict["chain_mask"] = chain_mask

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    key = jax.random.PRNGKey(42)
    result = jax_model.sample(feature_dict_jax, key=key)

    S_true = feature_dict["S"].numpy()
    S_sampled = np.array(result["S"])

    # Fixed positions should match true sequence
    np.testing.assert_array_equal(
        S_sampled[:, :10],
        S_true[:, :10],
        err_msg="First 10 fixed positions should match true sequence",
    )

    np.testing.assert_array_equal(
        S_sampled[:, 40:],
        S_true[:, 40:],
        err_msg="Last 10 fixed positions should match true sequence",
    )


def test_sample_proteinmpnn_probs_sum_to_one():
    """Test that sampling probabilities sum to approximately 1."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_sample_feature_dict(B, L)

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    key = jax.random.PRNGKey(42)
    result = jax_model.sample(feature_dict_jax, key=key)

    # Sampling probs should sum to ~1 for each position (excluding X)
    prob_sums = np.sum(np.array(result["sampling_probs"]), axis=-1)
    np.testing.assert_allclose(
        prob_sums,
        np.ones((B, L)),
        atol=1e-5,
        err_msg="Sampling probs should sum to 1",
    )


def test_sample_proteinmpnn_valid_amino_acids():
    """Test that sampled amino acids are valid (0-19)."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_sample_feature_dict(B, L)

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    key = jax.random.PRNGKey(42)
    result = jax_model.sample(feature_dict_jax, key=key)

    S = np.array(result["S"])
    assert np.all(S >= 0), "Amino acids should be >= 0"
    assert np.all(S <= 20), "Amino acids should be <= 20"


def test_sample_proteinmpnn_temperature():
    """Test that temperature affects sampling diversity."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 30
    feature_dict = create_sample_feature_dict(B, L)

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    # Low temperature should produce more confident predictions
    key = jax.random.PRNGKey(42)
    result_low_temp = jax_model.sample(feature_dict_jax, key=key, temperature=0.1)

    # High temperature should produce less confident predictions
    result_high_temp = jax_model.sample(feature_dict_jax, key=key, temperature=2.0)

    # Max probability should be higher for low temperature
    max_prob_low = np.max(np.array(result_low_temp["sampling_probs"]), axis=-1)
    max_prob_high = np.max(np.array(result_high_temp["sampling_probs"]), axis=-1)

    assert np.mean(max_prob_low) > np.mean(max_prob_high), \
        "Low temperature should produce more confident predictions"


def test_sample_proteinmpnn_decoding_order_matches_torch():
    """Test that decoding order matches PyTorch implementation."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    # Use B=1 for PyTorch comparison since PyTorch's batch_size parameter
    # is for generating multiple samples from a single structure
    B, L = 1, 50
    feature_dict = create_sample_feature_dict(B, L)

    # Run PyTorch to get decoding order
    with torch.no_grad():
        torch_result = torch_model.sample(feature_dict)

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    key = jax.random.PRNGKey(42)
    jax_result = jax_model.sample(feature_dict_jax, key=key)

    # Decoding order should match (it's deterministic based on randn)
    np.testing.assert_array_equal(
        np.array(jax_result["decoding_order"]),
        torch_result["decoding_order"].numpy(),
        err_msg="Decoding order should match PyTorch",
    )


def test_sample_ligandmpnn_output_shapes():
    """Test that LigandMPNN sample() produces correct output shapes."""
    torch_model = load_pretrained_ligandmpnn()
    jax_model = from_torch(torch_model)

    B, L, M = 2, 50, 25
    feature_dict = create_sample_feature_dict_ligand(B, L, M)

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    key = jax.random.PRNGKey(42)
    result = jax_model.sample(feature_dict_jax, key=key)

    # Check shapes
    assert result["S"].shape == (B, L), f"S shape: {result['S'].shape}"
    assert result["sampling_probs"].shape == (B, L, 20)
    assert result["log_probs"].shape == (B, L, 21)
    assert result["decoding_order"].shape == (B, L)


def test_sample_proteinmpnn_jit():
    """Test that sample() works correctly with eqx.filter_jit."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    # JIT compile the sample method
    sample_jit = eqx.filter_jit(jax_model.sample)

    B, L = 2, 50
    feature_dict = create_sample_feature_dict(B, L)

    feature_dict_jax = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict_jax[k] = jnp.array(v.numpy())
        else:
            feature_dict_jax[k] = v

    key = jax.random.PRNGKey(42)

    # First call triggers compilation
    result1 = sample_jit(feature_dict_jax, key=key)

    # Second call uses cached compilation
    result2 = sample_jit(feature_dict_jax, key=key)

    # Results should be identical
    np.testing.assert_array_equal(
        np.array(result1["S"]),
        np.array(result2["S"]),
        err_msg="JIT sample() should be deterministic",
    )

    np.testing.assert_allclose(
        np.array(result1["sampling_probs"]),
        np.array(result2["sampling_probs"]),
        atol=1e-6,
        err_msg="JIT sample() probs should match",
    )
