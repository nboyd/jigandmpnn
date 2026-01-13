"""Tests for ProteinMPNN.score() comparing JAX and PyTorch implementations."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special
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


def create_score_feature_dict(B: int, L: int, seed: int = 42):
    """Create synthetic protein data for score testing."""
    torch.manual_seed(seed)

    X = torch.randn(B, L, 4, 3) * 3.0
    S = torch.randint(0, 21, (B, L))
    mask = torch.ones(B, L)
    R_idx = torch.arange(L).unsqueeze(0).expand(B, -1)
    chain_labels = torch.zeros(B, L, dtype=torch.long)
    chain_mask = torch.ones(B, L)
    randn = torch.randn(B, L)

    return {
        "X": X,
        "S": S,
        "mask": mask,
        "R_idx": R_idx,
        "chain_labels": chain_labels,
        "chain_mask": chain_mask,
        "randn": randn,
        "symmetry_residues": [[]],
        "symmetry_weights": [[]],
        "batch_size": 1,
    }


def create_score_feature_dict_ligand(B: int, L: int, M: int = 25, seed: int = 42):
    """Create synthetic protein + ligand data for score testing."""
    feature_dict = create_score_feature_dict(B, L, seed)
    torch.manual_seed(seed + 1000)

    X = feature_dict["X"]
    Y = torch.randn(B, L, M, 3) * 2.0 + X[:, :, 1:2, :]
    Y_t = torch.randint(6, 9, (B, L, M))
    Y_m = torch.zeros(B, L, M)
    Y_m[:, :, :10] = 1.0

    feature_dict.update({"Y": Y, "Y_t": Y_t, "Y_m": Y_m})
    return feature_dict


def to_jax_score_kwargs(feature_dict, include_ligand=False):
    """Convert PyTorch feature dict to JAX kwargs for score()."""
    kwargs = {
        "X": jnp.array(feature_dict["X"].numpy()),
        "S": jnp.array(feature_dict["S"].numpy()),
        "mask": jnp.array(feature_dict["mask"].numpy()),
        "R_idx": jnp.array(feature_dict["R_idx"].numpy()),
        "chain_labels": jnp.array(feature_dict["chain_labels"].numpy()),
        "chain_mask": jnp.array(feature_dict["chain_mask"].numpy()),
        "decoding_order_noise": jnp.array(feature_dict["randn"].numpy()),
    }
    if include_ligand and "Y" in feature_dict:
        kwargs.update({
            "Y": jnp.array(feature_dict["Y"].numpy()),
            "Y_t": jnp.array(feature_dict["Y_t"].numpy()),
            "Y_m": jnp.array(feature_dict["Y_m"].numpy()),
        })
    return kwargs


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


def test_score_proteinmpnn_use_sequence():
    """Test ProteinMPNN.score() with use_sequence=True."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_score_feature_dict(B, L)

    with torch.no_grad():
        torch_result = torch_model.score(feature_dict, use_sequence=True)

    jax_result = jax_model.score(
        **to_jax_score_kwargs(feature_dict),
        key=jax.random.PRNGKey(0),
        use_sequence=True,
    )

    np.testing.assert_array_equal(
        np.array(jax_result.decoding_order),
        torch_result["decoding_order"].numpy(),
        err_msg="score() decoding_order mismatch",
    )

    np.testing.assert_allclose(
        np.array(jax_result.log_probs),
        torch_result["log_probs"].numpy(),
        atol=1e-4,
        err_msg="score() log_probs mismatch",
    )

    np.testing.assert_allclose(
        np.array(jax_result.logits),
        torch_result["logits"].numpy(),
        atol=1e-4,
        err_msg="score() logits mismatch",
    )


def test_score_proteinmpnn_no_sequence():
    """Test ProteinMPNN.score() with use_sequence=False."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_score_feature_dict(B, L)

    with torch.no_grad():
        torch_result = torch_model.score(feature_dict, use_sequence=False)

    jax_result = jax_model.score(
        **to_jax_score_kwargs(feature_dict),
        key=jax.random.PRNGKey(0),
        use_sequence=False,
    )

    np.testing.assert_allclose(
        np.array(jax_result.log_probs),
        torch_result["log_probs"].numpy(),
        atol=5e-4,
        err_msg="score(use_sequence=False) log_probs mismatch",
    )


def test_score_proteinmpnn_partial_mask():
    """Test ProteinMPNN.score() with partial chain_mask (some fixed positions)."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_score_feature_dict(B, L, seed=123)

    # Fix some positions
    chain_mask = torch.ones(B, L)
    chain_mask[:, :10] = 0.0
    chain_mask[:, 40:] = 0.0
    feature_dict["chain_mask"] = chain_mask

    with torch.no_grad():
        torch_result = torch_model.score(feature_dict, use_sequence=True)

    jax_result = jax_model.score(
        **to_jax_score_kwargs(feature_dict),
        key=jax.random.PRNGKey(0),
        use_sequence=True,
    )

    np.testing.assert_array_equal(
        np.array(jax_result.decoding_order),
        torch_result["decoding_order"].numpy(),
        err_msg="score() partial mask decoding_order mismatch",
    )

    np.testing.assert_allclose(
        np.array(jax_result.log_probs),
        torch_result["log_probs"].numpy(),
        atol=1e-4,
        err_msg="score() partial mask log_probs mismatch",
    )


def test_score_proteinmpnn_output_shapes():
    """Test that score() produces correct output shapes."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_score_feature_dict(B, L)

    result = jax_model.score(
        **to_jax_score_kwargs(feature_dict),
        key=jax.random.PRNGKey(0),
        use_sequence=True,
    )

    assert result.S.shape == (B, L), f"S shape: {result.S.shape}"
    assert result.log_probs.shape == (B, L, 21), f"log_probs shape: {result.log_probs.shape}"
    assert result.logits.shape == (B, L, 21), f"logits shape: {result.logits.shape}"
    assert result.decoding_order.shape == (B, L), f"decoding_order shape: {result.decoding_order.shape}"

    assert result.log_probs.dtype == jnp.float32
    assert result.logits.dtype == jnp.float32
    assert result.decoding_order.dtype == jnp.int32


def test_score_ligandmpnn_use_sequence():
    """Test LigandMPNN.score() with use_sequence=True."""
    torch_model = load_pretrained_ligandmpnn()
    jax_model = from_torch(torch_model)

    B, L, M = 2, 50, 25
    feature_dict = create_score_feature_dict_ligand(B, L, M)

    with torch.no_grad():
        torch_result = torch_model.score(feature_dict, use_sequence=True)

    jax_result = jax_model.score(
        **to_jax_score_kwargs(feature_dict, include_ligand=True),
        key=jax.random.PRNGKey(0),
        use_sequence=True,
    )

    np.testing.assert_array_equal(
        np.array(jax_result.decoding_order),
        torch_result["decoding_order"].numpy(),
        err_msg="ligand_mpnn score() decoding_order mismatch",
    )

    np.testing.assert_allclose(
        np.array(jax_result.log_probs),
        torch_result["log_probs"].numpy(),
        atol=1e-4,
        err_msg="ligand_mpnn score() log_probs mismatch",
    )


def test_score_proteinmpnn_jit():
    """Test ProteinMPNN.score() works correctly with eqx.filter_jit."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    score_jit = eqx.filter_jit(jax_model.score)

    B, L = 2, 50
    feature_dict = create_score_feature_dict(B, L)

    with torch.no_grad():
        torch_result = torch_model.score(feature_dict, use_sequence=True)

    jax_kwargs = to_jax_score_kwargs(feature_dict)

    jax_result = score_jit(**jax_kwargs, key=jax.random.PRNGKey(0), use_sequence=True)

    np.testing.assert_allclose(
        np.array(jax_result.log_probs),
        torch_result["log_probs"].numpy(),
        atol=1e-4,
        err_msg="JIT score() log_probs mismatch",
    )

    jax_result2 = score_jit(**jax_kwargs, key=jax.random.PRNGKey(0), use_sequence=True)

    np.testing.assert_allclose(
        np.array(jax_result.log_probs),
        np.array(jax_result2.log_probs),
        atol=1e-6,
        err_msg="JIT score() not deterministic",
    )


def test_score_log_probs_sum():
    """Test that log_probs sum to 0 (probabilities sum to 1) for each position."""
    torch_model = load_pretrained_proteinmpnn()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    feature_dict = create_score_feature_dict(B, L)

    result = jax_model.score(
        **to_jax_score_kwargs(feature_dict),
        key=jax.random.PRNGKey(0),
        use_sequence=True,
    )

    log_sum = jax.scipy.special.logsumexp(result.log_probs, axis=-1)
    np.testing.assert_allclose(
        np.array(log_sum),
        np.zeros((B, L)),
        atol=1e-5,
        err_msg="log_probs don't sum to 0 (probs don't sum to 1)",
    )
