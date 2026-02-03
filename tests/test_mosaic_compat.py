"""Tests for mosaic-compatible API: decode(), __call__(), from_pretrained(), unbatched encode()."""

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import pytest
import torch

from jigandmpnn import get_weight_path, load_protein_mpnn, load_mpnn, load_mpnn_sol
from jigandmpnn.backend import from_torch
from jigandmpnn.modules.model import ProteinMPNN

# Import PyTorch reference implementation
from jigandmpnn.vendor.ligandmpnn import ProteinMPNN as TorchProteinMPNN


CHECKPOINT_PATH = get_weight_path("protein_mpnn")


def load_pretrained_model():
    """Load pretrained ProteinMPNN model."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

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


def create_test_data(B, L, seed=42):
    """Create synthetic protein data."""
    torch.manual_seed(seed)
    X = torch.randn(B, L, 4, 3) * 3.0
    S = torch.randint(0, 21, (B, L))
    mask = torch.ones(B, L)
    R_idx = torch.arange(L).unsqueeze(0).expand(B, -1)
    chain_labels = torch.zeros(B, L, dtype=torch.long)
    return X, S, mask, R_idx, chain_labels


# --- Unbatched encode tests ---

def test_encode_unbatched():
    """Test that encode() works with unbatched (N, 4, 3) inputs."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    B, L = 1, 50
    X, S, mask, R_idx, chain_labels = create_test_data(B, L)

    # Batched reference
    h_V_b, h_E_b, E_idx_b = jax_model.encode(
        X=jnp.array(X.numpy()),
        mask=jnp.array(mask.numpy()),
        residue_idx=jnp.array(R_idx.numpy()),
        chain_encoding_all=jnp.array(chain_labels.numpy()),
    )

    # Unbatched (squeeze batch dim)
    h_V_u, h_E_u, E_idx_u = jax_model.encode(
        X=jnp.array(X[0].numpy()),
        mask=jnp.array(mask[0].numpy()),
        residue_idx=jnp.array(R_idx[0].numpy()),
        chain_encoding_all=jnp.array(chain_labels[0].numpy()),
    )

    # Unbatched should add batch dim, giving same result
    assert h_V_u.shape == (1, L, 128)
    np.testing.assert_allclose(
        np.array(h_V_b), np.array(h_V_u), atol=1e-5,
        err_msg="Unbatched encode should match batched",
    )
    np.testing.assert_array_equal(
        np.array(E_idx_b), np.array(E_idx_u),
        err_msg="Unbatched E_idx should match batched",
    )


# --- decode() tests ---

def test_decode_matches_score():
    """Test that decode() produces same log_probs as score() with teacher forcing."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    B, L = 2, 50
    X, S, mask, R_idx, chain_labels = create_test_data(B, L)

    jax_X = jnp.array(X.numpy())
    jax_S = jnp.array(S.numpy())
    jax_mask = jnp.array(mask.numpy())
    jax_R = jnp.array(R_idx.numpy())
    jax_C = jnp.array(chain_labels.numpy())

    # Use score() as reference
    randn = jnp.array(torch.randn(B, L).numpy())
    score_result = jax_model.score(
        X=jax_X, S=jax_S, mask=jax_mask,
        residue_idx=jax_R, chain_encoding_all=jax_C,
        key=jax.random.PRNGKey(0),
        decoding_order_noise=randn,
        use_sequence=True,
    )

    # Now use encode + decode
    h_V, h_E, E_idx = jax_model.encode(
        X=jax_X, mask=jax_mask, residue_idx=jax_R, chain_encoding_all=jax_C,
    )

    # Convert integer S to one-hot for decode
    S_onehot = jnn.one_hot(jax_S, 21)

    # decode() expects float noise that it will argsort. score() computes
    # decoding_order from chain_mask and noise. With chain_mask=1 everywhere:
    #   decoding_order = argsort((1 + 0.0001) * abs(randn))
    # So we pass (1.0001) * abs(randn) as decoding_order to decode()
    decode_order_noise = (1.0 + 0.0001) * jnp.abs(randn)

    log_probs = jax_model.decode(
        S=S_onehot, h_V=h_V, h_E=h_E, E_idx=E_idx,
        mask=jax_mask, decoding_order=decode_order_noise,
    )

    assert log_probs.shape == (B, L, 21)
    np.testing.assert_allclose(
        np.array(log_probs),
        np.array(score_result.log_probs),
        atol=1e-5,
        err_msg="decode() should match score() log_probs",
    )


def test_decode_unbatched():
    """Test that decode() works with unbatched inputs."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    B, L = 1, 30
    X, S, mask, R_idx, chain_labels = create_test_data(B, L)

    jax_X = jnp.array(X.numpy())
    jax_mask = jnp.array(mask.numpy())
    jax_R = jnp.array(R_idx.numpy())
    jax_C = jnp.array(chain_labels.numpy())

    h_V, h_E, E_idx = jax_model.encode(
        X=jax_X, mask=jax_mask, residue_idx=jax_R, chain_encoding_all=jax_C,
    )

    S_onehot = jnn.one_hot(jnp.array(S[0].numpy()), 21)  # (L, 21) unbatched
    order = jnp.array(torch.randn(L).numpy())  # (L,) unbatched

    log_probs = jax_model.decode(
        S=S_onehot, h_V=h_V, h_E=h_E, E_idx=E_idx,
        mask=jax_mask[0], decoding_order=order,
    )

    assert log_probs.shape == (1, L, 21)
    # Check it's valid log probs
    log_sum = jax.scipy.special.logsumexp(log_probs, axis=-1)
    np.testing.assert_allclose(
        np.array(log_sum), np.zeros((1, L)), atol=1e-5,
        err_msg="decode() log_probs should sum to 0",
    )


def test_decode_output_shapes():
    """Test decode() output shapes."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    B, L = 2, 40
    X, S, mask, R_idx, chain_labels = create_test_data(B, L)

    jax_X = jnp.array(X.numpy())
    jax_mask = jnp.array(mask.numpy())
    jax_R = jnp.array(R_idx.numpy())
    jax_C = jnp.array(chain_labels.numpy())

    h_V, h_E, E_idx = jax_model.encode(
        X=jax_X, mask=jax_mask, residue_idx=jax_R, chain_encoding_all=jax_C,
    )

    S_onehot = jnn.one_hot(jnp.array(S.numpy()), 21)
    order = jnp.array(torch.randn(B, L).numpy())

    log_probs = jax_model.decode(
        S=S_onehot, h_V=h_V, h_E=h_E, E_idx=E_idx,
        mask=jax_mask, decoding_order=order,
    )

    assert log_probs.shape == (B, L, 21)
    assert log_probs.dtype == jnp.float32


# --- __call__() tests ---

def test_call_matches_encode_decode():
    """Test that __call__ produces same results as encode + decode."""
    torch_model = load_pretrained_model()
    jax_model = from_torch(torch_model)

    B, L = 2, 40
    X, S, mask, R_idx, chain_labels = create_test_data(B, L)

    jax_X = jnp.array(X.numpy())
    jax_mask = jnp.array(mask.numpy())
    jax_R = jnp.array(R_idx.numpy())
    jax_C = jnp.array(chain_labels.numpy())
    S_onehot = jnn.one_hot(jnp.array(S.numpy()), 21)
    order = jnp.array(torch.randn(B, L).numpy())

    # Via __call__
    log_probs_call = jax_model(
        jax_X, S_onehot, jax_mask, jax_R, jax_C, order,
    )

    # Via encode + decode
    h_V, h_E, E_idx = jax_model.encode(
        X=jax_X, mask=jax_mask, residue_idx=jax_R, chain_encoding_all=jax_C,
    )
    log_probs_manual = jax_model.decode(
        S=S_onehot, h_V=h_V, h_E=h_E, E_idx=E_idx,
        mask=jax_mask, decoding_order=order,
    )

    np.testing.assert_allclose(
        np.array(log_probs_call),
        np.array(log_probs_manual),
        atol=1e-6,
        err_msg="__call__ should match encode+decode",
    )


# --- from_pretrained() tests ---

def test_from_pretrained_default():
    """Test from_pretrained() with default weights."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    model = ProteinMPNN.from_pretrained()
    assert model.model_type == "protein_mpnn"
    assert model.hidden_dim == 128


def test_from_pretrained_with_path():
    """Test from_pretrained() with explicit checkpoint path."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    model = ProteinMPNN.from_pretrained(checkpoint_path=str(CHECKPOINT_PATH))
    assert model.model_type == "protein_mpnn"
    assert model.hidden_dim == 128


def test_from_pretrained_matches_load_function():
    """Test that from_pretrained() produces same model as load_protein_mpnn()."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    model_fp = ProteinMPNN.from_pretrained()
    model_lp = load_protein_mpnn()

    B, L = 1, 30
    X, S, mask, R_idx, chain_labels = create_test_data(B, L)
    jax_X = jnp.array(X.numpy())
    jax_mask = jnp.array(mask.numpy())
    jax_R = jnp.array(R_idx.numpy())
    jax_C = jnp.array(chain_labels.numpy())

    h_V_fp, _, _ = model_fp.encode(
        X=jax_X, mask=jax_mask, residue_idx=jax_R, chain_encoding_all=jax_C,
    )
    h_V_lp, _, _ = model_lp.encode(
        X=jax_X, mask=jax_mask, residue_idx=jax_R, chain_encoding_all=jax_C,
    )
    np.testing.assert_array_equal(
        np.array(h_V_fp), np.array(h_V_lp),
        err_msg="from_pretrained should match load_protein_mpnn",
    )


# --- Alias tests ---

def test_load_aliases():
    """Test that load_mpnn and load_mpnn_sol are valid aliases."""
    assert load_mpnn is load_protein_mpnn
    from jigandmpnn import load_soluble_mpnn
    assert load_mpnn_sol is load_soluble_mpnn
