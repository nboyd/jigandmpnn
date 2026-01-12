"""Tests for augment_eps coordinate noise feature."""

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
import torch

from jigandmpnn.backend import from_torch
from jigandmpnn.modules.features import ProteinFeatures, ProteinFeaturesLigand

# Import PyTorch reference implementations
from jigandmpnn.vendor.ligandmpnn import (
    ProteinFeatures as TorchProteinFeatures,
    ProteinFeaturesLigand as TorchProteinFeaturesLigand,
)


def create_test_protein_features(B: int, L: int, seed: int = 42):
    """Create test protein features."""
    torch.manual_seed(seed)
    X = torch.randn(B, L, 4, 3) * 3.0
    mask = torch.ones(B, L)
    R_idx = torch.arange(L).unsqueeze(0).expand(B, -1)
    chain_labels = torch.zeros(B, L, dtype=torch.long)
    return {"X": X, "mask": mask, "R_idx": R_idx, "chain_labels": chain_labels}


def create_test_ligand_features(B: int, L: int, M: int = 25, seed: int = 42):
    """Create test ligand features."""
    features = create_test_protein_features(B, L, seed)
    torch.manual_seed(seed + 1000)
    X = features["X"]
    Y = torch.randn(B, L, M, 3) * 2.0 + X[:, :, 1:2, :]
    Y_t = torch.randint(6, 9, (B, L, M))
    Y_m = torch.zeros(B, L, M)
    Y_m[:, :, :10] = 1.0
    features.update({"Y": Y, "Y_t": Y_t, "Y_m": Y_m})
    return features


def test_protein_features_augment_eps_no_key():
    """Test that ProteinFeatures without key produces same results as augment_eps=0."""
    edge_features = 128
    node_features = 128
    top_k = 48
    augment_eps = 0.1

    torch.manual_seed(42)
    # Create module with augment_eps > 0
    torch_module = TorchProteinFeatures(
        edge_features=edge_features,
        node_features=node_features,
        top_k=top_k,
        augment_eps=augment_eps,
    )
    torch_module.eval()
    jax_module = from_torch(torch_module)

    # Create module with augment_eps = 0 for reference
    torch.manual_seed(42)
    torch_module_no_aug = TorchProteinFeatures(
        edge_features=edge_features,
        node_features=node_features,
        top_k=top_k,
        augment_eps=0.0,
    )
    torch_module_no_aug.eval()
    jax_module_no_aug = from_torch(torch_module_no_aug)

    B, L = 2, 50
    input_features = create_test_protein_features(B, L)
    input_features_jax = {k: jnp.array(v.numpy()) for k, v in input_features.items()}

    # Without key, should produce deterministic results (no noise added)
    E_jax, E_idx_jax = jax_module(input_features_jax, key=None)
    E_jax_no_aug, E_idx_jax_no_aug = jax_module_no_aug(input_features_jax, key=None)

    # Results should be identical since no noise is applied without key
    np.testing.assert_allclose(
        np.array(E_jax),
        np.array(E_jax_no_aug),
        atol=1e-5,
        err_msg="augment_eps without key should match augment_eps=0",
    )


def test_protein_features_augment_eps_with_key():
    """Test that ProteinFeatures with key adds noise."""
    edge_features = 128
    node_features = 128
    top_k = 48
    augment_eps = 0.1

    torch.manual_seed(42)
    torch_module = TorchProteinFeatures(
        edge_features=edge_features,
        node_features=node_features,
        top_k=top_k,
        augment_eps=augment_eps,
    )
    torch_module.eval()
    jax_module = from_torch(torch_module)

    B, L = 2, 50
    input_features = create_test_protein_features(B, L)
    input_features_jax = {k: jnp.array(v.numpy()) for k, v in input_features.items()}

    # Without key (deterministic)
    E_no_noise, E_idx_no_noise = jax_module(input_features_jax, key=None)

    # With key (adds noise)
    key = jax.random.PRNGKey(0)
    E_with_noise, E_idx_with_noise = jax_module(input_features_jax, key=key)

    # E_idx may differ due to different neighbor ordering from noise
    # E values should differ due to noise
    assert not np.allclose(
        np.array(E_no_noise),
        np.array(E_with_noise),
        atol=1e-3,
    ), "augment_eps with key should produce different results"


def test_protein_features_augment_eps_deterministic():
    """Test that same key produces same results."""
    edge_features = 128
    node_features = 128
    top_k = 48
    augment_eps = 0.1

    torch.manual_seed(42)
    torch_module = TorchProteinFeatures(
        edge_features=edge_features,
        node_features=node_features,
        top_k=top_k,
        augment_eps=augment_eps,
    )
    torch_module.eval()
    jax_module = from_torch(torch_module)

    B, L = 2, 50
    input_features = create_test_protein_features(B, L)
    input_features_jax = {k: jnp.array(v.numpy()) for k, v in input_features.items()}

    key = jax.random.PRNGKey(42)

    # Same key should produce same results
    E1, E_idx1 = jax_module(input_features_jax, key=key)
    E2, E_idx2 = jax_module(input_features_jax, key=key)

    np.testing.assert_array_equal(
        np.array(E_idx1),
        np.array(E_idx2),
        err_msg="Same key should produce same E_idx",
    )

    np.testing.assert_allclose(
        np.array(E1),
        np.array(E2),
        atol=1e-6,
        err_msg="Same key should produce same E",
    )


def test_protein_features_augment_eps_different_keys():
    """Test that different keys produce different results."""
    edge_features = 128
    node_features = 128
    top_k = 48
    augment_eps = 0.1

    torch.manual_seed(42)
    torch_module = TorchProteinFeatures(
        edge_features=edge_features,
        node_features=node_features,
        top_k=top_k,
        augment_eps=augment_eps,
    )
    torch_module.eval()
    jax_module = from_torch(torch_module)

    B, L = 2, 50
    input_features = create_test_protein_features(B, L)
    input_features_jax = {k: jnp.array(v.numpy()) for k, v in input_features.items()}

    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(123)

    E1, _ = jax_module(input_features_jax, key=key1)
    E2, _ = jax_module(input_features_jax, key=key2)

    # Different keys should produce different results
    assert not np.allclose(
        np.array(E1),
        np.array(E2),
        atol=1e-3,
    ), "Different keys should produce different results"


def test_ligand_features_augment_eps_with_key():
    """Test that ProteinFeaturesLigand with key adds noise to both X and Y."""
    edge_features = 128
    node_features = 128
    top_k = 32
    augment_eps = 0.1
    atom_context_num = 25

    torch.manual_seed(42)
    torch_module = TorchProteinFeaturesLigand(
        edge_features=edge_features,
        node_features=node_features,
        top_k=top_k,
        augment_eps=augment_eps,
        atom_context_num=atom_context_num,
        use_side_chains=False,
    )
    torch_module.eval()
    jax_module = from_torch(torch_module)

    B, L, M = 2, 50, atom_context_num
    input_features = create_test_ligand_features(B, L, M)
    input_features_jax = {k: jnp.array(v.numpy()) for k, v in input_features.items()}

    # Without key (deterministic)
    V_no_noise, E_no_noise, _, _, _, _ = jax_module(input_features_jax, key=None)

    # With key (adds noise)
    key = jax.random.PRNGKey(0)
    V_with_noise, E_with_noise, _, _, _, _ = jax_module(input_features_jax, key=key)

    # Results should differ due to noise
    assert not np.allclose(
        np.array(E_no_noise),
        np.array(E_with_noise),
        atol=1e-3,
    ), "ligand augment_eps with key should produce different E"

    assert not np.allclose(
        np.array(V_no_noise),
        np.array(V_with_noise),
        atol=1e-3,
    ), "ligand augment_eps with key should produce different V"


def test_ligand_features_augment_eps_deterministic():
    """Test that same key produces same results for ligand features."""
    edge_features = 128
    node_features = 128
    top_k = 32
    augment_eps = 0.1
    atom_context_num = 25

    torch.manual_seed(42)
    torch_module = TorchProteinFeaturesLigand(
        edge_features=edge_features,
        node_features=node_features,
        top_k=top_k,
        augment_eps=augment_eps,
        atom_context_num=atom_context_num,
        use_side_chains=False,
    )
    torch_module.eval()
    jax_module = from_torch(torch_module)

    B, L, M = 2, 50, atom_context_num
    input_features = create_test_ligand_features(B, L, M)
    input_features_jax = {k: jnp.array(v.numpy()) for k, v in input_features.items()}

    key = jax.random.PRNGKey(42)

    V1, E1, E_idx1, Y_nodes1, Y_edges1, _ = jax_module(input_features_jax, key=key)
    V2, E2, E_idx2, Y_nodes2, Y_edges2, _ = jax_module(input_features_jax, key=key)

    np.testing.assert_allclose(
        np.array(E1),
        np.array(E2),
        atol=1e-6,
        err_msg="Same key should produce same E for ligand",
    )

    np.testing.assert_allclose(
        np.array(V1),
        np.array(V2),
        atol=1e-6,
        err_msg="Same key should produce same V for ligand",
    )
