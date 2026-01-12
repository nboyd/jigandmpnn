#!/usr/bin/env python3
"""Demo: Sequence sampling and log-likelihood evaluation using JAX/Equinox ProteinMPNN.

This script demonstrates:
1. Loading a PDB structure
2. Converting a PyTorch model to JAX
3. Sampling new sequences with the JAX implementation
4. Evaluating log-likelihoods of sequences

Usage:
    python demos/demo_jax.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from jigandmpnn import get_weight_path
from jigandmpnn.backend import from_torch
from jigandmpnn.modules.model import ProteinMPNN  # Registers the from_torch converter
from jigandmpnn.vendor.ligandmpnn import (
    ProteinMPNN as TorchProteinMPNN,
    parse_PDB,
    featurize,
    restype_int_to_str,
)


def load_jax_model(model_type: str = "protein_mpnn"):
    """Load a pretrained ProteinMPNN model and convert to JAX.

    Args:
        model_type: One of "protein_mpnn", "ligand_mpnn", "soluble_mpnn"

    Returns:
        JAX/Equinox model
    """
    checkpoint_path = get_weight_path(model_type)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Weights should be included in the package."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    k_neighbors = checkpoint["num_edges"]
    atom_context_num = checkpoint.get("atom_context_num", 1)

    # Create PyTorch model
    torch_model = TorchProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device="cpu",
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=0,
    )

    torch_model.load_state_dict(checkpoint["model_state_dict"])
    torch_model.eval()

    # Convert to JAX
    jax_model = from_torch(torch_model)

    return jax_model


def sequence_to_string(S: jnp.ndarray) -> str:
    """Convert integer-encoded sequence to string."""
    return "".join([restype_int_to_str[int(aa)] for aa in np.array(S)])


def compute_sequence_log_likelihood(log_probs: jnp.ndarray, S: jnp.ndarray, mask: jnp.ndarray) -> float:
    """Compute per-residue log-likelihood of a sequence.

    Args:
        log_probs: Log probabilities [B, L, 21]
        S: Sequence [B, L]
        mask: Position mask [B, L]

    Returns:
        Mean per-residue log-likelihood
    """
    # Gather log probabilities for the actual sequence
    log_probs_seq = jnp.take_along_axis(log_probs, S[..., None], axis=-1)[..., 0]  # [B, L]
    # Mask and compute mean
    log_probs_masked = log_probs_seq * mask
    return float(log_probs_masked.sum() / mask.sum())


def torch_dict_to_jax(feature_dict: dict) -> dict:
    """Convert a feature dictionary from PyTorch tensors to JAX arrays."""
    jax_dict = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            jax_dict[k] = jnp.array(v.numpy())
        else:
            jax_dict[k] = v
    return jax_dict


def main():
    print("=" * 60)
    print("JAX/Equinox ProteinMPNN Demo")
    print("=" * 60)

    # Configuration
    pdb_path = Path(__file__).parent.parent / "3DI3.pdb"
    model_type = "protein_mpnn"
    num_samples = 4
    temperature = 0.1
    seed = 42

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    print(f"\nPDB file: {pdb_path}")
    print(f"Model type: {model_type}")
    print(f"Temperature: {temperature}")
    print(f"Number of samples: {num_samples}")
    print(f"JAX backend: {jax.default_backend()}")

    # Load model
    print("\n[1] Loading model and converting to JAX...")
    model = load_jax_model(model_type=model_type)
    print(f"    Model converted successfully")

    # Parse PDB (using PyTorch utilities, then convert)
    print("\n[2] Parsing PDB structure...")
    protein_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(
        str(pdb_path),
        device="cpu",
        chains=[],  # Parse all chains
        parse_all_atoms=False,
    )

    L = protein_dict["X"].shape[0]
    chains = list(set(protein_dict["chain_letters"]))
    native_seq = sequence_to_string(jnp.array(protein_dict["S"].numpy()))

    print(f"    Sequence length: {L}")
    print(f"    Chains: {chains}")
    print(f"    Native sequence: {native_seq[:50]}..." if len(native_seq) > 50 else f"    Native sequence: {native_seq}")

    # Prepare features
    print("\n[3] Preparing features...")
    protein_dict["chain_mask"] = torch.ones(L)  # Design all positions

    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=1,
        model_type=model_type,
    )

    B, L_feat = feature_dict["mask"].shape

    # Add required fields
    feature_dict["bias"] = torch.zeros(B, L_feat, 21)
    feature_dict["randn"] = torch.randn(B, L_feat)

    # Convert to JAX
    feature_dict_jax = torch_dict_to_jax(feature_dict)

    print(f"    Feature dict prepared with {L_feat} positions")

    # Evaluate log-likelihood of native sequence
    print("\n[4] Evaluating log-likelihood of native sequence...")
    score_result = model.score(feature_dict_jax, use_sequence=True)

    native_ll = compute_sequence_log_likelihood(
        score_result["log_probs"],
        feature_dict_jax["S"],
        feature_dict_jax["mask"].astype(jnp.float32)
    )
    print(f"    Native sequence log-likelihood: {native_ll:.4f} (per residue)")

    # Sample new sequences
    print(f"\n[5] Sampling {num_samples} new sequences...")
    sampled_sequences = []
    sampled_log_likelihoods = []

    for i in range(num_samples):
        # Split key for this sample
        key, sample_key = jax.random.split(key)

        # Generate random decoding order
        key, randn_key = jax.random.split(key)
        feature_dict_jax["randn"] = jax.random.normal(randn_key, (B, L_feat))

        # Sample
        sample_result = model.sample(feature_dict_jax, key=sample_key, temperature=temperature)

        seq = sequence_to_string(sample_result["S"][0])
        sampled_sequences.append(seq)

        # Compute log-likelihood of sampled sequence
        # Update S in feature dict temporarily
        original_S = feature_dict_jax["S"]
        feature_dict_jax["S"] = sample_result["S"]
        score_result = model.score(feature_dict_jax, use_sequence=True)
        ll = compute_sequence_log_likelihood(
            score_result["log_probs"],
            sample_result["S"],
            feature_dict_jax["mask"].astype(jnp.float32)
        )
        feature_dict_jax["S"] = original_S

        sampled_log_likelihoods.append(ll)

        # Compute sequence identity to native
        identity = sum(a == b for a, b in zip(seq, native_seq)) / len(seq) * 100

        print(f"\n    Sample {i+1}:")
        print(f"      Sequence: {seq[:50]}..." if len(seq) > 50 else f"      Sequence: {seq}")
        print(f"      Log-likelihood: {ll:.4f}")
        print(f"      Identity to native: {identity:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Native sequence LL: {native_ll:.4f}")
    print(f"Sampled sequences LL: {np.mean(sampled_log_likelihoods):.4f} +/- {np.std(sampled_log_likelihoods):.4f}")
    print(f"Best sampled LL: {max(sampled_log_likelihoods):.4f}")

    # JIT compilation demo
    print("\n" + "=" * 60)
    print("JIT Compilation Demo")
    print("=" * 60)

    import equinox as eqx
    import time

    print("\nCompiling sample() with eqx.filter_jit...")

    # JIT compile the sample method
    sample_jit = eqx.filter_jit(model.sample)

    # Warm-up call (triggers compilation)
    key, warmup_key = jax.random.split(key)
    start = time.time()
    _ = sample_jit(feature_dict_jax, key=warmup_key, temperature=temperature)
    compile_time = time.time() - start
    print(f"    First call (includes compilation): {compile_time:.2f}s")

    # Subsequent calls use cached compilation
    key, test_key = jax.random.split(key)
    start = time.time()
    _ = sample_jit(feature_dict_jax, key=test_key, temperature=temperature)
    run_time = time.time() - start
    print(f"    Subsequent calls: {run_time*1000:.1f}ms")
    print(f"    Speedup: {compile_time/run_time:.1f}x")


if __name__ == "__main__":
    main()
