#!/usr/bin/env python3
"""Demo: Sequence sampling and log-likelihood evaluation using JAX/Equinox ProteinMPNN.

This script demonstrates:
1. Loading a PDB structure
2. Loading a pretrained JAX model
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

from jigandmpnn import load_protein_mpnn
from jigandmpnn.vendor.ligandmpnn import (
    parse_PDB,
    featurize,
    restype_int_to_str,
)


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
    log_probs_seq = jnp.take_along_axis(log_probs, S[..., None], axis=-1)[..., 0]
    log_probs_masked = log_probs_seq * mask
    return float(log_probs_masked.sum() / mask.sum())


def main():
    print("=" * 60)
    print("JAX/Equinox ProteinMPNN Demo")
    print("=" * 60)

    # Configuration
    pdb_path = Path(__file__).parent.parent / "3DI3.pdb"
    num_samples = 4
    temperature = 0.1
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    print(f"\nPDB file: {pdb_path}")
    print(f"Temperature: {temperature}")
    print(f"Number of samples: {num_samples}")
    print(f"JAX backend: {jax.default_backend()}")

    # Load model
    print("\n[1] Loading JAX model...")
    model = load_protein_mpnn()
    print(f"    Model loaded successfully")

    # Parse PDB (using PyTorch utilities, then convert)
    print("\n[2] Parsing PDB structure...")
    protein_dict, *_ = parse_PDB(
        str(pdb_path), device="cpu", chains=[], parse_all_atoms=False,
    )

    L = protein_dict["X"].shape[0]
    chains = list(set(protein_dict["chain_letters"]))
    native_seq = sequence_to_string(jnp.array(protein_dict["S"].numpy()))

    print(f"    Sequence length: {L}")
    print(f"    Chains: {chains}")
    print(f"    Native sequence: {native_seq[:50]}..." if len(native_seq) > 50 else f"    Native sequence: {native_seq}")

    # Prepare features
    print("\n[3] Preparing features...")
    protein_dict["chain_mask"] = torch.ones(L)

    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=1,
        model_type="protein_mpnn",
    )

    B, L_feat = feature_dict["mask"].shape

    # Convert to JAX keyword arguments
    X = jnp.array(feature_dict["X"].numpy())
    S = jnp.array(feature_dict["S"].numpy())
    mask = jnp.array(feature_dict["mask"].numpy())
    residue_idx = jnp.array(feature_dict["R_idx"].numpy())
    chain_encoding_all = jnp.array(feature_dict["chain_labels"].numpy())

    print(f"    Feature dict prepared with {L_feat} positions")

    # Evaluate log-likelihood of native sequence
    print("\n[4] Evaluating log-likelihood of native sequence...")
    score_result = model.score(
        X=X, S=S, mask=mask,
        residue_idx=residue_idx, chain_encoding_all=chain_encoding_all,
        key=key, use_sequence=True,
    )

    native_ll = compute_sequence_log_likelihood(score_result.log_probs, S, mask)
    print(f"    Native sequence log-likelihood: {native_ll:.4f} (per residue)")

    # Sample new sequences
    print(f"\n[5] Sampling {num_samples} new sequences...")
    sampled_sequences = []
    sampled_log_likelihoods = []

    for i in range(num_samples):
        key, sample_key = jax.random.split(key)

        result = model.sample(
            X=X, S=S, mask=mask,
            residue_idx=residue_idx, chain_encoding_all=chain_encoding_all,
            key=sample_key, temperature=temperature,
        )

        seq = sequence_to_string(result.S[0])
        sampled_sequences.append(seq)

        # Compute log-likelihood of sampled sequence
        score_result = model.score(
            X=X, S=result.S, mask=mask,
            residue_idx=residue_idx, chain_encoding_all=chain_encoding_all,
            key=sample_key, use_sequence=True,
        )
        ll = compute_sequence_log_likelihood(score_result.log_probs, result.S, mask)
        sampled_log_likelihoods.append(ll)

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

    sample_jit = eqx.filter_jit(model.sample)

    # Warm-up call (triggers compilation)
    key, warmup_key = jax.random.split(key)
    start = time.time()
    _ = sample_jit(
        X=X, S=S, mask=mask,
        residue_idx=residue_idx, chain_encoding_all=chain_encoding_all,
        key=warmup_key, temperature=temperature,
    )
    compile_time = time.time() - start
    print(f"    First call (includes compilation): {compile_time:.2f}s")

    # Subsequent calls use cached compilation
    key, test_key = jax.random.split(key)
    start = time.time()
    _ = sample_jit(
        X=X, S=S, mask=mask,
        residue_idx=residue_idx, chain_encoding_all=chain_encoding_all,
        key=test_key, temperature=temperature,
    )
    run_time = time.time() - start
    print(f"    Subsequent calls: {run_time*1000:.1f}ms")
    print(f"    Speedup: {compile_time/run_time:.1f}x")


if __name__ == "__main__":
    main()
