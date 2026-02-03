#!/usr/bin/env python3
"""Benchmark JAX/Equinox LigandMPNN on GPU."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import equinox as eqx

from jigandmpnn import load_protein_mpnn
from jigandmpnn.vendor.ligandmpnn import (
    parse_PDB,
    featurize,
)


def benchmark_sampling(num_samples: int = 512, num_warmup: int = 3, num_runs: int = 5):
    """Benchmark sequence sampling with vmap."""
    backend = jax.default_backend()
    print(f"JAX backend: {backend}")
    if backend == "gpu":
        print(f"GPU: {jax.devices()[0]}")

    # Load model
    print("\nLoading JAX model...")
    model = load_protein_mpnn()

    # Parse PDB
    pdb_path = Path(__file__).parent.parent / "3DI3.pdb"
    print(f"Parsing {pdb_path}...")
    protein_dict, *_ = parse_PDB(str(pdb_path), device="cpu", chains=[])

    L = protein_dict["X"].shape[0]
    print(f"Sequence length: {L}")

    # Prepare features
    protein_dict["chain_mask"] = torch.ones(L)
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=1,
        model_type="protein_mpnn",
    )

    B, L_feat = feature_dict["mask"].shape

    # Convert to JAX arrays
    X = jnp.array(feature_dict["X"].numpy())
    S = jnp.array(feature_dict["S"].numpy())
    mask = jnp.array(feature_dict["mask"].numpy())
    residue_idx = jnp.array(feature_dict["R_idx"].numpy())
    chain_encoding_all = jnp.array(feature_dict["chain_labels"].numpy())

    print(f"\nBenchmarking {num_samples} samples...")
    print(f"Warmup runs: {num_warmup}")
    print(f"Timed runs: {num_runs}")

    # Create vmapped sample function
    @eqx.filter_jit
    def sample_batch(model, X, S, mask, residue_idx, chain_encoding_all, keys):
        """Sample a batch of sequences using vmap over keys."""
        def sample_single(key):
            return model.sample(
                X=X, S=S, mask=mask,
                residue_idx=residue_idx, chain_encoding_all=chain_encoding_all,
                key=key, temperature=0.1,
            )
        return jax.vmap(sample_single)(keys)

    # Also test sequential JIT version for comparison
    sample_jit = eqx.filter_jit(model.sample)

    # Generate keys
    master_key = jax.random.PRNGKey(42)

    # Warmup
    print("\nWarming up (vmap)...")
    for i in range(num_warmup):
        master_key, *sample_keys = jax.random.split(master_key, num_samples + 1)
        keys = jnp.stack(sample_keys)
        result = sample_batch(model, X, S, mask, residue_idx, chain_encoding_all, keys)
        jax.block_until_ready(result.S)
        print(f"  Warmup {i+1}/{num_warmup}")

    # Timed runs (vmap)
    print("\nTimed runs (vmap)...")
    times_vmap = []
    for i in range(num_runs):
        master_key, *sample_keys = jax.random.split(master_key, num_samples + 1)
        keys = jnp.stack(sample_keys)

        start = time.perf_counter()
        result = sample_batch(model, X, S, mask, residue_idx, chain_encoding_all, keys)
        jax.block_until_ready(result.S)
        elapsed = time.perf_counter() - start

        times_vmap.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s ({num_samples/elapsed:.1f} seq/s)")

    # Also test sequential for comparison (smaller batch)
    seq_samples = min(32, num_samples)
    print(f"\nTimed runs (sequential, {seq_samples} samples)...")
    times_seq = []
    for i in range(num_runs):
        start = time.perf_counter()
        for j in range(seq_samples):
            master_key, sample_key = jax.random.split(master_key)
            result = sample_jit(
                X=X, S=S, mask=mask,
                residue_idx=residue_idx, chain_encoding_all=chain_encoding_all,
                key=sample_key, temperature=0.1,
            )
            jax.block_until_ready(result.S)
        elapsed = time.perf_counter() - start

        times_seq.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s ({seq_samples/elapsed:.1f} seq/s)")

    # Results
    times_vmap = np.array(times_vmap)
    times_seq = np.array(times_seq)

    print("\n" + "=" * 60)
    print("RESULTS (JAX)")
    print("=" * 60)
    print(f"\n[vmap - {num_samples} samples]")
    print(f"Mean time: {times_vmap.mean():.3f}s +/- {times_vmap.std():.3f}s")
    print(f"Throughput: {num_samples/times_vmap.mean():.1f} seq/s")
    print(f"Time per sequence: {times_vmap.mean()/num_samples*1000:.2f}ms")

    print(f"\n[sequential JIT - {seq_samples} samples]")
    print(f"Mean time: {times_seq.mean():.3f}s +/- {times_seq.std():.3f}s")
    print(f"Throughput: {seq_samples/times_seq.mean():.1f} seq/s")
    print(f"Time per sequence: {times_seq.mean()/seq_samples*1000:.2f}ms")


if __name__ == "__main__":
    benchmark_sampling(num_samples=512, num_warmup=3, num_runs=5)
