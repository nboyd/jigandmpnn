#!/usr/bin/env python3
"""Benchmark JAX/Equinox LigandMPNN on GPU."""

import time
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
import equinox as eqx

from jigandmpnn import get_weight_path
from jigandmpnn.backend import from_torch
from jigandmpnn.modules.model import ProteinMPNN
from jigandmpnn.vendor.ligandmpnn import (
    ProteinMPNN as TorchProteinMPNN,
    parse_PDB,
    featurize,
)


def load_jax_model(model_type: str = "protein_mpnn"):
    """Load a pretrained model and convert to JAX."""
    checkpoint_path = get_weight_path(model_type)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    k_neighbors = checkpoint["num_edges"]
    atom_context_num = checkpoint.get("atom_context_num", 1)

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

    return from_torch(torch_model)


def torch_dict_to_jax(feature_dict: dict) -> dict:
    """Convert feature dictionary from PyTorch to JAX."""
    jax_dict = {}
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            jax_dict[k] = jnp.array(v.numpy())
        else:
            jax_dict[k] = v
    return jax_dict


def benchmark_sampling(num_samples: int = 512, num_warmup: int = 3, num_runs: int = 5):
    """Benchmark sequence sampling with vmap."""
    backend = jax.default_backend()
    print(f"JAX backend: {backend}")
    if backend == "gpu":
        print(f"GPU: {jax.devices()[0]}")

    # Load model
    print("\nLoading model and converting to JAX...")
    model = load_jax_model()

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
    feature_dict["bias"] = torch.zeros(B, L_feat, 21)
    feature_dict["randn"] = torch.randn(B, L_feat)

    # Convert to JAX
    feature_dict_jax = torch_dict_to_jax(feature_dict)

    print(f"\nBenchmarking {num_samples} samples...")
    print(f"Warmup runs: {num_warmup}")
    print(f"Timed runs: {num_runs}")

    # Create vmapped sample function
    # vmap over different random keys
    @eqx.filter_jit
    def sample_batch(model, feature_dict, keys):
        """Sample a batch of sequences using vmap over keys."""
        def sample_single(key):
            return model.sample(feature_dict, key=key, temperature=0.1)
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
        result = sample_batch(model, feature_dict_jax, keys)
        jax.block_until_ready(result["S"])
        print(f"  Warmup {i+1}/{num_warmup}")

    # Timed runs (vmap)
    print("\nTimed runs (vmap)...")
    times_vmap = []
    for i in range(num_runs):
        master_key, *sample_keys = jax.random.split(master_key, num_samples + 1)
        keys = jnp.stack(sample_keys)

        start = time.perf_counter()
        result = sample_batch(model, feature_dict_jax, keys)
        jax.block_until_ready(result["S"])
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
            master_key, randn_key = jax.random.split(master_key)
            feature_dict_jax["randn"] = jax.random.normal(randn_key, (B, L_feat))
            result = sample_jit(feature_dict_jax, key=sample_key, temperature=0.1)
            jax.block_until_ready(result["S"])
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
