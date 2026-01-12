#!/usr/bin/env python3
"""Benchmark PyTorch LigandMPNN on GPU."""

import time
from pathlib import Path

import numpy as np
import torch

from jigandmpnn import get_weight_path
from jigandmpnn.vendor.ligandmpnn import (
    ProteinMPNN,
    parse_PDB,
    featurize,
)


def load_model(model_type: str = "protein_mpnn", device: str = "cuda") -> ProteinMPNN:
    """Load a pretrained ProteinMPNN model."""
    checkpoint_path = get_weight_path(model_type)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    k_neighbors = checkpoint["num_edges"]
    atom_context_num = checkpoint.get("atom_context_num", 1)

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=0,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def benchmark_sampling(num_samples: int = 512, num_warmup: int = 3, num_runs: int = 5):
    """Benchmark sequence sampling."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load model
    print("\nLoading model...")
    model = load_model(device=device)

    # Parse PDB
    pdb_path = Path(__file__).parent.parent / "3DI3.pdb"
    print(f"Parsing {pdb_path}...")
    protein_dict, *_ = parse_PDB(str(pdb_path), device=device, chains=[])

    L = protein_dict["X"].shape[0]
    print(f"Sequence length: {L}")

    # Prepare features
    protein_dict["chain_mask"] = torch.ones(L, device=device)
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=1,
        model_type="protein_mpnn",
    )

    # Move to device
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict[k] = v.to(device)

    B, L_feat = feature_dict["mask"].shape
    feature_dict["batch_size"] = num_samples
    feature_dict["symmetry_residues"] = [[]] * num_samples
    feature_dict["symmetry_weights"] = [[]] * num_samples
    feature_dict["temperature"] = 0.1
    feature_dict["bias"] = torch.zeros(B, L_feat, 21, device=device)

    # Expand batch dimension for batched sampling
    # PyTorch LigandMPNN uses batch_size parameter internally

    print(f"\nBenchmarking {num_samples} samples...")
    print(f"Warmup runs: {num_warmup}")
    print(f"Timed runs: {num_runs}")

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for i in range(num_warmup):
            feature_dict["randn"] = torch.randn(B, L_feat, device=device)
            _ = model.sample(feature_dict)
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"  Warmup {i+1}/{num_warmup}")

    # Timed runs
    print("\nTimed runs...")
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            feature_dict["randn"] = torch.randn(B, L_feat, device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            result = model.sample(feature_dict)

            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s ({num_samples/elapsed:.1f} seq/s)")

    # Results
    times = np.array(times)
    print("\n" + "=" * 60)
    print("RESULTS (PyTorch)")
    print("=" * 60)
    print(f"Samples per run: {num_samples}")
    print(f"Sequence length: {L_feat}")
    print(f"Mean time: {times.mean():.3f}s +/- {times.std():.3f}s")
    print(f"Throughput: {num_samples/times.mean():.1f} seq/s")
    print(f"Time per sequence: {times.mean()/num_samples*1000:.2f}ms")


if __name__ == "__main__":
    benchmark_sampling(num_samples=512, num_warmup=3, num_runs=5)
