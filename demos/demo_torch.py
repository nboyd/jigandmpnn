#!/usr/bin/env python3
"""Demo: Sequence sampling and log-likelihood evaluation using PyTorch LigandMPNN.

This script demonstrates:
1. Loading a PDB structure
2. Sampling new sequences with ProteinMPNN
3. Evaluating log-likelihoods of sequences

Usage:
    python demos/demo_torch.py
"""

from pathlib import Path

import numpy as np
import torch

from jigandmpnn import get_weight_path
from jigandmpnn.vendor.ligandmpnn import (
    ProteinMPNN,
    parse_PDB,
    featurize,
    restype_int_to_str,
)


def load_model(model_type: str = "protein_mpnn", device: str = "cpu") -> ProteinMPNN:
    """Load a pretrained ProteinMPNN model.

    Args:
        model_type: One of "protein_mpnn", "ligand_mpnn", "soluble_mpnn"
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint_path = get_weight_path(model_type)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Weights should be included in the package."
        )

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


def sequence_to_string(S: torch.Tensor) -> str:
    """Convert integer-encoded sequence to string."""
    return "".join([restype_int_to_str[int(aa)] for aa in S])


def compute_sequence_log_likelihood(log_probs: torch.Tensor, S: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute per-residue log-likelihood of a sequence.

    Args:
        log_probs: Log probabilities [B, L, 21]
        S: Sequence [B, L]
        mask: Position mask [B, L]

    Returns:
        Mean per-residue log-likelihood
    """
    # Gather log probabilities for the actual sequence
    log_probs_seq = torch.gather(log_probs, -1, S.unsqueeze(-1)).squeeze(-1)  # [B, L]
    # Mask and compute mean
    log_probs_masked = log_probs_seq * mask
    return (log_probs_masked.sum() / mask.sum()).item()


def main():
    print("=" * 60)
    print("PyTorch ProteinMPNN Demo")
    print("=" * 60)

    # Configuration
    pdb_path = Path(__file__).parent.parent / "3DI3.pdb"
    device = "cpu"
    model_type = "protein_mpnn"
    num_samples = 4
    temperature = 0.1
    seed = 42

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\nPDB file: {pdb_path}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"Temperature: {temperature}")
    print(f"Number of samples: {num_samples}")

    # Load model
    print("\n[1] Loading model...")
    model = load_model(model_type=model_type, device=device)
    print(f"    Model loaded successfully")

    # Parse PDB
    print("\n[2] Parsing PDB structure...")
    protein_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(
        str(pdb_path),
        device=device,
        chains=[],  # Parse all chains
        parse_all_atoms=False,
    )

    L = protein_dict["X"].shape[0]
    chains = list(set(protein_dict["chain_letters"]))
    native_seq = sequence_to_string(protein_dict["S"])

    print(f"    Sequence length: {L}")
    print(f"    Chains: {chains}")
    print(f"    Native sequence: {native_seq[:50]}..." if len(native_seq) > 50 else f"    Native sequence: {native_seq}")

    # Prepare features
    print("\n[3] Preparing features...")
    protein_dict["chain_mask"] = torch.ones(L, device=device)  # Design all positions

    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=1,
        model_type=model_type,
    )

    B, L_feat = feature_dict["mask"].shape
    feature_dict["batch_size"] = 1
    feature_dict["symmetry_residues"] = [[]]
    feature_dict["symmetry_weights"] = [[]]
    feature_dict["temperature"] = temperature
    feature_dict["bias"] = torch.zeros(B, L_feat, 21, device=device)

    print(f"    Feature dict prepared with {L_feat} positions")

    # Evaluate log-likelihood of native sequence
    print("\n[4] Evaluating log-likelihood of native sequence...")
    with torch.no_grad():
        feature_dict["randn"] = torch.randn(B, L_feat, device=device)
        score_result = model.score(feature_dict, use_sequence=True)

    native_ll = compute_sequence_log_likelihood(
        score_result["log_probs"],
        feature_dict["S"],
        feature_dict["mask"].float()
    )
    print(f"    Native sequence log-likelihood: {native_ll:.4f} (per residue)")

    # Sample new sequences
    print(f"\n[5] Sampling {num_samples} new sequences...")
    sampled_sequences = []
    sampled_log_likelihoods = []

    with torch.no_grad():
        for i in range(num_samples):
            # Generate random decoding order
            feature_dict["randn"] = torch.randn(B, L_feat, device=device)

            # Sample
            sample_result = model.sample(feature_dict)

            seq = sequence_to_string(sample_result["S"][0])
            sampled_sequences.append(seq)

            # Compute log-likelihood of sampled sequence
            # Update S in feature dict temporarily
            original_S = feature_dict["S"].clone()
            feature_dict["S"] = sample_result["S"]
            score_result = model.score(feature_dict, use_sequence=True)
            ll = compute_sequence_log_likelihood(
                score_result["log_probs"],
                sample_result["S"],
                feature_dict["mask"].float()
            )
            feature_dict["S"] = original_S

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


if __name__ == "__main__":
    main()
