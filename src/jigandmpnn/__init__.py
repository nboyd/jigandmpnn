"""JAX/Equinox translation of LigandMPNN."""

from pathlib import Path
from typing import Literal, TYPE_CHECKING

from jigandmpnn.backend import from_torch, register_from_torch
from jigandmpnn.modules.model import SampleResult, ScoreResult

if TYPE_CHECKING:
    from jigandmpnn.modules.model import ProteinMPNN

# Package paths
PACKAGE_DIR = Path(__file__).parent
WEIGHTS_DIR = PACKAGE_DIR / "weights"

# All available weight files
WEIGHT_FILES = {
    # ProteinMPNN variants (different noise levels: 0.02, 0.10, 0.20, 0.30)
    "proteinmpnn_v_48_002": "proteinmpnn_v_48_002.pt",
    "proteinmpnn_v_48_010": "proteinmpnn_v_48_010.pt",
    "proteinmpnn_v_48_020": "proteinmpnn_v_48_020.pt",
    "proteinmpnn_v_48_030": "proteinmpnn_v_48_030.pt",
    # LigandMPNN variants (different noise levels)
    "ligandmpnn_v_32_005_25": "ligandmpnn_v_32_005_25.pt",
    "ligandmpnn_v_32_010_25": "ligandmpnn_v_32_010_25.pt",
    "ligandmpnn_v_32_020_25": "ligandmpnn_v_32_020_25.pt",
    "ligandmpnn_v_32_030_25": "ligandmpnn_v_32_030_25.pt",
    # SolubleMPNN variants
    "solublempnn_v_48_002": "solublempnn_v_48_002.pt",
    "solublempnn_v_48_010": "solublempnn_v_48_010.pt",
    "solublempnn_v_48_020": "solublempnn_v_48_020.pt",
    "solublempnn_v_48_030": "solublempnn_v_48_030.pt",
    # Membrane MPNNs
    "per_residue_label_membrane_mpnn_v_48_020": "per_residue_label_membrane_mpnn_v_48_020.pt",
    "global_label_membrane_mpnn_v_48_020": "global_label_membrane_mpnn_v_48_020.pt",
    # Side-chain packing
    "ligandmpnn_sc_v_32_002_16": "ligandmpnn_sc_v_32_002_16.pt",
    # Aliases for convenience
    "protein_mpnn": "proteinmpnn_v_48_020.pt",
    "ligand_mpnn": "ligandmpnn_v_32_010_25.pt",
    "soluble_mpnn": "solublempnn_v_48_020.pt",
}

# Map weight names to model types for loading
_MODEL_TYPES = {
    "proteinmpnn": "protein_mpnn",
    "protein_mpnn": "protein_mpnn",
    "solublempnn": "soluble_mpnn",
    "soluble_mpnn": "soluble_mpnn",
    "ligandmpnn": "ligand_mpnn",
    "ligand_mpnn": "ligand_mpnn",
    "per_residue_label_membrane_mpnn": "per_residue_label_membrane_mpnn",
    "global_label_membrane_mpnn": "global_label_membrane_mpnn",
}


def get_weight_path(model_name: str = "protein_mpnn") -> Path:
    """Get path to pretrained model weights.

    Args:
        model_name: Model name or alias. Options:
            - Aliases: "protein_mpnn", "ligand_mpnn", "soluble_mpnn"
            - ProteinMPNN: "proteinmpnn_v_48_002", "proteinmpnn_v_48_010",
              "proteinmpnn_v_48_020", "proteinmpnn_v_48_030"
            - LigandMPNN: "ligandmpnn_v_32_005_25", "ligandmpnn_v_32_010_25",
              "ligandmpnn_v_32_020_25", "ligandmpnn_v_32_030_25"
            - SolubleMPNN: "solublempnn_v_48_002", "solublempnn_v_48_010",
              "solublempnn_v_48_020", "solublempnn_v_48_030"
            - Membrane: "per_residue_label_membrane_mpnn_v_48_020",
              "global_label_membrane_mpnn_v_48_020"
            - Side-chain: "ligandmpnn_sc_v_32_002_16"

    Returns:
        Path to the checkpoint file
    """
    if model_name not in WEIGHT_FILES:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Available: {sorted(WEIGHT_FILES.keys())}"
        )
    return WEIGHTS_DIR / WEIGHT_FILES[model_name]


def list_weights() -> list[str]:
    """List all available pretrained weight names."""
    return sorted(WEIGHT_FILES.keys())


def _get_model_type(model_name: str) -> str:
    """Infer model_type from weight name."""
    # Check direct mapping first
    for prefix, model_type in _MODEL_TYPES.items():
        if model_name.startswith(prefix):
            return model_type
    # Default fallback
    return "protein_mpnn"


def _load_model(checkpoint_path: Path, model_type: str) -> "ProteinMPNN":
    """Internal helper to load a model from checkpoint."""
    import torch
    from jigandmpnn.modules.model import ProteinMPNN  # noqa: F811
    from jigandmpnn.vendor.ligandmpnn import ProteinMPNN as TorchProteinMPNN

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    torch_model = TorchProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=checkpoint["num_edges"],
        device="cpu",
        atom_context_num=checkpoint.get("atom_context_num", 1),
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=0,
    )
    torch_model.load_state_dict(checkpoint["model_state_dict"])
    torch_model.eval()

    return from_torch(torch_model)


def load_protein_mpnn() -> "ProteinMPNN":
    """Load pretrained ProteinMPNN model.

    Returns:
        JAX/Equinox ProteinMPNN model ready for inference.

    Example:
        >>> from jigandmpnn import load_protein_mpnn
        >>> model = load_protein_mpnn()
        >>> result = model.sample(features, key=jax.random.PRNGKey(0))
    """
    return _load_model(WEIGHTS_DIR / "proteinmpnn_v_48_020.pt", "protein_mpnn")


def load_soluble_mpnn() -> "ProteinMPNN":
    """Load pretrained SolubleMPNN model.

    SolubleMPNN is trained on soluble proteins only, which can improve
    sequence design for soluble protein targets.

    Returns:
        JAX/Equinox ProteinMPNN model ready for inference.

    Example:
        >>> from jigandmpnn import load_soluble_mpnn
        >>> model = load_soluble_mpnn()
        >>> result = model.sample(features, key=jax.random.PRNGKey(0))
    """
    return _load_model(WEIGHTS_DIR / "solublempnn_v_48_020.pt", "soluble_mpnn")


__all__ = [
    "from_torch",
    "register_from_torch",
    "get_weight_path",
    "list_weights",
    "load_protein_mpnn",
    "load_soluble_mpnn",
    "SampleResult",
    "ScoreResult",
    "WEIGHTS_DIR",
    "WEIGHT_FILES",
]
