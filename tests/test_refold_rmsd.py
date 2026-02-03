"""Test inverse folding + refolding RMSD for PDL1 monomer.

Compares autoregressive sampling (jigandmpnn) and Jacobi decoding (mosaic)
by inverse-folding a PDL1 structure, refolding with Boltz2, and computing
backbone CA RMSD against the original.

Uses PDB 4Z18 chain A — apo human PD-L1 extracellular domain (IgV+IgC).
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import pytest
import gemmi
import torch

from jigandmpnn import load_protein_mpnn
from jigandmpnn.vendor.ligandmpnn import parse_PDB, featurize, restype_int_to_str

from mosaic.models.boltz2 import Boltz2, target_only_features
from mosaic.losses.boltz2 import set_binder_sequence
from mosaic.structure_prediction import TargetChain
from mosaic.proteinmpnn.mpnn import ProteinMPNN as MosaicMPNN

NUM_SAMPLES = 32


# --- Utilities ---

def ca_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Kabsch-aligned CA RMSD between two backbone coordinate arrays.

    Args:
        coords1, coords2: [N, 4, 3] backbone coords (N, CA, C, O).

    Returns:
        CA RMSD in Angstroms after optimal superposition.
    """
    ca1 = coords1[:, 1, :].copy()
    ca2 = coords2[:, 1, :].copy()

    # Center
    c1 = ca1.mean(axis=0)
    c2 = ca2.mean(axis=0)
    ca1 -= c1
    ca2 -= c2

    # Kabsch
    H = ca1.T @ ca2
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T

    ca1_aligned = (R @ (coords1[:, 1, :] - c1).T).T + c2
    return float(np.sqrt(np.mean(np.sum((ca1_aligned - coords2[:, 1, :]) ** 2, axis=-1))))


def extract_backbone_coords(st: gemmi.Structure, chain_id: str | None = None) -> np.ndarray:
    """Extract N, CA, C, O coordinates from a gemmi Structure."""
    model = st[0]
    coords = []
    for chain in model:
        if chain_id is not None and chain.name != chain_id:
            continue
        for residue in chain:
            atom_coords = np.zeros((4, 3))
            for i, name in enumerate(["N", "CA", "C", "O"]):
                atom = residue.find_atom(name, " ")
                if atom is not None:
                    atom_coords[i] = [atom.pos.x, atom.pos.y, atom.pos.z]
            coords.append(atom_coords)
    return np.array(coords)


def seq_to_string(S: jnp.ndarray) -> str:
    return "".join(restype_int_to_str[int(aa)] for aa in np.array(S))


def seq_identity(a: str, b: str) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


# --- Inverse folding methods ---

def autoregressive_inverse_fold_batch(mpnn, *, X, S, mask, residue_idx,
                                      chain_encoding_all, keys, temperature=0.1):
    """Sample N sequences via vmap over autoregressive sampling."""
    @eqx.filter_jit
    def _sample_batch(mpnn, X, S, mask, residue_idx, chain_encoding_all, keys):
        def _single(key):
            return mpnn.sample(
                X=X, S=S, mask=mask,
                residue_idx=residue_idx,
                chain_encoding_all=chain_encoding_all,
                key=key, temperature=temperature,
            ).S
        return jax.vmap(_single)(keys)

    # [NUM_SAMPLES, B, L]
    all_S = _sample_batch(mpnn, X, S, mask, residue_idx, chain_encoding_all, keys)
    return all_S[:, 0, :]  # [NUM_SAMPLES, L]


def jacobi_inverse_fold_batch(mpnn, *, X, mask, residue_idx, chain_encoding_all,
                              keys, temperature=0.1, jacobi_iterations=10):
    """Sample N sequences via vmap over Jacobi decoding.

    Encodes structure once, then vmaps the iterative decode loop.
    """
    # Encode once (unbatched)
    h_V, h_E, E_idx = mpnn.encode(
        X=X, mask=mask,
        residue_idx=residue_idx,
        chain_encoding_all=chain_encoding_all,
    )

    N = X.shape[0]

    @eqx.filter_jit
    def _jacobi_batch(h_V, h_E, E_idx, mask, keys):
        def _single(key):
            decoding_order = jax.random.uniform(key, shape=(N,))
            gumbel = jax.random.gumbel(key, (N, 21))
            seq = jax.random.randint(key=key, minval=0, maxval=20, shape=(N,))

            def _step(seq, _):
                S_onehot = jnn.one_hot(seq, 21)
                log_probs = mpnn.decode(
                    S=S_onehot, h_V=h_V, h_E=h_E, E_idx=E_idx,
                    mask=mask, decoding_order=decoding_order,
                )
                logits = log_probs[0]  # remove batch dim
                return (logits + temperature * gumbel).argmax(-1), None

            seq, _ = jax.lax.scan(_step, seq, None, length=jacobi_iterations)
            return seq

        return jax.vmap(_single)(keys)

    return _jacobi_batch(h_V, h_E, E_idx, mask, keys)  # [NUM_SAMPLES, L]


def mosaic_jacobi_inverse_fold_batch(mosaic_mpnn, *, X, mask, residue_idx,
                                     chain_encoding_all, keys,
                                     temperature=0.1, jacobi_iterations=10):
    """Jacobi decoding using mosaic's own ProteinMPNN implementation.

    Same algorithm as jacobi_inverse_fold_batch but drives mosaic's model
    to verify both implementations produce comparable results.
    """
    h_V, h_E, E_idx = mosaic_mpnn.encode(
        X=X, mask=mask,
        residue_idx=residue_idx,
        chain_encoding_all=chain_encoding_all,
        key=keys[0],
    )

    N = X.shape[0]

    @eqx.filter_jit
    def _jacobi_batch(h_V, h_E, E_idx, mask, keys):
        def _single(key):
            decoding_order = jax.random.uniform(key, shape=(N,))
            gumbel = jax.random.gumbel(key, (N, 21))
            seq = jax.random.randint(key=key, minval=0, maxval=20, shape=(N,))

            def _step(seq, _):
                S_onehot = jnn.one_hot(seq, 21)
                log_probs = mosaic_mpnn.decode(
                    S=S_onehot, h_V=h_V, h_E=h_E, E_idx=E_idx,
                    mask=mask, decoding_order=decoding_order,
                )
                logits = log_probs[0]  # remove batch dim
                return (logits + temperature * gumbel).argmax(-1), None

            seq, _ = jax.lax.scan(_step, seq, None, length=jacobi_iterations)
            return seq

        return jax.vmap(_single)(keys)

    return _jacobi_batch(h_V, h_E, E_idx, mask, keys)


# --- Boltz2 batch folding ---

class Boltz2Folder:
    """Precomputes Boltz2 features for a fixed-length monomer and refolds
    arbitrary sequences of that length without re-featurizing."""

    def __init__(self, boltz_model: Boltz2, seq_length: int):
        self.boltz = boltz_model
        # Precompute features using a poly-alanine template
        template_seq = "A" * seq_length
        chains = [TargetChain(sequence=template_seq, use_msa=False)]
        self.features, self.writer = target_only_features(chains)
        self.seq_length = seq_length
        # Warm up JIT
        self._predict_coords(self.features, jax.random.key(0))

    @eqx.filter_jit
    def _predict_coords(self, features, key):
        output = self.boltz.model_output(features=features, key=key)
        return output.backbone_coordinates

    def fold(self, sequence: str, key: jax.Array) -> np.ndarray:
        """Fold a sequence and return backbone coords [N, 4, 3]."""
        assert len(sequence) == self.seq_length
        # Encode sequence as one-hot over standard 20 AAs
        from mosaic.common import TOKENS
        indices = [TOKENS.index(aa) if aa in TOKENS else 0 for aa in sequence]
        seq_onehot = jnn.one_hot(jnp.array(indices), 20)
        features = set_binder_sequence(seq_onehot, self.features)
        coords = self._predict_coords(features, key)
        return np.array(coords)

    def fold_batch(self, sequences: list[str], keys: list[jax.Array]) -> list[np.ndarray]:
        """Fold multiple sequences, returning list of [N, 4, 3] coord arrays."""
        return [self.fold(seq, k) for seq, k in zip(sequences, keys)]


# --- Fixtures ---

@pytest.fixture(scope="module")
def mpnn():
    return load_protein_mpnn()


@pytest.fixture(scope="module")
def mosaic_mpnn():
    return MosaicMPNN.from_pretrained()


@pytest.fixture(scope="module")
def boltz():
    return Boltz2()


@pytest.fixture(scope="module")
def pdl1_data():
    """Download PDL1 monomer (PDB 4Z18 chain A) and prepare MPNN features."""
    import prody
    pdb_path = prody.fetchPDB("4Z18", compressed=False)

    protein_dict, *_ = parse_PDB(str(pdb_path), device="cpu", chains=["A"])
    L = protein_dict["X"].shape[0]
    protein_dict["chain_mask"] = torch.ones(L)

    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=1,
        model_type="protein_mpnn",
    )

    X = jnp.array(feature_dict["X"].numpy())
    S = jnp.array(feature_dict["S"].numpy())
    mask = jnp.array(feature_dict["mask"].numpy())
    residue_idx = jnp.array(feature_dict["R_idx"].numpy())
    chain_encoding_all = jnp.array(feature_dict["chain_labels"].numpy())
    native_seq = seq_to_string(S[0])

    # Native backbone coords from crystal structure
    st_native = gemmi.read_structure(pdb_path)
    st_native.remove_ligands_and_waters()
    native_coords = extract_backbone_coords(st_native, chain_id="A")[:len(native_seq)]

    return {
        "X": X,
        "S": S,
        "mask": mask,
        "residue_idx": residue_idx,
        "chain_encoding_all": chain_encoding_all,
        "native_seq": native_seq,
        "native_coords": native_coords,
    }


# --- Tests ---

def test_autoregressive_inverse_fold(mpnn, pdl1_data):
    """Autoregressive inverse folding produces valid sequences."""
    d = pdl1_data
    result = mpnn.sample(
        X=d["X"], S=d["S"], mask=d["mask"],
        residue_idx=d["residue_idx"],
        chain_encoding_all=d["chain_encoding_all"],
        key=jax.random.PRNGKey(42), temperature=0.1,
    )
    designed_seq = seq_to_string(result.S[0])
    identity = seq_identity(designed_seq, d["native_seq"])

    print(f"\nAutoregressive sequence identity to native: {identity:.1%}")
    assert len(designed_seq) == len(d["native_seq"])
    assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in designed_seq)


def test_jacobi_inverse_fold(mpnn, pdl1_data):
    """Jacobi inverse folding produces valid sequences."""
    d = pdl1_data
    all_S = jacobi_inverse_fold_batch(
        mpnn,
        X=d["X"][0], mask=d["mask"][0],
        residue_idx=d["residue_idx"][0],
        chain_encoding_all=d["chain_encoding_all"][0],
        keys=jax.random.split(jax.random.PRNGKey(42), 1),
        temperature=0.1,
    )
    designed_seq = seq_to_string(all_S[0])
    identity = seq_identity(designed_seq, d["native_seq"])

    print(f"\nJacobi sequence identity to native: {identity:.1%}")
    assert len(designed_seq) == len(d["native_seq"])
    assert all(c in "ACDEFGHIKLMNPQRSTVWYX" for c in designed_seq)


def test_refold_rmsd(mpnn, mosaic_mpnn, boltz, pdl1_data):
    """Inverse fold PDL1 N times, refold each with Boltz2, compare RMSDs.

    Compares three methods:
      1. Autoregressive sampling (jigandmpnn model)
      2. Jacobi decoding via jigandmpnn's encode/decode
      3. Jacobi decoding via mosaic's own ProteinMPNN
    """
    d = pdl1_data
    N = len(d["native_seq"])
    native_coords = d["native_coords"]

    # --- Batch inverse fold ---
    ar_keys = jax.random.split(jax.random.PRNGKey(0), NUM_SAMPLES)
    jacobi_keys = jax.random.split(jax.random.PRNGKey(1000), NUM_SAMPLES)
    mosaic_keys = jax.random.split(jax.random.PRNGKey(1000), NUM_SAMPLES)  # same seeds

    print(f"\nSampling {NUM_SAMPLES} autoregressive sequences (vmap)...")
    ar_S = autoregressive_inverse_fold_batch(
        mpnn,
        X=d["X"], S=d["S"], mask=d["mask"],
        residue_idx=d["residue_idx"],
        chain_encoding_all=d["chain_encoding_all"],
        keys=ar_keys, temperature=0.1,
    )

    print(f"Sampling {NUM_SAMPLES} Jacobi sequences via jigandmpnn (vmap)...")
    jacobi_S = jacobi_inverse_fold_batch(
        mpnn,
        X=d["X"][0], mask=d["mask"][0],
        residue_idx=d["residue_idx"][0],
        chain_encoding_all=d["chain_encoding_all"][0],
        keys=jacobi_keys, temperature=0.1,
    )

    print(f"Sampling {NUM_SAMPLES} Jacobi sequences via mosaic (vmap)...")
    mosaic_S = mosaic_jacobi_inverse_fold_batch(
        mosaic_mpnn,
        X=d["X"][0], mask=d["mask"][0],
        residue_idx=d["residue_idx"][0],
        chain_encoding_all=d["chain_encoding_all"][0],
        keys=mosaic_keys, temperature=0.1,
    )

    ar_seqs = [seq_to_string(ar_S[i]) for i in range(NUM_SAMPLES)]
    jacobi_seqs = [seq_to_string(jacobi_S[i]) for i in range(NUM_SAMPLES)]
    mosaic_seqs = [seq_to_string(mosaic_S[i]) for i in range(NUM_SAMPLES)]

    # --- Batch refold with precomputed Boltz2 features ---
    print(f"Precomputing Boltz2 features for length {N}...")
    folder = Boltz2Folder(boltz, N)

    fold_keys = jax.random.split(jax.random.key(2000), 3 * NUM_SAMPLES)

    print(f"Refolding {NUM_SAMPLES} autoregressive designs...")
    ar_coords_list = folder.fold_batch(ar_seqs, fold_keys[:NUM_SAMPLES])

    print(f"Refolding {NUM_SAMPLES} jigandmpnn Jacobi designs...")
    jacobi_coords_list = folder.fold_batch(jacobi_seqs, fold_keys[NUM_SAMPLES:2*NUM_SAMPLES])

    print(f"Refolding {NUM_SAMPLES} mosaic Jacobi designs...")
    mosaic_coords_list = folder.fold_batch(mosaic_seqs, fold_keys[2*NUM_SAMPLES:])

    # --- Compute RMSDs ---
    def stats(seqs, coords_list):
        rmsds = np.array([ca_rmsd(c, native_coords) for c in coords_list])
        ids = np.array([seq_identity(s, d["native_seq"]) for s in seqs])
        return ids, rmsds

    ar_ids, ar_rmsds = stats(ar_seqs, ar_coords_list)
    jacobi_ids, jacobi_rmsds = stats(jacobi_seqs, jacobi_coords_list)
    mosaic_ids, mosaic_rmsds = stats(mosaic_seqs, mosaic_coords_list)

    # --- Report ---
    def report(name, ids, rmsds):
        print(f"\n{name}:")
        for i in range(NUM_SAMPLES):
            print(f"  {i+1:2d}: identity={ids[i]:.1%}  RMSD={rmsds[i]:.2f} A")
        print(f"  Mean identity: {ids.mean():.1%} +/- {ids.std():.1%}")
        print(f"  Mean CA RMSD:  {rmsds.mean():.2f} +/- {rmsds.std():.2f} A")
        print(f"  Median CA RMSD: {np.median(rmsds):.2f} A")

    print(f"\n{'='*60}")
    print(f"PDL1 (4Z18-A) Inverse Folding + Refolding ({NUM_SAMPLES} samples)")
    print(f"{'='*60}")
    print(f"Native sequence length: {N}")

    report("Autoregressive (jigandmpnn)", ar_ids, ar_rmsds)
    report("Jacobi via jigandmpnn encode/decode", jacobi_ids, jacobi_rmsds)
    report("Jacobi via mosaic ProteinMPNN", mosaic_ids, mosaic_rmsds)

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Method':<35} {'Identity':>10} {'RMSD':>10} {'Median':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10}")
    print(f"{'Autoregressive (jigandmpnn)':<35} {ar_ids.mean():.1%}{'':<5} {ar_rmsds.mean():.2f} A{'':<3} {np.median(ar_rmsds):.2f} A")
    print(f"{'Jacobi (jigandmpnn)':<35} {jacobi_ids.mean():.1%}{'':<5} {jacobi_rmsds.mean():.2f} A{'':<3} {np.median(jacobi_rmsds):.2f} A")
    print(f"{'Jacobi (mosaic)':<35} {mosaic_ids.mean():.1%}{'':<5} {mosaic_rmsds.mean():.2f} A{'':<3} {np.median(mosaic_rmsds):.2f} A")

    # Sanity
    for name, rmsds in [("autoregressive", ar_rmsds), ("jacobi-jigandmpnn", jacobi_rmsds), ("jacobi-mosaic", mosaic_rmsds)]:
        assert np.all(np.isfinite(rmsds)), f"{name} has non-finite RMSDs"
        assert rmsds.mean() < 20.0, f"Mean {name} RMSD {rmsds.mean():.2f}A too high"
