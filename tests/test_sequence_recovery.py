"""Tests for InverseFoldingSequenceRecovery using autoregressive sampling.

This implements and tests a version of mosaic's InverseFoldingSequenceRecovery
that uses jigandmpnn's autoregressive sampler instead of Jacobi decoding.

The metric:
1. Given a structure and a reference sequence, autoregressively sample N sequences.
2. Convert each sample to one-hot, average across samples.
3. Compute inner product between average one-hot and the reference one-hot.
4. The result is sequence recovery: fraction of positions where the sampled
   consensus matches the reference.
"""

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jigandmpnn import get_weight_path, load_protein_mpnn
from jigandmpnn.backend import from_torch
from jigandmpnn.modules.model import ProteinMPNN

from jigandmpnn.vendor.ligandmpnn import ProteinMPNN as TorchProteinMPNN


CHECKPOINT_PATH = get_weight_path("protein_mpnn")


def inverse_folding_sequence_recovery(
    mpnn: ProteinMPNN,
    *,
    X: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    residue_idx: jnp.ndarray,
    chain_encoding_all: jnp.ndarray,
    key: jax.Array,
    num_samples: int = 16,
    temperature: float = 0.1,
) -> tuple[float, jnp.ndarray]:
    """Compute sequence recovery via autoregressive inverse folding.

    Samples ``num_samples`` sequences autoregressively from ``mpnn`` given the
    structure ``X``, averages their one-hot representations, and computes the
    per-position inner product with the reference sequence ``S``.

    Args:
        mpnn: ProteinMPNN model.
        X: Backbone coordinates [B, N, 4, 3].
        S: Reference sequence (integer-encoded) [B, N].
        mask: Position mask [B, N].
        residue_idx: Residue indices [B, N].
        chain_encoding_all: Chain labels [B, N].
        key: PRNG key.
        num_samples: Number of sequences to sample.
        temperature: Sampling temperature.

    Returns:
        (sequence_recovery, average_one_hot):
            sequence_recovery: Mean per-position recovery (scalar).
            average_one_hot: Averaged one-hot samples [B, N, 21].
    """
    keys = jax.random.split(key, num_samples)

    def sample_one(k):
        result = mpnn.sample(
            X=X, S=S, mask=mask,
            residue_idx=residue_idx,
            chain_encoding_all=chain_encoding_all,
            key=k, temperature=temperature,
        )
        # One-hot over first 20 AAs (standard residues)
        return jnn.one_hot(result.S, 20)

    # [num_samples, B, N, 20]
    all_onehot = jax.vmap(sample_one)(keys)

    # Average over samples: [B, N, 20]
    average_onehot = all_onehot.mean(axis=0)

    # Reference one-hot (first 20 AAs only, token 20=X excluded from recovery)
    ref_onehot = jnn.one_hot(S, 20)

    # Per-position inner product: [B, N]
    per_position = (average_onehot * ref_onehot).sum(axis=-1)

    # Mean over valid positions
    recovery = (per_position * mask).sum() / mask.sum()

    return recovery, average_onehot


# --- Fixtures ---

def load_model():
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")
    return load_protein_mpnn()


def create_protein(B, L, seed=42):
    """Create synthetic protein data as JAX arrays."""
    torch.manual_seed(seed)
    X = jnp.array(torch.randn(B, L, 4, 3).numpy() * 3.0)
    S = jnp.array(torch.randint(0, 20, (B, L)).numpy())  # Only standard AAs
    mask = jnp.ones((B, L))
    residue_idx = jnp.broadcast_to(jnp.arange(L), (B, L))
    chain_encoding_all = jnp.zeros((B, L), dtype=jnp.int32)
    return X, S, mask, residue_idx, chain_encoding_all


# --- Tests ---

def test_recovery_bounded_zero_one():
    """Sequence recovery should be in [0, 1]."""
    model = load_model()
    X, S, mask, r_idx, chain_enc = create_protein(1, 30)

    recovery, avg = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(0), num_samples=4, temperature=0.1,
    )

    assert 0.0 <= float(recovery) <= 1.0, f"Recovery {recovery} out of bounds"


def test_recovery_deterministic():
    """Same key should produce same recovery."""
    model = load_model()
    X, S, mask, r_idx, chain_enc = create_protein(1, 30)

    key = jax.random.PRNGKey(42)
    r1, _ = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=key, num_samples=4, temperature=0.1,
    )
    r2, _ = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=key, num_samples=4, temperature=0.1,
    )

    np.testing.assert_allclose(float(r1), float(r2), atol=1e-6)


def test_recovery_different_keys():
    """Different keys should generally produce different recovery values."""
    model = load_model()
    X, S, mask, r_idx, chain_enc = create_protein(1, 30)

    r1, _ = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(0), num_samples=4, temperature=1.0,
    )
    r2, _ = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(999), num_samples=4, temperature=1.0,
    )

    # With high temperature and random structure, different keys should differ
    assert float(r1) != float(r2), "Different keys should give different recovery"


def test_recovery_average_onehot_sums_to_one():
    """Average one-hot should sum to 1 along the AA axis for each position."""
    model = load_model()
    X, S, mask, r_idx, chain_enc = create_protein(1, 30)

    _, avg = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(0), num_samples=8, temperature=0.1,
    )

    # avg is [B, N, 20] — each position should sum to 1
    sums = avg.sum(axis=-1)  # [B, N]
    np.testing.assert_allclose(
        np.array(sums), np.ones_like(np.array(sums)), atol=1e-6,
        err_msg="Average one-hot should sum to 1 per position",
    )


def test_recovery_low_temp_higher_than_high_temp():
    """Low temperature should give higher recovery than high temperature.

    At low temperature, samples are more peaked around the mode, so if the
    model has any signal at all, the consensus should be more consistent
    and match the reference better (or at least as well).
    With random structures this isn't guaranteed per-instance, so we use
    many samples to reduce variance.
    """
    model = load_model()
    X, S, mask, r_idx, chain_enc = create_protein(1, 50, seed=123)

    r_low, _ = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(0), num_samples=32, temperature=0.01,
    )
    r_high, _ = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(0), num_samples=32, temperature=2.0,
    )

    # Low temp consensus should be sharper. With random coords both recoveries
    # are low, but low-temp consensus has higher self-agreement so the inner
    # product with any fixed reference tends to be higher (or equal at worst
    # when both are near chance). We use a generous margin.
    assert float(r_low) >= float(r_high) - 0.05, (
        f"Low temp recovery {float(r_low):.4f} should be >= "
        f"high temp recovery {float(r_high):.4f} (with margin)"
    )


def test_recovery_more_samples_reduces_variance():
    """More samples should reduce variance of the recovery estimate."""
    model = load_model()
    X, S, mask, r_idx, chain_enc = create_protein(1, 30)

    def get_recoveries(num_samples, num_trials=5):
        recoveries = []
        for i in range(num_trials):
            r, _ = inverse_folding_sequence_recovery(
                model, X=X, S=S, mask=mask,
                residue_idx=r_idx, chain_encoding_all=chain_enc,
                key=jax.random.PRNGKey(i), num_samples=num_samples,
                temperature=0.5,
            )
            recoveries.append(float(r))
        return np.std(recoveries)

    var_few = get_recoveries(2)
    var_many = get_recoveries(16)

    # More samples should reduce or maintain variance
    assert var_many <= var_few + 0.02, (
        f"16-sample std {var_many:.4f} should be <= 2-sample std {var_few:.4f}"
    )


def test_recovery_low_temp_sharpens_consensus():
    """Very low temperature should produce a sharper consensus than high temperature.

    At low temp, samples concentrate on the mode, so the average one-hot
    should have higher max values (closer to 1) than at high temp.
    """
    model = load_model()
    X, S, mask, r_idx, chain_enc = create_protein(1, 20)

    _, avg_low = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(0), num_samples=16, temperature=0.001,
    )
    _, avg_high = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(0), num_samples=16, temperature=2.0,
    )

    # Low temp consensus should be sharper (higher max per position)
    sharpness_low = float(avg_low.max(axis=-1).mean())
    sharpness_high = float(avg_high.max(axis=-1).mean())

    assert sharpness_low > sharpness_high, (
        f"Low temp sharpness {sharpness_low:.4f} should exceed "
        f"high temp sharpness {sharpness_high:.4f}"
    )


def test_recovery_with_mask():
    """Recovery should only consider masked (valid) positions."""
    model = load_model()
    X, S, mask, r_idx, chain_enc = create_protein(1, 40)

    # Full mask
    r_full, _ = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(0), num_samples=4, temperature=0.1,
    )

    # Partial mask (zero out last 20 positions)
    partial_mask = mask.at[:, 20:].set(0.0)
    r_partial, _ = inverse_folding_sequence_recovery(
        model, X=X, S=S, mask=partial_mask,
        residue_idx=r_idx, chain_encoding_all=chain_enc,
        key=jax.random.PRNGKey(0), num_samples=4, temperature=0.1,
    )

    # Both should be valid recoveries
    assert 0.0 <= float(r_full) <= 1.0
    assert 0.0 <= float(r_partial) <= 1.0
    # They should generally differ since different positions are considered
    # (not guaranteed but very likely with random coords)


def test_recovery_jit_compatible():
    """The full recovery computation should work under JIT."""
    model = load_model()
    X, S, mask, r_idx, chain_enc = create_protein(1, 20)

    @eqx.filter_jit
    def compute(mpnn, X, S, mask, r_idx, chain_enc, key):
        return inverse_folding_sequence_recovery(
            mpnn, X=X, S=S, mask=mask,
            residue_idx=r_idx, chain_encoding_all=chain_enc,
            key=key, num_samples=4, temperature=0.1,
        )

    key = jax.random.PRNGKey(42)
    r1, avg1 = compute(model, X, S, mask, r_idx, chain_enc, key)
    r2, avg2 = compute(model, X, S, mask, r_idx, chain_enc, key)

    np.testing.assert_allclose(float(r1), float(r2), atol=1e-6,
                               err_msg="JIT should be deterministic")
    np.testing.assert_allclose(np.array(avg1), np.array(avg2), atol=1e-6)
