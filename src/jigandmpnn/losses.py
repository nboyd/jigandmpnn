"""Mosaic-compatible loss terms using jigandmpnn's autoregressive sampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from mosaic.common import LossTerm
from mosaic.losses.protein_mpnn import boltz_to_mpnn_matrix
from mosaic.losses.structure_prediction import AbstractStructureOutput

from jigandmpnn.modules.model import ProteinMPNN


class AutoregressiveSequenceRecovery(LossTerm):
    """Inner product of binder sequence and average autoregressive MPNN samples.

    Like mosaic's ``InverseFoldingSequenceRecovery`` but uses jigandmpnn's
    autoregressive ``sample()`` instead of Jacobi decoding.

    Args:
        mpnn: jigandmpnn ProteinMPNN model.
        temp: Sampling temperature.
        num_samples: Number of sequences to sample and average.
        bias: Optional per-position amino acid bias [N, 20] in Boltz token order.
    """

    mpnn: ProteinMPNN
    temp: float
    num_samples: int = 16
    bias: Float[Array, "N 20"] | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        coords = output.backbone_coordinates  # (total_N, 4, 3)
        full_sequence = output.full_sequence  # (total_N, 20)
        asym_id = output.asym_id  # (total_N,)
        residue_idx_raw = output.residue_idx  # (total_N,)

        binder_length = sequence.shape[0]
        total_length = full_sequence.shape[0]

        # --- Prepare residue_idx (same logic as mosaic ProteinMPNNLoss) ---
        chain_lengths = (asym_id[:, None] == np.arange(16)[None]).sum(-2)
        res_idx_adjustment = jnp.cumsum(chain_lengths, -1) - chain_lengths
        residue_idx = (
            residue_idx_raw
            + (asym_id[:, None] == np.arange(16)[None]) @ res_idx_adjustment
        )
        residue_idx += 100 * asym_id

        # --- Build full sequence with current binder optimisation variable ---
        full_seq = full_sequence.at[:binder_length].set(sequence)

        # Convert TOKENS order → MPNN alphabet
        conv = boltz_to_mpnn_matrix()  # (20, 21)
        seq_mpnn = full_seq @ conv
        S_int = seq_mpnn.argmax(-1)  # integer sequence for sample()

        mask = jnp.ones(total_length)
        # chain_mask: only redesign binder positions
        chain_mask = jnp.zeros(total_length).at[:binder_length].set(1.0)

        # --- Prepare bias in MPNN alphabet if provided ---
        if self.bias is not None:
            # bias is (binder_length, 20) in Boltz order → pad to (total, 21) in MPNN order
            bias_boltz_full = jnp.zeros((total_length, 20))
            bias_boltz_full = bias_boltz_full.at[:binder_length].set(self.bias)
            bias_mpnn_20 = bias_boltz_full @ conv[:, :20]  # (total, 20)
            # Pad with zero column for X token → (total, 21)
            bias_mpnn = jnp.concatenate(
                [bias_mpnn_20, jnp.zeros((total_length, 1))], axis=-1
            )
        else:
            bias_mpnn = None

        # --- Add batch dim (sample() expects (B, N, ...)) ---
        X = coords[None]
        S = S_int[None]
        mask_b = mask[None]
        residue_idx_b = residue_idx[None].astype(jnp.int32)
        chain_enc_b = asym_id[None].astype(jnp.int32)
        chain_mask_b = chain_mask[None]
        bias_b = bias_mpnn[None] if bias_mpnn is not None else None

        # --- vmap over num_samples keys ---
        def _single_sample(k):
            result = self.mpnn.sample(
                X=X,
                S=S,
                mask=mask_b,
                residue_idx=residue_idx_b,
                chain_encoding_all=chain_enc_b,
                chain_mask=chain_mask_b,
                bias=bias_b,
                key=k,
                temperature=self.temp,
            )
            # result.S is (1, N) in MPNN alphabet; take binder portion
            sampled_S = result.S[0, :binder_length]
            # Convert MPNN int → one-hot(21) → Boltz order(20)
            return jax.nn.one_hot(sampled_S, 21) @ conv.T  # (binder_len, 20)

        keys = jax.random.split(key, self.num_samples)
        sequences = jax.vmap(_single_sample)(keys)  # (num_samples, binder_len, 20)

        # --- Average and stop gradient (same pattern as mosaic's version) ---
        average_sequence = jax.lax.stop_gradient(sequences.mean(0))

        # --- Inner product loss ---
        ip = (average_sequence * sequence).sum(-1).mean()
        return -ip, {"sequence_recovery": ip}
