"""ProteinMPNN model implementation in JAX/Equinox."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jaxtyping import Array, Float, Int, PRNGKeyArray

from jigandmpnn.backend import Linear, LayerNorm, Embedding, from_torch, register_from_torch
from jigandmpnn.modules.features import ProteinFeatures, ProteinFeaturesLigand
from jigandmpnn.modules.layers import EncLayer, DecLayer, DecLayerJ
from jigandmpnn.modules.utils import gather_nodes, cat_neighbors_nodes
from jigandmpnn.vendor.ligandmpnn import ProteinMPNN as TorchProteinMPNN

if TYPE_CHECKING:
    pass


@register_from_torch(TorchProteinMPNN)
class ProteinMPNN(eqx.Module):
    """ProteinMPNN model for protein sequence design.

    Supports `protein_mpnn`, `soluble_mpnn`, and `ligand_mpnn` model types.
    Inference only.
    """

    model_type: str = eqx.field(static=True)
    node_features: int = eqx.field(static=True)
    edge_features: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    # Feature extractor
    features: Union[ProteinFeatures, ProteinFeaturesLigand]

    # Encoder
    W_e: Linear
    encoder_layers: list[EncLayer]

    # Decoder (for sample/score, not encode)
    W_s: Embedding
    decoder_layers: list[DecLayer]
    W_out: Linear

    # LigandMPNN-specific layers (None for protein_mpnn/soluble_mpnn)
    W_v: Linear | None = None
    W_c: Linear | None = None
    W_nodes_y: Linear | None = None
    W_edges_y: Linear | None = None
    V_C: Linear | None = None
    V_C_norm: LayerNorm | None = None
    context_encoder_layers: list[DecLayer] | None = None
    y_context_encoder_layers: list[DecLayerJ] | None = None

    def encode(
        self,
        feature_dict: dict,
        key: PRNGKeyArray | None = None,
    ) -> tuple[
        Float[Array, "B N hidden_dim"],
        Float[Array, "B N K hidden_dim"],
        Int[Array, "B N K"],
    ]:
        """Encode protein structure into node and edge embeddings.

        Args:
            feature_dict: Dictionary containing:
                - X: Backbone coordinates [B, N, 4, 3]
                - S: Sequence (integer encoded) [B, N]
                - mask: Position mask [B, N]
                - R_idx: Residue indices [B, N]
                - chain_labels: Chain labels [B, N]
                For ligand_mpnn, additionally:
                - Y: Ligand atom coordinates [B, L, M, 3]
                - Y_t: Ligand atom types [B, L, M]
                - Y_m: Ligand atom mask [B, L, M]
            key: Optional PRNG key for coordinate noise (augment_eps).
                 If None and augment_eps > 0, no noise is added.

        Returns:
            h_V: Node embeddings [B, N, hidden_dim]
            h_E: Edge embeddings [B, N, K, hidden_dim]
            E_idx: Neighbor indices [B, N, K]
        """
        S_true = feature_dict["S"]
        mask = feature_dict["mask"]

        B, L = S_true.shape

        if self.model_type == "ligand_mpnn":
            # Extract features from ProteinFeaturesLigand
            V, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(feature_dict, key=key)

            # Initialize node embeddings to zero
            h_V = jnp.zeros((B, L, E.shape[-1]))

            # Project edge features
            h_E = self.W_e(E)

            # Project context features
            h_E_context = self.W_v(V)

            # Compute attention mask
            mask_attend = gather_nodes(mask[:, :, None], E_idx)[:, :, :, 0]
            mask_attend = mask[:, :, None] * mask_attend

            # Run encoder layers
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

            # Ligand context processing
            h_V_C = self.W_c(h_V)
            Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
            Y_nodes = self.W_nodes_y(Y_nodes)
            Y_edges = self.W_edges_y(Y_edges)

            # Interleaved context encoder layers
            for i in range(len(self.context_encoder_layers)):
                # Update ligand node embeddings
                Y_nodes = self.y_context_encoder_layers[i](
                    Y_nodes, Y_edges, Y_m, Y_m_edges
                )
                # Concatenate context features
                h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
                # Update protein node embeddings with context
                h_V_C = self.context_encoder_layers[i](
                    h_V_C, h_E_context_cat, mask, Y_m
                )

            # Final context projection and residual
            h_V_C = self.V_C(h_V_C)
            # Note: dropout is identity at inference
            h_V = h_V + self.V_C_norm(h_V_C)

        elif self.model_type in ("protein_mpnn", "soluble_mpnn"):
            # Extract features
            E, E_idx = self.features(feature_dict, key=key)

            # Initialize node embeddings to zero
            h_V = jnp.zeros((B, L, E.shape[-1]))

            # Project edge features
            h_E = self.W_e(E)

            # Compute attention mask
            mask_attend = gather_nodes(mask[:, :, None], E_idx)[:, :, :, 0]
            mask_attend = mask[:, :, None] * mask_attend

            # Run encoder layers
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        else:
            raise NotImplementedError(
                f"model_type '{self.model_type}' not yet implemented. "
                f"Only 'protein_mpnn', 'soluble_mpnn', and 'ligand_mpnn' are supported."
            )

        return h_V, h_E, E_idx

    def score(
        self,
        feature_dict: dict,
        *,
        key: PRNGKeyArray,
        use_sequence: bool = True,
        key_augment: PRNGKeyArray | None = None,
    ) -> dict:
        """Compute log-probabilities for a given sequence.

        This method computes the log-likelihood of a sequence given the structure,
        using autoregressive decoding with a specified decoding order.

        Args:
            feature_dict: Dictionary containing:
                - X: Backbone coordinates [B, N, 4, 3]
                - S: Sequence (integer encoded) [B, N]
                - mask: Position mask [B, N]
                - R_idx: Residue indices [B, N]
                - chain_labels: Chain labels [B, N]
                - chain_mask: Design mask [B, N] (1.0 = design, 0.0 = fixed).
                    Defaults to all 1s (design all positions).
                - decoding_order_noise: Random numbers for decoding order [B, N].
                    Positions are decoded in order of (chain_mask + eps) * |noise|.
                    Defaults to random values generated from the provided key.
                For ligand_mpnn, additionally:
                - Y, Y_t, Y_m: Ligand features
            key: PRNG key for random decoding order.
            use_sequence: If True, use teacher forcing (see true sequence during decoding).
                         If False, only use encoder information.
            key_augment: Optional PRNG key for coordinate noise (augment_eps).

        Returns:
            Dictionary containing:
                - S: Input sequence [B, N]
                - log_probs: Log probabilities [B, N, 21]
                - logits: Raw logits [B, N, 21]
                - decoding_order: The order in which positions were decoded [B, N]
        """
        S_true = feature_dict["S"]
        mask = feature_dict["mask"]
        B, L = S_true.shape

        # Default chain_mask: design all positions
        chain_mask = feature_dict.get("chain_mask", jnp.ones((B, L)))

        # Default decoding_order_noise: random values for randomized decoding order
        # Also support legacy "randn" key for backwards compatibility
        if "decoding_order_noise" in feature_dict:
            decoding_order_noise = feature_dict["decoding_order_noise"]
        elif "randn" in feature_dict:
            decoding_order_noise = feature_dict["randn"]
        else:
            # Generate random values for decoding order using the provided key
            decoding_order_noise = jax.random.normal(key, (B, L))

        # Encode structure
        h_V, h_E, E_idx = self.encode(feature_dict, key=key_augment)

        # Update chain_mask to include missing regions
        chain_mask = mask * chain_mask

        # Compute decoding order: positions with chain_mask=0 decoded first (fixed),
        # then positions with chain_mask=1 (to be designed)
        decoding_order = jnp.argsort((chain_mask + 0.0001) * jnp.abs(decoding_order_noise), axis=-1)

        # Build permutation matrix for decoding order
        permutation_matrix_reverse = jnn.one_hot(decoding_order, num_classes=L)

        # Build backward mask: which positions can attend to which
        # Lower triangular means position i can only see positions j < i in decoding order
        lower_tri = 1.0 - jnp.triu(jnp.ones((L, L)))
        order_mask_backward = jnp.einsum(
            "ij, biq, bjp->bqp",
            lower_tri,
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )

        # Gather the backward mask for each neighbor
        # order_mask_backward[b, i, j] = 1 if position j is decoded before position i
        mask_attend = jnp.take_along_axis(
            order_mask_backward,
            E_idx,
            axis=2,
        )[..., None]

        mask_1D = mask[:, :, None, None]
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        # Embed the true sequence
        h_S = self.W_s(S_true)

        # Build context from sequence and edges
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder-only context (no sequence information)
        h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        # Forward mask: positions that have NOT been decoded yet
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        if not use_sequence:
            # Only use encoder information (no teacher forcing)
            for layer in self.decoder_layers:
                h_V = layer(h_V, h_EXV_encoder_fw, mask)
        else:
            # Teacher forcing: use true sequence during decoding
            for layer in self.decoder_layers:
                # Combine sequence context (for decoded positions) with encoder context
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

        # Compute logits and log probabilities
        logits = self.W_out(h_V)
        log_probs = jnn.log_softmax(logits, axis=-1)

        return {
            "S": S_true,
            "log_probs": log_probs,
            "logits": logits,
            "decoding_order": decoding_order,
        }

    def sample(
        self,
        feature_dict: dict,
        key: PRNGKeyArray,
        temperature: float = 1.0,
        key_augment: PRNGKeyArray | None = None,
    ) -> dict:
        """Sample sequences autoregressively.

        This method generates new sequences by sampling from the predicted
        distribution at each position, following a decoding order.

        Args:
            feature_dict: Dictionary containing:
                - X: Backbone coordinates [B, N, 4, 3]
                - S: Sequence (integer encoded) [B, N] - used for fixed positions
                - mask: Position mask [B, N]
                - R_idx: Residue indices [B, N]
                - chain_labels: Chain labels [B, N]
                - chain_mask: Design mask [B, N] (1.0 = design, 0.0 = fixed).
                    Defaults to all 1s (design all positions).
                - decoding_order_noise: Random numbers for decoding order [B, N].
                    Positions are decoded in order of (chain_mask + eps) * |noise|.
                    Defaults to random values generated from the provided key.
                - bias: Amino acid bias per position [B, N, 21].
                    Defaults to zeros (no bias).
                For ligand_mpnn, additionally:
                - Y, Y_t, Y_m: Ligand features
            key: PRNG key for sampling.
            temperature: Sampling temperature. Higher = more random.
            key_augment: Optional PRNG key for coordinate noise (augment_eps).

        Returns:
            Dictionary containing:
                - S: Sampled sequence [B, N]
                - sampling_probs: Sampling probabilities [B, N, 20]
                - log_probs: Log probabilities [B, N, 21]
                - decoding_order: The order in which positions were decoded [B, N]
        """
        S_true = feature_dict["S"]
        mask = feature_dict["mask"]
        B, L = S_true.shape

        # Default chain_mask: design all positions
        chain_mask = feature_dict.get("chain_mask", jnp.ones((B, L)))

        # Default decoding_order_noise: random values for randomized decoding order
        # Also support legacy "randn" key for backwards compatibility
        if "decoding_order_noise" in feature_dict:
            decoding_order_noise = feature_dict["decoding_order_noise"]
        elif "randn" in feature_dict:
            decoding_order_noise = feature_dict["randn"]
        else:
            # Generate random values for decoding order using a split of the key
            key, order_key = jax.random.split(key)
            decoding_order_noise = jax.random.normal(order_key, (B, L))

        # Default bias: no bias
        bias = feature_dict.get("bias", jnp.zeros((B, L, 21)))

        # Encode structure
        h_V, h_E, E_idx = self.encode(feature_dict, key=key_augment)

        # Update chain_mask to include missing regions
        chain_mask = mask * chain_mask

        # Compute decoding order
        decoding_order = jnp.argsort((chain_mask + 0.0001) * jnp.abs(decoding_order_noise), axis=-1)

        # Build permutation matrix for decoding order
        permutation_matrix_reverse = jnn.one_hot(decoding_order, num_classes=L)

        # Build backward mask
        lower_tri = 1.0 - jnp.triu(jnp.ones((L, L)))
        order_mask_backward = jnp.einsum(
            "ij, biq, bjp->bqp",
            lower_tri,
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )

        # Gather the backward mask for each neighbor
        mask_attend = jnp.take_along_axis(order_mask_backward, E_idx, axis=2)[..., None]
        mask_1D = mask[:, :, None, None]
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        # Precompute encoder context (doesn't depend on sequence)
        h_EX_encoder = cat_neighbors_nodes(jnp.zeros((B, L, self.hidden_dim)), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        # Initialize state for scan
        h_S = jnp.zeros((B, L, self.hidden_dim))
        S = jnp.full((B, L), 20, dtype=jnp.int32)  # 20 = X (unknown)
        all_probs = jnp.zeros((B, L, 20))
        all_log_probs = jnp.zeros((B, L, 21))

        # Initialize h_V_stack: h_V_stack[0] = h_V, rest are zeros
        # h_V_stack[l] holds output of layer l-1 (or h_V for l=0)
        num_layers = len(self.decoder_layers)
        h_V_stack = jnp.zeros((num_layers + 1, B, L, self.hidden_dim))
        h_V_stack = h_V_stack.at[0].set(h_V)

        # Split key for each decoding step
        keys = jax.random.split(key, L)

        # Batch indices for advanced indexing
        batch_idx = jnp.arange(B)

        # Define the scan function
        def decode_step(carry, step_data):
            h_S, S, h_V_stack, all_probs, all_log_probs = carry
            t_idx, step_key = step_data  # t_idx is the step index (0 to L-1)

            # Get position to decode at this step
            t = decoding_order[:, t_idx]  # [B]

            # Gather masks for position t using advanced indexing
            chain_mask_t = chain_mask[batch_idx, t]  # [B]
            mask_t = mask[batch_idx, t]  # [B]
            bias_t = bias[batch_idx, t]  # [B, 21]

            # Gather edge info for position t
            E_idx_t = E_idx[batch_idx, t][:, None, :]  # [B, 1, K]
            h_E_t = h_E[batch_idx, t][:, None, :, :]  # [B, 1, K, H]

            # Build sequence context for position t
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)  # [B, 1, K, 2H]

            # Gather encoder context for position t
            h_EXV_encoder_t = h_EXV_encoder_fw[batch_idx, t][:, None, :, :]  # [B, 1, K, 3H]

            # Gather backward mask for position t
            mask_bw_t = mask_bw[batch_idx, t][:, None, :, :]  # [B, 1, K, 1]

            # Run through decoder layers
            h_V_stack_new = h_V_stack
            for l, layer in enumerate(self.decoder_layers):
                # Get neighbor node embeddings from previous layer
                # Note: must use h_V_stack_new since it's updated each iteration
                h_ESV_decoder_t = cat_neighbors_nodes(
                    h_V_stack_new[l], h_ES_t, E_idx_t
                )  # [B, 1, K, 3H]

                # Get node embedding for position t
                h_V_t = h_V_stack_new[l, batch_idx, t][:, None, :]  # [B, 1, H]

                # Combine decoder and encoder context
                h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t  # [B, 1, K, 3H]

                # Run decoder layer
                h_V_t_new = layer(h_V_t, h_ESV_t, mask_t[:, None])  # [B, 1, H]

                # Update h_V_stack at position t for layer l+1
                h_V_stack_new = h_V_stack_new.at[l + 1, batch_idx, t].set(h_V_t_new[:, 0])

            # Get final node embedding for position t
            h_V_t_final = h_V_stack_new[-1, batch_idx, t]  # [B, H]

            # Compute logits and probabilities
            logits = self.W_out(h_V_t_final)  # [B, 21]
            log_probs_t = jnn.log_softmax(logits, axis=-1)  # [B, 21]

            # Sample with temperature and bias
            probs = jnn.softmax((logits + bias_t) / temperature, axis=-1)  # [B, 21]
            # Renormalize to exclude X (index 20)
            probs_sample = probs[:, :20] / jnp.sum(probs[:, :20], axis=-1, keepdims=True)

            # Sample from categorical distribution
            S_t = jax.random.categorical(step_key, jnp.log(probs_sample + 1e-10), axis=-1)  # [B]

            # Use true sequence for fixed positions
            S_true_t = S_true[batch_idx, t]  # [B]
            S_t = jnp.where(chain_mask_t > 0.5, S_t, S_true_t).astype(jnp.int32)

            # Update outputs
            all_probs_new = all_probs.at[batch_idx, t].set(
                chain_mask_t[:, None] * probs_sample
            )
            all_log_probs_new = all_log_probs.at[batch_idx, t].set(
                chain_mask_t[:, None] * log_probs_t
            )

            # Update sequence embeddings
            h_S_t_new = self.W_s(S_t)  # [B, H]
            h_S_new = h_S.at[batch_idx, t].set(h_S_t_new)

            # Update sampled sequence
            S_new = S.at[batch_idx, t].set(S_t)

            return (h_S_new, S_new, h_V_stack_new, all_probs_new, all_log_probs_new), None

        # Run scan over decoding steps
        init_carry = (h_S, S, h_V_stack, all_probs, all_log_probs)
        xs = (jnp.arange(L), keys)
        (h_S_final, S_final, _, all_probs_final, all_log_probs_final), _ = jax.lax.scan(
            decode_step, init_carry, xs
        )

        return {
            "S": S_final,
            "sampling_probs": all_probs_final,
            "log_probs": all_log_probs_final,
            "decoding_order": decoding_order,
        }

    @staticmethod
    def from_torch(m: TorchProteinMPNN) -> "ProteinMPNN":
        """Convert PyTorch ProteinMPNN to JAX/Equinox.

        Supports protein_mpnn, soluble_mpnn, and ligand_mpnn model types.
        """
        model_type = m.model_type

        if model_type not in ("protein_mpnn", "soluble_mpnn", "ligand_mpnn"):
            raise NotImplementedError(
                f"model_type '{model_type}' not yet implemented. "
                f"Only 'protein_mpnn', 'soluble_mpnn', and 'ligand_mpnn' are supported."
            )

        # Common layers
        kwargs = {
            "model_type": model_type,
            "node_features": m.node_features,
            "edge_features": m.edge_features,
            "hidden_dim": m.hidden_dim,
            "features": from_torch(m.features),
            "W_e": from_torch(m.W_e),
            "encoder_layers": [from_torch(layer) for layer in m.encoder_layers],
            "W_s": from_torch(m.W_s),
            "decoder_layers": [from_torch(layer) for layer in m.decoder_layers],
            "W_out": from_torch(m.W_out),
        }

        # LigandMPNN-specific layers
        if model_type == "ligand_mpnn":
            kwargs.update({
                "W_v": from_torch(m.W_v),
                "W_c": from_torch(m.W_c),
                "W_nodes_y": from_torch(m.W_nodes_y),
                "W_edges_y": from_torch(m.W_edges_y),
                "V_C": from_torch(m.V_C),
                "V_C_norm": from_torch(m.V_C_norm),
                "context_encoder_layers": [from_torch(layer) for layer in m.context_encoder_layers],
                "y_context_encoder_layers": [from_torch(layer) for layer in m.y_context_encoder_layers],
            })

        return ProteinMPNN(**kwargs)
