"""Protein feature extraction modules in JAX/Equinox."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from jigandmpnn.backend import Linear, LayerNorm, from_torch, register_from_torch
from jigandmpnn.modules.layers import PositionalEncodings
from jigandmpnn.modules.utils import gather_edges
from jigandmpnn.vendor.ligandmpnn import (
    ProteinFeatures as TorchProteinFeatures,
    ProteinFeaturesLigand as TorchProteinFeaturesLigand,
)

if TYPE_CHECKING:
    pass


@register_from_torch(TorchProteinFeatures)
class ProteinFeatures(eqx.Module):
    """Extract edge features from protein backbone coordinates."""

    edge_features: int = eqx.field(static=True)
    node_features: int = eqx.field(static=True)
    top_k: int = eqx.field(static=True)
    augment_eps: float = eqx.field(static=True)
    num_rbf: int = eqx.field(static=True)
    num_positional_embeddings: int = eqx.field(static=True)

    embeddings: PositionalEncodings
    edge_embedding: Linear
    norm_edges: LayerNorm

    def _dist(
        self,
        X: Float[Array, "B N 3"],
        mask: Float[Array, "B N"],
        eps: float = 1e-6,
    ) -> tuple[Float[Array, "B N K"], Int[Array, "B N K"]]:
        """Compute pairwise distances and find top-k nearest neighbors.

        Args:
            X: Coordinates (typically Ca atoms) [B, N, 3]
            mask: Mask for valid positions [B, N]

        Returns:
            D_neighbors: Distances to k nearest neighbors [B, N, K]
            E_idx: Indices of k nearest neighbors [B, N, K]
        """
        # Compute 2D mask
        mask_2D = mask[:, :, None] * mask[:, None, :]  # [B, N, N]

        # Compute pairwise distances
        dX = X[:, :, None, :] - X[:, None, :, :]  # [B, N, N, 3]
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=-1) + eps)  # [B, N, N]

        # Adjust distances for masked positions (set to max)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max

        # Find top-k smallest distances
        # Note: JAX top_k returns largest, so we negate
        k = min(self.top_k, X.shape[1])
        neg_D_neighbors, E_idx = jax.lax.top_k(-D_adjust, k)
        D_neighbors = -neg_D_neighbors

        return D_neighbors, E_idx

    def _rbf(self, D: Float[Array, "..."]) -> Float[Array, "... num_rbf"]:
        """Compute radial basis function features.

        Args:
            D: Distance values

        Returns:
            RBF features
        """
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu.reshape([1] * len(D.shape) + [-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = D[..., None]
        RBF = jnp.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(
        self,
        A: Float[Array, "B N 3"],
        B: Float[Array, "B N 3"],
        E_idx: Int[Array, "B N K"],
    ) -> Float[Array, "B N K num_rbf"]:
        """Compute RBF features between two atom types.

        Args:
            A: First atom coordinates [B, N, 3]
            B: Second atom coordinates [B, N, 3]
            E_idx: Neighbor indices [B, N, K]

        Returns:
            RBF features [B, N, K, num_rbf]
        """
        # Compute all pairwise distances [B, N, N]
        D_A_B = jnp.sqrt(
            jnp.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, axis=-1) + 1e-6
        )

        # Gather neighbor distances [B, N, K]
        D_A_B_neighbors = gather_edges(D_A_B[..., None], E_idx)[..., 0]

        # Compute RBF
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def __call__(
        self,
        input_features: dict,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "B N K edge_features"], Int[Array, "B N K"]]:
        """Extract edge features from protein structure.

        Args:
            input_features: Dictionary containing:
                - X: Backbone coordinates [B, N, 4, 3] (N, CA, C, O)
                - mask: Position mask [B, N]
                - R_idx: Residue indices [B, N]
                - chain_labels: Chain labels [B, N]
            key: Optional PRNG key for coordinate noise (augment_eps).
                 If None and augment_eps > 0, no noise is added.

        Returns:
            E: Edge features [B, N, K, edge_features]
            E_idx: Neighbor indices [B, N, K]
        """
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        # Add coordinate noise if augment_eps > 0 and key is provided
        if self.augment_eps > 0 and key is not None:
            X = X + self.augment_eps * jax.random.normal(key, X.shape)

        # Extract backbone atoms
        N = X[:, :, 0, :]   # [B, L, 3]
        Ca = X[:, :, 1, :]  # [B, L, 3]
        C = X[:, :, 2, :]   # [B, L, 3]
        O = X[:, :, 3, :]   # [B, L, 3]

        # Compute Cb (virtual beta carbon)
        b = Ca - N
        c = C - Ca
        a = jnp.cross(b, c)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        # Compute distances and find neighbors
        D_neighbors, E_idx = self._dist(Ca, mask)

        # Compute all 25 RBF features (symmetric and asymmetric pairs)
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))       # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))   # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))   # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))   # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))   # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))   # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))   # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))   # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))   # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))   # C-O

        RBF_all = jnp.concatenate(RBF_all, axis=-1)  # [B, N, K, 25*num_rbf]

        # Compute positional encodings
        # offset = R_idx[i] - R_idx[j] for neighbors
        offset = R_idx[:, :, None] - R_idx[:, None, :]  # [B, N, N]
        offset = gather_edges(offset[..., None], E_idx)[..., 0]  # [B, N, K]

        # Chain mask: 1 if same chain, 0 if different
        d_chains = (chain_labels[:, :, None] == chain_labels[:, None, :]).astype(jnp.int32)
        E_chains = gather_edges(d_chains[..., None], E_idx)[..., 0]  # [B, N, K]

        E_positional = self.embeddings(offset.astype(jnp.int32), E_chains)

        # Concatenate positional and RBF features
        E = jnp.concatenate([E_positional, RBF_all], axis=-1)

        # Project through linear and normalize
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx

    @staticmethod
    def from_torch(m: TorchProteinFeatures) -> "ProteinFeatures":
        return ProteinFeatures(
            edge_features=m.edge_features,
            node_features=m.node_features,
            top_k=m.top_k,
            augment_eps=m.augment_eps,
            num_rbf=m.num_rbf,
            num_positional_embeddings=m.num_positional_embeddings,
            embeddings=from_torch(m.embeddings),
            edge_embedding=from_torch(m.edge_embedding),
            norm_edges=from_torch(m.norm_edges),
        )


@register_from_torch(TorchProteinFeaturesLigand)
class ProteinFeaturesLigand(eqx.Module):
    """Extract features for ligand-aware protein design."""

    edge_features: int = eqx.field(static=True)
    node_features: int = eqx.field(static=True)
    top_k: int = eqx.field(static=True)
    augment_eps: float = eqx.field(static=True)
    num_rbf: int = eqx.field(static=True)
    num_positional_embeddings: int = eqx.field(static=True)
    atom_context_num: int = eqx.field(static=True)
    use_side_chains: bool = eqx.field(static=True)

    # Backbone edge features
    embeddings: PositionalEncodings
    edge_embedding: Linear
    norm_edges: LayerNorm

    # Ligand context features
    node_project_down: Linear
    norm_nodes: LayerNorm
    type_linear: Linear

    # Ligand graph features
    y_nodes: Linear
    y_edges: Linear
    norm_y_edges: LayerNorm
    norm_y_nodes: LayerNorm

    # Constant tensors
    periodic_table_features: jnp.ndarray
    side_chain_atom_types: jnp.ndarray

    def _dist(
        self,
        X: Float[Array, "B N 3"],
        mask: Float[Array, "B N"],
        eps: float = 1e-6,
    ) -> tuple[Float[Array, "B N K"], Int[Array, "B N K"]]:
        """Compute pairwise distances and find top-k nearest neighbors."""
        mask_2D = mask[:, :, None] * mask[:, None, :]
        dX = X[:, :, None, :] - X[:, None, :, :]
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=-1) + eps)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        k = min(self.top_k, X.shape[1])
        neg_D_neighbors, E_idx = jax.lax.top_k(-D_adjust, k)
        D_neighbors = -neg_D_neighbors
        return D_neighbors, E_idx

    def _rbf(self, D: Float[Array, "..."]) -> Float[Array, "... num_rbf"]:
        """Compute radial basis function features."""
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu.reshape([1] * len(D.shape) + [-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = D[..., None]
        RBF = jnp.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(
        self,
        A: Float[Array, "B N 3"],
        B: Float[Array, "B N 3"],
        E_idx: Int[Array, "B N K"],
    ) -> Float[Array, "B N K num_rbf"]:
        """Compute RBF features between two atom types."""
        D_A_B = jnp.sqrt(
            jnp.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, axis=-1) + 1e-6
        )
        D_A_B_neighbors = gather_edges(D_A_B[..., None], E_idx)[..., 0]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _make_angle_features(
        self,
        A: Float[Array, "B L 3"],
        B: Float[Array, "B L 3"],
        C: Float[Array, "B L 3"],
        Y: Float[Array, "B L M 3"],
    ) -> Float[Array, "B L M 4"]:
        """Compute angular features for ligand atoms relative to backbone frame."""
        v1 = A - B
        v2 = C - B

        # Normalize v1 to get e1
        e1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)

        # Gram-Schmidt to get e2
        e1_v2_dot = jnp.sum(e1 * v2, axis=-1, keepdims=True)
        u2 = v2 - e1 * e1_v2_dot
        e2 = u2 / (jnp.linalg.norm(u2, axis=-1, keepdims=True) + 1e-8)

        # Cross product for e3
        e3 = jnp.cross(e1, e2)

        # Build rotation matrix [B, L, 3, 3]
        R_residue = jnp.stack([e1, e2, e3], axis=-1)

        # Transform Y to local coordinates
        # Y - B[:, :, None, :] gives [B, L, M, 3]
        # R_residue is [B, L, 3, 3], need to do einsum
        local_vectors = jnp.einsum("blqp,blyq->blyp", R_residue, Y - B[:, :, None, :])

        # Compute angular features
        rxy = jnp.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = jnp.linalg.norm(local_vectors, axis=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        f = jnp.stack([f1, f2, f3, f4], axis=-1)
        return f

    def __call__(
        self,
        input_features: dict,
        key: PRNGKeyArray | None = None,
    ) -> tuple[
        Float[Array, "B L M node_features"],  # V
        Float[Array, "B L K edge_features"],  # E
        Int[Array, "B L K"],                   # E_idx
        Float[Array, "B L M node_features"],  # Y_nodes
        Float[Array, "B L M M node_features"], # Y_edges
        Float[Array, "B L M"],                 # Y_m
    ]:
        """Extract features for ligand-aware design.

        Args:
            input_features: Dictionary containing:
                - X: Backbone coordinates [B, L, 4, 3]
                - Y: Ligand atom coordinates [B, L, M, 3]
                - Y_t: Ligand atom types [B, L, M]
                - Y_m: Ligand atom mask [B, L, M]
                - mask: Position mask [B, L]
                - R_idx: Residue indices [B, L]
                - chain_labels: Chain labels [B, L]
            key: Optional PRNG key for coordinate noise (augment_eps).
                 If None and augment_eps > 0, no noise is added.

        Returns:
            V: Ligand context node features [B, L, M, node_features]
            E: Edge features [B, L, K, edge_features]
            E_idx: Neighbor indices [B, L, K]
            Y_nodes: Ligand node features [B, L, M, node_features]
            Y_edges: Ligand edge features [B, L, M, M, node_features]
            Y_m: Ligand mask [B, L, M]
        """
        Y = input_features["Y"]
        Y_m = input_features["Y_m"]
        Y_t = input_features["Y_t"]
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        # Add coordinate noise if augment_eps > 0 and key is provided
        if self.augment_eps > 0 and key is not None:
            key_x, key_y = jax.random.split(key)
            X = X + self.augment_eps * jax.random.normal(key_x, X.shape)
            Y = Y + self.augment_eps * jax.random.normal(key_y, Y.shape)

        B, L, _, _ = X.shape

        # Extract backbone atoms
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        # Compute Cb
        b = Ca - N
        c = C - Ca
        a = jnp.cross(b, c)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        # Compute distances and neighbors
        D_neighbors, E_idx = self._dist(Ca, mask)

        # Compute 25 RBF features
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))
        RBF_all.append(self._get_rbf(N, N, E_idx))
        RBF_all.append(self._get_rbf(C, C, E_idx))
        RBF_all.append(self._get_rbf(O, O, E_idx))
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))
        RBF_all.append(self._get_rbf(Ca, N, E_idx))
        RBF_all.append(self._get_rbf(Ca, C, E_idx))
        RBF_all.append(self._get_rbf(Ca, O, E_idx))
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))
        RBF_all.append(self._get_rbf(N, C, E_idx))
        RBF_all.append(self._get_rbf(N, O, E_idx))
        RBF_all.append(self._get_rbf(N, Cb, E_idx))
        RBF_all.append(self._get_rbf(Cb, C, E_idx))
        RBF_all.append(self._get_rbf(Cb, O, E_idx))
        RBF_all.append(self._get_rbf(O, C, E_idx))
        RBF_all.append(self._get_rbf(N, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, Ca, E_idx))
        RBF_all.append(self._get_rbf(O, Ca, E_idx))
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, N, E_idx))
        RBF_all.append(self._get_rbf(O, N, E_idx))
        RBF_all.append(self._get_rbf(Cb, N, E_idx))
        RBF_all.append(self._get_rbf(C, Cb, E_idx))
        RBF_all.append(self._get_rbf(O, Cb, E_idx))
        RBF_all.append(self._get_rbf(C, O, E_idx))
        RBF_all = jnp.concatenate(RBF_all, axis=-1)

        # Positional encodings
        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[..., None], E_idx)[..., 0]
        d_chains = (chain_labels[:, :, None] == chain_labels[:, None, :]).astype(jnp.int32)
        E_chains = gather_edges(d_chains[..., None], E_idx)[..., 0]
        E_positional = self.embeddings(offset.astype(jnp.int32), E_chains)

        # Edge features
        E = jnp.concatenate([E_positional, RBF_all], axis=-1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        # Ligand atom type features
        Y_t = Y_t.astype(jnp.int32)
        Y_t_g = self.periodic_table_features[1][Y_t]  # group
        Y_t_p = self.periodic_table_features[2][Y_t]  # period

        Y_t_g_1hot = jax.nn.one_hot(Y_t_g, 19)
        Y_t_p_1hot = jax.nn.one_hot(Y_t_p, 8)
        Y_t_1hot = jax.nn.one_hot(Y_t, 120)

        Y_t_1hot_full = jnp.concatenate([Y_t_1hot, Y_t_g_1hot, Y_t_p_1hot], axis=-1)
        Y_t_1hot_proj = self.type_linear(Y_t_1hot_full.astype(jnp.float32))

        # RBF distances from backbone to ligand atoms
        D_N_Y = self._rbf(jnp.sqrt(jnp.sum((N[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6))
        D_Ca_Y = self._rbf(jnp.sqrt(jnp.sum((Ca[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6))
        D_C_Y = self._rbf(jnp.sqrt(jnp.sum((C[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6))
        D_O_Y = self._rbf(jnp.sqrt(jnp.sum((O[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6))
        D_Cb_Y = self._rbf(jnp.sqrt(jnp.sum((Cb[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6))

        # Angular features
        f_angles = self._make_angle_features(N, Ca, C, Y)

        # Node features for ligand context
        D_all = jnp.concatenate([D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot_proj, f_angles], axis=-1)
        V = self.node_project_down(D_all)
        V = self.norm_nodes(V)

        # Ligand graph edges (pairwise distances)
        Y_edges = self._rbf(
            jnp.sqrt(jnp.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, axis=-1) + 1e-6)
        )
        Y_edges = self.y_edges(Y_edges)
        Y_edges = self.norm_y_edges(Y_edges)

        # Ligand graph nodes
        Y_nodes = self.y_nodes(Y_t_1hot_full.astype(jnp.float32))
        Y_nodes = self.norm_y_nodes(Y_nodes)

        return V, E, E_idx, Y_nodes, Y_edges, Y_m

    @staticmethod
    def from_torch(m: TorchProteinFeaturesLigand) -> "ProteinFeaturesLigand":
        return ProteinFeaturesLigand(
            edge_features=m.edge_features,
            node_features=m.node_features,
            top_k=m.top_k,
            augment_eps=m.augment_eps,
            num_rbf=m.num_rbf,
            num_positional_embeddings=m.num_positional_embeddings,
            atom_context_num=m.atom_context_num,
            use_side_chains=m.use_side_chains,
            embeddings=from_torch(m.embeddings),
            edge_embedding=from_torch(m.edge_embedding),
            norm_edges=from_torch(m.norm_edges),
            node_project_down=from_torch(m.node_project_down),
            norm_nodes=from_torch(m.norm_nodes),
            type_linear=from_torch(m.type_linear),
            y_nodes=from_torch(m.y_nodes),
            y_edges=from_torch(m.y_edges),
            norm_y_edges=from_torch(m.norm_y_edges),
            norm_y_nodes=from_torch(m.norm_y_nodes),
            periodic_table_features=jnp.array(m.periodic_table_features.cpu().numpy()),
            side_chain_atom_types=jnp.array(m.side_chain_atom_types.cpu().numpy()),
        )
