#!/usr/bin/env python3
"""Design a binder to human PD-L1: compare vanilla mosaic vs jigandmpnn.

Runs 10 independent trials for each method and compares final ipTM scores:
  A) Vanilla mosaic: Jacobi-based InverseFoldingSequenceRecovery
  B) jigandmpnn:     Autoregressive InverseFoldingSequenceRecovery

Both share the same structure-prediction losses (IPTM, Contact, pLDDT) and
differ only in the sequence-recovery term.

Usage:
    python demos/design_pdl1_binder.py
"""

import numpy as np
import jax
import jax.numpy as jnp

from mosaic.common import TOKENS
from mosaic.models.boltz2 import Boltz2
from mosaic.structure_prediction import TargetChain
from mosaic.losses.structure_prediction import (
    IPTMLoss,
    BinderTargetContact,
    PLDDTLoss,
)
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.proteinmpnn.mpnn import load_mpnn as load_mosaic_mpnn
from mosaic.optimizers import simplex_APGM

from jigandmpnn import load_protein_mpnn
from jigandmpnn.losses import AutoregressiveSequenceRecovery

# ---------------------------------------------------------------------------
# PD-L1 target: PDB 4Z18 chain A (apo human PD-L1, IgV+IgC, 222 residues)
# ---------------------------------------------------------------------------
PDL1_SEQUENCE = (
    "MFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRAR"
    "LLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYNKINQRILVVDPVTSEHELTCQA"
    "EGYPKAEVIWTSSDHQVLSGKTTTTNSKREEKLFNVTSTLRINTTTNEIFYCTFRRLDPEENHTAELVIPE"
    "LPLAHPPNERT"
)

BINDER_LENGTH = 60
N_STEPS = 100
STEPSIZE = 0.01
MOMENTUM = 0.9
N_TRIALS = 10


def pssm_to_seq(x):
    return "".join(TOKENS[i] for i in x.argmax(-1))


def run_single_trial(loss_fn, x0, key):
    """Run one optimisation trial, return x_best."""
    _, x_best = simplex_APGM(
        loss_function=loss_fn,
        x=x0,
        n_steps=N_STEPS,
        stepsize=STEPSIZE,
        momentum=MOMENTUM,
        key=key,
    )
    return x_best


def evaluate_iptm(boltz, x_best, features, writer):
    """Predict structure and return ipTM."""
    prediction = boltz.predict(
        PSSM=x_best,
        features=features,
        writer=writer,
        recycling_steps=3,
        key=jax.random.key(0),
    )
    return float(prediction.iptm), float(prediction.plddt.mean())


def main():
    print("=" * 60)
    print("PDL1 Binder Design: mosaic Jacobi vs jigandmpnn AR")
    print(f"  {N_TRIALS} trials each, {N_STEPS} optimisation steps per trial")
    print("=" * 60)
    print(f"Target: PD-L1 (PDB 4Z18 chain A, {len(PDL1_SEQUENCE)} residues)")
    print(f"Binder length: {BINDER_LENGTH}")
    print(f"JAX backend: {jax.default_backend()}")

    # ------------------------------------------------------------------
    # 1. Load models (sequentially to manage GPU memory)
    # ------------------------------------------------------------------
    print("\n[1] Loading models...")

    # Load mosaic MPNN first (triggers torch import), then free torch tensors
    mosaic_mpnn = load_mosaic_mpnn()
    import gc, torch
    torch.cuda.empty_cache()
    gc.collect()

    jig_mpnn = load_protein_mpnn()
    torch.cuda.empty_cache()
    gc.collect()

    boltz = Boltz2()
    print("    Boltz2, mosaic ProteinMPNN, and jigandmpnn ProteinMPNN loaded.")

    # ------------------------------------------------------------------
    # 2. Build features (shared by all runs)
    # ------------------------------------------------------------------
    print("\n[2] Building features...")
    target = TargetChain(sequence=PDL1_SEQUENCE, use_msa=False)
    features, writer = boltz.binder_features(
        binder_length=BINDER_LENGTH,
        chains=[target],
    )
    print(f"    Features ready (binder={BINDER_LENGTH}, target={len(PDL1_SEQUENCE)})")

    # ------------------------------------------------------------------
    # 3. Compose losses
    # ------------------------------------------------------------------
    print("\n[3] Composing loss functions...")
    structure_losses = (
        1.0 * IPTMLoss()
        + 0.5 * BinderTargetContact(contact_distance=20.0)
        + 0.3 * PLDDTLoss()
    )

    loss_jacobi = boltz.build_loss(
        loss=structure_losses + 10.0 * InverseFoldingSequenceRecovery(
            mpnn=mosaic_mpnn, temp=0.1, num_samples=8,
        ),
        features=features,
        recycling_steps=3,
        sampling_steps=25,
    )
    loss_ar = boltz.build_loss(
        loss=structure_losses + 10.0 * AutoregressiveSequenceRecovery(
            mpnn=jig_mpnn, temp=0.1, num_samples=8,
        ),
        features=features,
        recycling_steps=3,
        sampling_steps=25,
    )
    print("    Jacobi:        1.0*IPTM + 0.5*Contact + 0.3*pLDDT + 10.0*JacobiSeqRecov")
    print("    Autoregressive: 1.0*IPTM + 0.5*Contact + 0.3*pLDDT + 10.0*AR-SeqRecov")

    # ------------------------------------------------------------------
    # 4. Run N_TRIALS for each method
    #    Each trial gets a fresh random PSSM on the simplex (Dirichlet),
    #    shared between Jacobi and AR so the comparison is paired.
    # ------------------------------------------------------------------
    jacobi_iptms = []
    jacobi_plddts = []
    ar_iptms = []
    ar_plddts = []

    print(f"\n[4] Running {N_TRIALS} trials (random Dirichlet init per trial)...")
    print(f"\n{'trial':>5}  {'Jacobi ipTM':>12} {'Jacobi pLDDT':>13}  {'AR ipTM':>12} {'AR pLDDT':>13}")
    print("─" * 62)

    for i in range(N_TRIALS):
        init_key, opt_key = jax.random.split(jax.random.key(i))

        # Random PSSM: Dirichlet(1,...,1) = uniform over the simplex
        x0 = jax.random.dirichlet(init_key, alpha=jnp.ones(20), shape=(BINDER_LENGTH,))

        # --- Jacobi ---
        x_best_j = run_single_trial(loss_jacobi, x0, opt_key)
        iptm_j, plddt_j = evaluate_iptm(boltz, x_best_j, features, writer)
        jacobi_iptms.append(iptm_j)
        jacobi_plddts.append(plddt_j)

        # --- Autoregressive ---
        x_best_a = run_single_trial(loss_ar, x0, opt_key)
        iptm_a, plddt_a = evaluate_iptm(boltz, x_best_a, features, writer)
        ar_iptms.append(iptm_a)
        ar_plddts.append(plddt_a)

        seq_j = pssm_to_seq(x_best_j)
        seq_a = pssm_to_seq(x_best_a)
        print(f"  {i+1:3d}    {iptm_j:10.4f}   {plddt_j:10.4f}    {iptm_a:10.4f}   {plddt_a:10.4f}   {seq_j[:20]}.. / {seq_a[:20]}..")

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    jacobi_iptms = np.array(jacobi_iptms)
    ar_iptms = np.array(ar_iptms)
    jacobi_plddts = np.array(jacobi_plddts)
    ar_plddts = np.array(ar_plddts)

    print(f"\n{'=' * 62}")
    print(f"  Summary over {N_TRIALS} trials")
    print(f"{'=' * 62}")
    print(f"{'':>18} {'Jacobi':>14} {'Autoregressive':>16}")
    print(f"  {'ipTM mean':>14}   {jacobi_iptms.mean():>10.4f}     {ar_iptms.mean():>10.4f}")
    print(f"  {'ipTM std':>14}   {jacobi_iptms.std():>10.4f}     {ar_iptms.std():>10.4f}")
    print(f"  {'ipTM min':>14}   {jacobi_iptms.min():>10.4f}     {ar_iptms.min():>10.4f}")
    print(f"  {'ipTM max':>14}   {jacobi_iptms.max():>10.4f}     {ar_iptms.max():>10.4f}")
    print(f"  {'pLDDT mean':>14}   {jacobi_plddts.mean():>10.4f}     {ar_plddts.mean():>10.4f}")
    print(f"  {'pLDDT std':>14}   {jacobi_plddts.std():>10.4f}     {ar_plddts.std():>10.4f}")

    # Per-trial comparison
    ar_wins = int((ar_iptms > jacobi_iptms).sum())
    jacobi_wins = int((jacobi_iptms > ar_iptms).sum())
    ties = N_TRIALS - ar_wins - jacobi_wins
    print(f"\n  AR wins: {ar_wins}   Jacobi wins: {jacobi_wins}   Ties: {ties}")
    print(f"  Mean ipTM difference (AR - Jacobi): {(ar_iptms - jacobi_iptms).mean():+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
