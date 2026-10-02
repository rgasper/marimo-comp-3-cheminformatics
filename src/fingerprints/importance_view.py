"""Feature-importance views for the cliff pair: which fingerprint bits/dimensions
a RandomForest leans on to predict D3/D4 activity, projected back onto the
molecule and onto the fingerprint strip.

Importances are precomputed offline (``fingerprints.analyses.importance``) and
cached; this module just maps them onto a given molecule so the notebook stays
instant.

Two fingerprints:
  * ECFP (Morgan) - bit -> atom environments is exact (RDKit bit info map), so a
    bit's importance is spread over the atoms of every environment that set it.
  * CheMeleon - dimension k's per-atom contribution is exact (mean-pool), so a
    dimension's importance is spread over atoms by their contribution to it.

Both are honest *estimates* of "where the model looks", not ground-truth
substructure definitions.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

from fingerprints import chemeleon_fp as chf
from fingerprints import morgan_explorer as me
from fingerprints import paths

CACHE = paths.IMPORTANCE
N_BITS = 2048
_MORGAN = fpg.GetMorganGenerator(radius=2, fpSize=N_BITS)


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads(CACHE.read_text())


def has_data() -> bool:
    return CACHE.exists()


def config() -> dict:
    """Training hyperparameters stored in the cache (n_bits, rf_trees)."""
    d = _data()
    return {"n_bits": d["n_bits"], "rf_trees": d["rf_trees"]}


def endpoints() -> list[str]:
    return list(_data()["endpoints"].keys()) if has_data() else []


def importances(endpoint: str, fp: str) -> np.ndarray:
    """Per-bit importances (length N_BITS) for one endpoint + fingerprint
    ('ecfp' or 'chemeleon')."""
    return np.asarray(_data()["endpoints"][endpoint][fp]["importances"], dtype=float)


def metrics(endpoint: str, fp: str) -> dict:
    d = _data()["endpoints"][endpoint][fp]
    return {"r2": d["r2"], "rmse": d["rmse"], "n_train": d["n_train"], "n_test": d["n_test"]}


def predict(endpoint: str, fp: str, mol: Chem.Mol) -> float | None:
    """Cached RF prediction (pKi) for a curated cliff molecule, looked up by
    canonical SMILES. Returns None if this molecule wasn't precomputed."""
    preds = _data()["endpoints"][endpoint][fp].get("predictions", {})
    return preds.get(Chem.MolToSmiles(mol))


def test_scatter(endpoint: str, fp: str) -> tuple[list[float], list[float]]:
    """Held-out test-fold (measured, predicted) pKi for one endpoint + fp, so
    the notebook can plot how good the model is. Returns ([], []) if absent."""
    d = _data()["endpoints"][endpoint][fp]
    return d.get("test_measured", []), d.get("test_pred", [])


def fold_scatter(endpoint: str, fp: str, fold: str) -> tuple[list[float], list[float]]:
    """(measured, predicted) pKi for one endpoint + fp on the given ``fold``
    ('train' or 'test'). Returns ([], []) if not present in the cache."""
    d = _data()["endpoints"][endpoint][fp]
    return d.get(f"{fold}_measured", []), d.get(f"{fold}_pred", [])


# ---- per-atom importance ------------------------------------------------
def ecfp_on_bits(mol: Chem.Mol) -> list[int]:
    return me.on_bits(mol, n_bits=N_BITS)


def atom_importance_ecfp(mol: Chem.Mol, imp: np.ndarray) -> np.ndarray:
    """Spread each ON Morgan bit's importance over the atoms of its
    environment(s). Returns (n_atoms,) importance mass."""
    w = np.zeros(mol.GetNumAtoms(), dtype=float)
    for bit in ecfp_on_bits(mol):
        hit = me.bit_hit(mol, bit, n_bits=N_BITS)
        atoms = hit.atoms
        if not atoms:
            continue
        share = imp[bit] / len(atoms)
        for a in atoms:
            w[a] += share
    return w


def atom_importance_chemeleon(mol: Chem.Mol, imp: np.ndarray, top_k: int = 64) -> np.ndarray:
    """Weight each atom by sum_k importance[k] * |contribution of atom to dim k|,
    over the top_k most-important dimensions (keeps it cheap and focused)."""
    H = chf.atom_hidden(mol)  # (n_atoms, EMBED_DIM)
    n = H.shape[0]
    top = np.argsort(imp)[::-1][:top_k]
    contrib = np.abs(H[:, top]) / n  # (n_atoms, top_k)
    return (contrib * imp[top][None, :]).sum(axis=1)


def atom_importance(mol: Chem.Mol, endpoint: str, fp: str) -> np.ndarray:
    imp = importances(endpoint, fp)
    if fp == "ecfp":
        return atom_importance_ecfp(mol, imp)
    return atom_importance_chemeleon(mol, imp)


def atom_importance_diff(
    mol: Chem.Mol, ep_a: str, ep_b: str, fp: str
) -> np.ndarray:
    """Per-atom *change* in model attention between two endpoints:
    importance(ep_a) - importance(ep_b), each L1-normalised first so the two
    endpoints are on the same scale. Positive => the atom matters more for
    ep_a's model; negative => more for ep_b's."""
    wa = atom_importance(mol, ep_a, fp)
    wb = atom_importance(mol, ep_b, fp)
    sa = wa.sum() or 1.0
    sb = wb.sum() or 1.0
    return wa / sa - wb / sb


def importance_diff_stats(
    mol: Chem.Mol, ep_a: str, ep_b: str, fp: str, changed: list[int]
) -> dict:
    """Quantify whether the model even *notices* the target swap, and how much
    of the re-weighting actually lands on the atoms that cause the cliff.

    Rather than a misleading yes/no "is the single biggest mover a changed
    atom?", we report a **gradual** share-of-attention metric:

      changed_share = sum(|diff| over changed atoms) / sum(|diff| over all atoms)

    i.e. what fraction of the model's total attention *shift* (between the two
    targets) concentrates on the handful of atoms that structurally changed.
    We compare it to the **fair share** a spatially-blind model would give those
    atoms just by their count, ``n_changed / n_atoms``. An ``enrichment`` > 1
    means the shift is *concentrated* on the change; ~1 means the model
    re-weights the changed atoms no more than any others; < 1 means it actually
    leans elsewhere. Also returns a rank check (are the changed atoms among the
    top movers) for a secondary, discrete read.

    Returns:
      changed_share: fraction of total |shift| on the changed atoms (0..1).
      fair_share: n_changed / n_atoms (the count-only baseline).
      enrichment: changed_share / fair_share (1.0 == no concentration).
      changed_in_top: True if any changed atom is among the top-``top_k`` movers.
      top_k: neighbourhood size for the rank check (5, capped at n_atoms).
      top_hits: how many changed atoms are in that top-k.
      total_shift: sum |diff| (0 => identical, ~2 => disjoint). Reference only.
      peak_atom / peak_on_changed: kept for backward compat (the single biggest
        mover); no longer foregrounded in the prose.
      n_changed: number of changed atoms.
    """
    diff = np.abs(atom_importance_diff(mol, ep_a, ep_b, fp))
    n_atoms = int(mol.GetNumAtoms())
    changed_set = set(changed)
    total = float(diff.sum()) or 1.0
    changed_mass = float(diff[list(changed_set)].sum()) if changed_set else 0.0
    changed_share = changed_mass / total
    fair = (len(changed_set) / n_atoms) if n_atoms else 0.0
    enrichment = (changed_share / fair) if fair > 0 else 0.0

    order = np.argsort(diff)[::-1]
    k = min(5, n_atoms)
    top = set(int(a) for a in order[:k])
    return {
        "changed_share": changed_share,
        "fair_share": fair,
        "enrichment": enrichment,
        "changed_in_top": bool(changed_set & top),
        "top_k": k,
        "top_hits": int(len(changed_set & top)),
        "total_shift": float(diff.sum()),
        "peak_atom": int(order[0]),
        "peak_on_changed": bool(int(order[0]) in changed_set),
        "n_changed": len(changed_set),
    }


def importance_diff_heatmap_svg(
    mol: Chem.Mol,
    ep_a: str,
    ep_b: str,
    fp: str,
    *,
    width: int = 300,
    height: int = 220,
) -> str:
    """Diverging heatmap of the per-atom attention *difference* between the two
    endpoints' models (red = leans more on this atom for ep_a; blue = for
    ep_b). A near-blank map means the static fingerprint barely re-weights
    anything when the target changes."""
    w = atom_importance_diff(mol, ep_a, ep_b, fp)
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d.drawOptions().addStereoAnnotation = False
    if mol.GetNumAtoms() < 2 or float(np.ptp(w)) == 0.0:
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        [float(x) for x in w],
        draw2d=d,
        contourLines=3,
        gridResolution=0.45,
    )
    d.FinishDrawing()
    return d.GetDrawingText()


def importance_heatmap_svg(
    mol: Chem.Mol, endpoint: str, fp: str, *, width: int = 420, height: int = 320
) -> str:
    """Color the molecule by aggregated feature importance (green = the model
    leans on this region to predict activity)."""
    w = atom_importance(mol, endpoint, fp)
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d.drawOptions().addStereoAnnotation = False
    if mol.GetNumAtoms() < 2 or float(np.ptp(w)) == 0.0:
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    # All-positive importance: use a single-sided (white->green) feel by
    # centering weights so SimilarityMaps' diverging map reads as intensity.
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        [float(x) for x in w],
        draw2d=d,
        contourLines=3,
        gridResolution=0.45,
    )
    d.FinishDrawing()
    return d.GetDrawingText()


# ---- importance-tinted fingerprint strip --------------------------------
def strip_svg(
    mol: Chem.Mol,
    endpoint: str,
    fp: str,
    *,
    width: int = 920,
    height: int = 40,
) -> str:
    """Fingerprint strip where each cell's colour intensity encodes that bit's
    importance to the model. For ECFP only ON bits are drawn (an off bit means
    'no environment here'); for CheMeleon every dimension is dense, so we draw
    the full vector shaded by importance."""
    imp = importances(endpoint, fp)
    mx = float(imp.max()) or 1.0
    pad = 2
    inner_w = width - 2 * pad
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="{pad}" y="{pad}" width="{inner_w}" height="{height - 2 * pad}" '
        f'fill="#f1f3f5" stroke="#dee2e6" stroke-width="0.5" />',
    ]
    if fp == "ecfp":
        bits = ecfp_on_bits(mol)
    else:
        bits = list(range(N_BITS))
    tick_w = max(inner_w / N_BITS, 0.6)
    for bit in bits:
        frac = imp[bit] / mx
        if frac <= 0.02 and fp == "chemeleon":
            continue  # skip near-zero dims to keep the dense strip legible
        x = pad + (bit / N_BITS) * inner_w
        # green with alpha proportional to importance
        alpha = 0.15 + 0.85 * min(frac, 1.0)
        parts.append(
            f'<rect x="{x:.2f}" y="{pad}" width="{max(tick_w, 1.0):.2f}" '
            f'height="{height - 2 * pad}" fill="#2f9e44" fill-opacity="{alpha:.2f}" />'
        )
    parts.append("</svg>")
    return "".join(parts)
