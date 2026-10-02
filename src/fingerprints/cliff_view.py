"""Rendering + scoring helpers for the context-dependent activity-cliff section.

Ties together three existing pieces for a single curated cliff pair:

- ``mcs_diff_atoms`` (from clustering.cliffs) - which atoms changed between the
  two molecules, so we can highlight the (small) difference;
- the classical fingerprints - to score how *similar* each one thinks the pair
  is;
- the pair's two endpoint pKi values - to contrast "the fingerprint's guess"
  against reality on each target.

The point the section makes: a fingerprint sees only structure, so it assigns
one similarity to the pair. That single number is simultaneously right for the
endpoint where the pair is flat and wrong for the endpoint where it's a cliff.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from fingerprints.clustering.cliffs import mcs_diff_atoms
from fingerprints.data.context_cliffs import ContextCliff
from fingerprints.fingerprint_methods.rdkit_fps import all_classical
from fingerprints.fingerprint_methods.similarity import pairwise_similarity

# Reuse the same short id -> display label the rest of the repo uses.
FP_LABELS: dict[str, str] = {
    "morgan": "Morgan",
    "rdkit_topo": "RDKit topological",
    "atom_pair": "Atom pair",
    "top_torsion": "Topological torsion",
    "maccs": "MACCS",
    "avalon": "Avalon",
    "chemeleon": "CheMeleon (learned)",
}
FP_DISPLAY_ORDER: tuple[str, ...] = (
    "morgan", "maccs", "atom_pair", "top_torsion", "rdkit_topo",
)

_CHANGE_COLOR = (0.95, 0.45, 0.15)  # orange for the atoms that differ


def pair_mols(cliff: ContextCliff) -> tuple[Chem.Mol, Chem.Mol]:
    return Chem.MolFromSmiles(cliff.smiles_1), Chem.MolFromSmiles(cliff.smiles_2)


def changed_atoms(cliff: ContextCliff) -> tuple[list[int], list[int]]:
    """Atoms in (mol1, mol2) that are NOT part of the shared MCS core."""
    m1, m2 = pair_mols(cliff)
    diff = mcs_diff_atoms(m1, m2)
    if diff is None:
        return [], []
    return diff[0], diff[1]


def _draw(mol: Chem.Mol, highlight: list[int], width: int, height: int) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = False
    colors = {a: _CHANGE_COLOR for a in highlight}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=highlight, highlightAtomColors=colors
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def pair_svgs(
    cliff: ContextCliff, *, width: int = 340, height: int = 260
) -> tuple[str, str]:
    """Both molecules drawn side by side with the changed atoms highlighted."""
    m1, m2 = pair_mols(cliff)
    diff1, diff2 = changed_atoms(cliff)
    return _draw(m1, diff1, width, height), _draw(m2, diff2, width, height)


@dataclass(frozen=True)
class FPScore:
    key: str
    label: str
    similarity: float


def fingerprint_scores(
    cliff: ContextCliff, *, classical_only: bool = False
) -> list[FPScore]:
    """How similar does each fingerprint think this pair is?

    Classical fingerprints use Tanimoto; the learned **CheMeleon** embedding is
    dense/continuous, so it uses cosine similarity (its natural metric).
    """
    m1, m2 = pair_mols(cliff)
    fps = all_classical([m1, m2])
    out: list[FPScore] = []
    for key in FP_DISPLAY_ORDER:
        sim = float(pairwise_similarity(fps[key])[0, 1])
        out.append(FPScore(key=key, label=FP_LABELS[key], similarity=sim))
    if classical_only:
        # A learned embedding (CheMeleon) has no principled similarity metric of
        # its own — it is meant to feed a trained head, not a cosine. So for the
        # honest "every fingerprint calls this pair similar" comparison we show
        # only the classical fingerprints, whose native metric IS Tanimoto.
        return out
    # Append the learned fingerprint (cosine similarity of CheMeleon vectors).
    try:
        from fingerprints import chemeleon_fp as chf

        v1 = chf.fingerprint(m1)
        v2 = chf.fingerprint(m2)
        denom = float((v1 @ v1) ** 0.5 * (v2 @ v2) ** 0.5)
        cos = float(v1 @ v2) / denom if denom > 0 else 0.0
        out.append(
            FPScore(key="chemeleon", label=FP_LABELS["chemeleon"], similarity=cos)
        )
    except Exception:
        # If the CheMeleon weights aren't available, just omit it - the
        # classical bars still tell the story and nothing throws.
        pass
    return out


def similarity_to_reference(
    smiles: list[str], *, include_learned: bool = True
) -> list[dict]:
    """Similarity of ``smiles[0]`` to every other molecule, per fingerprint.

    Returns long-form rows ``{fingerprint, other_index, similarity, metric}``
    for the classical fingerprints (Tanimoto) and, if available, CheMeleon
    (cosine). Used by the "what does similarity mean?" explainer.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = all_classical(mols)
    rows: list[dict] = []
    for key in FP_DISPLAY_ORDER:
        sim = pairwise_similarity(fps[key])[0]
        for j in range(1, len(mols)):
            rows.append(
                {"fingerprint": FP_LABELS[key], "other_index": j,
                 "similarity": float(sim[j]), "metric": "Tanimoto"}
            )
    if include_learned:
        try:
            import numpy as np

            from fingerprints import chemeleon_fp as chf

            vecs = np.stack([chf.fingerprint(m) for m in mols])
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            cos = vecs @ vecs[0]
            for j in range(1, len(mols)):
                rows.append(
                    {"fingerprint": FP_LABELS["chemeleon"], "other_index": j,
                     "similarity": float(cos[j]), "metric": "cosine"}
                )
        except Exception:
            pass
    return rows


def plif_similarity(pair_key: str, index: int, target: str) -> float | None:
    """Tanimoto between the two ligands' interaction fingerprints in one pocket.

    Unlike the structure fingerprints (which read only the 2D graph), the PLIF
    is read off each ligand's *3D pose in the binding site*: every 'on' bit is a
    specific contact (an H-bond to a residue, a π-stack, ...). We binarise each
    pose's ``(residue, interaction-type) -> count`` fingerprint to presence and
    take Tanimoto over the union of contacts.

    Returns ``None`` when this cliff has no cached poses (so the caller can just
    omit the bar rather than fabricate a number). Requires the four Boltz poses
    produced offline by ``fingerprints.rebuild_poses``.
    """
    try:
        from fingerprints import pose_view as pv

        if not pv.has_poses(pair_key, index):
            return None
        poses = pv.load_all(pair_key, index)
        fp1 = pv.interaction_fingerprint(poses[f"mol1_{target}"])
        fp2 = pv.interaction_fingerprint(poses[f"mol2_{target}"])
    except (KeyError, OSError):
        return None
    bits1, bits2 = set(fp1), set(fp2)
    if not bits1 and not bits2:
        return None
    union = len(bits1 | bits2)
    return (len(bits1 & bits2) / union) if union else 0.0


def fold_change(delta_pki: float) -> str:
    """Turn a |ΔpKi| into a readable potency-fold string, e.g. '≈250×'."""
    fold = 10.0 ** delta_pki
    if fold < 10:
        return f"{fold:.1f}×"
    if fold < 1000:
        return f"≈{round(fold, -1):.0f}×"
    return f"≈{fold:,.0f}×"
