"""Rendering + scoring helpers for the context-dependent activity-cliff section.

Ties together three things for a single curated cliff pair:

- the maximum common substructure (MCS) of the two molecules, to find and
  highlight the (small) set of atoms that differ between them;
- five classical binary fingerprints — to score how *similar* each one thinks
  the pair is (Tanimoto, their native metric);
- the pair's two endpoint pKi values — to contrast "the fingerprint's guess"
  against reality on each target.

The point the section makes: a fingerprint sees only structure, so it assigns
one similarity to the pair. That single number is simultaneously right for the
endpoint where the pair is flat and wrong for the endpoint where it's a cliff.
"""

from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

from fingerprints.data.context_cliffs import ContextCliff

N_BITS = 2048

# Five classical fingerprints, each binary, each scored by Tanimoto (their
# native similarity metric). Built fresh per call — these are only ever
# compared pairwise (2-3 molecules at a time), so there's no need for the
# batched-matrix machinery a larger survey would want.
_GENERATORS = {
    "morgan": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS),
    "atom_pair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS),
    "top_torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=N_BITS),
    "rdkit_topo": rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=N_BITS),
}

FP_LABELS: dict[str, str] = {
    "morgan": "Morgan",
    "rdkit_topo": "RDKit topological",
    "atom_pair": "Atom pair",
    "top_torsion": "Topological torsion",
    "maccs": "MACCS",
    "chemeleon": "CheMeleon (learned)",
}
FP_DISPLAY_ORDER: tuple[str, ...] = (
    "morgan", "maccs", "atom_pair", "top_torsion", "rdkit_topo",
)

_CHANGE_COLOR = (0.95, 0.45, 0.15)  # orange for the atoms that differ


def _classical_fp(mol: Chem.Mol, key: str):
    """One classical bit vector for ``mol`` under fingerprint ``key``."""
    if key == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    return _GENERATORS[key].GetFingerprint(mol)


def _tanimoto(mol_a: Chem.Mol, mol_b: Chem.Mol, key: str) -> float:
    return DataStructs.TanimotoSimilarity(
        _classical_fp(mol_a, key), _classical_fp(mol_b, key)
    )


def mcs_diff_atoms(
    mol_i: Chem.Mol, mol_j: Chem.Mol, timeout: int = 2
) -> tuple[list[int], list[int]] | None:
    """Atoms in each molecule that are NOT part of their maximum common
    substructure (MCS) — i.e. the atoms that changed between the two.

    Settings:
        - atomCompare=CompareElements: distinct elements never match.
        - bondCompare=CompareOrderExact: aromatic bonds match aromatic only,
          single matches single, etc.
        - completeRingsOnly=False: allow partial ring matches, so a ring
          expansion/contraction is still mostly "the same" structure.
        - ringMatchesRingOnly=True: a ring atom can only match a ring atom
          (otherwise a long chain falsely matches a same-size ring).

    Returns (diff_atoms_i, diff_atoms_j) as atom-index lists, or None if the
    MCS search times out or fails to resolve a match.
    """
    res = rdFMCS.FindMCS(
        [mol_i, mol_j],
        timeout=timeout,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        completeRingsOnly=False,
        ringMatchesRingOnly=True,
    )
    if res.canceled:
        return None
    patt = res.queryMol
    match_i = mol_i.GetSubstructMatch(patt)
    match_j = mol_j.GetSubstructMatch(patt)
    if not match_i or not match_j:
        return None
    diff_i = [a.GetIdx() for a in mol_i.GetAtoms() if a.GetIdx() not in match_i]
    diff_j = [a.GetIdx() for a in mol_j.GetAtoms() if a.GetIdx() not in match_j]
    return diff_i, diff_j


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


def similarity_to_reference(
    smiles: list[str], *, include_learned: bool = True
) -> list[dict]:
    """Similarity of ``smiles[0]`` to every other molecule, per fingerprint.

    Returns long-form rows ``{fingerprint, other_index, similarity, metric}``
    for the classical fingerprints (Tanimoto) and, if available, CheMeleon
    (cosine). Used by the "what does similarity mean?" explainer.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    rows: list[dict] = []
    for key in FP_DISPLAY_ORDER:
        for j in range(1, len(mols)):
            rows.append(
                {"fingerprint": FP_LABELS[key], "other_index": j,
                 "similarity": _tanimoto(mols[0], mols[j], key), "metric": "Tanimoto"}
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


def fold_change(delta_pki: float) -> str:
    """Turn a |ΔpKi| into a readable potency-fold string, e.g. '≈250×'."""
    fold = 10.0 ** delta_pki
    if fold < 10:
        return f"{fold:.1f}×"
    if fold < 1000:
        return f"≈{round(fold, -1):.0f}×"
    return f"≈{fold:,.0f}×"
