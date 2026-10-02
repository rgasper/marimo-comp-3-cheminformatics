"""Pure-RDKit "context-dependent computation recovers the hidden signal"
demonstration for the ADMET solubility-cliff appetizer.

The census showed that fingerprints (and even flat descriptors like TPSA) call
the sharpest solubility cliffs *identical*. This module makes the honest
counter-move: a **physically-motivated computed property** - hydrophobicity
integrated over the whole molecule - tracks the cliff the fingerprint cannot.

We use RDKit's Crippen logP, whose per-atom contributions are an **exact**
decomposition (they sum to MolLogP), so we can color a molecule by "how much
each atom adds to its hydrophobicity" - the analog of the Morgan/CheMeleon
per-atom heatmaps elsewhere in the notebook. Nothing here is new science: it is
a textbook lipophilicity model, used to make a hard idea legible.

Everything is cheap and deterministic (no 3D embedding needed for the scalar
story), so it runs live in the notebook - no cache, no extra dependencies, no
heavy quantum toolchain. That portability is the whole point of staying in
RDKit.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit.Chem import rdMolDescriptors as rd
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

_MORGAN = fpg.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class CliffProbe:
    """How different 'lenses' score one solubility-cliff pair."""

    tanimoto: float       # ECFP4 similarity (structure lens - blind)
    d_tpsa: float         # |Delta TPSA| (polar-surface lens - often blind here)
    d_hbd: int            # |Delta H-bond donors| (counting lens)
    d_logp: float         # |Delta Crippen logP| (hydrophobicity lens - sees it)
    logp_1: float
    logp_2: float
    measured_gap: float   # |Delta logS| from the data (the truth)


def _mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def probe_pair(smiles_1: str, smiles_2: str, measured_gap: float) -> CliffProbe | None:
    """Score one cliff pair through structure vs physicochemical lenses."""
    m1, m2 = _mol(smiles_1), _mol(smiles_2)
    if m1 is None or m2 is None:
        return None
    fp1 = _MORGAN.GetFingerprint(m1)
    fp2 = _MORGAN.GetFingerprint(m2)
    lp1, lp2 = Crippen.MolLogP(m1), Crippen.MolLogP(m2)
    return CliffProbe(
        tanimoto=float(DataStructs.TanimotoSimilarity(fp1, fp2)),
        d_tpsa=abs(rd.CalcTPSA(m1) - rd.CalcTPSA(m2)),
        d_hbd=abs(rd.CalcNumHBD(m1) - rd.CalcNumHBD(m2)),
        d_logp=abs(lp1 - lp2),
        logp_1=lp1,
        logp_2=lp2,
        measured_gap=measured_gap,
    )


def logp_atom_contribs(mol: Chem.Mol) -> list[float]:
    """Per-atom Crippen logP contributions (an exact decomposition of MolLogP:
    they sum to the molecule's logP). Positive = hydrophobic (drives insolubility
    / desolvation cost), negative = polar/hydrophilic."""
    return [float(c[0]) for c in rd._CalcCrippenContribs(mol)]


def logp_heatmap_svg(mol: Chem.Mol, *, width: int = 300, height: int = 220) -> str:
    """Color atoms by hydrophobicity (per-atom logP contribution).

    Uses RDKit's similarity-map convention (green up / pink down). Here 'up'
    = more hydrophobic. The long alkyl chains that a fingerprint ignores light
    up as exactly the accumulated hydrophobic burden that tanks solubility.
    """
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d.drawOptions().addStereoAnnotation = False
    if mol.GetNumAtoms() < 2:
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        logp_atom_contribs(mol),
        draw2d=d,
        contourLines=6,
        gridResolution=0.25,
    )
    d.FinishDrawing()
    return d.GetDrawingText()
