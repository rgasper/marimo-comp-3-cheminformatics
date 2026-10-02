"""A curated gallery of recognizable molecules for the notebook's selector.

The notebook uses a single upstream molecule selector; every downstream
visualization reacts to whatever molecule is chosen. This module supplies the
default menu — familiar drugs and chemically instructive structures — plus the
existing hand-picked comparison pairs, each with a display name and a one-line
"why it's interesting" note.

Kept deliberately small and readable. Users can always type their own SMILES;
this is just a good starting set so the selector is useful out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class GalleryMol:
    label: str
    smiles: str
    note: str


# Familiar molecules spanning scaffolds, functional groups, and size — chosen so
# that flipping between them makes fingerprint differences obvious.
GALLERY: tuple[GalleryMol, ...] = (
    GalleryMol("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "ester + carboxylic acid on benzene"),
    GalleryMol("Caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "fused N-heterocycle, no rotatable core"),
    GalleryMol("Paracetamol", "CC(=O)Nc1ccc(O)cc1", "amide + phenol — simple, drug-like"),
    GalleryMol("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "aliphatic + aromatic, chiral center"),
    GalleryMol("Benzene", "c1ccccc1", "the minimal aromatic ring"),
    GalleryMol("Glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "polyol, many stereocenters"),
    GalleryMol("Gefitinib", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "decorated 4-aminoquinazoline kinase inhibitor"),
    GalleryMol("Celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", "pyrazole sulfonamide, CF3 group"),
    GalleryMol("Imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "large multi-ring kinase drug"),
    GalleryMol("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "benzodiazepine, fused 7-membered ring"),
    GalleryMol("Penicillin G", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O", "beta-lactam, strained ring + stereo"),
    GalleryMol("Nicotine", "CN1CCC[C@H]1c1cccnc1", "two N-heterocycles, small"),
)


def default_label() -> str:
    return "Penicillin G"


def by_label() -> dict[str, GalleryMol]:
    return {g.label: g for g in GALLERY}


def canonical_smiles(smiles: str) -> str | None:
    """Round-trip a SMILES to RDKit canonical form; None if it won't parse.

    Used to validate user input and to display a normalized structure.
    """
    mol = Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None
    return Chem.MolToSmiles(mol) if mol is not None else None
