"""Bit-provenance explorer for the classical RDKit fingerprints (beyond Morgan).

Each classical fingerprint stores "which atoms/bonds set this bit" in a
*different* place, because each encodes a different kind of substructure:

- **RDKit-topological**: hashed linear *paths*; provenance is a set of bond
  indices per bit (``AdditionalOutput.GetBitPaths``).
- **Topological torsion**: 4-atom paths (torsions); provenance is atom tuples
  per bit (``GetBitPaths``).
- **Atom pair**: pairs of atoms at a topological distance; provenance is
  (atom_i, atom_j) tuples per bit (``GetBitInfoMap``).
- **Avalon**: computed by a separate C++ toolkit that returns only the bit
  vector - **no per-bit atom mapping is available**, so it can't be
  highlighted. That opacity is itself worth showing.

This module gives every highlightable fingerprint the same interface as
``morgan_explorer`` / ``maccs_explorer`` (``on_bits`` / ``bit_hit`` /
``highlight_svg`` / ``fingerprint_strip_svg``) so the notebook can drive them
all with one tabbed UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit.Chem.Draw import rdMolDraw2D

FPKey = Literal["rdkit_topo", "atom_pair", "top_torsion"]

DEFAULT_BITS = 2048


@dataclass(frozen=True)
class ClassicalFPInfo:
    key: FPKey
    label: str
    blurb: str  # one-line "what makes this fingerprint different"


# Metadata for the highlightable classical FPs (Avalon handled separately).
FP_INFO: dict[FPKey, ClassicalFPInfo] = {
    "rdkit_topo": ClassicalFPInfo(
        key="rdkit_topo",
        label="RDKit topological",
        blurb=(
            "Hashes every linear **path** through the molecular graph (up to a "
            "max length). Dense and connectivity-focused - it lights up far more "
            "bits than Morgan and rewards shared backbones."
        ),
    ),
    "atom_pair": ClassicalFPInfo(
        key="atom_pair",
        label="Atom pair",
        blurb=(
            "Encodes **pairs of atoms** by their types and the topological "
            "distance between them. Sees long-range relationships Morgan's local "
            "circles miss - 'a nitrogen 5 bonds from an oxygen'."
        ),
    ),
    "top_torsion": ClassicalFPInfo(
        key="top_torsion",
        label="Topological torsion",
        blurb=(
            "Encodes every **4-atom linear path** (a torsion) by the atom types "
            "along it. Captures short backbone motifs; sparse, like Morgan."
        ),
    ),
}


def _make_gen(key: FPKey, n_bits: int):
    if key == "rdkit_topo":
        return fpg.GetRDKitFPGenerator(fpSize=n_bits)
    if key == "atom_pair":
        return fpg.GetAtomPairGenerator(fpSize=n_bits)
    if key == "top_torsion":
        return fpg.GetTopologicalTorsionGenerator(fpSize=n_bits)
    raise ValueError(f"unknown fingerprint key: {key}")


@dataclass(frozen=True)
class ClassicalBitHit:
    key: FPKey
    bit: int
    n_instances: int  # how many substructures set this bit
    atoms: tuple[int, ...]  # union of atoms across instances (for highlight)
    bonds: tuple[int, ...]  # union of bonds (only for path-based FPs)


def _atoms_bonds_for_bit(
    mol: Chem.Mol, key: FPKey, bit: int, n_bits: int
) -> tuple[set[int], set[int], int]:
    """Resolve a bit to its atoms/bonds, handling each FP's provenance format."""
    ao = fpg.AdditionalOutput()
    ao.AllocateBitInfoMap()
    ao.AllocateBitPaths()
    _make_gen(key, n_bits).GetFingerprint(mol, additionalOutput=ao)

    atoms: set[int] = set()
    bonds: set[int] = set()
    n_instances = 0

    if key == "atom_pair":
        # bitInfoMap: bit -> ((atom_i, atom_j), ...)
        info = ao.GetBitInfoMap()
        instances = info.get(bit, ())
        n_instances = len(instances)
        for pair in instances:
            atoms.update(pair)
    elif key == "rdkit_topo":
        # bitPaths: bit -> (bond-index-tuple, ...)  (linear paths)
        paths = ao.GetBitPaths()
        instances = paths.get(bit, ())
        n_instances = len(instances)
        for bond_ids in instances:
            for b in bond_ids:
                bonds.add(int(b))
                bd = mol.GetBondWithIdx(int(b))
                atoms.add(bd.GetBeginAtomIdx())
                atoms.add(bd.GetEndAtomIdx())
    elif key == "top_torsion":
        # bitPaths: bit -> (atom-index-4-tuple, ...)
        paths = ao.GetBitPaths()
        instances = paths.get(bit, ())
        n_instances = len(instances)
        for atom_ids in instances:
            atoms.update(int(a) for a in atom_ids)
        # add bonds internal to each torsion path for a cleaner highlight
        for a in list(atoms):
            for bd in mol.GetAtomWithIdx(a).GetBonds():
                other = bd.GetOtherAtomIdx(a)
                if other in atoms:
                    bonds.add(bd.GetIdx())

    return atoms, bonds, n_instances


def on_bits(mol: Chem.Mol, key: FPKey, n_bits: int = DEFAULT_BITS) -> list[int]:
    """Sorted indices of bits set for this molecule under the given FP."""
    ao = fpg.AdditionalOutput()
    ao.AllocateBitInfoMap()
    ao.AllocateBitPaths()
    _make_gen(key, n_bits).GetFingerprint(mol, additionalOutput=ao)
    if key == "atom_pair":
        return sorted(ao.GetBitInfoMap().keys())
    return sorted(ao.GetBitPaths().keys())


def bit_hit(
    mol: Chem.Mol, key: FPKey, bit: int, n_bits: int = DEFAULT_BITS
) -> ClassicalBitHit:
    atoms, bonds, n = _atoms_bonds_for_bit(mol, key, bit, n_bits)
    return ClassicalBitHit(
        key=key,
        bit=bit,
        n_instances=n,
        atoms=tuple(sorted(atoms)),
        bonds=tuple(sorted(bonds)),
    )


def highlight_svg(
    mol: Chem.Mol,
    hit: ClassicalBitHit,
    *,
    width: int = 460,
    height: int = 340,
    color: tuple[float, float, float] = (0.20, 0.55, 0.95),
) -> str:
    """Render mol with the bit's atoms/bonds highlighted."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = False
    atom_colors = {a: color for a in hit.atoms}
    bond_colors = {b: color for b in hit.bonds}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(hit.atoms),
        highlightBonds=list(hit.bonds),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def fingerprint_strip_svg(
    mol: Chem.Mol,
    key: FPKey,
    current_bit: int,
    *,
    n_bits: int = DEFAULT_BITS,
    width: int = 920,
    height: int = 36,
    on_color: str = "#2f9e44",
    off_color: str = "#f1f3f5",
    current_color: str = "#1c7ed6",
) -> str:
    """Sparse bit strip (like Morgan): ON green, current blue, rest blank."""
    on = set(on_bits(mol, key, n_bits))
    pad = 2
    inner_w = width - 2 * pad
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="{pad}" y="{pad}" width="{inner_w}" height="{height - 2 * pad}" '
        f'fill="{off_color}" stroke="#dee2e6" stroke-width="0.5" />',
    ]
    tick_w = max(1.0, inner_w / n_bits)
    current_x = None
    for bit in sorted(on):
        x = pad + (bit / n_bits) * inner_w
        if bit == current_bit:
            current_x = x
            continue  # draw the current bit last, on top, so it's never hidden
        parts.append(
            f'<rect x="{x:.2f}" y="{pad}" width="{max(tick_w, 1.0):.2f}" '
            f'height="{height - 2 * pad}" fill="{on_color}" />'
        )
    if current_x is not None:
        # a bold, wide, full-height marker so the selected bit is unmistakable
        cw = max(tick_w, 4.0)
        parts.append(
            f'<rect x="{current_x - cw / 2:.2f}" y="0" width="{cw:.2f}" '
            f'height="{height}" fill="{current_color}" />'
        )
    parts.append("</svg>")
    return "".join(parts)


def plain_svg(mol: Chem.Mol, *, width: int = 460, height: int = 340) -> str:
    """Render mol with nothing highlighted (used for the opaque Avalon tab)."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
