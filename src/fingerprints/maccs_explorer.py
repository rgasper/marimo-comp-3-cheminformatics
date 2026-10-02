"""Helpers for interactively exploring MACCS keys on a single molecule.

MACCS keys are 166 predefined structural patterns (RDKit exposes 167 bits;
index 0 is unused). Each key is a SMARTS pattern plus a count threshold, so a
bit is "on" when the molecule contains at least that many matches. This makes
MACCS the most human-readable fingerprint: every bit maps to a named
substructure you can point at on the molecule.

This module turns that mapping into something a UI can drive: given a molecule
and a bit index, tell me which atoms/bonds lit the bit and render a highlighted
depiction.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MACCSkeys import smartsPatts

from fingerprints import maccs_descriptions as descs

# Bits with no SMARTS pattern (count-based / hardcoded keys in RDKit).
SPECIAL_BITS: frozenset[int] = frozenset(
    i for i, (smarts, _count) in smartsPatts.items() if smarts == "?"
)

# The three keys RDKit computes procedurally instead of via a SMARTS match,
# with a plain-language explanation of what each one actually tests.
SPECIAL_DESCRIPTIONS: dict[int, str] = {
    1: "ISOTOPE — reserved/undefined in RDKit's implementation; it never fires.",
    125: "Aromatic rings > 1 — fires when the molecule has more than one aromatic "
    "ring. Counting rings can't be written as a single substructure pattern, so "
    "RDKit computes it directly rather than with SMARTS.",
    166: "Fragments — fires when the molecule is made of more than one disconnected "
    "piece (e.g. a salt: a drug cation plus a separate counter-ion). \"Is this "
    "more than one molecule?\" also can't be expressed as one SMARTS pattern.",
}


@dataclass(frozen=True)
class BitHit:
    """The result of asking 'does bit N fire on this molecule, and where?'"""

    bit: int
    is_on: bool
    smarts: str | None  # None for the three procedurally-computed special keys
    name: str  # plain-English description of what the key looks for
    official: str  # terse authoritative MDL/MayaChemTools definition
    match_count: int  # number of distinct substructure matches
    threshold: int  # bit fires when match_count > threshold (usually 0)
    is_special: bool  # True for keys 1/125/166 (no SMARTS)
    atoms: tuple[int, ...]  # union of atoms across all matches (for highlight)
    bonds: tuple[int, ...]  # bonds fully inside the matched atom set


def bit_name(bit: int) -> str:
    """Plain-English description of what a MACCS key looks for."""
    return descs.plain(bit)


def describe_special(bit: int) -> str | None:
    """Plain-language explanation for the three procedurally-computed keys."""
    return SPECIAL_DESCRIPTIONS.get(bit)


def blank_hit() -> BitHit:
    """A no-op BitHit for rendering a molecule with nothing highlighted."""
    return BitHit(
        bit=-1,
        is_on=False,
        smarts=None,
        name="",
        official="",
        match_count=0,
        threshold=0,
        is_special=False,
        atoms=(),
        bonds=(),
    )


def on_bits(mol: Chem.Mol) -> list[int]:
    """MACCS bit indices that are set for this molecule (1..166)."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return [b for b in fp.GetOnBits() if b != 0]


def bit_hit(mol: Chem.Mol, bit: int) -> BitHit:
    """Resolve a MACCS bit against a molecule: is it on, and which atoms fired?"""
    fp = MACCSkeys.GenMACCSKeys(mol)
    is_on = bool(fp[bit])
    patt = smartsPatts.get(bit)
    is_special = patt is None or patt[0] == "?"
    smarts = None if is_special else patt[0]
    threshold = 0 if patt is None else patt[1]

    atoms: set[int] = set()
    match_count = 0
    if smarts is not None:
        query = Chem.MolFromSmarts(smarts)
        if query is not None:
            matches = mol.GetSubstructMatches(query)
            match_count = len(matches)
            for m in matches:
                atoms.update(m)

    bonds: list[int] = []
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms:
            bonds.append(b.GetIdx())

    return BitHit(
        bit=bit,
        is_on=is_on,
        smarts=smarts,
        name=bit_name(bit),
        official=descs.official(bit),
        match_count=match_count,
        threshold=threshold,
        is_special=is_special,
        atoms=tuple(sorted(atoms)),
        bonds=tuple(bonds),
    )


def highlight_svg(
    mol: Chem.Mol,
    hit: BitHit,
    *,
    width: int = 480,
    height: int = 360,
    color: tuple[float, float, float] = (1.0, 0.55, 0.0),
) -> str:
    """Render mol as SVG with the bit's matched atoms/bonds highlighted."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False
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


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse SMILES, returning None on failure (for UI validation)."""
    return Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None


def all_bits() -> list[int]:
    """Every MACCS key index (1..166), so the UI can scrub OFF bits too."""
    return sorted(smartsPatts.keys())


def query_svg(
    bit: int,
    *,
    width: int = 260,
    height: int = 200,
) -> str | None:
    """Render the *pattern a MACCS bit looks for*, independent of any molecule.

    Draws the key's SMARTS as a query depiction. Wildcards (``*``), generic
    bonds (``~``), and element lists (``[F,Cl,Br,I]``) are shown as RDKit draws
    them — this is the honest picture of what the bit matches. Returns None for
    the three count-based keys that have no SMARTS, so the UI can show a note
    instead.
    """
    patt = smartsPatts.get(bit)
    if patt is None or patt[0] == "?":
        return None
    query = Chem.MolFromSmarts(patt[0])
    if query is None:
        return None
    rdDepictor.Compute2DCoords(query)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = False
    drawer.DrawMolecule(query)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def fingerprint_strip_svg(
    mol: Chem.Mol,
    current_bit: int,
    *,
    width: int = 940,
    height: int = 56,
    on_color: str = "#2f9e44",
    off_color: str = "#ffffff",
    current_color: str = "#1c7ed6",
    grid_color: str = "#dee2e6",
) -> str:
    """Render the whole 166-bit MACCS vector as a strip of colored cells.

    ON bits are green, OFF bits are blank (white), and the currently-scrubbed
    bit is drawn in blue on top so the reader can locate it within the whole
    fingerprint. This is a plain hand-built SVG (no matplotlib) so it renders
    crisply and cheaply on every scrub.
    """
    bits = all_bits()  # 1..166 in order
    n = len(bits)
    fp = MACCSkeys.GenMACCSKeys(mol)
    pad = 2
    cell_w = (width - 2 * pad) / n
    cell_h = height - 2 * pad

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]
    for i, bit in enumerate(bits):
        x = pad + i * cell_w
        is_on = bool(fp[bit])
        is_current = bit == current_bit
        if is_current:
            fill = current_color
        elif is_on:
            fill = on_color
        else:
            fill = off_color
        parts.append(
            f'<rect x="{x:.2f}" y="{pad}" width="{cell_w:.2f}" height="{cell_h}" '
            f'fill="{fill}" stroke="{grid_color}" stroke-width="0.3" />'
        )
        if is_current:
            # outline the current cell so it's unmissable
            parts.append(
                f'<rect x="{x - 0.5:.2f}" y="0" width="{cell_w + 1:.2f}" height="{height}" '
                f'fill="none" stroke="{current_color}" stroke-width="1.5" />'
            )
    parts.append("</svg>")
    return "".join(parts)
