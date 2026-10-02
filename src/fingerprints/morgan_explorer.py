"""Interactively explore Morgan (ECFP-like) bits on a single molecule.

Morgan fingerprints are the workhorse of modern cheminformatics and the
conceptual opposite of MACCS. Instead of a fixed checklist of named
substructures, Morgan asks, for every atom: "what does the circular
neighborhood out to radius R look like?" It hashes each such atom environment
into a bit. So a Morgan bit has no human-given name — but RDKit can tell us
exactly which atom(s) and radius produced it, which is all we need to highlight
the environment on the molecule.

This module mirrors ``maccs_explorer`` so the notebook can drive both with the
same interaction (scrub bits -> highlight substructure), letting the reader feel
the difference between an expert checklist and a learned/hashed representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

DEFAULT_RADIUS = 2
DEFAULT_BITS = 2048


@dataclass(frozen=True)
class MorganBitHit:
    """Which atoms/bonds set a Morgan bit, and at what radius."""

    bit: int
    n_instances: int  # how many atom centers produced this bit
    radii: tuple[int, ...]  # distinct radii across those centers
    centers: tuple[int, ...]  # the center atom indices
    atoms: tuple[int, ...]  # union of environment atoms (for highlight)
    bonds: tuple[int, ...]  # union of environment bonds


def _bit_info(mol: Chem.Mol, radius: int, n_bits: int):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetFingerprint(mol, additionalOutput=ao)
    return ao.GetBitInfoMap()


def _env_atoms_bonds(mol: Chem.Mol, center: int, radius: int) -> tuple[set[int], set[int]]:
    """Atoms and bonds of the circular environment around `center` at `radius`."""
    if radius == 0:
        return {center}, set()
    env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
    atom_map: dict[int, int] = {}
    Chem.PathToSubmol(mol, env_bonds, atomMap=atom_map)
    atoms = set(atom_map.keys()) | {center}
    return atoms, set(env_bonds)


def on_bits(mol: Chem.Mol, radius: int = DEFAULT_RADIUS, n_bits: int = DEFAULT_BITS) -> list[int]:
    """Sorted Morgan bit indices that are set for this molecule."""
    return sorted(_bit_info(mol, radius, n_bits).keys())


def distinct_environments(mol: Chem.Mol, radius: int = DEFAULT_RADIUS) -> int:
    """How many distinct atom environments the molecule actually has.

    This is the collision-free count: the number of unique Morgan identifiers
    before they get hashed (folded) down into a fixed-length bit vector. It is
    the ceiling that a long-enough fingerprint would reach.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    sparse = gen.GetSparseCountFingerprint(mol)
    return len(sparse.GetNonzeroElements())


@dataclass(frozen=True)
class CollisionPoint:
    n_bits: int
    bits_set: int  # distinct bits turned on at this length
    distinct_envs: int  # true number of environments (constant across lengths)
    collisions: int  # distinct_envs - bits_set: environments that shared a bit


def collision_curve(
    mol: Chem.Mol,
    radius: int = DEFAULT_RADIUS,
    lengths: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
) -> list[CollisionPoint]:
    """Trace how hash collisions vanish as the fingerprint gets longer.

    At short lengths many distinct atom environments are forced to share a bit
    (a collision); lengthening the vector spreads them out until nearly every
    environment gets its own bit.
    """
    distinct = distinct_environments(mol, radius)
    out: list[CollisionPoint] = []
    for n in lengths:
        bits_set = len(_bit_info(mol, radius, n))
        out.append(
            CollisionPoint(
                n_bits=n,
                bits_set=bits_set,
                distinct_envs=distinct,
                collisions=distinct - bits_set,
            )
        )
    return out


def bit_hit(
    mol: Chem.Mol,
    bit: int,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_BITS,
) -> MorganBitHit:
    """Resolve a Morgan bit: which atom environment(s) produced it, and where."""
    info = _bit_info(mol, radius, n_bits)
    instances = info.get(bit, ())
    atoms: set[int] = set()
    bonds: set[int] = set()
    centers: list[int] = []
    radii: set[int] = set()
    for center, r in instances:
        centers.append(center)
        radii.add(r)
        a, b = _env_atoms_bonds(mol, center, r)
        atoms |= a
        bonds |= b
    return MorganBitHit(
        bit=bit,
        n_instances=len(instances),
        radii=tuple(sorted(radii)),
        centers=tuple(centers),
        atoms=tuple(sorted(atoms)),
        bonds=tuple(sorted(bonds)),
    )


def highlight_svg(
    mol: Chem.Mol,
    hit: MorganBitHit,
    *,
    width: int = 460,
    height: int = 340,
    env_color: tuple[float, float, float] = (0.20, 0.55, 0.95),
    center_color: tuple[float, float, float] = (0.95, 0.35, 0.25),
) -> str:
    """Render mol as SVG with a Morgan bit's environment highlighted.

    The environment atoms are shaded blue; the center atom(s) that define the
    bit are shaded a distinct red so the "grows outward from here" idea is
    visible.
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = False
    atom_colors = {a: env_color for a in hit.atoms}
    for c in hit.centers:
        atom_colors[c] = center_color
    bond_colors = {b: env_color for b in hit.bonds}
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
    current_bit: int,
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_BITS,
    width: int = 940,
    height: int = 40,
    on_color: str = "#2f9e44",
    off_color: str = "#f1f3f5",
    current_color: str = "#1c7ed6",
) -> str:
    """Render the Morgan bit vector as a strip: ON green, current blue.

    Unlike MACCS, a Morgan vector is long and sparse — most bits are off and an
    off bit carries no specific meaning (it just means 'no environment hashed
    here'). So the strip is mostly blank; the value is seeing *how sparse* it is
    and locating the handful of ON bits (and the current one) within the whole
    length. Individual cells are sub-pixel at 2048 bits, so we draw ON/current
    bits as full-height ticks that stay visible.
    """
    on = set(on_bits(mol, radius, n_bits))
    pad = 2
    inner_w = width - 2 * pad
    parts: list[str] = [
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
        cw = max(tick_w, 4.0)
        parts.append(
            f'<rect x="{current_x - cw / 2:.2f}" y="0" width="{cw:.2f}" '
            f'height="{height}" fill="{current_color}" />'
        )
    parts.append("</svg>")
    return "".join(parts)


def _env_signature(mol: Chem.Mol, center: int, radius: int) -> str:
    """A canonical string identifying an atom environment's *shape*.

    Two environments with the same signature are genuinely the same substructure;
    different signatures that land on the same bit are a real hash collision.
    """
    if radius == 0:
        a = mol.GetAtomWithIdx(center)
        return f"atom:{a.GetSymbol()}{'(ar)' if a.GetIsAromatic() else ''}"
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
    sub = Chem.PathToSubmol(mol, bonds)
    return Chem.MolToSmiles(sub)


@dataclass(frozen=True)
class Collision:
    """One bit that two or more *different* environments were forced to share."""

    bit: int
    n_distinct: int  # number of distinct environment shapes on this bit
    signatures: tuple[str, ...]  # the distinct environment signatures
    atom_groups: tuple[tuple[int, ...], ...]  # atoms per distinct environment


def find_collisions(
    mol: Chem.Mol,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = 128,
) -> list[Collision]:
    """Bits where two or more genuinely different environments hashed together.

    At a short length like 128, several distinct substructures are forced onto
    the same bit — the fingerprint can no longer tell them apart. Returns those
    bits with the atoms of each colliding environment, so the UI can highlight
    them on the molecule and *show* the collision.
    """
    info = _bit_info(mol, radius, n_bits)
    collisions: list[Collision] = []
    for bit, instances in info.items():
        groups: dict[str, set[int]] = {}
        for center, r in instances:
            sig = _env_signature(mol, center, r)
            atoms, _bonds = _env_atoms_bonds(mol, center, r)
            groups.setdefault(sig, set()).update(atoms)
        if len(groups) > 1:
            sigs = tuple(groups.keys())
            collisions.append(
                Collision(
                    bit=bit,
                    n_distinct=len(groups),
                    signatures=sigs,
                    atom_groups=tuple(tuple(sorted(groups[s])) for s in sigs),
                )
            )
    return sorted(collisions, key=lambda c: (-c.n_distinct, c.bit))


# A small qualitative palette for distinguishing colliding environments.
COLLISION_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.30, 0.24),  # red
    (0.20, 0.55, 0.95),  # blue
    (0.20, 0.65, 0.35),  # green
    (0.85, 0.55, 0.10),  # amber
    (0.60, 0.35, 0.80),  # purple
)


def collision_svg(
    mol: Chem.Mol,
    collision: Collision,
    *,
    width: int = 520,
    height: int = 380,
) -> str:
    """Render mol with each colliding environment shaded a different color.

    Makes a hash collision tangible: two differently-colored regions that the
    short fingerprint nonetheless records as the *same* bit.
    """
    atom_colors: dict[int, tuple[float, float, float]] = {}
    highlight_atoms: list[int] = []
    for i, atoms in enumerate(collision.atom_groups):
        color = COLLISION_PALETTE[i % len(COLLISION_PALETTE)]
        for a in atoms:
            atom_colors[a] = color
            highlight_atoms.append(a)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
