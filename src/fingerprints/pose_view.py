"""Load cached Boltz-2 poses for the activity-cliff section.

The offline ``fingerprints.rebuild_poses`` co-folded a cliff pair's two ligands
into both receptors and cached the predicted complexes (ModelCIF) plus metadata
under ``data/boltz_poses/``. This module reads those cached files so the notebook
can render them instantly - it never calls Boltz.

It also parses the ModelCIF ``_atom_site`` table (this RDKit build has no mmCIF
reader) to find which protein residues line the pocket, so the notebook can name
the conserved anchor (the D3.32 aspartate) that validates the pose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from fingerprints import paths

POSE_DIR = paths.BOLTZ_POSES


@dataclass(frozen=True)
class Pose:
    tag: str
    mol_id: str
    target: str
    smiles: str
    binding_confidence: float
    ligand_iptm: float
    structure_confidence: float
    cif_text: str


def load_manifest(pair: str = "mu_vs_kappa", index: int = 2) -> list[dict]:
    path = POSE_DIR / f"{pair}_{index}_manifest.json"
    return json.loads(path.read_text())


def has_poses(pair: str, index: int) -> bool:
    """True if all four poses for this cliff have been folded and cached."""
    manifest = POSE_DIR / f"{pair}_{index}_manifest.json"
    if not manifest.exists():
        return False
    try:
        entries = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return len(entries) == 4 and all(
        (POSE_DIR / e["cif_file"]).exists() for e in entries
    )


def load_pose(entry: dict) -> Pose:
    cif = (POSE_DIR / entry["cif_file"]).read_text()
    return Pose(
        tag=entry["tag"],
        mol_id=entry["mol_id"],
        target=entry["target"],
        smiles=entry["smiles"],
        binding_confidence=entry.get("binding_confidence") or 0.0,
        ligand_iptm=entry.get("ligand_iptm") or 0.0,
        structure_confidence=entry.get("structure_confidence") or 0.0,
        cif_text=cif,
    )


def load_all(pair: str = "mu_vs_kappa", index: int = 2) -> dict[str, Pose]:
    """All four poses keyed by (mol_id, target) tag suffix."""
    out: dict[str, Pose] = {}
    for entry in load_manifest(pair, index):
        out[f"{entry['mol_id']}_{entry['target']}"] = load_pose(entry)
    return out


def load_interactions(pose: Pose) -> list[dict]:
    """PLIP interaction records for a pose (empty list if not yet computed).

    Written by ``fingerprints.rebuild_poses`` next to each pose's CIF.
    """
    path = POSE_DIR / f"{pose.tag}.interactions.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


# Interaction types in a stable display order + labels for charts/legends.
INTERACTION_TYPES: tuple[tuple[str, str], ...] = (
    ("saltbridge", "salt bridge"),
    ("hbond", "H-bond"),
    ("pistack", "π-stack"),
    ("pication", "π-cation"),
    ("hydrophobic", "hydrophobic"),
)


def interaction_counts(pose: Pose) -> dict[str, int]:
    """Count of each interaction type for a pose (0 for types not present)."""
    counts = {t: 0 for t, _ in INTERACTION_TYPES}
    for rec in load_interactions(pose):
        if rec["type"] in counts:
            counts[rec["type"]] += 1
    return counts


def interaction_fingerprint(pose: Pose) -> dict[tuple[str, str], int]:
    """Encode a pose as an *interaction fingerprint*: (residue, type) -> count.

    This is the same idea as a structural fingerprint, but the bits are read off
    the binding event itself — each 'on' bit is a specific contact the pose
    makes (e.g. an H-bond to ASP149). The data defines the features, rather than
    us imposing them.
    """
    fp: dict[tuple[str, str], int] = {}
    for rec in load_interactions(pose):
        key = (rec["residue"], rec["type"])
        fp[key] = fp.get(key, 0) + 1
    return fp


def aligned_interaction_fingerprints(
    poses: dict[str, Pose], keys: list[str]
) -> tuple[list[tuple[str, str]], dict[str, dict[tuple[str, str], int]]]:
    """Interaction fingerprints for several poses over a shared, sorted key set.

    Returns (all_keys, {pose_key: fingerprint}) where all_keys is the union of
    (residue, interaction-type) bits across the given poses — so the poses'
    fingerprints line up column-for-column for a comparison grid. Keys are
    sorted by interaction type (in display order) then residue number.
    """
    type_order = {t: i for i, (t, _) in enumerate(INTERACTION_TYPES)}
    fps = {k: interaction_fingerprint(poses[k]) for k in keys}
    all_keys: set[tuple[str, str]] = set()
    for fp in fps.values():
        all_keys |= set(fp)

    def _resnum(residue: str) -> int:
        digits = "".join(ch for ch in residue if ch.isdigit())
        return int(digits) if digits else 0

    ordered = sorted(
        all_keys, key=lambda k: (type_order.get(k[1], 99), _resnum(k[0]), k[0])
    )
    return ordered, fps


def _atom_site(cif_text: str) -> tuple[dict[str, int], list[list[str]]]:
    lines = cif_text.splitlines()
    i = 0
    cols: list[str] = []
    start: int | None = None
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            hdr: list[str] = []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                hdr.append(lines[j].strip())
                j += 1
            if hdr:
                cols = [h.split(".")[1] for h in hdr]
                start = j
                break
        i += 1
    if start is None:
        return {}, []
    idx = {c: k for k, c in enumerate(cols)}
    rows: list[list[str]] = []
    for line in lines[start:]:
        s = line.strip()
        if s in ("", "#") or s.startswith("loop_") or s.startswith("_"):
            break
        parts = s.split()
        if len(parts) >= len(cols):
            rows.append(parts)
    return idx, rows


def pocket_residues(cif_text: str, cutoff: float = 4.0) -> list[tuple[str, str, float]]:
    """Protein residues within `cutoff` Angstroms of the ligand, nearest first.

    Returns (residue_name, residue_seq, min_distance) tuples.
    """
    idx, rows = _atom_site(cif_text)
    if not rows:
        return []
    gx = idx["group_PDB"]
    cx = (idx["Cartn_x"], idx["Cartn_y"], idx["Cartn_z"])
    cres, cseq = idx["label_comp_id"], idx["label_seq_id"]
    lig: list[np.ndarray] = []
    prot: list[tuple[str, str, np.ndarray]] = []
    for r in rows:
        xyz = np.array([float(r[cx[0]]), float(r[cx[1]]), float(r[cx[2]])])
        if r[gx] == "HETATM":
            lig.append(xyz)
        else:
            prot.append((r[cres], r[cseq], xyz))
    if not lig:
        return []
    lig_arr = np.array(lig)
    contacts: dict[tuple[str, str], float] = {}
    for rn, seq, p in prot:
        d = float(np.min(np.linalg.norm(lig_arr - p, axis=1)))
        if d < cutoff:
            key = (rn, seq)
            contacts[key] = min(contacts.get(key, 1e9), d)
    return sorted(
        ((rn, seq, d) for (rn, seq), d in contacts.items()), key=lambda x: x[2]
    )


def _ligand_block(cif_text: str) -> tuple[list[int], list[str], list[tuple[float, float, float]]]:
    """Ligand atoms from the CIF: (serials, elements, coords), in file order."""
    idx, rows = _atom_site(cif_text)
    gx = idx["group_PDB"]
    ai, ts = idx["id"], idx["type_symbol"]
    cx = (idx["Cartn_x"], idx["Cartn_y"], idx["Cartn_z"])
    serials: list[int] = []
    elements: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for r in rows:
        if r[gx] == "HETATM":
            serials.append(int(r[ai]))
            elements.append(r[ts])
            coords.append((float(r[cx[0]]), float(r[cx[1]]), float(r[cx[2]])))
    return serials, elements, coords


def _ligand_atom_names(cif_text: str) -> list[str]:
    """Ligand atom names (label_atom_id) in CIF file order."""
    idx, rows = _atom_site(cif_text)
    gx = idx["group_PDB"]
    la = idx["label_atom_id"]
    return [r[la] for r in rows if r[gx] == "HETATM"]


def changed_atom_names(pose: Pose, other: Pose) -> list[str]:
    """CIF atom *names* of `pose`'s ligand atoms that differ from `other`'s.

    Determined chemically via the maximum common substructure of the two
    ligand SMILES (not geometrically — the two ligands were folded into
    different receptors, so their coordinate frames don't superimpose). Boltz
    orders the CIF ligand atoms to match the input SMILES atom order, so the
    RDKit changed-atom indices map onto the CIF ligand atoms by position; we
    return their unique atom *names* (e.g. "O26") because 3Dmol selects reliably
    by atom name, not by the CIF numeric id.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    mol = Chem.MolFromSmiles(pose.smiles)
    ref = Chem.MolFromSmiles(other.smiles)
    if mol is None or ref is None:
        return []
    mcs = rdFMCS.FindMCS(
        [mol, ref],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        timeout=10,
    )
    patt = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
    core = set(mol.GetSubstructMatch(patt)) if patt is not None else set()
    changed_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in core]

    names = _ligand_atom_names(pose.cif_text)
    # CIF ligand atoms are in SMILES (heavy-atom) order; map index -> name.
    return [names[i] for i in changed_idx if i < len(names)]
