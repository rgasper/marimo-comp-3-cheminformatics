"""CheMeleon: a *pretrained* learned molecular fingerprint, and per-atom
attribution of individual embedding dimensions.

CheMeleon (Burns et al., 2025) is a message-passing neural network pretrained
on large molecular data; its learned graph embedding is a 2048-dim fingerprint.
Unlike MACCS/Morgan we did NOT design its features - the network learned them.

We load only the message-passing block (`chemeleon_mp.pt`) and use its
per-atom hidden vectors:

    H = mp(batch_mol_graph)          # (n_atoms, 2048)
    fingerprint(mol) = mean(H, dim=0)  # the 2048-dim molecule vector

Because the graph fingerprint is a *mean over atoms*, each atom's contribution
to embedding dimension ``k`` is exactly ``H[i, k] / n_atoms`` - an exact
decomposition (not a saliency approximation). That lets us color a molecule by
"how much each atom drives fingerprint dimension k", the learned analog of the
Morgan bit-scrubber.

Weights are cached under ``.cache/chemeleon``; download once from Zenodo.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

from fingerprints import paths

CACHE_DIR = paths.CHEMELEON_DIR
WEIGHTS = CACHE_DIR / "chemeleon_mp.pt"
WEIGHTS_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
EMBED_DIM = 2048


def pick_device() -> str:
    """cuda > mps > cpu, whatever this machine has."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_weights() -> Path:
    """Download the CheMeleon MP checkpoint if not already cached."""
    if not WEIGHTS.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    return WEIGHTS


@lru_cache(maxsize=1)
def _load_mp():
    """Load the pretrained BondMessagePassing block onto the best device."""
    import torch
    from chemprop import featurizers, nn

    ckpt = torch.load(ensure_weights(), map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    mp = nn.BondMessagePassing(
        d_v=hp["d_v"],
        d_e=hp["d_e"],
        d_h=hp["d_h"],
        depth=hp["depth"],
        bias=hp["bias"],
        dropout=hp["dropout"],
        activation=hp["activation"],
        undirected=hp["undirected"],
    )
    mp.load_state_dict(ckpt["state_dict"], strict=True)
    mp.eval()
    device = pick_device()
    mp.to(device)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    return mp, feat, device


def atom_hidden(mol: Chem.Mol) -> np.ndarray:
    """Per-atom hidden vectors after message passing: (n_atoms, EMBED_DIM)."""
    import torch
    from chemprop.data import BatchMolGraph

    mp, feat, device = _load_mp()
    bmg = BatchMolGraph([feat(mol)])
    bmg.to(device)
    with torch.no_grad():
        H = mp(bmg).cpu().numpy()
    return H  # (n_atoms, EMBED_DIM)


def fingerprint(mol: Chem.Mol) -> np.ndarray:
    """The 2048-dim CheMeleon fingerprint (mean over atoms)."""
    return atom_hidden(mol).mean(axis=0)


def fingerprint_matrix(mols, batch_size: int = 256) -> np.ndarray:
    """Mean-pooled CheMeleon fingerprints for many molecules, batched.

    Runs message passing on batches of molecules at once (one graph batch per
    forward pass) and mean-pools each molecule's atom rows. Far faster than
    calling :func:`fingerprint` in a loop. Returns ``(len(mols), EMBED_DIM)``.
    """
    import torch
    from chemprop.data import BatchMolGraph

    mp, feat, device = _load_mp()
    out = np.empty((len(mols), EMBED_DIM), dtype=np.float32)
    for start in range(0, len(mols), batch_size):
        chunk = mols[start : start + batch_size]
        graphs = [feat(m) for m in chunk]
        bmg = BatchMolGraph(graphs)
        bmg.to(device)
        with torch.no_grad():
            H = mp(bmg).cpu().numpy()  # (total atoms in chunk, EMBED_DIM)
        # BatchMolGraph concatenates atoms; split back per molecule by count.
        offset = 0
        for i, m in enumerate(chunk):
            n = m.GetNumAtoms()
            out[start + i] = H[offset : offset + n].mean(axis=0)
            offset += n
    return out

def atom_contributions(mol: Chem.Mol, dim: int) -> np.ndarray:
    """Each atom's exact contribution to embedding dimension ``dim``.

    Since fingerprint[dim] = mean_i H[i, dim], atom i contributes
    H[i, dim] / n_atoms. Returns a (n_atoms,) array.
    """
    H = atom_hidden(mol)
    n = H.shape[0]
    return H[:, dim] / n


@dataclass(frozen=True)
class DimInfo:
    dim: int
    value: float  # fingerprint[dim] (the mean)
    top_atom: int  # atom contributing most (abs)


def dim_sensitivity(mol: Chem.Mol) -> np.ndarray:
    """Per-dimension **structure-sensitivity**: how much a dimension's per-atom
    contribution varies across the molecule's atoms.

    For dimension k, with per-atom hidden values H[i, k], we use the spread

        s_k = max_i H[i, k] - min_i H[i, k]

    A large s_k means different atoms push dimension k very differently - the
    dimension is reading a *local* structural feature, so a heatmap of it is
    informative. A small s_k means every atom contributes about the same, so
    the dimension encodes something diffuse/global and its heatmap is flat.
    Returns a (EMBED_DIM,) array.
    """
    H = atom_hidden(mol)
    return H.max(axis=0) - H.min(axis=0)


def most_active_dims(mol: Chem.Mol, *, floor_frac: float = 0.5) -> list[int]:
    """Structure-sensitive dimensions worth scrubbing, in ascending index order.

    A dimension qualifies when its sensitivity ``s_k`` (see ``dim_sensitivity``)
    is at least ``floor_frac`` of this molecule's most-sensitive dimension - a
    **relative floor**, no fixed count. So the number of “active” dimensions
    varies from molecule to molecule (a big, decorated molecule lights up many;
    a small or symmetric one lights up few) - exactly like the varying on-bit
    count of ECFP/MACCS. Returned sorted by index so scrubbing moves
    left-to-right.

    Symmetric/tiny molecules (e.g. benzene) can legitimately have *no* sensitive
    dimension; we then fall back to the single most-varying one so the UI still
    has something to show.
    """
    spread = dim_sensitivity(mol)
    mx = float(spread.max())
    order = np.argsort(spread)[::-1]
    if mx <= 1e-6:
        return [int(order[0])]
    threshold = floor_frac * mx
    qualifying = [int(d) for d in order if spread[d] >= threshold]
    picked = qualifying if qualifying else [int(order[0])]
    return sorted(picked)


def strip_svg(
    mol: Chem.Mol,
    current_dim: int,
    *,
    active_dims: list[int] | None = None,
    width: int = 920,
    height: int = 40,
) -> str:
    """The dense 2048-dim CheMeleon vector as a strip: each dimension's cell is
    shaded by |fingerprint[k]| (how strongly this molecule activates it). The
    currently-selected dimension is marked blue; the structure-sensitive
    dimensions the scrubber visits are ticked green underneath.

    Unlike the Morgan strip (sparse 0/1 bits), CheMeleon dimensions are
    continuous, so we encode *magnitude* rather than on/off.
    """
    fp = np.abs(fingerprint(mol))
    mx = float(fp.max()) or 1.0
    active = set(active_dims or [])
    pad = 2
    inner_w = width - 2 * pad
    band_h = height - 2 * pad
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="{pad}" y="{pad}" width="{inner_w}" height="{band_h}" '
        f'fill="#f1f3f5" stroke="#dee2e6" stroke-width="0.5" />',
    ]
    tick_w = max(inner_w / EMBED_DIM, 0.6)
    # Value-magnitude shading across all dims.
    for k in range(EMBED_DIM):
        frac = fp[k] / mx
        if frac <= 0.02:
            continue
        x = pad + (k / EMBED_DIM) * inner_w
        alpha = 0.12 + 0.88 * min(frac, 1.0)
        parts.append(
            f'<rect x="{x:.2f}" y="{pad}" width="{max(tick_w, 1.0):.2f}" '
            f'height="{band_h * 0.7:.1f}" fill="#7048e8" fill-opacity="{alpha:.2f}" />'
        )
    # Green ticks for the structure-sensitive dims the scrubber can visit.
    for k in active:
        x = pad + (k / EMBED_DIM) * inner_w
        parts.append(
            f'<rect x="{x:.2f}" y="{pad + band_h * 0.72:.1f}" '
            f'width="{max(tick_w, 1.5):.2f}" height="{band_h * 0.28:.1f}" '
            f'fill="#2f9e44" />'
        )
    # Current dimension: full-height blue marker on top.
    cx = pad + (current_dim / EMBED_DIM) * inner_w
    cw = max(tick_w, 4.0)
    parts.append(
        f'<rect x="{cx - cw / 2:.2f}" y="0" width="{cw:.2f}" height="{height}" '
        f'fill="#1c7ed6" />'
    )
    parts.append("</svg>")
    return "".join(parts)


def heatmap_svg(mol: Chem.Mol, dim: int, *, width: int = 460, height: int = 340) -> str:
    """Color the molecule by each atom's contribution to dimension ``dim``.

    Green = pushes the dimension up, pink = pushes it down (RDKit similarity-map
    convention). This is an *estimated* read of what the learned dimension keys
    on for this molecule - not a fixed substructure definition like Morgan.
    """
    weights = [float(w) for w in atom_contributions(mol, dim)]
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d.drawOptions().addStereoAnnotation = False
    if mol.GetNumAtoms() < 2:
        # Similarity maps need >=2 atoms; just draw the structure plainly.
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    # If every atom contributes ~equally the map is flat; that's fine.
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, weights, draw2d=d, contourLines=5, gridResolution=0.3
    )
    d.FinishDrawing()
    return d.GetDrawingText()
