"""Uniform fingerprint interface and registry.

All fingerprint methods produce a (n_molecules, n_features) numpy array.
Some are binary bit vectors (Morgan-bit, RDKit, AtomPair, TopTorsion, MACCS, Avalon),
some are float embeddings (CheMeleon, MIST). Each method declares its `kind`
so downstream code (similarity, clustering) can pick a sensible metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from rdkit.Chem import Mol

FingerprintKind = Literal["binary", "count", "continuous"]


@dataclass(frozen=True)
class FingerprintResult:
    name: str
    kind: FingerprintKind
    array: np.ndarray  # shape (n_mols, n_features)
    n_features: int

    def __post_init__(self) -> None:
        if self.array.ndim != 2:
            raise ValueError(f"expected 2D array, got shape {self.array.shape}")


class FingerprintMethod(Protocol):
    name: str
    kind: FingerprintKind

    def __call__(self, mols: list[Mol]) -> FingerprintResult: ...


def default_metric_for(kind: FingerprintKind) -> str:
    """Default distance metric matching the fingerprint kind.

    - binary -> jaccard (1 - tanimoto)
    - count -> jaccard on binarized, but cosine often used; we use cosine
    - continuous -> cosine
    """
    if kind == "binary":
        return "jaccard"
    return "cosine"


def similarity_for(kind: FingerprintKind) -> str:
    """Human-readable similarity name corresponding to default metric."""
    if kind == "binary":
        return "Tanimoto"
    return "cosine"
