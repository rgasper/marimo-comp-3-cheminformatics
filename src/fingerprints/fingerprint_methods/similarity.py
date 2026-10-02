"""Pairwise similarity / distance computation across fingerprint kinds."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from fingerprints.fingerprint_methods.base import FingerprintKind, FingerprintResult


def pairwise_similarity(fp: FingerprintResult) -> np.ndarray:
    """Return n x n similarity matrix appropriate for the fingerprint kind.

    binary -> Tanimoto (Jaccard similarity = 1 - jaccard distance)
    continuous -> cosine similarity (1 - cosine distance), shifted to [0, 1] with (s+1)/2

    Note: cosine similarity can be negative for continuous embeddings; we
    return raw cosine in [-1, 1] for continuous so plots can show that.
    """
    arr = fp.array
    if fp.kind == "binary":
        # tanimoto for bit vectors
        d = pdist(arr.astype(bool), metric="jaccard")  # in [0, 1]
        sim = 1.0 - squareform(d)
        np.fill_diagonal(sim, 1.0)
        return sim
    # continuous: cosine similarity (note: not bounded to [0,1])
    d = pdist(arr.astype(np.float32), metric="cosine")
    sim = 1.0 - squareform(d)
    np.fill_diagonal(sim, 1.0)
    return sim


def pairwise_distance(fp: FingerprintResult) -> np.ndarray:
    """Return n x n distance matrix (1 - similarity for the kind's natural metric).

    binary -> Jaccard distance
    continuous -> cosine distance
    """
    arr = fp.array
    metric = "jaccard" if fp.kind == "binary" else "cosine"
    return squareform(pdist(arr.astype(bool) if fp.kind == "binary" else arr.astype(np.float32), metric=metric))


def similarity_label(kind: FingerprintKind) -> str:
    return "Tanimoto" if kind == "binary" else "cosine"
