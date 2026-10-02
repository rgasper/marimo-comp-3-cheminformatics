"""Activity-cliff analysis: how does each fingerprint score known cliff pairs?

Activity cliffs are pairs of structurally-similar molecules with very
different activity. We define them in a fingerprint-agnostic way using
graph properties only:

    - |Delta y| >= 2.0 (>=100x potency difference)
    - graph_distance(i, j) <= 5

where graph_distance is

    graph_distance(i, j) = n_atoms(i) + n_atoms(j) - 2 * mcs_atoms(i, j)

i.e. the number of heavy atoms not shared via the maximum common
substructure. This counts the symmetric difference of atom sets under
the best MCS mapping. With strict atom-type matching it captures small
chemical changes the way a chemist would: a methyl->ethyl is distance 1,
a C->N ring swap is distance 2, a halogen swap is distance 2 (one delete
+ one insert), a ring expansion / contraction is distance 1-2, and
adding a phenyl ring is distance 6+ (rejected). Importantly it does
*not* require scaffold equality, so ring expansions and contractions are
admitted.

The MCS step is expensive but easily cached. We pre-filter cheaply on
heavy-atom count difference and atom-composition L1 distance (both lower
bounds for graph distance) to keep the MCS workload manageable.

Once the cliff pair set is built (no fingerprint involved), the per-FP
metric is purely:

    For each cliff pair (i, j), compute fp_similarity(i, j).

The aggregate is the distribution of those similarities. Lower is better
- a fingerprint that scores cliffs as low-similarity correctly recognizes
that they are different. A fingerprint that scores them high is "cliff-
blind" and would confidently propagate one molecule's activity to the
other in any kNN-style retrieval.

Example:
    >>> # cliff_pairs = find_cliff_pairs(mols, y)
    >>> # sims = cliff_similarities(morgan_fp, cliff_pairs)
    >>> # np.median(sims)  # lower = better cliff resolution
    0.31
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from rdkit.Chem import Mol, rdFMCS
from sklearn.metrics.pairwise import paired_distances
from typeguard import typechecked

from fingerprints.fingerprint_methods.base import (
    FingerprintResult,
    default_metric_for,
)

DEFAULT_DELTA_Y = 2.0
DEFAULT_GRAPH_DIST = 5
DEFAULT_DELTA_ATOMS = 3
DEFAULT_L1 = 5


@dataclass(frozen=True)
class CliffPair:
    """One graph-distance-defined activity cliff.

    Attributes:
        i, j: dataset indices, i < j.
        graph_distance: n_atoms(i) + n_atoms(j) - 2 * mcs_atoms(i, j).
            Symmetric atom-difference under the best MCS mapping.
        delta_y: |y_i - y_j|, the activity gap.
        mcs_atoms: number of atoms in the MCS, kept for diagnostics.
    """

    i: int
    j: int
    graph_distance: int
    delta_y: float
    mcs_atoms: int


def _atom_composition(mol: Mol) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "H":
            continue
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def _composition_l1(a: dict[str, int], b: dict[str, int]) -> int:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def _mcs_atoms(mol_i: Mol, mol_j: Mol, timeout: int = 2) -> int | None:
    """Return MCS atom count or None if MCS computation is canceled.

    Settings:
        - atomCompare=CompareElements: distinct elements never match.
        - bondCompare=CompareOrderExact: aromatic bonds match aromatic
          only, single matches single, etc.
        - completeRingsOnly=False: allow partial ring matches so that
          ring contraction / expansion (size 5 vs 6 ring of same scaffold)
          is admitted.
        - ringMatchesRingOnly=True: an atom in a ring can only match an
          atom in a ring. Without this, a long carbon chain matches the
          atoms of a ring of the same size, falsely calling open-vs-
          cyclic pairs structurally similar.
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
    return int(res.numAtoms)


def mcs_diff_atoms(
    mol_i: Mol, mol_j: Mol, timeout: int = 2,
) -> tuple[list[int], list[int]] | None:
    """Atoms in each molecule that are NOT part of the MCS.

    Used by the example-figure plot to highlight the differing atoms.
    Returns (diff_atoms_i, diff_atoms_j) as lists of atom indices, or
    None if MCS computation fails.
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


@typechecked
def find_cliff_pairs(
    mols: list,
    y: np.ndarray,
    delta_y_min: float = DEFAULT_DELTA_Y,
    graph_dist_max: int = DEFAULT_GRAPH_DIST,
    delta_atoms_max: int = DEFAULT_DELTA_ATOMS,
    l1_max: int = DEFAULT_L1,
    n_jobs: int = -1,
    mcs_timeout: int = 2,
) -> list[CliffPair]:
    """Enumerate fingerprint-agnostic activity cliffs in a dataset.

    Pipeline (each stage is a strict lower bound for the next, so pruning
    here is safe):

    1. Activity gap: |y_i - y_j| >= delta_y_min.
    2. Heavy-atom count: |n_atoms(i) - n_atoms(j)| <= delta_atoms_max.
       Necessary because graph_distance >= |delta n_atoms|.
    3. Composition L1: sum over elements of |count_e(i) - count_e(j)|
       <= l1_max. Necessary because graph_distance >= L1.
    4. MCS verification (parallel): graph_distance computed exactly, kept
       if <= graph_dist_max.

    Parallelization uses joblib threads (RDKit MCS releases the GIL).
    """
    n = len(mols)
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} rows, mols has {n}")

    n_atoms = np.array([m.GetNumHeavyAtoms() for m in mols])
    comps = [_atom_composition(m) for m in mols]

    # Stage 1: |delta y|
    delta_y_full = np.abs(y[:, None] - y[None, :])
    i_arr, j_arr = np.where(delta_y_full >= delta_y_min)
    mask = i_arr < j_arr
    i_arr, j_arr = i_arr[mask], j_arr[mask]
    logger.info(
        f"cliff candidates: |\u0394y| >= {delta_y_min} -> {len(i_arr)} pairs"
    )

    # Stage 2: |delta n_atoms|
    mask = np.abs(n_atoms[i_arr] - n_atoms[j_arr]) <= delta_atoms_max
    i_arr, j_arr = i_arr[mask], j_arr[mask]
    logger.info(
        f"  + |\u0394n_atoms| <= {delta_atoms_max} -> {len(i_arr)} pairs"
    )

    # Stage 3: composition L1
    mask = np.array([
        _composition_l1(comps[int(i)], comps[int(j)]) <= l1_max
        for i, j in zip(i_arr, j_arr)
    ])
    i_arr, j_arr = i_arr[mask], j_arr[mask]
    logger.info(
        f"  + composition L1 <= {l1_max} -> {len(i_arr)} pairs"
    )

    if len(i_arr) == 0:
        return []

    # Stage 4: parallel MCS
    logger.info(f"  running parallel MCS on {len(i_arr)} pairs (n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_mcs_atoms)(mols[int(i)], mols[int(j)], mcs_timeout)
        for i, j in zip(i_arr, j_arr)
    )

    pairs: list[CliffPair] = []
    for k, mcs in enumerate(results):
        if mcs is None:
            continue
        i, j = int(i_arr[k]), int(j_arr[k])
        gd = int(n_atoms[i]) + int(n_atoms[j]) - 2 * mcs
        if gd > graph_dist_max:
            continue
        pairs.append(CliffPair(
            i=i, j=j, graph_distance=gd,
            delta_y=float(abs(y[i] - y[j])),
            mcs_atoms=mcs,
        ))
    logger.info(
        f"  + graph_distance <= {graph_dist_max} -> {len(pairs)} cliff pairs"
    )
    return pairs


@typechecked
def cliff_similarities(
    fp: FingerprintResult,
    cliff_pairs: list[CliffPair],
) -> np.ndarray:
    """For each cliff pair, return the fingerprint's pairwise similarity.

    Binary fingerprints use Tanimoto, continuous use cosine. The two are
    on different absolute scales but both are in [0, 1] for non-negative
    inputs, both have "1 = identical" and both increase with similarity,
    so they're directly comparable as ordinal "this FP thinks they're
    similar" measurements.

    Returns:
        sims: (n_pairs,) array of similarities in [0, 1].
    """
    metric = default_metric_for(fp.kind)
    arr = fp.array
    if fp.kind == "binary":
        arr_a = arr[[p.i for p in cliff_pairs]].astype(bool)
        arr_b = arr[[p.j for p in cliff_pairs]].astype(bool)
        intersection = np.sum(arr_a & arr_b, axis=1)
        union = np.sum(arr_a | arr_b, axis=1)
        sims = np.where(union == 0, 0.0, intersection / np.maximum(union, 1))
    else:
        arr_f = arr.astype(np.float32, copy=False)
        a = arr_f[[p.i for p in cliff_pairs]]
        b = arr_f[[p.j for p in cliff_pairs]]
        dists = paired_distances(a, b, metric=metric)
        sims = 1.0 - dists
    return sims.astype(np.float64)


@dataclass(frozen=True)
class CliffSimResult:
    """Per-fingerprint cliff-similarity statistics for one dataset.

    Attributes:
        name: fingerprint display name.
        sims: similarity values for every cliff pair (length = n_pairs).
        n_pairs: number of cliff pairs (== sims.shape[0]).
        median_sim: median similarity across cliff pairs.
        mean_sim: mean similarity.
        cliff_blind_rate_05: fraction with similarity >= 0.5
            (cliff-blind by a generous threshold).
        cliff_blind_rate_07: fraction with similarity >= 0.7
            (cliff-blind by a stricter threshold; in classical Morgan
            terms this is the conventional "very similar" cutoff).
    """

    name: str
    sims: np.ndarray
    n_pairs: int
    median_sim: float
    mean_sim: float
    cliff_blind_rate_05: float
    cliff_blind_rate_07: float


@typechecked
def cliff_similarity_for(
    fp: FingerprintResult,
    cliff_pairs: list[CliffPair],
) -> CliffSimResult:
    """Compute the cliff similarity distribution + summary stats for one FP."""
    sims = cliff_similarities(fp, cliff_pairs)
    n = sims.shape[0]
    if n == 0:
        return CliffSimResult(
            name=fp.name, sims=sims, n_pairs=0,
            median_sim=float("nan"), mean_sim=float("nan"),
            cliff_blind_rate_05=float("nan"),
            cliff_blind_rate_07=float("nan"),
        )
    median_sim = float(np.median(sims))
    mean_sim = float(np.mean(sims))
    cb05 = float((sims >= 0.5).mean())
    cb07 = float((sims >= 0.7).mean())
    logger.info(
        f"cliff sims: {fp.name} n={n} median={median_sim:.3f} "
        f"mean={mean_sim:.3f} P(sim>=0.5)={cb05:.3f} P(sim>=0.7)={cb07:.3f}"
    )
    return CliffSimResult(
        name=fp.name,
        sims=sims,
        n_pairs=n,
        median_sim=median_sim,
        mean_sim=mean_sim,
        cliff_blind_rate_05=cb05,
        cliff_blind_rate_07=cb07,
    )


@typechecked
def cliff_similarity_for_all(
    fps: dict[str, FingerprintResult],
    cliff_pairs: list[CliffPair],
) -> dict[str, CliffSimResult]:
    return {sid: cliff_similarity_for(fp, cliff_pairs) for sid, fp in fps.items()}


@dataclass(frozen=True)
class CliffExamplePair:
    """A single cliff pair selected as an illustration for one fingerprint.

    Used for the per-fingerprint deep-dive example figures.
    """

    name: str
    i: int
    j: int
    similarity: float
    delta_y: float
    graph_distance: int


@typechecked
def select_cliff_examples(
    fp: FingerprintResult,
    cliff_pairs: list[CliffPair],
    n_top: int = 3,
) -> tuple[list[CliffExamplePair], list[CliffExamplePair]]:
    """Return (most_cliff_blind, most_cliff_aware) lists for this fingerprint.

    most_cliff_blind: the n_top cliff pairs with the highest FP similarity.
        These are the FP's worst-case misses - cases where it scored a true
        activity cliff as very similar.

    most_cliff_aware: the n_top cliff pairs with the lowest FP similarity.
        These are the FP's best catches - cases where it correctly
        recognized cliff pairs as different.

    Both lists are returned in descending order of "extremeness": index 0
    is the single most blind / most aware example, index n_top-1 is the
    third most.
    """
    if not cliff_pairs:
        raise ValueError("cliff_pairs is empty")
    if n_top < 1:
        raise ValueError(f"n_top must be >= 1, got {n_top}")
    sims = cliff_similarities(fp, cliff_pairs)
    n_top = min(n_top, len(cliff_pairs))

    # Indices sorted by similarity (ascending) and (descending)
    order_asc = np.argsort(sims, kind="stable")
    order_desc = order_asc[::-1]

    def _make(idx: int) -> CliffExamplePair:
        p = cliff_pairs[idx]
        return CliffExamplePair(
            name=fp.name, i=p.i, j=p.j,
            similarity=float(sims[idx]),
            delta_y=p.delta_y,
            graph_distance=p.graph_distance,
        )

    most_blind = [_make(int(order_desc[k])) for k in range(n_top)]
    most_aware = [_make(int(order_asc[k])) for k in range(n_top)]
    return most_blind, most_aware


@typechecked
def select_cliff_examples_for_all(
    fps: dict[str, FingerprintResult],
    cliff_pairs: list[CliffPair],
    n_top: int = 3,
) -> dict[str, tuple[list[CliffExamplePair], list[CliffExamplePair]]]:
    return {
        sid: select_cliff_examples(fp, cliff_pairs, n_top=n_top)
        for sid, fp in fps.items()
    }
