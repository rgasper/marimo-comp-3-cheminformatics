"""Read the cached kNN cliff-failure analysis (``fingerprints.analyses.knn``)
for the notebook's "why a similarity model can't see the cliff" section.

kNN regression on ECFP is the simplest model whose behaviour *is* the
fingerprint's similarity: a molecule's prediction is the mean activity of its
Tanimoto-nearest neighbours. So its blindness to a cliff is the fingerprint's
blindness. This module just serves the precomputed numbers; it never trains.
"""

from __future__ import annotations

import json
from functools import lru_cache

from fingerprints import paths

CACHE = paths.KNN_CLIFFS


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads(CACHE.read_text())


def has_data() -> bool:
    return CACHE.exists()


def endpoints() -> list[str]:
    return list(_data()["endpoints"].keys()) if has_data() else []


def k_grid() -> list[int]:
    return list(_data()["k_grid"]) if has_data() else []


def k_curve(endpoint: str) -> list[dict]:
    """[{k, r2}, ...] held-out R^2 vs neighbourhood size for an endpoint."""
    return _data()["endpoints"][endpoint]["k_curve"]


def k_curves_by_fp(endpoint: str) -> dict:
    """{fp_key: {label, k_curve:[{k,r2}]}} - the held-out curve under each
    fingerprint's similarity, for the overlaid multi-line plot."""
    ep = _data()["endpoints"][endpoint]
    return ep.get("k_curves_by_fp", {})


def fp_colors() -> dict:
    """{fingerprint label: hex color} - one fixed palette shared by every
    chart that splits by fingerprint (k-curve lines + cliff scatter)."""
    return dict(_data().get("fp_colors", {}))


def endpoint_meta(endpoint: str) -> dict:
    ep = _data()["endpoints"][endpoint]
    return {"n_total": ep["n_total"], "n_train": ep["n_train"], "n_test": ep["n_test"]}


def smoothness(endpoint: str) -> dict:
    """The 'why any structure-only model must be smooth' census for an endpoint.

    Shape: {sim_threshold, n_similar_pairs, frac_flat, frac_cliff, flat_gap,
    cliff_gap, gap_hist:[{lo,hi,count}], flat_pool:[{smiles_1,smiles_2,tanimoto,
    act_1,act_2}]}. Among all molecule pairs that are structurally similar
    (Tanimoto >= sim_threshold), what fraction are flat (|dpKi| < flat_gap) vs
    cliffs (|dpKi| > cliff_gap); ``flat_pool`` is a random sample of the flat
    majority for display.
    """
    return _data()["endpoints"][endpoint]["smoothness"]


def sample_flat_pairs(endpoint: str, n: int, seed: int) -> list[dict]:
    """Deterministically sample ``n`` flat similar pairs from the cached pool."""
    import random

    pool = smoothness(endpoint).get("flat_pool", [])
    if not pool:
        return []
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def cliff_pair(endpoint: str, index: int) -> dict | None:
    """The analysis record for one curated cliff pair, or None if not present.

    Shape: {index, cliff_on, change, mol1, mol2} where each mol is
    {smiles, true, neighbor_acts:[float], pred_by_k:[{k,pred}],
    by_fp:{fp_key:{label, neighbor_acts, pred_by_k}}}.
    """
    if not has_data() or endpoint not in _data()["endpoints"]:
        return None
    for p in _data()["endpoints"][endpoint]["cliff_pairs"]:
        if p["index"] == index:
            return p
    return None


def best_k(endpoint: str) -> dict:
    """The k that maximises held-out R^2 (the model's honest 'best setting')."""
    curve = k_curve(endpoint)
    return max(curve, key=lambda d: d["r2"])


def pred_at_k(mol_report: dict, k: int, fp: str | None = None) -> float:
    """A molecule's kNN prediction at neighbourhood size k (nearest grid point).

    If ``fp`` (a fingerprint key like 'morgan') is given, use that fingerprint's
    neighbourhood; otherwise use the default (ECFP) neighbourhood.
    """
    src = mol_report
    if fp is not None:
        src = mol_report.get("by_fp", {}).get(fp, mol_report)
    rows = src["pred_by_k"]
    exact = next((r for r in rows if r["k"] == k), None)
    if exact is not None:
        return float(exact["pred"])
    # fall back to the closest available k on the grid
    closest = min(rows, key=lambda r: abs(r["k"] - k))
    return float(closest["pred"])


def neighbor_acts_at_k(mol_report: dict, k: int, fp: str | None = None) -> list[float]:
    """The activities of the k nearest neighbours (the values kNN averages).

    If ``fp`` is given, use that fingerprint's neighbourhood. Falls back to
    whatever was cached if k exceeds the stored count.
    """
    src = mol_report
    if fp is not None:
        src = mol_report.get("by_fp", {}).get(fp, mol_report)
    acts = src.get("neighbor_acts", [])
    return [float(a) for a in acts[:k]]


def cliff_fps(mol_report: dict) -> dict:
    """{fp_key: label} for the fingerprints this molecule report was scored
    under (order preserved), or empty if only the default is present."""
    return {k: v["label"] for k, v in mol_report.get("by_fp", {}).items()}


def cliff_rmse_by_fp(endpoint: str) -> dict:
    """Per-fingerprint kNN RMSE **on the cliff molecules only**, vs k.

    For every cliff pair in the endpoint we already cache each molecule's
    predicted activity at every k (per fingerprint) and its true activity. This
    pools *all* cliff molecules (both ends of every pair) and returns, for each
    fingerprint, the RMSE of those predictions at each k:

        {fp_key: {"label": str, "rmse_curve": [{"k": k, "rmse": v}, ...]}}

    It is the cliff-focused analogue of ``k_curves_by_fp`` (which is the RMSE /
    R² over the whole held-out set): where the global curve asks "how accurate
    is kNN on average", this asks "how badly does it miss precisely the pairs
    that break the smoothness assumption".
    """
    if not has_data() or endpoint not in _data()["endpoints"]:
        return {}
    import math

    pairs = _data()["endpoints"][endpoint]["cliff_pairs"]
    if not pairs:
        return {}
    # discover fingerprints + grid from the first molecule
    first_mol = pairs[0]["mol1"]
    by_fp_labels = {k: v["label"] for k, v in first_mol.get("by_fp", {}).items()}
    if not by_fp_labels:
        return {}
    ks = [r["k"] for r in first_mol["by_fp"][next(iter(by_fp_labels))]["pred_by_k"]]

    out: dict[str, dict] = {}
    for fp_key, label in by_fp_labels.items():
        curve = []
        for k in ks:
            sq_errs = []
            for pair in pairs:
                for mol_key in ("mol1", "mol2"):
                    mol = pair[mol_key]
                    fp_src = mol.get("by_fp", {}).get(fp_key)
                    if fp_src is None:
                        continue
                    row = next(
                        (r for r in fp_src["pred_by_k"] if r["k"] == k), None
                    )
                    if row is None:
                        continue
                    sq_errs.append((row["pred"] - mol["true"]) ** 2)
            if sq_errs:
                curve.append(
                    {"k": k, "rmse": math.sqrt(sum(sq_errs) / len(sq_errs))}
                )
        out[fp_key] = {"label": label, "rmse_curve": curve}
    return out


def n_cliff_pairs(endpoint: str) -> int:
    """How many curated cliff pairs back the cliff-RMSE curve for an endpoint."""
    if not has_data() or endpoint not in _data()["endpoints"]:
        return 0
    return len(_data()["endpoints"][endpoint]["cliff_pairs"])
