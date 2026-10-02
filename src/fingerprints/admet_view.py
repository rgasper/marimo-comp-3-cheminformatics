"""Read the cached ADMET single-endpoint cliff census
(``fingerprints.analyses.admet``) for the notebook's "cliffs aren't
exclusive to protein binding" section.

Activity cliffs are *easiest to see* in the two-target binding case, because a
second target gives you a built-in control ("flat on the other target"). But
the same "similar structure -> similar property" assumption - and the same
cliffs that break it - show up in single-endpoint ADMET data too, where there
is no second target. This module serves the precomputed AqSolDB solubility and
AstraZeneca lipophilicity census; it never trains.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache

from fingerprints import paths

CACHE = paths.ADMET_CLIFFS


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads(CACHE.read_text())


def has_data() -> bool:
    return CACHE.exists()


def endpoints() -> list[str]:
    return list(_data()["endpoints"].keys()) if has_data() else []


def meta(endpoint: str) -> dict:
    """{unit, blurb, n_total, n_train, n_test} for an endpoint."""
    ep = _data()["endpoints"][endpoint]
    return {
        "unit": ep["unit"],
        "blurb": ep["blurb"],
        "n_total": ep["n_total"],
        "n_train": ep["n_train"],
        "n_test": ep["n_test"],
    }


def smoothness(endpoint: str) -> dict:
    """Flat-vs-cliff census among structurally similar pairs.

    Shape: {sim_threshold, n_similar_pairs, frac_flat, frac_cliff, flat_gap,
    cliff_gap, gap_hist:[{lo,hi,count}], flat_pool:[{smiles_1,smiles_2,
    tanimoto,act_1,act_2}]}.
    """
    return _data()["endpoints"][endpoint]["smoothness"]


def cliff_gallery(endpoint: str) -> list[dict]:
    """The sharpest discovered cliffs: [{smiles_1,smiles_2,tanimoto,act_1,
    act_2,gap}], sorted by property gap descending."""
    return _data()["endpoints"][endpoint]["cliff_gallery"]


def per_fp(endpoint: str) -> dict:
    """{fp_key: {label, n_similar_pairs, frac_flat, frac_cliff, gap_hist,
    top_cliff}} - the census run under each fingerprint's similarity, plus that
    fingerprint's own sharpest cliff."""
    return _data()["endpoints"][endpoint]["smoothness"].get("per_fp", {})


def k_curve(endpoint: str) -> list[dict]:
    """[{k, r2}, ...] held-out (scaffold-split) R^2 vs neighbourhood size."""
    return _data()["endpoints"][endpoint]["k_curve"]


def best_k(endpoint: str) -> dict:
    return max(k_curve(endpoint), key=lambda d: d["r2"])


def k_grid() -> list[int]:
    return list(_data()["k_grid"]) if has_data() else []


def sample_flat_pairs(endpoint: str, n: int, seed: int) -> list[dict]:
    """Deterministically sample n flat similar pairs from the cached pool."""
    pool = smoothness(endpoint).get("flat_pool", [])
    if not pool:
        return []
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))
