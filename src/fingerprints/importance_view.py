"""Feature-importance scaffold scatter for the cliff pair: how good is a
RandomForest trained on ECFP vs CheMeleon at predicting D3/D4/mu/kappa
activity, on both its train and its held-out scaffold-split test fold.

Importances are precomputed offline (``fingerprints.analyses.importance``)
and cached; this module just reads that cache so the notebook stays instant.
"""

from __future__ import annotations

import json
from functools import lru_cache

from fingerprints import paths

CACHE = paths.IMPORTANCE


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads(CACHE.read_text())


def has_data() -> bool:
    return CACHE.exists()


def endpoints() -> list[str]:
    return list(_data()["endpoints"].keys()) if has_data() else []


def metrics(endpoint: str, fp: str) -> dict:
    d = _data()["endpoints"][endpoint][fp]
    return {"r2": d["r2"], "rmse": d["rmse"], "n_train": d["n_train"], "n_test": d["n_test"]}


def predict(endpoint: str, fp: str, mol) -> float | None:
    """Cached RF prediction (pKi) for a curated cliff molecule, looked up by
    canonical SMILES. Returns None if this molecule wasn't precomputed."""
    from rdkit import Chem

    preds = _data()["endpoints"][endpoint][fp].get("predictions", {})
    return preds.get(Chem.MolToSmiles(mol))


def fold_scatter(endpoint: str, fp: str, fold: str) -> tuple[list[float], list[float]]:
    """(measured, predicted) pKi for one endpoint + fp on the given ``fold``
    ('train' or 'test'). Returns ([], []) if not present in the cache."""
    d = _data()["endpoints"][endpoint][fp]
    return d.get(f"{fold}_measured", []), d.get(f"{fold}_pred", [])
