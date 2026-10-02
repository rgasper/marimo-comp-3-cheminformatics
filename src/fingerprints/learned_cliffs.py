"""Does the *learned* fingerprint capture activity cliffs any better?

The accumulation spotlight showed CheMeleon nearly matches count fingerprints on
smooth, accumulating properties. That begs the question the rest of the notebook
implies but never tests head-on: on the **binding activity cliffs** themselves
(the D3/D4, mu/kappa pairs), does a model built on the learned CheMeleon
fingerprint do any better than one built on a classical fingerprint?

We answer it directly. For each MoleculeACE binding endpoint we train the same
fair linear head (RidgeCV, standardised) on two representations - ECFP (fixed)
and CheMeleon (learned) - using MoleculeACE's own train/test split, then compare
their held-out RMSE **on the cliff molecules** (``cliff_mol == 1``) against the
non-cliff molecules. MoleculeACE labels a molecule a cliff member when it is very
similar to another molecule (Tanimoto >= 0.9) yet differs in activity by >= 10x,
so "cliff RMSE" is exactly error on the pairs that break smoothness.

Runs live; CheMeleon features are batched. The takeaway (spoiler, but it's the
honest one): the learned fingerprint does **not** rescue the cliffs - both
representations are worse on cliff molecules than on ordinary ones, because the
missing information isn't in *any* 2D structure encoding, learned or fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as fpg

from fingerprints import paths

RDLogger.DisableLog("rdApp.*")

N_BITS = 2048
_GEN = fpg.GetMorganGenerator(radius=2, fpSize=N_BITS)
_ALPHAS = np.array([1.0, 10.0, 100.0, 1000.0])

# The decision heads we bolt onto each *frozen* fingerprint. The question is
# whether a fancier head can recover the cliff that a linear read-out misses:
# a similarity/clustering read-out (kNN), or a nonlinear net (MLP). Spoiler -
# none can, because the head only ever sees the (collapsed) representation.
_HEADS = ("linear", "kNN", "MLP")

# MoleculeACE binding endpoints (label -> dataset id). Two pairs of related
# targets, matching the curated cliffs shown earlier.
_ENDPOINTS = {
    "Dopamine D3": "CHEMBL234_Ki",
    "Dopamine D4": "CHEMBL219_Ki",
    "mu-opioid": "CHEMBL233_Ki",
    "kappa-opioid": "CHEMBL237_Ki",
}


def endpoints() -> list[str]:
    return list(_ENDPOINTS)


@lru_cache(maxsize=8)
def _load(name: str):
    """Return (mols, y, cliff_flag, split) for a MoleculeACE dataset, deduped."""
    import polars as pl

    from fingerprints.data import molace

    path = paths.CACHE_DIR / "molace" / f"{name}.csv"
    if not path.exists():
        ds = molace.MolACEDataset(
            name=name, target_label=name, target_class="", assay_type="Ki"
        )
        molace.download_molace(ds, path)
    df = pl.read_csv(path)
    smis, ys, cliff, split, seen = [], [], [], [], set()
    for r in df.iter_rows(named=True):
        m = Chem.MolFromSmiles(r["smiles"])
        y = r["y [pEC50/pKi]"]
        if m is None or y is None:
            continue
        cs = Chem.MolToSmiles(m)
        if cs in seen:
            continue
        seen.add(cs)
        smis.append(cs)
        ys.append(float(y))
        cliff.append(int(r.get("cliff_mol", 0) or 0))
        split.append(r["split"])
    mols = [Chem.MolFromSmiles(s) for s in smis]
    return (
        mols,
        np.asarray(ys, dtype=float),
        np.asarray(cliff, dtype=int),
        np.asarray(split),
    )


def _rmse(yt, yp) -> float:
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if len(yt) else float("nan")


def _fit_predict(x, y, tr, head: str):
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if head == "linear":
        est = RidgeCV(alphas=_ALPHAS)
    elif head == "kNN":
        est = KNeighborsRegressor(n_neighbors=5)
    elif head == "MLP":
        est = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            alpha=1e-3,
            max_iter=600,
            random_state=0,
            early_stopping=True,
        )
    else:
        raise ValueError(head)
    model = make_pipeline(StandardScaler(with_mean=False), est).fit(x[tr], y[tr])
    return model.predict(x)


@dataclass(frozen=True)
class CliffFit:
    endpoint: str
    fingerprint: str  # "ECFP (fixed)" | "CheMeleon (learned)"
    head: str  # "linear" | "kNN" | "MLP"
    cliff_rmse: float
    noncliff_rmse: float
    n_cliff: int
    n_noncliff: int


def heads() -> list[str]:
    return list(_HEADS)


def fingerprints() -> list[str]:
    return ["ECFP (fixed)", "CheMeleon (learned)"]


@lru_cache(maxsize=1)
def results() -> tuple[CliffFit, ...]:
    """Cliff vs non-cliff held-out RMSE for {ECFP, CheMeleon} x {linear, kNN,
    MLP}, every endpoint. Reads the precomputed cache under ``data/`` when it
    exists (instant); otherwise trains live (~50 s) and is memoised for the
    session. Regenerate the cache with ``python -m fingerprints.learned_cliffs``
    (or via the from-scratch recompute path)."""
    if has_data():
        return _load_cache()
    return _compute()


def has_data() -> bool:
    return paths.LEARNED_CLIFFS.exists()


def _load_cache() -> tuple[CliffFit, ...]:
    import json

    rows = json.loads(paths.LEARNED_CLIFFS.read_text())
    return tuple(CliffFit(**r) for r in rows)


def _compute() -> tuple[CliffFit, ...]:
    """Train every {fingerprint, head} combo on every endpoint (the ~50 s live
    path). Kept separate from :func:`results` so the cache writer can call it
    directly."""
    from fingerprints import chemeleon_fp as chf

    out: list[CliffFit] = []
    for label, name in _ENDPOINTS.items():
        mols, y, cliff, split = _load(name)
        xe = np.vstack(
            [np.asarray(_GEN.GetFingerprintAsNumPy(m), dtype=np.float32) for m in mols]
        )
        xc = chf.fingerprint_matrix(mols)
        tr = split == "train"
        te = split == "test"
        te_c = te & (cliff == 1)
        te_n = te & (cliff == 0)
        for fp_label, x in [("ECFP (fixed)", xe), ("CheMeleon (learned)", xc)]:
            for head in _HEADS:
                pred = _fit_predict(x, y, tr, head)
                out.append(
                    CliffFit(
                        endpoint=label,
                        fingerprint=fp_label,
                        head=head,
                        cliff_rmse=_rmse(y[te_c], pred[te_c]),
                        noncliff_rmse=_rmse(y[te_n], pred[te_n]),
                        n_cliff=int(te_c.sum()),
                        n_noncliff=int(te_n.sum()),
                    )
                )
    return tuple(out)


def main() -> None:
    """Train everything and write the JSON cache read by :func:`results`."""
    import json
    from dataclasses import asdict

    rows = [asdict(r) for r in _compute()]
    paths.LEARNED_CLIFFS.parent.mkdir(parents=True, exist_ok=True)
    paths.LEARNED_CLIFFS.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} rows -> {paths.LEARNED_CLIFFS}")


if __name__ == "__main__":
    main()


def mean_cliff_rmse(fingerprint: str, head: str) -> float:
    """Cliff RMSE averaged over endpoints for one (fingerprint, head) combo."""
    vals = [
        r.cliff_rmse
        for r in results()
        if r.fingerprint == fingerprint and r.head == head
    ]
    return float(np.mean(vals)) if vals else float("nan")
