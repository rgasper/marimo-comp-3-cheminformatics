"""Can a model *accumulate* a whole-molecule quantity from a fingerprint?

The activity-cliff census argues that solubility-type cliffs arise because a
fingerprint can't track a property that **accumulates over the whole molecule**
(add one more -CH2- and solubility drops, but a binary fingerprint barely
moves). This module turns that claim into a controlled experiment across
several fingerprint encodings:

  * **binary Morgan** — the usual fingerprint. Binarising throws away *how
    many times* each substructure occurs, i.e. exactly the count information
    an accumulator needs.
  * **count Morgan** — same bits, but each slot holds a count, so a linear
    head can literally sum them.
  * **CheMeleon (learned)** — a pretrained message-passing fingerprint. It is
    *mean*-pooled over atoms, which is telling: a mean is size-*intensive*, so
    even a learned representation is not automatically good at a
    size-extensive raw count.

Two targets: **heavy-atom count** (a *pure* accumulator we control exactly —
by definition a sum over atoms) and **real AqSolDB solubility**.

Every number in the notebook comes from the same **5x5 repeated scaffold
cross-validation** (:func:`cv_results`): the molecules are dealt into 5
Bemis-Murcko-scaffold folds, each fold takes a turn as the held-out test set,
and the whole thing repeats 5 times with a different shuffle, for 25 scores
per model on identical splits. :func:`tukey` runs Tukey's HSD test over those
25 scores to say which models are *statistically* best vs merely best-looking.

A second analysis (:func:`cliff_summary`) reuses those same CV folds but keeps
every molecule's *out-of-fold* prediction, so it can ask the sharper question:
of the actual activity cliffs in AqSolDB (structurally similar pairs with a
big solubility gap), how much of each cliff's gap does each model's prediction
actually reproduce?

Rebuild everything with ``python -m fingerprints.recompute`` (or, to touch
only this module, ``python -m fingerprints.accumulation``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold

from fingerprints import paths
from fingerprints.data import tdc

RDLogger.DisableLog("rdApp.*")

N_BITS = 2048
_GEN = fpg.GetMorganGenerator(radius=2, fpSize=N_BITS)
_SALT = SaltRemover()
_MAX_N = 4000  # cap for a snappy fit; sampled deterministically
_ALPHAS = np.logspace(-2, 4, 13)  # RidgeCV search grid

CV_REPEATS = 5
CV_FOLDS = 5

# (key, label, family) for each encoding/head compared throughout this module.
# family drives the notebook's chart colour: red = binary, green = count,
# purple = learned.
CV_METHODS: tuple[tuple[str, str, str], ...] = (
    ("binary_linear", "binary Morgan + linear", "binary"),
    ("count_linear", "count Morgan + linear", "count"),
    ("chemeleon_linear", "CheMeleon (learned) + linear", "learned"),
)

# Activity-cliff definition reused from the ADMET census: binary-Morgan
# Tanimoto >= CLIFF_SIM is "structurally similar"; among similar pairs, a
# solubility gap above CLIFF_GAP is a cliff and below FLAT_GAP is flat.
CLIFF_SIM = 0.7
CLIFF_GAP = 1.5
FLAT_GAP = 0.5
CLIFF_TARGET = "aqueous solubility"


# ---------------------------------------------------------------------------
# Dataset: AqSolDB, desalted + deduplicated by canonical parent SMILES.
# ---------------------------------------------------------------------------


def _parent(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _SALT.StripMol(mol, dontRemoveEverything=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=1)
def _dataset():
    """Load, desalt, dedup AqSolDB → (mols, smiles, logS). Cached per session."""
    df = tdc.load_tdc(tdc.SOLUBILITY, paths.CACHE_DIR / "tdc")
    by_parent: dict[str, list[float]] = {}
    for row in df.iter_rows(named=True):
        p = _parent(row["smiles"])
        if p is None or row["y"] is None:
            continue
        by_parent.setdefault(p, []).append(float(row["y"]))
    smis = list(by_parent)
    ys = np.array([float(np.mean(by_parent[s])) for s in smis])
    if len(smis) > _MAX_N:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(smis), _MAX_N, replace=False)
        smis = [smis[i] for i in idx]
        ys = ys[idx]
    mols = [Chem.MolFromSmiles(s) for s in smis]
    return mols, smis, ys


@lru_cache(maxsize=1)
def _morgan_matrices():
    """(binary fp, count fp) Morgan matrices for the cached dataset."""
    mols, _, _ = _dataset()
    xb = np.vstack(
        [np.asarray(_GEN.GetFingerprintAsNumPy(m), dtype=np.float32) for m in mols]
    )
    xc = np.vstack(
        [np.asarray(_GEN.GetCountFingerprintAsNumPy(m), dtype=np.float32) for m in mols]
    )
    return xb, xc


@lru_cache(maxsize=1)
def _chemeleon_matrix():
    """Batched mean-pooled CheMeleon fingerprints for the cached dataset."""
    from fingerprints import chemeleon_fp as chf

    mols, _, _ = _dataset()
    return chf.fingerprint_matrix(mols)


def _targets() -> dict[str, np.ndarray]:
    """The two regression targets: a pure accumulator + real solubility."""
    mols, _, ys = _dataset()
    hac = np.array([m.GetNumHeavyAtoms() for m in mols], dtype=float)
    return {"heavy-atom count": hac, "aqueous solubility": ys}


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def target_labels() -> list[str]:
    return list(_targets().keys())


def n_molecules() -> int:
    if has_data():
        return _cache()["n_molecules"]
    return len(_dataset()[0])


# ---------------------------------------------------------------------------
# Scaffold cross-validation: fold assignment + per-split model fits.
# ---------------------------------------------------------------------------


def _scaffold_groups(mols) -> np.ndarray:
    """Integer scaffold-group id per molecule (acyclics are singletons)."""
    ids: dict[str, int] = {}
    out = np.empty(len(mols), dtype=int)
    for i, m in enumerate(mols):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            sc = ""
        key = sc or f"__none_{i}"
        out[i] = ids.setdefault(key, len(ids))
    return out


def cv_fold_ids(groups: np.ndarray, repeat: int, n_folds: int = CV_FOLDS) -> np.ndarray:
    """Fold id (0..n_folds-1) per molecule for one CV repeat.

    Whole scaffold groups are dealt to folds: groups are shuffled with a
    repeat-specific seed, then each group (largest first) goes to the currently
    smallest fold, so folds stay balanced and no scaffold spans train and test.
    """
    rng = np.random.default_rng(1000 + repeat)
    uniq, sizes = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    # stable sort by size (desc) after the shuffle -> random tie-breaking
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_of_group = np.empty(len(uniq), dtype=int)
    fill = np.zeros(n_folds, dtype=int)
    for gi in order:
        f = int(np.argmin(fill))
        fold_of_group[gi] = f
        fill[f] += sizes[gi]
    lookup = dict(zip(uniq.tolist(), fold_of_group.tolist()))
    return np.array([lookup[g] for g in groups.tolist()], dtype=int)


def _predict_split(xb, xc, xche, y, tr, te) -> dict[str, np.ndarray]:
    """Held-out predictions on ``te`` for every :data:`CV_METHODS` model."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    m = RidgeCV(alphas=_ALPHAS).fit(xb[tr], y[tr])
    out["binary_linear"] = m.predict(xb[te])
    m = RidgeCV(alphas=_ALPHAS).fit(xc[tr], y[tr])
    out["count_linear"] = m.predict(xc[te])
    if xche is not None:
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=_ALPHAS)).fit(
            xche[tr], y[tr]
        )
        out["chemeleon_linear"] = m.predict(xche[te])
    return out


def _fit_split(xb, xc, xche, y, tr, te) -> dict[str, tuple[float, float]]:
    """(R², RMSE) for every CV method on one train/test split."""
    yt = y[te]
    return {
        k: (_r2(yt, p), float(np.sqrt(np.mean((yt - p) ** 2))))
        for k, p in _predict_split(xb, xc, xche, y, tr, te).items()
    }


def _cv_live(n_jobs: int = 6) -> dict:
    """Run the 5×5 scaffold CV for every target; returns the JSON payload."""
    from joblib import Parallel, delayed

    mols, _, _ = _dataset()
    xb, xc = _morgan_matrices()
    try:
        xche = _chemeleon_matrix()
    except Exception:
        xche = None
    groups = _scaffold_groups(mols)
    folds = [cv_fold_ids(groups, r) for r in range(CV_REPEATS)]
    targets = _targets()

    jobs = [
        (t, r, f)
        for t in targets
        for r in range(CV_REPEATS)
        for f in range(CV_FOLDS)
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_split)(
            xb, xc, xche, targets[t], folds[r] != f, folds[r] == f
        )
        for t, r, f in jobs
    )
    payload: dict = {
        "n_repeats": CV_REPEATS,
        "n_folds": CV_FOLDS,
        "methods": [
            {"key": k, "label": lbl, "family": fam} for k, lbl, fam in CV_METHODS
        ],
        "targets": {t: {"r2": {}, "rmse": {}} for t in targets},
    }
    for (t, _r, _f), res in zip(jobs, results):
        for key, (r2, rmse) in res.items():
            payload["targets"][t]["r2"].setdefault(key, []).append(r2)
            payload["targets"][t]["rmse"].setdefault(key, []).append(rmse)
    return payload


def cv_results() -> dict:
    """The 5×5 CV payload (shipped cache when present, else fit live)."""
    if has_data() and "cv" in _cache():
        return _cache()["cv"]
    return _cv_live_cached()


@lru_cache(maxsize=1)
def _cv_live_cached() -> dict:
    return _cv_live()


def has_cv() -> bool:
    return has_data() and "cv" in _cache()


# ---------------------------------------------------------------------------
# Tukey-HSD statistics over the 5x5 CV scores.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TukeyRow:
    key: str
    label: str
    mean: float
    lo: float  # simultaneous-CI lower bound
    hi: float  # simultaneous-CI upper bound
    status: str  # "best" | "equivalent" | "different"
    p_vs_best: float


@dataclass(frozen=True)
class TukeyResult:
    target: str
    metric: str  # "r2" | "rmse"
    rows: tuple[TukeyRow, ...]
    anova_p: float
    n_splits: int


def tukey(target: str, metric: str = "r2", alpha: float = 0.05) -> TukeyResult:
    """Tukey-HSD comparison of every CV method on ``target`` / ``metric``.

    Mirrors statsmodels' ``plot_simultaneous`` view: each method gets a
    simultaneous confidence interval ``mean ± (q_crit/2)·sqrt(MSE/n)`` — two
    intervals fail to overlap exactly when Tukey's test calls the pair
    different. The best method (highest R² / lowest RMSE) is flagged, and every
    other method is labelled equivalent-to or different-from it by its Tukey
    p-value. A one-way ANOVA p-value across all methods is reported as well.
    """
    from scipy.stats import f_oneway, studentized_range, tukey_hsd

    cv = cv_results()
    meta = {m["key"]: m for m in cv["methods"]}
    vals = cv["targets"][target][metric]
    keys = [k for k in meta if k in vals]
    samples = [np.asarray(vals[k], dtype=float) for k in keys]
    k = len(samples)
    n = len(samples[0])
    means = np.array([s.mean() for s in samples])
    dof = k * n - k
    mse = float(sum(((s - s.mean()) ** 2).sum() for s in samples) / dof)
    q_crit = float(studentized_range.ppf(1 - alpha, k, dof))
    hw = 0.5 * q_crit * np.sqrt(mse / n)
    best = int(np.argmax(means) if metric == "r2" else np.argmin(means))
    pvals = tukey_hsd(*samples).pvalue
    rows = []
    for i, key in enumerate(keys):
        p = float(pvals[best, i]) if i != best else 1.0
        status = (
            "best" if i == best else ("different" if p <= alpha else "equivalent")
        )
        rows.append(
            TukeyRow(
                key, meta[key]["label"], float(means[i]),
                float(means[i] - hw), float(means[i] + hw), status, p,
            )
        )
    anova_p = float(f_oneway(*samples).pvalue)
    return TukeyResult(target, metric, tuple(rows), anova_p, n)


def paired_p(target: str, key_a: str, key_b: str, metric: str = "r2") -> float:
    """Paired t-test p-value between two methods across the shared CV splits."""
    from scipy.stats import ttest_rel

    vals = cv_results()["targets"][target][metric]
    return float(ttest_rel(vals[key_a], vals[key_b]).pvalue)


# ---------------------------------------------------------------------------
# Cliffs inside AqSolDB: do count Morgan / CheMeleon actually predict the
# activity cliffs better, or just the smooth bulk of the data?
#
# Same 5×5 scaffold CV as above, but we keep the *out-of-fold* prediction for
# every molecule in every repeat (each molecule is predicted exactly once per
# repeat, by a model that never saw its scaffold). Cliff pairs are defined
# exactly as in the ADMET census: binary-Morgan Tanimoto >= CLIFF_SIM and
# |ΔlogS| > CLIFF_GAP. Using one fixed pair definition means every model is
# judged on the very same cliffs.
# ---------------------------------------------------------------------------


def _tanimoto_matrix(xb: np.ndarray) -> np.ndarray:
    b = (xb > 0).astype(np.float32)
    inter = b @ b.T
    s = b.sum(1)
    union = s[:, None] + s[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(union > 0, inter / union, 0.0)
    return t


def _similar_pairs():
    """(i, j, tanimoto, |Δy|) for every structurally similar pair."""
    xb, _ = _morgan_matrices()
    y = _targets()[CLIFF_TARGET]
    t = _tanimoto_matrix(xb)
    i, j = np.triu_indices(len(y), 1)
    keep = t[i, j] >= CLIFF_SIM
    i, j = i[keep], j[keep]
    return i, j, t[i, j], np.abs(y[i] - y[j])


def _cliff_oof_live(n_jobs: int = 6) -> dict:
    """Out-of-fold predictions on solubility for every repeat × method."""
    from joblib import Parallel, delayed

    mols, _, _ = _dataset()
    xb, xc = _morgan_matrices()
    try:
        xche = _chemeleon_matrix()
    except Exception:
        xche = None
    y = _targets()[CLIFF_TARGET]
    groups = _scaffold_groups(mols)
    folds = [cv_fold_ids(groups, r) for r in range(CV_REPEATS)]
    jobs = [(r, f) for r in range(CV_REPEATS) for f in range(CV_FOLDS)]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_predict_split)(xb, xc, xche, y, folds[r] != f, folds[r] == f)
        for r, f in jobs
    )
    oof: dict[str, list[list[float]]] = {}
    for (r, f), preds in zip(jobs, res):
        te = folds[r] == f
        for key, p in preds.items():
            rows = oof.setdefault(key, [[0.0] * len(y) for _ in range(CV_REPEATS)])
            for idx, v in zip(np.flatnonzero(te).tolist(), p.tolist()):
                rows[r][idx] = round(float(v), 4)
    i, j, t, d = _similar_pairs()
    return {
        "y": [round(float(v), 4) for v in y],
        "oof": oof,
        "pairs": {
            "i": i.tolist(), "j": j.tolist(),
            "tanimoto": [round(float(v), 3) for v in t],
            "gap": [round(float(v), 3) for v in d],
        },
        "sim": CLIFF_SIM, "cliff_gap": CLIFF_GAP, "flat_gap": FLAT_GAP,
    }


@lru_cache(maxsize=1)
def _cliff_oof() -> dict:
    if has_data() and "cliff_oof" in _cache():
        return _cache()["cliff_oof"]
    return _cliff_oof_live()


def has_cliff_data() -> bool:
    return has_data() and "cliff_oof" in _cache()


def cliff_summary() -> dict:
    """Per-repeat cliff metrics for every method (5 values each).

    For each repeat's out-of-fold predictions:

    * ``gap_recovered`` — over cliff pairs, the predicted difference projected
      on the true direction, as a fraction of the true gap (1 = cliff fully
      reproduced, 0 = both molecules predicted the same);
    * ``rmse_cliff`` / ``rmse_rest`` — RMSE on molecules that sit in at least
      one cliff pair vs every other molecule;
    * ``pair_err_cliff`` / ``pair_err_flat`` — mean |Δpred − Δtrue| over cliff
      pairs vs flat (|Δy| < FLAT_GAP) similar pairs.
    """
    d = _cliff_oof()
    y = np.asarray(d["y"])
    pi, pj = np.asarray(d["pairs"]["i"]), np.asarray(d["pairs"]["j"])
    gap = np.asarray(d["pairs"]["gap"])
    cliff = gap > d["cliff_gap"]
    flat = gap < d["flat_gap"]
    ci, cj = pi[cliff], pj[cliff]
    true_diff = y[ci] - y[cj]
    in_cliff = np.zeros(len(y), dtype=bool)
    in_cliff[ci] = True
    in_cliff[cj] = True
    out: dict = {
        "n_cliff_pairs": int(cliff.sum()),
        "n_flat_pairs": int(flat.sum()),
        "n_similar_pairs": len(gap),
        "n_cliff_mols": int(in_cliff.sum()),
        "n_mols": len(y),
        "methods": {},
    }
    for key, reps in d["oof"].items():
        m = {k: [] for k in ("gap_recovered", "rmse_cliff", "rmse_rest",
                             "pair_err_cliff", "pair_err_flat")}
        for p in reps:
            p = np.asarray(p)
            pdiff = p[ci] - p[cj]
            m["gap_recovered"].append(float(np.mean(pdiff * np.sign(true_diff)
                                                    / np.abs(true_diff))))
            err = p - y
            m["rmse_cliff"].append(float(np.sqrt(np.mean(err[in_cliff] ** 2))))
            m["rmse_rest"].append(float(np.sqrt(np.mean(err[~in_cliff] ** 2))))
            m["pair_err_cliff"].append(float(np.mean(np.abs(pdiff - true_diff))))
            fd = y[pi[flat]] - y[pj[flat]]
            fp = p[pi[flat]] - p[pj[flat]]
            m["pair_err_flat"].append(float(np.mean(np.abs(fp - fd))))
        out["methods"][key] = m
    return out


def cliff_pairs_table() -> list[dict]:
    """One row per cliff pair × method: true vs predicted (repeat-averaged) gap."""
    d = _cliff_oof()
    y = np.asarray(d["y"])
    pi, pj = np.asarray(d["pairs"]["i"]), np.asarray(d["pairs"]["j"])
    gap = np.asarray(d["pairs"]["gap"])
    tan = np.asarray(d["pairs"]["tanimoto"])
    cliff = gap > d["cliff_gap"]
    ci, cj, ct = pi[cliff], pj[cliff], tan[cliff]
    # orient every pair so the true difference is positive
    sign = np.sign(y[ci] - y[cj])
    true_gap = (y[ci] - y[cj]) * sign
    rows = []
    for key, reps in d["oof"].items():
        p = np.mean(np.asarray(reps), axis=0)
        pred_gap = (p[ci] - p[cj]) * sign
        for n in range(len(ci)):
            rows.append({"key": key, "true_gap": float(true_gap[n]),
                         "pred_gap": float(pred_gap[n]), "tanimoto": float(ct[n])})
    return rows


def cliff_paired_p(metric: str, key_a: str, key_b: str) -> float:
    """Paired t-test across the CV repeats on a :func:`cliff_summary` metric."""
    from scipy.stats import ttest_rel

    m = cliff_summary()["methods"]
    return float(ttest_rel(m[key_a][metric], m[key_b][metric]).pvalue)


# ---------------------------------------------------------------------------
# Disk cache: both analyses above are precomputed and shipped under data/ so
# the notebook loads instantly. Rebuild with ``python -m fingerprints.recompute``
# (or just this module: ``python -m fingerprints.accumulation``).
# ---------------------------------------------------------------------------


def has_data() -> bool:
    return paths.ACCUMULATION.exists()


@lru_cache(maxsize=1)
def _cache() -> dict:
    return json.loads(paths.ACCUMULATION.read_text())


def main() -> None:
    """Fit everything live and write the JSON cache read by :func:`cv_results`,
    :func:`cliff_summary`, and :func:`n_molecules`."""
    payload = {
        "n_molecules": len(_dataset()[0]),
        "cv": _cv_live(),
        "cliff_oof": _cliff_oof_live(),
    }
    paths.ACCUMULATION.parent.mkdir(parents=True, exist_ok=True)
    paths.ACCUMULATION.write_text(json.dumps(payload))
    _cache.cache_clear()
    print(f"wrote {paths.ACCUMULATION}")


if __name__ == "__main__":
    main()
