"""Can a model *accumulate* a whole-molecule quantity from a fingerprint?

The activity-cliff census argues that solubility-type cliffs arise because a
fingerprint can't track a property that **accumulates over the whole molecule**
(add one more -CH2- and solubility drops, but a binary fingerprint barely
moves). This module turns that claim into a controlled, runs-live experiment
across several fingerprint encodings.

Encodings + heads compared (all with a fairly-tuned linear head via RidgeCV,
plus one deep MLP on the binary fingerprint to ask whether depth can rescue it):

  * **binary Morgan** — the usual fingerprint. Binarising throws away *how many
    times* each substructure occurs, i.e. exactly the count information an
    accumulator needs.
  * **count Morgan** — same bits, but each slot holds a count, so a linear head
    can literally sum them.
  * **binary Morgan + deep MLP** — can depth reconstruct the lost counts from
    which bits co-occur?
  * **CheMeleon (learned)** — a pretrained message-passing fingerprint. It is
    *mean*-pooled over atoms, which is telling: a mean is size-*intensive*, so
    even a learned representation is not automatically good at a size-extensive
    raw count.

Target 1 is a **pure accumulator** we control exactly (heavy-atom count — by
definition a sum over atoms). Target 2 is **real AqSolDB solubility**. The pure
target isolates the mechanism; the real one shows how much carries over.

Everything trains live on a Bemis-Murcko scaffold split (no leakage). CheMeleon
features are batched so the whole thing stays to a few seconds.
"""

from __future__ import annotations

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
_MAX_N = 4000  # cap for a snappy live fit; sampled deterministically
_ALPHAS = np.logspace(-2, 4, 13)  # RidgeCV search grid (headline section)
_SURVEY_MAX_N = 2500  # smaller cap for the multi-fingerprint survey (speed)
_SURVEY_ALPHAS = np.array([1.0, 10.0, 100.0, 1000.0])  # coarse but fair grid

# The classical RDKit fingerprints the survey compares, each in binary and
# count form. All fold to N_BITS bits so the only thing that changes between
# "binary" and "count" is whether a slot records presence or multiplicity.
_SURVEY_GENERATORS = {
    "Morgan (ECFP4)": fpg.GetMorganGenerator(radius=2, fpSize=N_BITS),
    "RDKit topological": fpg.GetRDKitFPGenerator(fpSize=N_BITS),
    "atom-pair": fpg.GetAtomPairGenerator(fpSize=N_BITS),
    "topological torsion": fpg.GetTopologicalTorsionGenerator(fpSize=N_BITS),
}

# One binding endpoint (MoleculeACE / ChEMBL) to contrast against solubility:
# does counting help or hurt when the property is molecular *recognition*
# rather than an accumulated bulk quantity?
_BINDING_ENDPOINT = ("Dopamine D3 (pKi)", "CHEMBL234_Ki")


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


@lru_cache(maxsize=1)
def _scaffold_split():
    """Boolean (train, test) masks by Bemis-Murcko scaffold (no leakage)."""
    mols, _smis, _ = _dataset()
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(mols):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            sc = ""
        groups.setdefault(sc or f"__none_{i}", []).append(i)
    n = len(mols)
    is_test = np.zeros(n, dtype=bool)
    target = int(round(0.2 * n))
    for grp in sorted(groups.values(), key=len):
        if is_test.sum() >= target:
            break
        for i in grp:
            is_test[i] = True
    return ~is_test, is_test


def _targets():
    """The two regression targets: a pure accumulator + real solubility."""
    mols, _, ys = _dataset()
    hac = np.array([m.GetNumHeavyAtoms() for m in mols], dtype=float)
    return {
        "heavy-atom count": {
            "y": hac,
            "kind": "pure accumulator (Σ over atoms)",
            "unit": "atoms",
        },
        "aqueous solubility": {
            "y": ys,
            "kind": "real measured property",
            "unit": "logS",
        },
    }


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


@dataclass(frozen=True)
class ModelScore:
    key: str
    label: str
    family: str  # "binary" | "count" | "learned" — drives the chart colour
    r2: float


def target_labels() -> list[str]:
    return list(_targets().keys())


def n_molecules() -> int:
    if has_data():
        return _cache()["n_molecules"]
    return len(_dataset()[0])


def target_meta(target: str) -> dict:
    return {k: v for k, v in _targets()[target].items() if k != "y"}


def _ridge_r2(x, y, tr, te, *, scale: bool) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [StandardScaler()] if scale else []
    steps.append(RidgeCV(alphas=_ALPHAS))
    model = make_pipeline(*steps).fit(x[tr], y[tr])
    return _r2(y[te], model.predict(x[te]))


def _mlp_r2(x, y, tr, te, *, depth: int, width: int) -> float:
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor(
        hidden_layer_sizes=(width,) * depth,
        max_iter=300,
        random_state=0,
        early_stopping=True,
    ).fit(x[tr], y[tr])
    return _r2(y[te], model.predict(x[te]))


@lru_cache(maxsize=8)
def scores(target: str, depth: int = 4, width: int = 48) -> tuple[ModelScore, ...]:
    """Held-out R² for every encoding/head on ``target``.

    Reads the shipped cache under ``data/`` when present (instant); otherwise
    fits live. Regenerate with ``python -m fingerprints.accumulation``.
    """
    if has_data():
        return tuple(
            ModelScore(**d) for d in _cache()["scores"].get(target, [])
        )
    return _scores_live(target, depth=depth, width=width)


def _scores_live(
    target: str, depth: int = 4, width: int = 48
) -> tuple[ModelScore, ...]:
    """Fit every encoding/head on ``target`` and return held-out R² for each.

    Linear heads use RidgeCV (fair, auto-tuned regularisation); CheMeleon's
    embedding is standardised first because a few of its dimensions have very
    large scale. The binary-Morgan MLP is the "can depth rescue it?" probe.
    """
    xb, xc = _morgan_matrices()
    tr, te = _scaffold_split()
    y = _targets()[target]["y"]

    out = [
        ModelScore(
            "binary_linear", "binary Morgan + linear", "binary",
            _ridge_r2(xb, y, tr, te, scale=False),
        ),
        ModelScore(
            "count_linear", "count Morgan + linear", "count",
            _ridge_r2(xc, y, tr, te, scale=False),
        ),
        ModelScore(
            "binary_mlp", f"binary Morgan + deep MLP ({width}×{depth})", "binary",
            _mlp_r2(xb, y, tr, te, depth=depth, width=width),
        ),
    ]
    try:
        xche = _chemeleon_matrix()
        out.append(
            ModelScore(
                "chemeleon_linear", "CheMeleon (learned) + linear", "learned",
                _ridge_r2(xche, y, tr, te, scale=True),
            )
        )
    except Exception:
        # CheMeleon weights unavailable — skip it rather than fail the cell.
        pass
    return tuple(out)


# ---------------------------------------------------------------------------
# Follow-up survey: binary vs count across every classical RDKit fingerprint,
# on a pure accumulator, on solubility, and on a binding endpoint.
# ---------------------------------------------------------------------------


def _scaffold_masks(mols, test_frac: float = 0.2):
    """Generic Bemis-Murcko scaffold split for an arbitrary molecule list."""
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(mols):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            sc = ""
        groups.setdefault(sc or f"__none_{i}", []).append(i)
    n = len(mols)
    is_test = np.zeros(n, dtype=bool)
    target = int(round(test_frac * n))
    for grp in sorted(groups.values(), key=len):
        if is_test.sum() >= target:
            break
        for i in grp:
            is_test[i] = True
    return ~is_test, is_test


@lru_cache(maxsize=1)
def _binding_dataset():
    """Load a MoleculeACE binding endpoint into (mols, pKi). Cached per session.

    Downloads the CSV via the MoleculeACE loader if not already present, so the
    survey is self-contained on a cold clone.
    """
    import polars as pl

    from fingerprints.data import molace

    _label, name = _BINDING_ENDPOINT
    path = paths.CACHE_DIR / "molace" / f"{name}.csv"
    if not path.exists():
        ds = molace.MolACEDataset(
            name=name, target_label=name, target_class="", assay_type="Ki"
        )
        molace.download_molace(ds, path)
    df = pl.read_csv(path)
    smis, ys, seen = [], [], set()
    for row in df.iter_rows(named=True):
        mol = Chem.MolFromSmiles(row["smiles"])
        y = row["y [pEC50/pKi]"]
        if mol is None or y is None:
            continue
        cs = Chem.MolToSmiles(mol)
        if cs in seen:
            continue
        seen.add(cs)
        smis.append(cs)
        ys.append(float(y))
    mols = [Chem.MolFromSmiles(s) for s in smis]
    if len(mols) > _SURVEY_MAX_N:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(mols), _SURVEY_MAX_N, replace=False)
        mols = [mols[i] for i in idx]
        ys = [ys[i] for i in idx]
    return mols, np.asarray(ys, dtype=float)


def _fp_matrix(mols, gen, *, count: bool) -> np.ndarray:
    fn = gen.GetCountFingerprintAsNumPy if count else gen.GetFingerprintAsNumPy
    return np.vstack([np.asarray(fn(m), dtype=np.float32) for m in mols])


def _survey_ridge_r2(x, y, tr, te) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(with_mean=False), RidgeCV(alphas=_SURVEY_ALPHAS)
    ).fit(x[tr], y[tr])
    return _r2(y[te], model.predict(x[te]))


@dataclass(frozen=True)
class SurveyCell:
    fingerprint: str
    encoding: str  # "binary" or "count"
    target: str
    r2: float


def survey_targets() -> list[str]:
    """Column labels for the survey, in display order."""
    return ["heavy-atom count", "aqueous solubility", _BINDING_ENDPOINT[0]]


def survey_fingerprints() -> list[str]:
    return list(_SURVEY_GENERATORS)


def survey_n() -> dict[str, int]:
    """Molecule counts backing each target family."""
    if has_data():
        return _cache()["survey_n"]
    return {
        "accumulator": len(_dataset()[0]),
        "binding": len(_binding_dataset()[0]),
    }


@lru_cache(maxsize=1)
def survey() -> tuple[SurveyCell, ...]:
    """binary vs count for every classical fingerprint on all survey targets.

    Reads the shipped cache under ``data/`` when present (instant); otherwise
    fits live. Regenerate with ``python -m fingerprints.accumulation``.
    """
    if has_data():
        return tuple(SurveyCell(**d) for d in _cache()["survey"])
    return _survey_live()


def _survey_live() -> tuple[SurveyCell, ...]:
    """binary vs count for every classical fingerprint on all survey targets.

    ADMET targets (heavy-atom count, solubility) share the AqSolDB molecule set
    and its scaffold split; the binding target uses its own MoleculeACE set and
    split. Every fit is a fairly-tuned RidgeCV on standardised features.
    """
    mols_s, _, ys_s = _dataset()
    tr_s, te_s = _scaffold_split()
    hac = np.array([m.GetNumHeavyAtoms() for m in mols_s], dtype=float)
    mols_b, ys_b = _binding_dataset()
    tr_b, te_b = _scaffold_masks(mols_b)
    bind_label = _BINDING_ENDPOINT[0]

    admet_targets = [("heavy-atom count", hac), ("aqueous solubility", ys_s)]
    out: list[SurveyCell] = []
    for fp_label, gen in _SURVEY_GENERATORS.items():
        for encoding, count in [("binary", False), ("count", True)]:
            xs = _fp_matrix(mols_s, gen, count=count)
            for tgt, y in admet_targets:
                out.append(
                    SurveyCell(
                        fp_label, encoding, tgt, _survey_ridge_r2(xs, y, tr_s, te_s)
                    )
                )
            xb = _fp_matrix(mols_b, gen, count=count)
            out.append(
                SurveyCell(
                    fp_label,
                    encoding,
                    bind_label,
                    _survey_ridge_r2(xb, ys_b, tr_b, te_b),
                )
            )
    return tuple(out)


def count_minus_binary(target: str) -> list[tuple[str, float]]:
    """Per-fingerprint count-minus-binary R2 delta on ``target``.

    Positive means counting helps; negative means counting hurts.
    """
    by = {(c.fingerprint, c.encoding): c.r2 for c in survey() if c.target == target}
    deltas = []
    for fp in survey_fingerprints():
        b = by.get((fp, "binary"))
        c = by.get((fp, "count"))
        if b is not None and c is not None:
            deltas.append((fp, c - b))
    return deltas


# ---------------------------------------------------------------------------
# Repeated scaffold cross-validation (5 repeats × 5 folds = 25 paired splits)
# for the headline encodings, plus Tukey-HSD / ANOVA statistics over them.
# A single train/test split gives one number per model with no error bar; the
# 5×5 CV gives a *distribution* per model on identical splits, so we can ask
# whether two encodings genuinely differ or just got a lucky split.
# ---------------------------------------------------------------------------

CV_REPEATS = 5
CV_FOLDS = 5

# (key, label, family) for each encoding/head in the CV comparison — same
# models as the single-split headline.
CV_METHODS: tuple[tuple[str, str, str], ...] = (
    ("binary_linear", "binary Morgan + linear", "binary"),
    ("count_linear", "count Morgan + linear", "count"),
    ("binary_mlp", "binary Morgan + deep MLP (48×4)", "binary"),
    ("chemeleon_linear", "CheMeleon (learned) + linear", "learned"),
)


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


def _fit_split(xb, xc, xche, y, tr, te) -> dict[str, tuple[float, float]]:
    """(R², RMSE) for every CV method on one train/test split."""
    yt = y[te]
    return {
        k: (_r2(yt, p), float(np.sqrt(np.mean((yt - p) ** 2))))
        for k, p in _predict_split(xb, xc, xche, y, tr, te).items()
    }


def _predict_split(xb, xc, xche, y, tr, te) -> dict[str, np.ndarray]:
    """Held-out predictions on ``te`` for every CV method."""
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    m = RidgeCV(alphas=_ALPHAS).fit(xb[tr], y[tr])
    out["binary_linear"] = m.predict(xb[te])
    m = RidgeCV(alphas=_ALPHAS).fit(xc[tr], y[tr])
    out["count_linear"] = m.predict(xc[te])
    m = MLPRegressor(
        hidden_layer_sizes=(48,) * 4, max_iter=300, random_state=0,
        early_stopping=True,
    ).fit(xb[tr], y[tr])
    out["binary_mlp"] = m.predict(xb[te])
    if xche is not None:
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=_ALPHAS)).fit(
            xche[tr], y[tr]
        )
        out["chemeleon_linear"] = m.predict(xche[te])
    return out


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
    targets = {t: v["y"] for t, v in _targets().items()}

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
        "targets": {},
    }
    for t in targets:
        payload["targets"][t] = {"r2": {}, "rmse": {}}
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


@dataclass(frozen=True)
class TukeyRow:
    key: str
    label: str
    family: str
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
                key, meta[key]["label"], meta[key]["family"], float(means[i]),
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
# exactly as in the ADMET census: binary-Morgan Tanimoto ≥ 0.7 and
# |ΔlogS| > 1.5. Using one fixed pair definition means every model is judged
# on the very same cliffs.
# ---------------------------------------------------------------------------

CLIFF_SIM = 0.7
CLIFF_GAP = 1.5
FLAT_GAP = 0.5
CLIFF_TARGET = "aqueous solubility"


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
    y = _targets()[CLIFF_TARGET]["y"]
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
    y = _targets()[CLIFF_TARGET]["y"]
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
      pairs vs flat (|Δy| < 0.5) similar pairs.
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
        "n_similar_pairs": int(len(gap)),
        "n_cliff_mols": int(in_cliff.sum()),
        "n_mols": int(len(y)),
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
# Disk cache: all live fits (scores over every target + the survey grid) are
# precomputed and shipped under data/ so the notebook loads instantly. Rebuild
# with ``python -m fingerprints.accumulation``.
# ---------------------------------------------------------------------------


def has_data() -> bool:
    return paths.ACCUMULATION.exists()


@lru_cache(maxsize=1)
def _cache() -> dict:
    import json

    return json.loads(paths.ACCUMULATION.read_text())


def main_cv() -> None:
    """(Re)compute only the 5×5 CV block and merge it into the JSON cache."""
    import json

    payload = json.loads(paths.ACCUMULATION.read_text()) if has_data() else {}
    payload["cv"] = _cv_live()
    payload["cliff_oof"] = _cliff_oof_live()
    paths.ACCUMULATION.parent.mkdir(parents=True, exist_ok=True)
    paths.ACCUMULATION.write_text(json.dumps(payload, indent=2))
    _cache.cache_clear()
    print(f"wrote 5x5 CV -> {paths.ACCUMULATION}")


def main_cliffs() -> None:
    """(Re)compute only the AqSolDB cliff out-of-fold block."""
    import json

    payload = json.loads(paths.ACCUMULATION.read_text()) if has_data() else {}
    payload["cliff_oof"] = _cliff_oof_live()
    paths.ACCUMULATION.parent.mkdir(parents=True, exist_ok=True)
    paths.ACCUMULATION.write_text(json.dumps(payload))
    print(f"wrote cliff OOF -> {paths.ACCUMULATION}")


def main() -> None:
    """Fit everything live and write the JSON cache read by :func:`scores`,
    :func:`survey`, :func:`cv_results`, :func:`n_molecules`, and
    :func:`survey_n`."""
    import json
    from dataclasses import asdict

    payload = {
        "n_molecules": len(_dataset()[0]),
        "survey_n": {
            "accumulator": len(_dataset()[0]),
            "binding": len(_binding_dataset()[0]),
        },
        "scores": {
            target: [asdict(s) for s in _scores_live(target)]
            for target in target_labels()
        },
        "survey": [asdict(c) for c in _survey_live()],
        "cv": _cv_live(),
        "cliff_oof": _cliff_oof_live(),
    }
    paths.ACCUMULATION.parent.mkdir(parents=True, exist_ok=True)
    paths.ACCUMULATION.write_text(json.dumps(payload, indent=2))
    n = sum(len(v) for v in payload["scores"].values()) + len(payload["survey"])
    print(f"wrote {n} fits -> {paths.ACCUMULATION}")


if __name__ == "__main__":
    import sys

    if "--cliffs-only" in sys.argv:
        main_cliffs()
    elif "--cv-only" in sys.argv:
        main_cv()
    else:
        main()
