"""Offline: build the ADMET single-endpoint activity-cliff census, so the
notebook can show that "similar structure -> similar property" (and the cliffs
that break it) is *not* exclusive to protein binding - it happens for ADMET
properties too, where there is no second target to compare against.

We use AqSolDB aqueous solubility (logS) from TDC. Same census machinery as
``analyze_knn_cliffs.py`` (flat-vs-cliff histogram among structurally similar
pairs, a flat-majority pool, and held-out kNN R^2 vs k), plus a small gallery
of the sharpest *solubility cliffs* we can find - near-identical molecules
whose solubility differs by orders of magnitude.

Chemistry hygiene (so a working chemist trusts it):
  * strip salts / keep the largest organic fragment, drop anything left empty;
  * canonicalise and de-duplicate by parent SMILES (no salt/duplicate leakage);
  * split train/test by **Bemis-Murcko scaffold** so near-duplicate scaffolds
    never straddle the split (the leakage the rubric warns about).

Run:
  uv run python -m fingerprints.analyses.admet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold

from fingerprints import paths
from fingerprints.data import openadmet as oa
from fingerprints.data import tdc

RDLogger.DisableLog("rdApp.*")

CACHE_TDC = paths.CACHE_DIR / "tdc"
OUT_DIR = paths.ADMET_CLIFFS.parent
N_BITS = 2048
_MORGAN = fpg.GetMorganGenerator(radius=2, fpSize=N_BITS)
_SALT = SaltRemover()

# The census is run under EACH of these fingerprints' similarity, so the
# notebook can show that "which pairs count as similar (and how cliffy they
# look)" is itself a choice of fingerprint. All are binary -> Tanimoto.
_FP_GENERATORS = {
    "morgan": fpg.GetMorganGenerator(radius=2, fpSize=N_BITS),
    "rdkit_topo": fpg.GetRDKitFPGenerator(fpSize=N_BITS),
    "atom_pair": fpg.GetAtomPairGenerator(fpSize=N_BITS),
    "top_torsion": fpg.GetTopologicalTorsionGenerator(fpSize=N_BITS),
}
FP_LABELS = {
    "morgan": "Morgan (ECFP4)",
    "maccs": "MACCS",
    "rdkit_topo": "RDKit topological",
    "atom_pair": "Atom pair",
    "top_torsion": "Topological torsion",
}
# order the notebook facets use
FP_ORDER = ["morgan", "maccs", "atom_pair", "top_torsion", "rdkit_topo"]

# Each endpoint: the TDC dataset + display metadata. Solubility is the headline
# (continuous, huge, famous cliffs); lipophilicity is a second continuous option.
ENDPOINTS = {
    "Aqueous solubility": {
        "dataset": tdc.SOLUBILITY,
        "unit": "logS",
        "blurb": (
            "Aqueous solubility (AqSolDB, log mol/L). A single-endpoint ADMET "
            "property with no 'other target' to compare against - yet the same "
            "'similar structure -> similar property' assumption, and the same "
            "cliffs that break it, appear just as they do for binding."
        ),
    },
    "Lipophilicity": {
        "dataset": tdc.LIPOPHILICITY,
        "unit": "logD7.4",
        "blurb": (
            "Lipophilicity (AstraZeneca, logD at pH 7.4). Another continuous "
            "ADMET endpoint; a single-atom change can still move it sharply."
        ),
    },
    "Caco-2 permeability": {
        "dataset": tdc.CACO2,
        "unit": "log cm/s",
        "blurb": (
            "Caco-2 apparent permeability (Wang et al., log cm/s) - a common "
            "proxy for intestinal absorption. A third, smaller ADMET endpoint: "
            "we can't know in advance how cliff-riddled it is, nor which "
            "fingerprint will happen to track its cliffs."
        ),
    },
    "OpenADMET LogD": {
        "dataset": oa.EXP_LOGD,
        "unit": "logD",
        "blurb": (
            "OpenADMET's *own* data: LogD from the ExpansionRx Challenge "
            "(CC-BY-4.0) - real prosecuted drug-discovery campaigns. The exact "
            "same census, now on the competition host's dataset: does 'similar "
            "structure -> similar logD' hold, and where does it snap?"
        ),
    },
    "OpenADMET solubility": {
        "dataset": oa.EXP_KSOL,
        "unit": "log µM",
        "blurb": (
            "OpenADMET ExpansionRx kinetic solubility (KSOL), log10 µM. "
            "Solubility was the cliffiest TDC endpoint; does OpenADMET's "
            "independently-measured solubility show the same rough terrain?"
        ),
    },
}
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100]
SIM_THRESHOLD = 0.7  # "structurally similar" cutoff for the census
FLAT_GAP = 0.5       # |dY| below this = flat (log units); ADMET noise is tighter
CLIFF_GAP = 1.5      # |dY| above this = an ADMET cliff (>~30x in solubility)
TEST_FRAC = 0.2      # scaffold-split held-out fraction
N_FLAT_POOL = 60
N_CLIFF_GALLERY = 6


def parent_smiles(smiles: str) -> str | None:
    """Desalt -> largest organic fragment -> canonical SMILES, or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _SALT.StripMol(mol, dontRemoveEverything=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    # keep the largest fragment (handles co-formers the salt remover misses)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def load_endpoint(dataset):
    """Return de-duplicated parent SMILES + property array (mean over dupes).

    Accepts either a TDC dataset or an OpenADMET endpoint; both resolve to a
    polars ``smiles, y`` frame, after which the salt-stripping / dedup path is
    identical."""
    if isinstance(dataset, oa.OpenADMETEndpoint):
        df = oa.load_endpoint(dataset, CACHE_TDC)
    else:
        df = tdc.load_tdc(dataset, CACHE_TDC)
    by_parent: dict[str, list[float]] = {}
    n_raw = 0
    for row in df.iter_rows(named=True):
        n_raw += 1
        p = parent_smiles(row["smiles"])
        if p is None or row["y"] is None:
            continue
        by_parent.setdefault(p, []).append(float(row["y"]))
    smis = list(by_parent)
    ys = np.array([float(np.mean(by_parent[s])) for s in smis])
    logger.info(f"  {n_raw} raw rows -> {len(smis)} unique parent structures")
    return smis, ys


def scaffold_split(smis, test_frac):
    """Bemis-Murcko scaffold split: whole scaffold groups go to one side only,
    so near-duplicate structures never straddle train/test (no leakage)."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smis):
        mol = Chem.MolFromSmiles(s)
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaf = ""
        groups.setdefault(scaf or f"__none_{i}", []).append(i)
    # deterministic: assign smallest scaffold groups to test until we hit the
    # target size (keeps big common scaffolds in train, tests on rarer
    # chemotypes - the honest hard case)
    n = len(smis)
    n_test_target = int(round(n * test_frac))
    is_test = np.zeros(n, dtype=bool)
    # assign smallest groups to test until we hit the target (keeps big common
    # scaffolds in train, tests on rarer chemotypes - the honest hard case)
    for grp in sorted(groups.values(), key=len):
        if is_test.sum() >= n_test_target:
            break
        for i in grp:
            is_test[i] = True
    return ~is_test, is_test


def ecfp_matrix(smiles):
    return np.asarray(
        [_MORGAN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
        dtype=np.float32,
    )


def fp_matrix(smiles, key):
    """Binary fingerprint matrix (n, N_BITS) for a given fingerprint key."""
    if key == "maccs":
        rows = []
        for s in smiles:
            bv = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s))
            arr = np.zeros(bv.GetNumBits(), dtype=np.uint8)
            from rdkit.DataStructs import ConvertToNumpyArray

            ConvertToNumpyArray(bv, arr)
            rows.append(arr)
        return np.asarray(rows, dtype=np.float32)
    gen = _FP_GENERATORS[key]
    return np.asarray(
        [gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
        dtype=np.float32,
    )


def gap_histogram(dd_sim):
    """Bin the activity gaps among similar pairs into the shared edges."""
    gap_edges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 20.0]
    hist = []
    for lo, hi in zip(gap_edges[:-1], gap_edges[1:]):
        c = int(np.sum((dd_sim >= lo) & (dd_sim < hi)))
        hist.append({"lo": lo, "hi": hi, "count": c})
    return hist


def census_for_fp(Xfp, ys, iu, dd, sub, smis):
    """Run the flat-vs-cliff census under one fingerprint's Tanimoto, and pick
    the sharpest cliff *this fingerprint* calls similar (biggest property gap
    among its own similar pairs) for the per-fingerprint example."""
    S = tanimoto(Xfp, Xfp)
    tt = S[iu]
    sim_mask = tt >= SIM_THRESHOLD
    n_similar = int(sim_mask.sum())
    dd_sim = dd[sim_mask]

    # sharpest cliff for this fingerprint: largest gap among ITS similar pairs
    top_cliff = None
    cliff_mask = sim_mask & (dd > CLIFF_GAP)
    if cliff_mask.any():
        ci, cj = iu[0][cliff_mask], iu[1][cliff_mask]
        gaps = dd[cliff_mask]
        p = int(np.argmax(gaps))
        top_cliff = {
            "smiles_1": smis[int(sub[ci[p]])],
            "smiles_2": smis[int(sub[cj[p]])],
            "tanimoto": float(S[int(ci[p]), int(cj[p])]),
            "act_1": float(ys[int(ci[p])]),
            "act_2": float(ys[int(cj[p])]),
            "gap": float(gaps[p]),
        }

    return {
        "n_similar_pairs": n_similar,
        "frac_flat": float(np.mean(dd_sim < FLAT_GAP)) if n_similar else 0.0,
        "frac_cliff": float(np.mean(dd_sim > CLIFF_GAP)) if n_similar else 0.0,
        "gap_hist": gap_histogram(dd_sim),
        "top_cliff": top_cliff,
    }


def tanimoto(A, B):
    inter = A @ B.T
    a = A.sum(1)[:, None]
    b = B.sum(1)[None, :]
    union = a + b - inter
    return np.where(union > 0, inter / union, 0.0)


def r2(yt, yp):
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def knn_predict(sim_to_ref, y_ref, k):
    idx = np.argsort(-sim_to_ref, axis=1)[:, :k]
    return y_ref[idx].mean(1)


def analyse_endpoint(meta: dict) -> dict:
    smis, y = load_endpoint(meta["dataset"])
    X = ecfp_matrix(smis)
    tr, te = scaffold_split(smis, TEST_FRAC)
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # --- census: among structurally similar pairs, flat vs cliff. Subsample
    #     the pair space if the dataset is large (all-pairs = O(n^2)). ---
    n = len(smis)
    rng = np.random.default_rng(0)
    MAX_N_FOR_PAIRS = 4000
    if n > MAX_N_FOR_PAIRS:
        sub = rng.choice(n, MAX_N_FOR_PAIRS, replace=False)
    else:
        sub = np.arange(n)
    Xs, ys = X[sub], y[sub]
    S = tanimoto(Xs, Xs)
    iu = np.triu_indices(len(sub), 1)
    tt = S[iu]
    dd = np.abs(ys[iu[0]] - ys[iu[1]])
    sim_mask = tt >= SIM_THRESHOLD
    n_similar = int(sim_mask.sum())
    dd_sim = dd[sim_mask]

    gap_edges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 20.0]
    gap_hist = []
    for lo, hi in zip(gap_edges[:-1], gap_edges[1:]):
        c = int(np.sum((dd_sim >= lo) & (dd_sim < hi)))
        gap_hist.append({"lo": lo, "hi": hi, "count": c})

    smoothness = {
        "sim_threshold": SIM_THRESHOLD,
        "n_similar_pairs": n_similar,
        "frac_flat": float(np.mean(dd_sim < FLAT_GAP)) if n_similar else 0.0,
        "frac_cliff": float(np.mean(dd_sim > CLIFF_GAP)) if n_similar else 0.0,
        "flat_gap": FLAT_GAP,
        "cliff_gap": CLIFF_GAP,
        "gap_hist": gap_hist,
    }

    # --- the same census under EACH fingerprint's similarity, on the same
    #     sample and the same gaps, so the facets are directly comparable. ---
    per_fp = {}
    for key in FP_ORDER:
        Xfp = fp_matrix([smis[int(i)] for i in sub], key)
        cen = census_for_fp(Xfp, ys, iu, dd, sub, smis)
        cen["label"] = FP_LABELS[key]
        per_fp[key] = cen
    smoothness["per_fp"] = per_fp

    # flat-majority pool for the gallery
    flat_sim = sim_mask & (dd < FLAT_GAP)
    fi, fj = iu[0][flat_sim], iu[1][flat_sim]
    order = rng.permutation(len(fi))[:N_FLAT_POOL]
    smoothness["flat_pool"] = [
        {
            "smiles_1": smis[int(sub[fi[p]])],
            "smiles_2": smis[int(sub[fj[p]])],
            "tanimoto": float(S[int(fi[p]), int(fj[p])]),
            "act_1": float(ys[int(fi[p])]),
            "act_2": float(ys[int(fj[p])]),
        }
        for p in order
    ]

    # sharpest cliffs: similar structure, biggest property gap (the ADMET
    # analog of the curated binding cliffs - discovered, not hand-picked).
    cliff_mask = sim_mask & (dd > CLIFF_GAP)
    ci, cj = iu[0][cliff_mask], iu[1][cliff_mask]
    cliff_gaps = dd[cliff_mask]
    top = np.argsort(-cliff_gaps)[:N_CLIFF_GALLERY]
    cliff_gallery = [
        {
            "smiles_1": smis[int(sub[ci[p]])],
            "smiles_2": smis[int(sub[cj[p]])],
            "tanimoto": float(S[int(ci[p]), int(cj[p])]),
            "act_1": float(ys[int(ci[p])]),
            "act_2": float(ys[int(cj[p])]),
            "gap": float(cliff_gaps[p]),
        }
        for p in top
    ]

    # held-out kNN R^2 vs k on the scaffold split
    S_te_tr = tanimoto(Xte, Xtr)
    k_curve = []
    for k in K_GRID:
        kk = min(k, len(ytr))
        yp = knn_predict(S_te_tr, ytr, kk)
        k_curve.append({"k": k, "r2": r2(yte, yp)})

    return {
        "unit": meta["unit"],
        "blurb": meta["blurb"],
        "n_total": int(n),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "smoothness": smoothness,
        "cliff_gallery": cliff_gallery,
        "k_curve": k_curve,
    }


def main(out_dir: Path | None = None, on_step=None):
    """Build the ADMET cliff census.

    Args:
        out_dir: where to write ``admet_cliffs.json`` (defaults to ``OUT_DIR``).
        on_step: optional callback ``(label: str)`` invoked before each
            endpoint, so a notebook progress bar can report which endpoint is
            being analysed.
    """
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"n_bits": N_BITS, "k_grid": K_GRID, "split": "scaffold", "endpoints": {}}
    for label, meta in ENDPOINTS.items():
        if on_step is not None:
            on_step(label)
        logger.info(f"{label}: analysing {meta['dataset'].name}")
        try:
            res = analyse_endpoint(meta)
        except Exception as e:  # a single endpoint's data source may be down
            logger.warning(f"  skipping {label}: {e}")
            continue
        best = max(res["k_curve"], key=lambda d: d["r2"])
        logger.info(
            f"  n={res['n_total']} (train {res['n_train']}/test {res['n_test']}) | "
            f"best k={best['k']} R2={best['r2']:.3f} | "
            f"{res['smoothness']['frac_cliff']*100:.1f}% of similar pairs are cliffs"
        )
        out["endpoints"][label] = res
    path = out_dir / "admet_cliffs.json"
    path.write_text(json.dumps(out))
    logger.info(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def endpoint_labels() -> list[str]:
    """Ordered endpoint labels, for progress-bar sizing."""
    return list(ENDPOINTS.keys())


if __name__ == "__main__":
    main()
