"""Offline: analyse *why a similarity model can't see activity cliffs*, using the
simplest model whose entire behaviour IS the fingerprint's notion of similarity:
k-nearest-neighbours regression on ECFP (Tanimoto).

For each endpoint (Dopamine D3, D4; mu-, kappa-opioid) we cache:

  * ``k_curve``: held-out R^2 as a function of k - the real bias/variance
    tradeoff (memorise locally at small k vs. smooth globally at large k). This
    is the honest analog of "move the decision boundary": the *only* knob kNN
    has is neighbourhood size, and it genuinely trades local sensitivity for
    global accuracy.

  * per curated cliff pair (that lives in this endpoint's data):
      - each molecule's nearest neighbours (smiles, Tanimoto, activity) from the
        rest of the dataset - so the notebook can SHOW that the two cliff
        molecules sit in the same fingerprint neighbourhood;
      - each molecule's leave-one-out kNN-predicted activity across a sweep of k,
        alongside its true activity - so the notebook can show the prediction
        tracks the neighbourhood (right for one end of the cliff, wrong for the
        other) no matter what k is chosen.

Why kNN and not a tree / MLP: kNN's prediction is literally the average activity
of the fingerprint-nearest molecules, so the fingerprint's similarity function
*is* the model - nothing is learned on top. Its blindness to the cliff is
therefore the fingerprint's blindness, laid bare, not an artefact of a fancier
learner.

Run:
  uv run python -m fingerprints.analyses.knn
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as fpg

from fingerprints import paths
from fingerprints.data import context_cliffs as ctx
from fingerprints.data import molace

RDLogger.DisableLog("rdApp.*")

CACHE_MOLACE = paths.CACHE_DIR / "molace"
OUT_DIR = paths.KNN_CLIFFS.parent
N_BITS = 2048
_MORGAN = fpg.GetMorganGenerator(radius=2, fpSize=N_BITS)

# We rebuild the held-out kNN k-curve under EACH fingerprint's similarity, so
# the notebook can show that the accuracy-vs-k tradeoff (and where it peaks) is
# itself a choice of fingerprint. All are binary -> Tanimoto.
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
FP_ORDER = ["morgan", "maccs", "atom_pair", "top_torsion", "rdkit_topo"]
# one fixed color per fingerprint, shared by every chart that splits by
# fingerprint (the k-curve lines AND the per-fingerprint cliff scatter).
FP_COLORS = {
    "morgan": "#4c6ef5",
    "maccs": "#e8590c",
    "atom_pair": "#2b8a3e",
    "top_torsion": "#ae3ec9",
    "rdkit_topo": "#f08c00",
}

# endpoint label -> molace dataset file (same mapping the RF importance uses)
ENDPOINTS = {
    "Dopamine D3": "CHEMBL234_Ki",
    "Dopamine D4": "CHEMBL219_Ki",
    "mu-opioid": "CHEMBL233_Ki",
    "kappa-opioid": "CHEMBL237_Ki",
}
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100]
SIM_THRESHOLD = 0.7  # "structurally similar" cutoff for the smoothness stat
FLAT_GAP = 1.0  # |dpKi| below this = a flat (smooth) pair
CLIFF_GAP = 2.0  # |dpKi| above this = an activity cliff


def load_endpoint(dataset: str):
    """canonical smiles -> (activity, split); de-duplicated by canonical SMILES."""
    df = pl.read_csv(CACHE_MOLACE / f"{dataset}.csv")
    smis, ys, splits = [], [], []
    seen = set()
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
        splits.append(row["split"])
    return smis, np.array(ys), np.array(splits)


def ecfp_matrix(smiles):
    return np.asarray(
        [_MORGAN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
        dtype=np.float32,
    )


def fp_matrix(smiles, key):
    """Binary fingerprint matrix (n, N_BITS) for a given fingerprint key."""
    if key == "maccs":
        from rdkit.Chem import MACCSkeys
        from rdkit.DataStructs import ConvertToNumpyArray

        rows = []
        for s in smiles:
            bv = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s))
            arr = np.zeros(bv.GetNumBits(), dtype=np.uint8)
            ConvertToNumpyArray(bv, arr)
            rows.append(arr)
        return np.asarray(rows, dtype=np.float32)
    gen = _FP_GENERATORS[key]
    return np.asarray(
        [gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
        dtype=np.float32,
    )


def tanimoto(A, B):
    """(n,d) x (m,d) binary -> (n,m) Tanimoto similarity."""
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
    """Mean activity of the top-k most-similar reference molecules, per query."""
    idx = np.argsort(-sim_to_ref, axis=1)[:, :k]
    return y_ref[idx].mean(1)


def analyse_endpoint(label: str, dataset: str) -> dict:
    smis, y, split = load_endpoint(dataset)
    X = ecfp_matrix(smis)
    idx_of = {s: i for i, s in enumerate(smis)}
    tr = split == "train"
    te = split == "test"
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # --- smoothness census: among structurally similar pairs, how many are
    #     flat vs cliffs? This is the load-bearing evidence for *why any*
    #     structure-only model must default to 'similar structure -> similar
    #     activity': that assumption holds for the vast majority of similar
    #     pairs, so honouring the rare cliff would wreck accuracy everywhere. ---
    S_all = tanimoto(X, X)
    n = len(smis)
    iu = np.triu_indices(n, 1)
    tt = S_all[iu]
    dd = np.abs(y[iu[0]] - y[iu[1]])
    sim_mask = tt >= SIM_THRESHOLD
    n_similar = int(sim_mask.sum())
    dd_sim = dd[sim_mask]
    # histogram of activity gaps among similar pairs (for the visual)
    gap_edges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 10.0]
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

    # A pool of *flat* similar pairs (near-identical structure, near-identical
    # activity) the notebook can randomly sample - the 85% majority that the
    # 'similar -> similar' assumption gets right, shown next to a click.
    rng = np.random.default_rng(0)
    flat_sim = sim_mask & (dd < FLAT_GAP)
    ii, jj = iu[0][flat_sim], iu[1][flat_sim]
    order = rng.permutation(len(ii))[:60]
    flat_pool = [
        {
            "smiles_1": smis[int(ii[p])],
            "smiles_2": smis[int(jj[p])],
            "tanimoto": float(S_all[int(ii[p]), int(jj[p])]),
            "act_1": float(y[int(ii[p])]),
            "act_2": float(y[int(jj[p])]),
        }
        for p in order
    ]
    smoothness["flat_pool"] = flat_pool

    # --- held-out R^2 vs k (test molecules predicted from train neighbours) ---
    S_te_tr = tanimoto(Xte, Xtr)
    k_curve = []
    for k in K_GRID:
        kk = min(k, len(ytr))
        yp = knn_predict(S_te_tr, ytr, kk)
        k_curve.append({"k": k, "r2": r2(yte, yp)})

    # --- the SAME held-out curve under each fingerprint's similarity, so the
    #     notebook can overlay them: the accuracy/k tradeoff is fingerprint-
    #     dependent, and so is where it peaks. ---
    k_curves_by_fp = {}
    for key in FP_ORDER:
        Xk = fp_matrix(smis, key)
        Sk = tanimoto(Xk[te], Xk[tr])
        curve = []
        for k in K_GRID:
            kk = min(k, int(tr.sum()))
            yp = knn_predict(Sk, ytr, kk)
            curve.append({"k": k, "r2": r2(yte, yp)})
        k_curves_by_fp[key] = {"label": FP_LABELS[key], "k_curve": curve}

    # --- per cliff pair: neighbours + predicted-vs-true across k ---
    # Reference set for a cliff molecule = the whole dataset minus itself (a
    # leave-one-out neighbourhood). This shows where the molecule actually sits.
    # Scan every curated target-pair's cliffs; keep the ones whose molecules
    # actually live in *this* endpoint's dataset. The notebook looks a cliff up
    # by (endpoint == cliff.cliff_on, index == position in its target pair), so
    # we preserve the cliff's original index within its own pair.
    indexed_cliffs = [
        (i, cl) for tp in ctx.by_key().values() for i, cl in enumerate(tp.cliffs)
    ]
    # per-fingerprint similarity matrices over the whole dataset, so each cliff
    # molecule's neighbour cloud + kNN prediction can be shown per fingerprint.
    fp_mats = {key: fp_matrix(smis, key) for key in FP_ORDER}
    pairs_out = []
    for i, cl in indexed_cliffs:
        c1 = Chem.MolToSmiles(Chem.MolFromSmiles(cl.smiles_1))
        c2 = Chem.MolToSmiles(Chem.MolFromSmiles(cl.smiles_2))
        if c1 not in idx_of or c2 not in idx_of:
            continue  # this endpoint doesn't contain the pair

        def molecule_report(self_smi):
            qi = idx_of[self_smi]
            sims = tanimoto(X[qi : qi + 1], X)[0]
            sims[qi] = -1.0  # exclude self
            order = np.argsort(-sims)
            # kNN prediction across the grid (leave-one-out neighbourhood).
            pred_by_k = []
            for k in K_GRID:
                kk = min(k, len(order))
                top = order[:kk]
                pred_by_k.append({"k": k, "pred": float(y[top].mean())})
            # activities of the top max(K_GRID) neighbours (floats only) so the
            # notebook can draw the *cloud* of values kNN averages at any k.
            _kmax = min(max(K_GRID), len(order))
            neighbor_acts = [float(y[j]) for j in order[:_kmax]]

            # the same neighbour cloud + kNN prediction under EACH fingerprint's
            # similarity, so the notebook can break the cliff scatter out per
            # fingerprint (each has its own nearest neighbours -> own cloud).
            by_fp = {}
            for key, Xk in fp_mats.items():
                sk = tanimoto(Xk[qi : qi + 1], Xk)[0]
                sk[qi] = -1.0
                ok = np.argsort(-sk)
                km = min(max(K_GRID), len(ok))
                pk = []
                for k in K_GRID:
                    kk = min(k, len(ok))
                    pk.append({"k": k, "pred": float(y[ok[:kk]].mean())})
                by_fp[key] = {
                    "label": FP_LABELS[key],
                    "neighbor_acts": [float(y[j]) for j in ok[:km]],
                    "pred_by_k": pk,
                }
            return {
                "smiles": self_smi,
                "true": float(y[qi]),
                "neighbor_acts": neighbor_acts,
                "pred_by_k": pred_by_k,
                "by_fp": by_fp,
            }

        pairs_out.append(
            {
                "index": i,
                "cliff_on": cl.cliff_on,
                "change": cl.change,
                "mol1": molecule_report(c1),
                "mol2": molecule_report(c2),
            }
        )

    return {
        "n_total": len(smis),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "smoothness": smoothness,
        "k_curve": k_curve,
        "k_curves_by_fp": k_curves_by_fp,
        "cliff_pairs": pairs_out,
    }


def _ensure_molace(dataset: str):
    """Download a MoleculeACE CSV into CACHE_MOLACE if not present."""
    ds = molace.MolACEDataset(
        name=dataset, target_label=dataset, target_class="", assay_type="Ki"
    )
    molace.download_molace(ds, CACHE_MOLACE / f"{dataset}.csv")


def main(out_dir=None, on_step=None):
    """Build the kNN activity-cliff analysis.

    Args:
        out_dir: where to write ``knn_cliffs.json`` (defaults to ``OUT_DIR``).
        on_step: optional ``(label)`` callback for a progress bar.
    """
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "n_bits": N_BITS,
        "k_grid": K_GRID,
        "fp_order": FP_ORDER,
        "fp_colors": {FP_LABELS[k]: FP_COLORS[k] for k in FP_ORDER},
        "endpoints": {},
    }
    for label, dataset in ENDPOINTS.items():
        if on_step is not None:
            on_step(label)
        _ensure_molace(dataset)
        logger.info(f"{label}: analysing {dataset}")
        res = analyse_endpoint(label, dataset)
        best = max(res["k_curve"], key=lambda d: d["r2"])
        logger.info(
            f"  n={res['n_total']} | best k={best['k']} R2={best['r2']:.3f} | "
            f"{len(res['cliff_pairs'])} cliff pairs present"
        )
        out["endpoints"][label] = res
    path = out_dir / "knn_cliffs.json"
    path.write_text(json.dumps(out))
    logger.info(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def endpoint_labels() -> list[str]:
    return list(ENDPOINTS.keys())


if __name__ == "__main__":
    main()
