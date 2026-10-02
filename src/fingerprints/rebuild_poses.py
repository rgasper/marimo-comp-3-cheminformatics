"""Offline: fold context-cliff ligands into both receptors with the Boltz API.

The activity-cliff section's payoff is a 3D "why": the same small change is a
potency cliff on one receptor and flat on a related one. To *show* that, we
co-fold each of a cliff pair's two ligands into *both* receptors (4 predicted
complexes) and look at where the changed atoms land in each pocket.

This calls the hosted Boltz-2 API (needs BOLTZ_API_KEY) and is slow + costs
credits, so it runs **offline**, once. It caches each predicted complex (mmCIF)
plus a small metadata JSON under ``data/boltz_poses/`` for the notebook to load
instantly. Predicted poses are hypotheses, not measurements - the notebook
labels them as such.

Usage:
    export BOLTZ_API_KEY=...   # or put it in .env
    uv run python -m fingerprints.rebuild_poses            # default pair
    uv run python -m fingerprints.rebuild_poses --pair mu_vs_kappa --index 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from loguru import logger

from fingerprints import paths
from fingerprints.data.context_cliffs import TARGET_PAIRS, by_key

# UniProt accessions for the receptors used in the curated target pairs.
UNIPROT = {
    "Dopamine D3": "P35462",
    "Dopamine D4": "P21917",
    "mu-opioid": "P35372",
    "kappa-opioid": "P41145",
}

OUT_DIR = paths.BOLTZ_POSES


def receptor_sequence(target_label: str) -> str:
    up = UNIPROT[target_label]
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{up}.fasta", timeout=60)
    r.raise_for_status()
    return "".join(r.text.splitlines()[1:])


def fold_one(client, seq: str, ligand_smiles: str, name: str, root_dir: Path) -> Path:
    """Submit one protein+ligand co-fold, wait, and return the output dir Path.

    ``.run()`` handles submit -> poll -> download -> extract and returns the
    directory containing ``outputs/files/prediction/*_predicted.cif`` and
    ``metrics.json``.
    """
    return client.predictions.structure_and_binding.run(
        input={
            "entities": [
                {"type": "protein", "chain_ids": ["A"], "value": seq},
                {"type": "ligand_smiles", "chain_ids": ["L"], "value": ligand_smiles},
            ],
            "binding": {
                "type": "ligand_protein_binding",
                "binder_chain_id": "L",
            },
            "num_samples": 1,
        },
        model="boltz-2.1",
        root_dir=str(root_dir),
        name=name,
    )


def _extract_from_output(out_dir: Path, cif_dest: Path) -> dict:
    """Copy the predicted complex CIF out of a .run() output dir and read metrics.

    ``.run()`` extracts to ``<out_dir>/outputs/files/prediction/`` containing
    ``*_predicted.cif`` and ``metrics.json``.
    """
    pred_dir = out_dir / "outputs" / "files" / "prediction"
    cifs = sorted(pred_dir.glob("*_predicted.cif"))
    if not cifs:
        raise FileNotFoundError(f"no *_predicted.cif under {pred_dir}")
    cif_dest.parent.mkdir(parents=True, exist_ok=True)
    cif_dest.write_bytes(cifs[0].read_bytes())
    metrics_path = pred_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return metrics


def fold_cliff(client, pair: str, index: int, force: bool) -> None:
    """Fold one cliff pair (2 ligands x 2 receptors) and write its manifest."""
    tp = by_key()[pair]
    cliff = tp.cliffs[index]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work_root = Path("boltz-experiments")

    seqs = {
        tp.target_a: receptor_sequence(tp.target_a),
        tp.target_b: receptor_sequence(tp.target_b),
    }

    manifest: list[dict] = []
    for mol_id, smiles in [("mol1", cliff.smiles_1), ("mol2", cliff.smiles_2)]:
        for target, seq in seqs.items():
            tag = f"{pair}_{index}_{mol_id}_{target.replace(' ', '')}"
            cif_path = OUT_DIR / f"{tag}.cif"
            meta_path = OUT_DIR / f"{tag}.json"

            # Idempotent: if this fold is already cached, reuse it and don't
            # spend Boltz credits again. Use --force to refold.
            if cif_path.exists() and meta_path.exists() and not force:
                logger.info(f"{tag}: already cached, skipping")
                manifest.append(json.loads(meta_path.read_text()))
                continue

            logger.info(f"folding {tag} ({len(seq)} aa + {smiles})")
            out_dir = fold_one(client, seq, smiles, name=tag, root_dir=work_root)
            metrics = _extract_from_output(Path(out_dir), cif_path)
            binding = metrics.get("binding_metrics", {}) or {}
            best = (metrics.get("best_sample", {}) or {}).get("metrics", {}) or {}
            entry = {
                "tag": tag,
                "pair": pair,
                "index": index,
                "mol_id": mol_id,
                "smiles": smiles,
                "target": target,
                "cif_file": cif_path.name,
                "binding_confidence": binding.get("binding_confidence"),
                "structure_confidence": best.get("structure_confidence"),
                "ligand_iptm": best.get("ligand_iptm"),
            }
            # Cache per-fold metadata too, so a partial re-run can resume.
            meta_path.write_text(json.dumps(entry, indent=2))
            manifest.append(entry)
            logger.info(
                f"  saved {cif_path.name} "
                f"(binding_confidence={entry['binding_confidence']}, "
                f"ligand_iptm={entry['ligand_iptm']})"
            )

    (OUT_DIR / f"{pair}_{index}_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"wrote manifest with {len(manifest)} poses for {pair}[{index}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="mu_vs_kappa", help="target-pair key")
    ap.add_argument("--index", type=int, default=2, help="cliff index within the pair")
    ap.add_argument(
        "--all",
        action="store_true",
        help="fold every cliff of every target pair (ignores --pair/--index)",
    )
    ap.add_argument(
        "--force", action="store_true", help="refold even if cached (re-spends credits)"
    )
    args = ap.parse_args()

    try:
        from boltz_api import Boltz
    except ImportError as e:
        raise SystemExit("pip/uv add boltz-api first") from e

    client = Boltz(base_url="https://api.boltz.bio")  # reads BOLTZ_API_KEY

    if args.all:
        jobs = [
            (tp.key, i)
            for tp in TARGET_PAIRS
            for i in range(len(tp.cliffs))
        ]
        logger.info(f"folding ALL {len(jobs)} cliffs ({len(jobs) * 4} complexes)")
    else:
        jobs = [(args.pair, args.index)]

    for pair, index in jobs:
        fold_cliff(client, pair, index, args.force)


if __name__ == "__main__":
    main()
