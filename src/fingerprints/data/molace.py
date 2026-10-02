"""Fetch and cache MoleculeACE benchmark datasets.

MoleculeACE (van Tilborg et al. 2022, J. Chem. Inf. Model. 62, 5938-5951)
is a curated activity-cliff benchmark covering 30 ChEMBL targets. Each
dataset is a CSV with columns:

- smiles: molecule SMILES
- exp_mean [nM]: assay value in nM
- y: log10(nM) used by the original paper for training
- cliff_mol: 1 if the molecule is involved in any cliff pair (Tanimoto
  >= 0.9 to some other molecule AND |Delta(pKi)| >= 1.0)
- split: "train" or "test" (the canonical MoleculeACE split)
- y [pEC50/pKi]: shifted to standard pKi/pEC50, == 9 - log10(nM). This is
  what we use as the regression target.

Datasets are pulled from the project's GitHub repo via raw.githubusercontent
and cached locally. We bypass the MoleculeACE pip package because it pins
old versions of TF/PyTorch/transformers that conflict with our env.

Reference: https://github.com/molML/MoleculeACE
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm
from typeguard import typechecked

from fingerprints.exceptions import DataLoadError

RAW_BASE = (
    "https://raw.githubusercontent.com/molML/MoleculeACE/main/"
    "MoleculeACE/Data/benchmark_data/"
)


@dataclass(frozen=True)
class MolACEDataset:
    """Metadata for one MoleculeACE benchmark dataset.

    Args:
        name: dataset id used in the CSV filename, e.g. "CHEMBL234_Ki".
        target_label: human-readable target name, e.g. "Dopamine D3 receptor".
    """

    name: str
    target_label: str


@typechecked
def download_molace(ds: MolACEDataset, dest: Path) -> Path:
    """Download a MoleculeACE CSV to dest if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"{ds.name} already present at {dest}")
        return dest
    url = f"{RAW_BASE}{ds.name}.csv"
    logger.info(f"downloading {ds.name} from {url} -> {dest}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=ds.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except requests.RequestException as e:
        raise DataLoadError(f"failed to download {url}: {e}") from e
    return dest
