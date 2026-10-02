"""Fetch ADME single-prediction datasets from Therapeutics Data Commons (TDC).

TDC datasets are hosted on Harvard Dataverse. We bypass the pytdc package
because its dependency pins (transformers, scikit-learn) conflict with the
versions we need for MIST and our other tooling. Instead, we fetch the few
datasets we care about directly via their dataverse file IDs.

Reference: https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests
from loguru import logger
from tqdm import tqdm
from typeguard import typechecked

from fingerprints.exceptions import DataLoadError

DATAVERSE_BASE = "https://dataverse.harvard.edu/api/access/datafile/"


@dataclass(frozen=True)
class TDCDataset:
    """Metadata for one TDC dataset.

    Args:
        name: short id (used for filenames and titles)
        dataverse_id: numeric id at dataverse.harvard.edu
        property_label: human-readable label for the y axis (e.g. "log S")
        task_type: 'regression' or 'classification'
        positive_label: short label to display for class 1 if classification
        negative_label: short label to display for class 0 if classification
        mirror_url: optional stable mirror serving a ``SMILES,Y`` CSV, used when
            Dataverse blocks direct file access (e.g. Caco-2). Preferred over
            the dataverse download when set.
    """

    name: str
    dataverse_id: int
    property_label: str
    task_type: str  # "regression" | "classification"
    positive_label: str | None = None
    negative_label: str | None = None
    mirror_url: str | None = None


# A small, curated set we'll use across the figures.
LIPOPHILICITY = TDCDataset(
    name="lipophilicity_astrazeneca",
    dataverse_id=4259595,
    property_label="lipophilicity (logD7.4)",
    task_type="regression",
)
SOLUBILITY = TDCDataset(
    name="solubility_aqsoldb",
    dataverse_id=4259610,
    property_label="aqueous solubility (logS)",
    task_type="regression",
)
CACO2 = TDCDataset(
    name="caco2_wang",
    dataverse_id=4259569,
    property_label="Caco-2 permeability (log cm/s)",
    task_type="regression",
    mirror_url=(
        "https://huggingface.co/datasets/scikit-fingerprints/TDC_caco2_wang/"
        "resolve/main/tdc_caco2_wang.csv"
    ),
)
BBB_MARTINS = TDCDataset(
    name="bbb_martins",
    dataverse_id=4259566,
    property_label="BBB penetration",
    task_type="classification",
    positive_label="penetrates",
    negative_label="excluded",
)


@typechecked
def download_tdc(ds: TDCDataset, dest: Path) -> Path:
    """Download a TDC tab-separated dataset to dest if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"{ds.name} already present at {dest}")
        return dest
    url = f"{DATAVERSE_BASE}{ds.dataverse_id}"
    logger.info(f"downloading {ds.name} from {url} -> {dest}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=ds.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except requests.RequestException as e:
        raise DataLoadError(f"failed to download {url}: {e}") from e
    return dest


@typechecked
def load_tdc(ds: TDCDataset, cache_dir: Path) -> pl.DataFrame:
    """Download (if needed) and parse a TDC dataset into a polars DataFrame.

    Returns a DataFrame with columns:
        - smiles: SMILES string
        - y: float property value (regression) or 0/1 (classification)

    The TDC tab-separated layout is:
        Drug_ID \t Drug \t Y
    where Drug is the SMILES string.

    Example:
        >>> df = load_tdc(SOLUBILITY, Path(".cache/tdc"))
        >>> df.head(2)
        shape: (2, 2)
        \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\u2500\u2500\u2500\u2500\u2500\u2510
        \u2502 smiles    \u2502 y     \u2502
        \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524\u2500\u2500\u2500\u2500\u2500\u2524
        \u2502 CCO       \u2502 0.10  \u2502
        \u2502 c1ccccc1  \u2502 -1.20 \u2502
        \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\u2500\u2500\u2500\u2500\u2500\u2518
    """
    if ds.mirror_url is not None:
        # Stable mirror path: a plain ``SMILES,Y`` CSV (used when Dataverse
        # blocks direct file access). Downloaded once, then read from cache.
        path = cache_dir / f"{ds.name}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"downloading {ds.name} from mirror {ds.mirror_url} -> {path}")
            try:
                with requests.get(ds.mirror_url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(path, "wb") as fh:
                        fh.writelines(r.iter_content(chunk_size=1024 * 1024))
            except requests.RequestException as e:
                raise DataLoadError(
                    f"failed to download mirror {ds.mirror_url}: {e}"
                ) from e
        df = pl.read_csv(path)
        cols = df.columns
        if "SMILES" not in cols or "Y" not in cols:
            raise DataLoadError(
                f"unexpected mirror schema for {ds.name}: columns={cols}"
            )
        out = df.select(
            pl.col("SMILES").alias("smiles"),
            pl.col("Y").cast(pl.Float64, strict=False).alias("y"),
        ).drop_nulls()
        logger.info(f"loaded {ds.name} (mirror): {out.height} rows")
        return out

    path = cache_dir / f"{ds.name}.tab"
    download_tdc(ds, path)

    # First line is a header; columns are tab-separated.
    df = pl.read_csv(path, separator="\t")
    # canonical column names per TDC: 'Drug_ID', 'Drug', 'Y'
    cols = df.columns
    if "Drug" not in cols or "Y" not in cols:
        raise DataLoadError(
            f"unexpected TDC schema for {ds.name}: columns={cols}"
        )
    out = df.select(
        pl.col("Drug").alias("smiles"),
        pl.col("Y").cast(pl.Float64, strict=False).alias("y"),
    ).drop_nulls()
    logger.info(f"loaded {ds.name}: {out.height} rows")
    return out
