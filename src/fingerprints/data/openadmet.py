"""Fetch OpenADMET challenge datasets (the competition host's own data).

This notebook was built for the OpenADMET x marimo molab competition, so the
ADMET activity-cliff census is run on OpenADMET's *own* released data in
addition to the public TDC benchmarks. The point is to show the "similar
structure -> similar property (and the cliffs that break it)" story on the
exact datasets the challenge is about.

We use the **ExpansionRx Challenge** training data (CC-BY-4.0): real-world
ADMET measurements from prosecuted drug-discovery campaigns by Expansion
Therapeutics, released by OpenADMET on Hugging Face. It carries several
continuous endpoints per molecule; we surface the two that drop straight into
the same continuous-endpoint cliff machinery the TDC section uses:

  * **LogD** (distribution coefficient, already on a log scale);
  * **KSOL** (kinetic aqueous solubility in µM) -> we take log10 so a "gap in
    log units" means the same thing it does for AqSolDB.

Source:
  https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests
from loguru import logger

from fingerprints.exceptions import DataLoadError

# Direct CSV on the OpenADMET Hugging Face dataset repo (CC-BY-4.0).
EXPANSIONRX_TRAIN_CSV = (
    "https://huggingface.co/datasets/openadmet/"
    "openadmet-expansionrx-challenge-train-data/resolve/main/"
    "expansion_data_train.csv"
)


@dataclass(frozen=True)
class OpenADMETEndpoint:
    """One continuous OpenADMET endpoint, mapped onto the census machinery.

    Args:
        name: short id (used for filenames/titles)
        column: the column in the ExpansionRx CSV
        property_label: human-readable y-axis label
        log10: apply log10 to the raw value (for µM solubility -> log units)
    """

    name: str
    column: str
    property_label: str
    log10: bool = False


EXP_LOGD = OpenADMETEndpoint(
    name="openadmet_expansionrx_logd",
    column="LogD",
    property_label="OpenADMET LogD (ExpansionRx)",
)
EXP_KSOL = OpenADMETEndpoint(
    name="openadmet_expansionrx_ksol",
    column="KSOL",
    property_label="OpenADMET solubility (ExpansionRx, log µM)",
    log10=True,
)


def download_expansionrx(dest: Path) -> Path:
    """Download the ExpansionRx training CSV to ``dest`` if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"ExpansionRx data already present at {dest}")
        return dest
    logger.info(f"downloading ExpansionRx train data -> {dest}")
    try:
        with requests.get(EXPANSIONRX_TRAIN_CSV, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                fh.writelines(r.iter_content(chunk_size=1024 * 1024))
    except requests.RequestException as e:
        raise DataLoadError(
            f"failed to download ExpansionRx data {EXPANSIONRX_TRAIN_CSV}: {e}"
        ) from e
    return dest


def load_endpoint(ep: OpenADMETEndpoint, cache_dir: Path) -> pl.DataFrame:
    """Download (if needed) and return a ``smiles, y`` DataFrame for ``ep``.

    Rows missing SMILES or the endpoint value are dropped; if ``ep.log10`` the
    value is log10-transformed (non-positive values dropped first).
    """
    path = cache_dir / "expansion_data_train.csv"
    download_expansionrx(path)
    df = pl.read_csv(path, infer_schema_length=10000)
    if "SMILES" not in df.columns or ep.column not in df.columns:
        raise DataLoadError(
            f"unexpected ExpansionRx schema: columns={df.columns}"
        )
    out = df.select(
        pl.col("SMILES").alias("smiles"),
        pl.col(ep.column).cast(pl.Float64, strict=False).alias("y"),
    ).drop_nulls()
    if ep.log10:
        out = out.filter(pl.col("y") > 0).with_columns(pl.col("y").log10())
    logger.info(f"loaded {ep.name}: {out.height} rows")
    return out
